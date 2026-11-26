"""Reconcile the vial pilot from original CSVs and the three pinned team branches."""
import argparse
import csv
import hashlib
import io
import json
import math
import statistics
import subprocess
from pathlib import Path

BASE = "6d7e8d8f0127838aae8f9c2dd3f2c85a87b94141"
EVAL = "56b0221824c457aa748d2e10638c8e11581f82f8"
PROF = "15ab4dac972d60885aa6ec229d68727d52523bfe"
ROOT = Path(__file__).resolve().parents[1]
RESULTS = "results/evaluation/pilot01/"
SMOKE = "docs/evidence/smoke-cpu-20261125/"


def blob(commit, name):
    data = subprocess.check_output(["git", "show", f"{commit}:{name}"], cwd=ROOT)
    return data.decode("utf-8-sig")


def rows(text):
    return list(csv.DictReader(io.StringIO(text)))


def original_csv(path, expected_hash):
    data = path.read_bytes()
    if hashlib.sha256(data).hexdigest() != expected_hash:
        raise ValueError(f"Input differs from the handed-over CSV: {path.name}")
    return rows(data.decode("utf-8-sig"))


def specimen_key(image_id):
    _, _, label, filename = image_id.split("/")
    prefix, lighting = filename.split("_", 1)
    return (label, prefix), lighting


def paired_alerts(groups, scores, threshold):
    # Keys are two successive alert bits (regular, underexposed).
    result = {label: dict.fromkeys(("00", "01", "10", "11"), 0) for label in ("good", "bad")}
    for (label, _), views in groups.items():
        bits = [int(scores[views[light]] >= threshold) for light in ("regular", "underexposed")]
        result[label][f"{bits[0]}{bits[1]}"] += 1
    return result


def audit_run(run_id, predictions, labels, masks, groups, quality, pixels):
    records = [row for row in predictions if row["run_id"] == run_id]
    scores = {r["image_id"]: float(r["image_score"]) for r in records}
    assert len(records) == len(scores) and scores.keys() == labels.keys()
    assert all(math.isfinite(v) for v in scores.values())
    validation = sorted(v for k, v in scores.items() if labels[k]["split"] == "validation")
    assert len(validation) == 41
    last_accepted = validation[len(validation) - math.floor(.05 * len(validation)) - 1]
    threshold = math.nextafter(last_accepted, math.inf)
    public = {k: v for k, v in scores.items() if labels[k]["split"] == "test_public"}
    positives = [public[k] for k in masks]
    negatives = [v for k, v in public.items() if k not in masks]
    small = [public[k] for k, area in masks.items() if area <= .005]
    counts = []
    for cohort in (negatives, positives, small):
        counts.extend((sum(score >= threshold for score in cohort), len(cohort)))
    auroc = sum((p > n) + .5 * (p == n) for p in positives for n in negatives) / (len(positives) * len(negatives))
    pixel_row, = [r for r in pixels if r["run_id"] == run_id and r["lighting_group"] == "all"]
    quality_row, = [r for r in quality if r["run_id"] == run_id and r["lighting_group"] == "all"
                    and float(r["target_calibration_fpr"]) == .05 and float(r["small_area_cutoff"]) == .005]
    count_columns = ("false_positives", "normal_count", "detected_anomalies", "anomaly_count", "small_detected", "small_count")
    assert counts == [int(quality_row[key]) for key in count_columns]
    assert abs(auroc - float(pixel_row["image_auroc"])) < 1e-12 and threshold == float(quality_row["threshold"])
    config = json.loads(blob(BASE, f"configs/pilot/{run_id}.json"))
    return dict(run_id=run_id, index_bytes=config["bank_storage"]["faiss_index_size_bytes"],
                regular_to_underexposed_alerts=paired_alerts(groups, public, threshold), auroc_recomputed=auroc,
                pixel_ap_from_team=float(pixel_row["pixel_ap"]), threshold=threshold,
                counts_fp_good_tp_bad_tpSmall_small=counts)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("manifest", type=Path)
    parser.add_argument("predictions", type=Path)
    args = parser.parse_args()
    hashes = json.loads(blob(EVAL, RESULTS + "audit.json"))
    manifest = original_csv(args.manifest, hashes["manifest_sha256"])
    predictions = original_csv(args.predictions, hashes["predictions_sha256"])
    assert len(manifest) == len({r["image_id"] for r in manifest}) == 472
    labels = {r["image_id"]: r for r in manifest if r["split"] in ("validation", "test_public")}
    assert len(predictions) == 543 and len(labels) == 181
    groups = {}
    for image_id, row in labels.items():
        if row["split"] == "test_public":
            key, light = specimen_key(image_id)
            groups.setdefault(key, {})[light] = image_id
    assert len(groups) == 20 and all(len(v) == 7 for v in groups.values())
    assert len({frozenset(v) for v in groups.values()}) == 1
    mask_rows = rows(blob(EVAL, RESULTS + "mask_audit.csv"))
    masks = {r["image_id"]: float(r["max_component_area_ratio"]) for r in mask_rows}
    assert len(mask_rows) == len(masks) == 105
    quality = rows(blob(EVAL, RESULTS + "quality.csv"))
    pixels = rows(blob(EVAL, RESULTS + "pixel_metrics.csv"))
    runs = [audit_run(f"pilot01-{mode}-s0", predictions, labels, masks, groups, quality, pixels)
            for mode in ("full", "random", "greedy")]

    timings = [float(r["latency_ms"]) for r in rows(blob(PROF, SMOKE + "timings.csv"))]
    p50 = statistics.median(timings)
    p95 = statistics.quantiles(timings, n=100, method="inclusive")[94]
    reference, = rows(blob(PROF, SMOKE + "efficiency.csv"))
    assert len(timings) == int(reference["n"]) == 50
    assert math.isclose(p50, float(reference["p50_ms"])) and math.isclose(p95, float(reference["p95_ms"]))
    result = dict(input_sha256={k: hashes[k] for k in ("manifest_sha256", "predictions_sha256")}, runs=runs,
                  physical_vials_good_bad=[sum(key[0] == label for key in groups) for label in ("good", "bad")],
                  small_cutoff_0_5pct_unique_bad_vials=sorted({specimen_key(k)[0][1] for k in masks if masks[k] <= .005}),
                  source_commits=dict(baseline=BASE, evaluation=EVAL, profiling=PROF),
                  instrument_only_cpu_ms=dict(p50=p50, p95=p95, n=len(timings)))
    print(json.dumps(result, indent=2, allow_nan=False))


if __name__ == "__main__":
    main()
