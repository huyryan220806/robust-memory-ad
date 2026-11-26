"""Replay the handed-over greedy bank on one raw vial PNG using the author's PatchCore."""
import argparse
import csv
import hashlib
import json
import subprocess
import sys
from pathlib import Path

import faiss
import numpy as np
import torch
import torchvision
from PIL import Image
from torchvision import models, transforms

UPSTREAM = "fcaa92f124fb1ad74a7acf56726decd4b27cbcad"
WEIGHTS_SHA256 = "95faca4d11227dddf8633dbb5ff6c8a9003c1aa5b8945c73834b8007b10950b8"
INDEX_SHA256 = "06b86b5133aaffb915172385628d5c70cb9d6ddd96c059b80bc670f9fbedfd99"


def sha256(path):
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("image", "weights", "index", "predictions", "patchcore_root"):
        parser.add_argument(name, type=Path)
    parser.add_argument("--image-id", default="vial/test_public/bad/000_regular")
    args = parser.parse_args()
    commit = subprocess.check_output(["git", "-C", args.patchcore_root, "rev-parse", "HEAD"], text=True).strip()
    if commit != UPSTREAM or sha256(args.index) != INDEX_SHA256 or sha256(args.weights) != WEIGHTS_SHA256:
        raise ValueError("PatchCore source, index or weights differ from the pilot")
    subprocess.run(["git", "-C", args.patchcore_root, "diff", "--quiet", "HEAD", "--", "src"], check=True)
    sys.path.insert(0, str(args.patchcore_root.resolve() / "src"))
    from patchcore.common import FaissNN
    from patchcore.patchcore import PatchCore

    torch.set_num_threads(4)
    backbone = models.wide_resnet50_2(weights=None)
    backbone.load_state_dict(torch.load(args.weights, map_location="cpu", weights_only=True))
    cpu = torch.device("cpu")
    search = FaissNN(on_gpu=False, num_workers=4)
    model = PatchCore(cpu)
    model.load(backbone, ["layer2", "layer3"], cpu, (3, 224, 224), 1024, 1024, nn_method=search)
    search.load(str(args.index))
    model.eval()
    preprocess = transforms.Compose([
        transforms.Resize((256, 256), interpolation=transforms.InterpolationMode.BILINEAR),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize([.485, .456, .406], [.229, .224, .225]),
    ])
    with Image.open(args.image) as image, torch.inference_mode():
        scores, maps = model._predict(preprocess(image.convert("RGB")).unsqueeze(0))
    score = float(scores[0])
    if maps[0].shape != (224, 224) or not np.isfinite(maps[0]).all():
        raise ValueError("Unexpected anomaly map")
    run_id = "pilot01-greedy-s0"
    with args.predictions.open(encoding="utf-8-sig", newline="") as stream:
        reference = float(next(r["image_score"] for r in csv.DictReader(stream)
                               if r["run_id"] == run_id and r["image_id"] == args.image_id))
    difference = abs(score - reference)
    report = dict(run_id=run_id, image_id=args.image_id, device="cpu", torch=torch.__version__,
                  torchvision=torchvision.__version__, faiss=faiss.__version__, patchcore_commit=commit,
                  image_sha256=sha256(args.image), weights_sha256=WEIGHTS_SHA256, index_sha256=INDEX_SHA256,
                  bank_entries=search.search_index.ntotal, map_shape=list(maps[0].shape),
                  score=score, reference=reference, absolute_difference=difference, absolute_tolerance=.001,
                  passed=difference <= .001)
    print(json.dumps(report, indent=2, allow_nan=False))
    if not report["passed"]:
        raise SystemExit("CPU score differs from the pilot beyond its tolerance")


if __name__ == "__main__":
    main()
