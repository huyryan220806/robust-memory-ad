# Evaluation methods and pilot results — working draft

26 November 2026 · Prepared for review by Đỗ Hoàng Phúc and coauthors. **Bản nháp tiếng Anh từ pilot đã kiểm; chưa phải phần kết quả xác nhận của paper.** Ngưỡng, định nghĩa lỗi nhỏ và thiết kế thí nghiệm tiếp theo còn chờ nhóm rà. Public test đã được xem trong quá trình phát triển.

## Evaluation methods

We evaluated three PatchCore-based baseline configurations—Full, Random, and Approximate Greedy—on the vial category of MVTec AD 2 with seed 0. The baseline handoff contained 291 normal training images used to construct the memory bank, 41 normal validation images, and 140 public-test images comprising 35 normal and 105 anomalous images. Each of seven lighting groups contained five normal and fifteen anomalous images. The Full bank contained 228,144 patches; Random and Greedy each retained 2,281 patches. PatchCore provides the underlying memory-bank approach; these pilot comparisons do not evaluate a new proposed method. [PatchCore paper](https://arxiv.org/abs/2106.08265), [MVTec AD 2 paper](https://arxiv.org/abs/2503.21622).

Higher scores indicated greater abnormality. Image AUROC was computed from the supplied image scores, recorded to six decimal places. These scores were derived before anomaly-map interpolation and smoothing; they were not recomputed as the maximum of the final maps. For localization, we computed pooled pixel average precision (AP) from the unnormalized maps across all test images, including zero masks for normal images. AP used the average-precision definition implemented in scikit-learn, rather than trapezoidal PR area. Original binary masks were resized to 256 × 256 using nearest-neighbor interpolation and center-cropped to 224 × 224 to match the maps.

For exploratory image-level operating points, we considered calibration targets alpha in {0.01, 0.05, 0.10}. For each run, let the 41 validation scores be sorted in ascending order and let k=floor(41 alpha). The threshold was the next representable float64 value above the score at zero-based index 41−k−1; an image was flagged when its score was greater than or equal to this threshold. This limits empirical validation alarms to at most k, including ties, without guaranteeing the population false-positive rate (FPR). Each threshold was held fixed across test lighting groups.

We also explored small-defect image recall. Eight-connected components were extracted from original-resolution masks. An anomalous image qualified only when every component occupied at most a specified fraction of the entire original image area. We considered cutoffs of 0.1%, 0.5%, and 1%. This metric measures image-level detection among eligible images, not localization recall of individual regions. Empty eligible sets were reported as unavailable. Operating points and size cutoffs remain exploratory and are not claimed to have been preregistered.

## Pilot results

| Configuration | Image AUROC | Pixel AP | Test FP / normal | Detected / anomalous | Detected / small-only, 0.5% |
|---|---:|---:|---:|---:|---:|
| Full | 0.887891 | 0.341058 | 0/35 | 48/105 | 8/21 |
| Random | 0.751837 | 0.339207 | 0/35 | 41/105 | 5/21 |
| Approximate Greedy | 0.891429 | 0.324905 | 2/35 | 54/105 | 7/21 |

The three count-based columns use alpha=0.05, for which every configuration flagged 2/41 validation images. Greedy had the highest image AUROC in this pilot, while Full had the highest pixel AP. Greedy's two test false positives occurred in the underexposed group, corresponding to 2/5 normal images in that group and 2/35 across all lighting conditions. Greedy AUROC was 0.9867 under regular lighting and 0.6800 under underexposure; Full achieved 0.8667 under underexposure. These are descriptive observations from small groups, not evidence of statistical significance.

At the 0.1% size cutoff no image qualified. At 1%, 56 images qualified, of which Full, Random, and Greedy detected 29, 26, and 30, respectively. Thus the reported sensitivity depends on the stated eligible set. All 105 anomalous masks retained positive pixels after preprocessing. However, seven masks sharing the bad/014 filename prefix lost 194 of 1,785 resized positive pixels during center cropping. Retaining a nonempty mask does not establish preservation of its full area.

## Limitations and pending evaluation

These results cover one category and one seed. The baseline contributor confirmed that each label-specific filename prefix identifies the same physical specimen and fixed scene under all seven lighting conditions: 5 normal and 15 anomalous specimens are each observed seven times. Consequently, images are repeated observations within specimens; statistical independence between specimens or across splits has not been established from the handoff. Paired comparisons should account for repeated measurements. The public test set has been inspected for development and error analysis and is not an untouched final evaluation set. We have not evaluated the proposed method, additional seeds or categories, hidden-test performance, or end-to-end efficiency. AU-PRO on the pilot has not been computed; synthetic checks exposed implementation differences that require resolution before reporting that metric. Confirmatory comparisons require a recorded protocol and a suitable subsequent evaluation design.

## Traceability for internal review

This table is an internal aid, not proposed manuscript text.

| Claim or subsection | Evidence |
|---|---|
| Dataset, settings and transforms | [Protocol](../evaluation.md), [baseline](../baseline.md), [audit](../../results/evaluation/pilot01/audit.json) |
| AUROC and AP | [pixel_metrics.csv](../../results/evaluation/pilot01/pixel_metrics.csv), [AUROC cross-check](../../results/evaluation/pilot01/anomalib-check.json) |
| Thresholds and count-based results | [calibration.csv](../../results/evaluation/pilot01/calibration.csv), [quality.csv](../../results/evaluation/pilot01/quality.csv) |
| Independent reconstruction of one result row | [Walkthrough](../evaluation-row-walkthrough.md) |
| Crop loss | [mask_audit.csv](../../results/evaluation/pilot01/mask_audit.csv) |
| Two inspected error cases | [Case analysis](../evaluation-case-analysis.md) |
| AU-PRO preparation and unresolved comparison | [Preparation notes](../aupro-preparation.md) |

Before integration, Huy should review scope and confirmatory design; Hữu Phúc should confirm baseline naming and explain the sampling across splits. Phúc should personally verify the numerical narrative. Replace provisional descriptions only after the corresponding decision or new experiment has been recorded.
