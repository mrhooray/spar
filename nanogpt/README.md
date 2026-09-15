# SlimFP8

Across eight fresh seeds on the GCP eight-H100 allocation, SlimFP8 averaged **70.554s** versus **75.772s** for upstream [f411b3d](https://github.com/KellerJordan/modded-nanogpt/commit/f411b3d346aa52d3504324ca93c230fd84c6c07f): **5.218s (6.9%) faster**. The 95% interval for mean paired savings is 5.001–5.435s.

The [upstream record history](https://github.com/KellerJordan/modded-nanogpt#world-record-history) lists its best result as **1.23 minutes (73.8s), dated 07/17/26**, from [PR #342: FP8 MLP down projection](https://github.com/KellerJordan/modded-nanogpt/pull/342).

Mean validation loss is **3.27633301**, with one-sided t-test **p=1.65748e-05** against 3.28, below the [upstream mean-loss criterion](https://github.com/KellerJordan/modded-nanogpt#rules) of p<0.01. All eight SlimFP8 runs reached ≤3.28. Upstream mean loss is 3.27931753 (p=0.278369); these eight upstream runs alone do not establish the mean-loss criterion. Both upstream individual loss misses are retained.

SlimFP8 uses 96-wide Q/K projections, FP8 kernels, and a first-stage batch of 12. It extends SPAR candidate `d3486b9a:bd4bd40b904c` with 60 extension updates after 1,130 scheduled updates (1,190 total). The three Python files are the exact frozen implementation tested here.

The implementation, 16 randomly drawn fresh seeds, sample size, and tests were fixed before the original confirmation. This report includes all eight pairs assigned to GCP; GCP-only reporting was requested after completion. These subset statistics are not a separate predeclared confirmation. Each seed ran upstream then SlimFP8 on the same sandbox and GPU UUIDs. A [final upstream repeat](logs/gcp/controls/upstream-f411b3d-seed-151749878.log) checks timing drift and is excluded from the eight independent samples. Training time excludes setup, compilation/warmup, and validation.

The reported allocation is `2a3517d7`, in GCP `us-east4`, with eight H100 80GB HBM3 GPUs, 32 requested CPUs, and 128 GiB requested memory. Modal assigned the other sandbox to AWS `ap-northeast-1`. It reported an AMD CPU, while the GCP sandbox reported Intel; GPU models, driver, image, and requested resources matched. Upstream ran substantially slower on the AWS allocation, while SlimFP8's timing differed less. One sandbox per provider does not establish a provider-wide effect or its cause. The eight AWS pairs and final upstream repeat are retained in [logs/aws/](logs/aws/) with their [manifest](logs/aws/manifest.json), and excluded from this comparison.

| Seed | Upstream s | Upstream loss | SlimFP8 s | SlimFP8 loss | Gain s | Raw logs |
|---:|---:|---:|---:|---:|---:|---|
| 584036170 | 75.568 | 3.27809978 | 70.648 | 3.27607536 | 4.920 | [upstream](logs/gcp/upstream-f411b3d-seed-584036170.log) · [SlimFP8](logs/gcp/slimfp8-seed-584036170.log) |
| 726063390 | 75.606 | 3.27543926 | 70.535 | 3.27543378 | 5.071 | [upstream](logs/gcp/upstream-f411b3d-seed-726063390.log) · [SlimFP8](logs/gcp/slimfp8-seed-726063390.log) |
| 202547310 | 75.964 | 3.27826405 | 70.348 | 3.27700782 | 5.616 | [upstream](logs/gcp/upstream-f411b3d-seed-202547310.log) · [SlimFP8](logs/gcp/slimfp8-seed-202547310.log) |
| 1339731838 | 75.909 | 3.27939439 | 70.545 | 3.27608037 | 5.364 | [upstream](logs/gcp/upstream-f411b3d-seed-1339731838.log) · [SlimFP8](logs/gcp/slimfp8-seed-1339731838.log) |
| 1730575045 | 75.774 | 3.27810574 | 70.493 | 3.27514529 | 5.281 | [upstream](logs/gcp/upstream-f411b3d-seed-1730575045.log) · [SlimFP8](logs/gcp/slimfp8-seed-1730575045.log) |
| 703678413 | 75.538 | 3.28023505 | 70.680 | 3.27543712 | 4.858 | [upstream](logs/gcp/upstream-f411b3d-seed-703678413.log) · [SlimFP8](logs/gcp/slimfp8-seed-703678413.log) |
| 1278925671 | 75.841 | 3.28626180 | 70.646 | 3.27845836 | 5.195 | [upstream](logs/gcp/upstream-f411b3d-seed-1278925671.log) · [SlimFP8](logs/gcp/slimfp8-seed-1278925671.log) |
| 151749878 | 75.973 | 3.27874017 | 70.536 | 3.27702594 | 5.437 | [upstream](logs/gcp/upstream-f411b3d-seed-151749878.log) · [SlimFP8](logs/gcp/slimfp8-seed-151749878.log) |

Unedited GCP logs, the extra control, source/evaluator hashes, hardware, statistical tests, and shutdown verification are recorded in [comparison.json](comparison.json).
