# SCRM-DETR

Official implementation of **SCRM-DETR: Scale-Consistent Residual Modulation for Prohibited Item Detection in X-Ray Images**.

> **Status:** repository under preparation. The paper and code are being organized for release.

## Overview

SCRM-DETR is built on RT-DETR for X-ray prohibited-item detection. Its core component, **Scale-Consistent Residual Modulation (SCRM)**, separates local enhancement generation from reliability verification.

SCRM contains two complementary modules:

- **LEM (Local Enhancement Module):** generates a candidate local enhancement residual.
- **SCM (Scale-Consistency Modulation):** uses the current feature, an aligned adjacent-scale reference, and their explicit absolute discrepancy to estimate a soft reliability gate.

The neighboring-scale feature is used as **reliability evidence rather than directly injected feature payload**.

## Method

For a current-scale feature \(X\), LEM first generates an enhanced representation \(X_e\) and candidate residual

\[
\Delta X = X_e - X.
\]

SCM aligns an adjacent-scale reference \(R\) to obtain \(R'\), and models

\[
[X, R', |X-R'|]
\]

to estimate a consistency gate. The residual is then written back through soft modulation:

\[
Y = X + \alpha G_s \odot \Delta X.
\]

For the final HiXray setting,

\[
G_s = 0.5 + 0.5G_c.
\]

SCRM is applied to the **P3 and P4** feature levels of the RT-DETR hybrid encoder.

## Main Results

### HiXray

| Model | Backbone | AP | AP50 | AP75 | APm | APl |
|---|---|---:|---:|---:|---:|---:|
| RT-DETR | R50VD | 49.58 | 83.66 | 53.82 | 29.14 | 50.41 |
| **SCRM-DETR** | R50VD | **50.64** | 83.53 | **55.29** | **30.70** | **50.79** |
| RT-DETR | R34VD | 49.92 | 81.62 | 55.26 | 29.55 | 49.68 |
| **SCRM-DETR** | R34VD | **50.46** | **82.22** | **55.73** | **30.36** | **51.26** |

With R50VD, SCRM-DETR improves the RT-DETR baseline by **+1.07 AP** and **+1.47 AP75**.

### PIDRay

| Model | AP | AP50 | AP75 | APs | APm | APl |
|---|---:|---:|---:|---:|---:|---:|
| RT-DETR | 67.10 | 79.07 | 73.03 | 13.01 | 55.82 | 65.71 |
| **SCRM-DETR** | **67.16** | **79.16** | **73.15** | **14.22** | **56.82** | 65.53 |

## Ablation on HiXray

| Variant | AP | AP50 | AP75 | APm | APl |
|---|---:|---:|---:|---:|---:|
| Baseline RT-DETR | 49.58 | 83.66 | 53.82 | 29.14 | 50.41 |
| LEM only | 48.20 | 82.85 | 50.65 | 29.68 | 48.48 |
| Hard residual gating | 48.55 | 83.51 | 51.28 | 29.58 | 49.07 |
| w/o explicit discrepancy | 49.20 | 83.86 | 52.24 | 30.44 | 49.65 |
| Cross-scale payload | 49.06 | 83.74 | 52.11 | 29.94 | 49.26 |
| **Full SCRM** | **50.64** | 83.53 | **55.29** | **30.70** | **50.79** |

## Complexity

| Model | Params (M) | GFLOPs | AP |
|---|---:|---:|---:|
| RT-DETR | 33.03 | 100.28 | 49.58 |
| SCRM-DETR | 35.91 | 123.35 | 50.64 |

## Repository Structure

```text
SCRM-DETR/
├── configs/       # HiXray/PIDRay experiment configurations
├── src/           # SCRM implementation
├── scripts/       # training and evaluation scripts
├── figures/       # paper figures
├── README.md
├── LICENSE
└── requirements.txt
```

The implementation, configuration files, and reproducibility instructions will be added progressively.

## Datasets

This project uses the public **HiXray** and **PIDRay** X-ray prohibited-item detection benchmarks.

Dataset files are **not** distributed in this repository. Please obtain them from their official sources and follow their respective licenses and terms of use.

## Acknowledgements

This project is built upon the open-source **RT-DETR** implementation. We thank the RT-DETR authors and contributors for their work.

RT-DETR: https://github.com/lyuwenyu/RT-DETR

## License

This repository follows the **Apache License 2.0**. See [LICENSE](LICENSE).

## Citation

Citation information will be added after the paper metadata is finalized.
