# SCRM-DETR Open-Source Checklist

This repository is being prepared as the reproducible implementation accompanying the SCRM-DETR paper.

## Files to collect from the final training project

### Core implementation
- `src/zoo/rtdetr/hybrid_encoder.py`
- Any additional source files modified specifically for SCRM registration or configuration

### Final configs
- HiXray RT-DETR baseline
- HiXray SCRM-DETR R50VD
- HiXray SCRM-DETR R34VD
- PIDRay RT-DETR baseline
- PIDRay SCRM-DETR R50VD

### Controlled ablations
- LEM only
- Hard residual gating
- w/o explicit discrepancy
- Cross-scale payload

### Reproducibility
- Environment / dependency versions
- Training commands
- Evaluation commands
- Dataset path templates

## Do not upload
- Datasets
- Private server paths or credentials
- Training outputs and logs
- Large checkpoints directly to GitHub
- Temporary debug files
- Cache directories
- Personal IDE settings

## Release goals
A user should be able to clone the repository, prepare HiXray or PIDRay, install dependencies, and reproduce training/evaluation with documented commands.
