# Environment

The experiments reported in the SCRM-DETR paper were conducted with:

- Python 3.10.20
- PyTorch 2.12.0.dev20260407+cu128
- CUDA 12.8

The repository also contains the original dependency specification inherited from the RT-DETR codebase in `requirements.txt`.

Because the exact PyTorch nightly build used in our experiments may not remain available permanently, users may install a compatible PyTorch/CUDA version according to their hardware environment.

Other Python dependencies can be installed with:

```bash
pip install -r requirements.txt
```
