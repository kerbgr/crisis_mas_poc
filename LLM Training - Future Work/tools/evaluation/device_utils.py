"""Pick the best available torch device: CUDA (NVIDIA) > MPS (Apple Silicon)
> CPU. Used by scripts that were originally hardcoded to `.to("cuda")` --
this keeps NVIDIA as the preferred backend where present while actually
using the GPU on Apple Silicon (M1-M4) instead of silently falling back
to CPU.
"""

import torch


def get_device():
    if torch.cuda.is_available():
        return "cuda"
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"
