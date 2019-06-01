import torch

def get_device(device):
    """Return the CPU device or GPU if requested and available."""
    if device.upper() in ("GPU", "CUDA") and torch.cuda.is_available():
        return torch.device("cuda")
    torch.device("cpu")
