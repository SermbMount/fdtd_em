import torch

def assert_on_device(x, device_prefix="cuda", name="tensor"):
    if isinstance(x, torch.Tensor):
        dev = str(x.device)
        assert dev.startswith(device_prefix), f"{name} is on {dev}, expected {device_prefix}*"
    return True

def log_cuda_mem(tag=""):
    if torch.cuda.is_available():
        torch.cuda.synchronize()
        alloc = torch.cuda.memory_allocated() / 1024**2
        reserv = torch.cuda.memory_reserved() / 1024**2
        print(f"[CUDA] {tag} allocated={alloc:.1f}MB reserved={reserv:.1f}MB")
