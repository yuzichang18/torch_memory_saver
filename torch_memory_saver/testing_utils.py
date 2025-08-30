"""Not to be used by end users, but only for tests of the package itself."""

import torch

def get_and_print_gpu_memory(message, gpu_id=0):
    """
    Print GPU/NPU physical memory usage with optional message.
    On NPU, uses mem_get_info to get physical memory used (like torch.cuda.device_memory_used).
    """
    if torch.npu.is_available():
        free, save  = torch.npu.mem_get_info(gpu_id)
        mem = save  - free
    else:
        mem = torch.cuda.device_memory_used(gpu_id)
    
    print(f"GPU {gpu_id} memory: {mem / 1024 ** 3:.2f} GB ({message})")
    return mem
