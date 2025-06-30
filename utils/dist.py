import torch.distributed as dist

def is_main_process() -> bool:
    """
    Returns True if:
      - torch.distributed is not available or not initialized, i.e. single-process
      - OR if initialized, current process rank == 0
    """
    if not dist.is_available() or not dist.is_initialized():
        return True
    return dist.get_rank() == 0
