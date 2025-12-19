from typing import List, Optional, Union
import torch
from torch import nn
import torch.distributed as dist
from accelerate import Accelerator

def is_model_sharded_by_deepspeed(model) -> bool:
    for param in model.parameters():
        if hasattr(param, 'ds_status'):
            return True
        break
    
    for param in model.parameters():
        if param.numel() == 0 or hasattr(param, 'ds_tensor'):
            return True
        break
    
    return False
    
def is_deepspeed_stage3(accelerator : Accelerator) -> bool:
    try:
        return accelerator.state.deepspeed_plugin and accelerator.state.deepspeed_plugin.zero_stage == 3
    except:
        return False

# -----------------------------------Tensor Gathering Utils---------------------------------------
def all_gather_tensor_list(
        accelerator: Accelerator,
        tensor_list: List[torch.Tensor],
        dtype: Optional[torch.dtype]=None,
        device: Union[str, torch.device]=torch.device("cpu")
    ) -> List[torch.Tensor]:
    """
    Gather a list of tensors from all processes, each process has a list of tensors.
    Each tensor can have a different shape (e.g., (C, H, W)).

    Args:
        accelerator (`Accelerator`): Accelerator object
        tensor_list (`List[torch.Tensor]`): list of tensors to gather, each tensor can have different shape but same dimension,  for example, [(3, 64, 64), (3, 128, 128), ...]. Each list can have different length on different processes.
        dtype (`torch.dtype`, *optional*): dtype of the gathered tensors, if None, use the dtype of the first tensor in tensor_list
        device (`Union[str, torch.device]`, *optional*, defaults to `torch.device("cpu")`): device of the gathered tensors

    Returns:
        gathered_tensors (`List[torch.Tensor]`): tensors from all processes, concatenated in rank order
    """
    if not tensor_list:
        return []
    
    assert all(isinstance(t, torch.Tensor) for t in tensor_list), "All elements in tensor_list must be torch.Tensor"
    assert all(t.dim() == tensor_list[0].dim() for t in tensor_list), "All tensors must have the same number of dimensions"

    tensor_dim = tensor_list[0].dim()
    tensor_dtype = tensor_list[0].dtype if dtype is None else dtype
    device = torch.device(device)

    # Step 1: Gather lengths of tensor_list from all ranks
    local_length = torch.tensor([len(tensor_list)], device=accelerator.device, dtype=torch.long)
    gathered_lengths = [torch.zeros(1, dtype=torch.long, device=accelerator.device) for _ in range(accelerator.num_processes)]
    dist.all_gather(gathered_lengths, local_length)
    gathered_lengths = [int(length.item()) for length in gathered_lengths]

    # Step 2: Gather shapes of each tensor in tensor_list from all ranks
    local_shapes = torch.tensor([list(t.shape) for t in tensor_list], device=accelerator.device, dtype=torch.long)
    gathered_shapes = [
        torch.zeros((length, tensor_dim), dtype=torch.long, device=accelerator.device)
        for length in gathered_lengths
    ]
    dist.all_gather(gathered_shapes, local_shapes)
    gathered_shapes = [shapes.cpu() for shapes in gathered_shapes]  # Move to CPU to save some GPU memory

    # Compute the total length of flattened tensors for each rank, [rank0_total_length, rank1_total_length, ...]
    flat_lengths = [
        sum(int(shape.prod().item()) for shape in this_rank_shapes)
        for this_rank_shapes in gathered_shapes
    ]

    # Step 3: Gather all tensors by flattening and concatenating
    local_flat_tensor = torch.cat([t.flatten() for t in tensor_list], dim=0).to(device=accelerator.device, dtype=tensor_dtype)
    gathered_flat_tensors = [
        torch.zeros(length, dtype=tensor_dtype, device=accelerator.device)
        for length in flat_lengths
    ]
    dist.all_gather(gathered_flat_tensors, local_flat_tensor)
    gathered_flat_tensors = [t.cpu() for t in gathered_flat_tensors]  # Move to CPU to save some GPU memory

    # Step 4: Reconstruct the original tensors from gathered shapes and flattened tensors
    gathered_tensors = []
    for rank, (this_rank_shapes, this_rank_flat_tensor) in enumerate(zip(gathered_shapes, gathered_flat_tensors)):
        offset = 0
        for shape in this_rank_shapes:
            length = int(shape.prod().item())
            # Reshape and move to the specified device
            this_tensor = this_rank_flat_tensor[offset:offset+length].reshape(shape.tolist()).to(device)
            gathered_tensors.append(this_tensor)
            offset += length

    return gathered_tensors