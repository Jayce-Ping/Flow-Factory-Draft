import re
import base64
from io import BytesIO
from typing import List, Union, Optional, Dict
from itertools import permutations, combinations, chain
import math
import hashlib

import torch.distributed as dist
from PIL import Image
import torch
import numpy as np
from accelerate import Accelerator

# ------------------------------------Random Utils---------------------------------------
def create_generator(prompts : List[str], base_seed : int) -> List[torch.Generator]:
    generators = []
    for batch_pos, prompt in enumerate(prompts):
        # Use a stable hash (SHA256), then convert it to an integer seed
        hash_digest = hashlib.sha256(prompt.encode()).digest()
        prompt_hash_int = int.from_bytes(hash_digest[:4], 'big')  # Take the first 4 bytes as part of the seed
        seed = (base_seed + prompt_hash_int) % (2**31) # Ensure the number is within a valid range
        gen = torch.Generator().manual_seed(seed)
        generators.append(gen)
    return generators

# ------------------------------------Combination Utils---------------------------------------

def num_to_base_tuple(num, base, length):
    """
        Convert a `num` to given `base` and pad left with 0 to form a `length`-tuple
    """
    result = np.zeros(length, dtype=int)
    for i in range(length - 1, -1, -1):
        result[i] = num % base
        num //= base
    return tuple(result.tolist())

# -------------------------------------Image Utils-------------------------------------

def hash_pil_image(image: Image.Image) -> str:
    """
        Generate a hash string for a PIL Image.
        Args:
            image (Image.Image): PIL Image object
        Returns:
            str: Hash string of the image
    """
    return hashlib.md5(image.tobytes()).hexdigest()

def pil_image_to_base64(image : Image.Image, format="JPEG") -> str:
    """
        Convert a PIL Image to a base64-encoded string.
        Args:
            image (Image.Image): PIL Image object
            format (str): Image format, e.g., "JPEG", "PNG"
        Returns:
            base64 string of the image
    """
    buffered = BytesIO()
    image.save(buffered, format="JPEG")
    encoded_image_text = base64.b64encode(buffered.getvalue()).decode("utf-8")
    base64_image = f"data:image/{format.lower()};base64,{encoded_image_text}"
    return base64_image

def pil_image_to_tensor(image: Union[Image.Image, List[Image.Image]]) -> torch.Tensor:
    """
        Convert a PIL Image or a list of PIL Images to a torch Tensor.
        Args:
            image (Union[Image.Image, List[Image.Image]]): PIL Image object or list of PIL Image objects
        Returns:
            torch.Tensor: Image tensor of shape (C, H, W) or (N, C, H, W)
    """
    if isinstance(image, Image.Image):
        image = [image]
    
    tensors = []
    for img in image:
        img_array = np.array(img).astype(np.float32) / 255.0  # Normalize to [0, 1]
        if img_array.ndim == 2:  # Grayscale image
            img_array = np.stack([img_array] * 3, axis=-1)  # Convert to RGB by duplicating channels
        elif img_array.shape[2] == 4:  # RGBA image
            img_array = img_array[:, :, :3]  # Discard alpha channel
        img_tensor = torch.from_numpy(img_array).permute(2, 0, 1)  # HWC to CHW
        tensors.append(img_tensor)
    
    return torch.stack(tensors, dim=0) if len(tensors) > 1 else tensors[0]


def tensor_to_pil_image(tensor: torch.Tensor) -> List[Image.Image]:
    """
        Convert a torch Tensor to a list of PIL Images.
        Args:
            tensor (torch.Tensor): Image tensor of shape (C, H, W) or (N, C, H, W)
        Returns:
            images (List[Image.Image]): list of PIL Image objects. If input is (C, H, W), returns a list with one image.
    """
    if len(tensor.shape) == 3:
        tensor = tensor.unsqueeze(0)
    
    images = (tensor * 255).round().clamp(0, 255).to(torch.uint8).cpu().numpy()
    images = images.transpose(0, 2, 3, 1)  # NCHW -> NHWC
    images = [Image.fromarray(image) for image in images]
    images = images
    return images

def numpy_to_pil_image(array: np.ndarray) -> List[Image.Image]:
    """
        Convert a NumPy array to a list of PIL Images.
        Args:
            array (np.ndarray): Image array of shape (C, H, W) or (N, C, H, W)
        Returns:
            images (List[Image.Image]): list of PIL Image objects. If input is (C, H, W), returns a list with one image.
        1. If the input array has shape (C, H, W), it is treated as a single image and converted to (1, C, H, W).
        2. The pixel values are assumed to be in the range [0, 1] or [0, 255]. If the maximum value is less than or equal to 1.0, the values are scaled to [0, 255].
        3. The array is clipped to ensure all values are within [0, 255] and converted to uint8.
    """
    if len(array.shape) == 3:
        array = array[np.newaxis, ...]
    
    # Clip and convert to uint8
    if array.max() <= 1.0:
        array = (array * 255).round()
    array = np.clip(array, 0, 255).astype(np.uint8)

    # Convert from NCHW to NHWC if needed
    if array.shape[1] == 3:  # NCHW format
        array = array.transpose(0, 2, 3, 1)  # NCHW -> NHWC

    images = [Image.fromarray(image) for image in array]
    images = images
    return images


def tensor_list_to_pil_image(tensor_list: List[torch.Tensor]) -> List[Image.Image]:
    """
        Convert a list of torch Tensors to a list of PIL Images.
        Args:
            tensor_list (List[torch.Tensor]): list of image tensors, each of shape (C, H, W) or (1, C, H, W). Each tensor can have different shape but same dimension.
        Returns:
            images (List[Image.Image]): list of PIL Image objects
        Note:
            If the input tensors have different shapes, they will be processed individually.
    """
    if not tensor_list:
        return []

    # If all image tensors have the same shape, stack them directly
    if all(tensor.shape == tensor_list[0].shape for tensor in tensor_list):
        batch = torch.stack([
            t if t.dim() == 3 else t.squeeze(0)
            for t in tensor_list
        ], dim=0)
        # Normalize, to uint8
        batch = (batch * 255).round().clamp(0, 255).to(torch.uint8).cpu().numpy()
        # NCHW -> NHWC
        if batch.shape[1] == 3:
            batch = batch.transpose(0, 2, 3, 1)
        return [Image.fromarray(img) for img in batch]
    else:
        # Process each tensor individually
        images = []
        for t in tensor_list:
            if t.dim() == 4 and t.shape[0] == 1:
                t = t.squeeze(0)
            img = (t * 255).round().clamp(0, 255).to(torch.uint8).cpu().numpy()
            if img.shape[0] == 3:
                img = img.transpose(1, 2, 0)  # CHW -> HWC
            images.append(Image.fromarray(img))
        return images

def numpy_list_to_pil_image(numpy_list: List[np.ndarray]) -> List[Image.Image]:
    """
        Convert a list of NumPy arrays to a list of PIL Images.
        Args:
            numpy_list (List[np.ndarray]): list of image arrays, each of shape (C, H, W) or (1, C, H, W). Each array can have different shape but same dimension.
        Returns:
            images (List[Image.Image]): list of PIL Image objects
        Note:
            If the input arrays have different shapes, they will be processed individually.
    """
    if not numpy_list:
        return []
    # If all image arrays have the same shape, stack them directly
    if all(arr.shape == numpy_list[0].shape for arr in numpy_list):
        batch = np.stack([
            arr if arr.ndim == 3 else arr.squeeze(0)
            for arr in numpy_list
        ], axis=0)
        # Normalize, to uint8
        if batch.max() <= 1.0:
            batch = (batch * 255).round()
        batch = np.clip(batch, 0, 255).astype(np.uint8)
        # NCHW -> NHWC
        if batch.shape[1] == 3:
            batch = batch.transpose(0, 2, 3, 1)
        return [Image.fromarray(img) for img in batch]
    else:
        # Process each array individually
        images = []
        for arr in numpy_list:
            if arr.ndim == 4 and arr.shape[0] == 1:
                arr = arr.squeeze(0)
            if arr.max() <= 1.0:
                arr = (arr * 255).round()
            arr = np.clip(arr, 0, 255).astype(np.uint8)
            if arr.shape[0] == 3:
                arr = arr.transpose(1, 2, 0)  # CHW -> HWC
            images.append(Image.fromarray(arr))
        return images

def divide_latents(latents: torch.Tensor, H: int, W: int, h: int, w: int) -> torch.Tensor:
    """
    Divide latents into sub-latents based on the specified sub-image size (h, w).
    Args:
        latents (torch.Tensor): The input latents tensor of shape (B, seq_len, C).
        H (int): Height of the original image.
        W (int): Width of the original image.
        h (int): Height of each sub-image.
        w (int): Width of each sub-image.

    Returns:
        torch.Tensor: A tensor of sub-latents of shape (B, rows, cols, sub_seq_len, C).
    """
    batch_size, image_seq_len, channels = latents.shape
    assert H % h == 0 and W % w == 0, "H and W must be divisible by h and w respectively."
    
    # Compute downsampling factor
    total_pixels = H * W
    downsampling_factor = total_pixels // image_seq_len

    # Check if downsampling factor is a perfect square
    downsample_ratio = int(math.sqrt(downsampling_factor))
    if downsample_ratio * downsample_ratio != downsampling_factor:
        raise ValueError(f"The downsampling ratio cannot be determined. Image pixels {total_pixels} and sequence length {image_seq_len} do not match.")
    
    # Calculate latent dimensions
    latent_H = H // downsample_ratio
    latent_W = W // downsample_ratio
    latent_h = h // downsample_ratio
    latent_w = w // downsample_ratio
    
    # Match check
    assert latent_H * latent_W == image_seq_len, f"Calculated latent dimensions {latent_H}x{latent_W} do not match sequence length {image_seq_len}"
    
    rows = latent_H // latent_h
    cols = latent_W // latent_w
    
    # Reshape latents to (B, latent_H, latent_W, C)
    latents = latents.view(batch_size, latent_H, latent_W, channels)
    
    # split into sub-grids: (B, rows, latent_h, cols, latent_w, C)
    latents = latents.view(batch_size, rows, latent_h, cols, latent_w, channels)

    # (B, rows, latent_h, cols, latent_w, C) -> (B, rows, cols, latent_h, latent_w, C)
    sub_latents = latents.permute(0, 1, 3, 2, 4, 5).contiguous()

    # (B, rows, cols, latent_h, latent_w, C) -> (B, rows, cols, sub_seq_len, C)
    sub_latents = sub_latents.view(batch_size, rows, cols, latent_h * latent_w, channels)

    return sub_latents


def merge_latents(sub_latents: torch.Tensor, H: int, W: int, h: int, w: int) -> torch.Tensor:
    """
    Merge sub-latents back into the original latents tensor.
    Args:
        sub_latents (torch.Tensor): A tensor of sub-latents of shape (B, rows, cols, sub_seq_len, C).
        H (int): Height of the original image.
        W (int): Width of the original image.
        h (int): Height of each sub-image.
        w (int): Width of each sub-image.
    Returns:
        torch.Tensor: The merged latents tensor of shape (B, seq_len, C).
    """
    batch_size, rows, cols, sub_seq_len, channels = sub_latents.shape
    
    vae_scale_factor = int(math.sqrt(h * w // sub_seq_len))
    # Calculate latent dimensions using the explicit parameters
    latent_h = h // vae_scale_factor
    latent_w = w // vae_scale_factor
    latent_H = H // vae_scale_factor
    latent_W = W // vae_scale_factor
    
    # Verify dimensions match
    assert latent_h * latent_w == sub_seq_len, f"sub_seq_len {sub_seq_len} does not match calculated sub-latent size {latent_h}x{latent_w}"
    assert rows * cols == (latent_H // latent_h) * (latent_W // latent_w), f"Grid size {rows}x{cols} does not match expected grid size"
    
    # Reshape sub_latents to (B, rows, cols, latent_h, latent_w, C)
    sub_latents = sub_latents.view(batch_size, rows, cols, latent_h, latent_w, channels)
    
    # Merge by rearranging dimensions
    # (B, rows, cols, latent_h, latent_w, C) -> (B, rows, latent_h, cols, latent_w, C)
    merged = sub_latents.permute(0, 1, 3, 2, 4, 5).contiguous()
    
    # Reshape to (B, latent_H, latent_W, C)
    merged = merged.view(batch_size, latent_H, latent_W, channels)
    
    # Final reshape to (B, seq_len, C)
    merged = merged.view(batch_size, latent_H * latent_W, channels)
    
    return merged


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


# -----------------------------------Tensor Utils---------------------------------------

def to_broadcast_tensor(value : Union[int, float, List[int], List[float], torch.Tensor], ref_tensor : torch.Tensor) -> torch.Tensor:
    """
    Convert a scalar, list, or tensor to a tensor that can be broadcasted with ref_tensor.
    The returned tensor will have shape (batch_size, 1, 1, ..., 1) where batch_size is the first dimension of ref_tensor,
    and the number of trailing singleton dimensions is equal to the number of dimensions in ref_tensor minus one.
    """
    # Convert to tensor if not already a tensor
    if not isinstance(value, torch.Tensor):
        value = torch.tensor(value if isinstance(value, list) else [value])

    # Move to the correct device and data type
    value = value.to(device=ref_tensor.device, dtype=ref_tensor.dtype)

    # If scalar, expand to batch size
    if value.numel() == 1:
        value = value.expand(ref_tensor.shape[0])

    # Adjust shape for broadcasting
    return value.view(-1, *([1] * (len(ref_tensor.shape) - 1)))
