# src/flow_factory/utils/base.py
import re
import base64
import inspect
from io import BytesIO
from typing import List, Union, Optional, Dict, Callable, Any
from itertools import permutations, combinations, chain
import math
import hashlib

import torch.distributed as dist
from PIL import Image
import torch
import numpy as np
from accelerate import Accelerator

# ----------------------------------- Type Check --------------------------------------

def is_pil_image_list(image_list: List[Any]) -> bool:
    """
    Check if the input is a list of PIL Images.
    Args:
        image_list (List[Any]): list to check
    Returns:
        bool: True if all elements are PIL Images, False otherwise
    """
    return isinstance(image_list, list) and all(isinstance(img, Image.Image) for img in image_list)

def is_pil_image_batch_list(image_batch_list: List[List[Any]]) -> bool:
    """
    Check if the input is a list of lists of PIL Images.
    Args:
        image_batch_list (List[List[Any]]): list of lists to check
    Returns:
        bool: True if all elements in all sublists are PIL Images, False otherwise
    """
    return isinstance(image_batch_list, list) and all(is_pil_image_list(batch) for batch in image_batch_list)

# ------------------------------------Function Utils-------------------------------------

def filter_kwargs(func: Callable, **kwargs: Any) -> dict[str, Any]:
    """
    Filter kwargs to only include parameters accepted by func.
    
    Args:
        func: Target function
        **kwargs: Keyword arguments to filter
    
    Returns:
        Filtered kwargs containing only valid parameters
    """
    sig = inspect.signature(func)
    
    # Check if function accepts **kwargs
    has_var_keyword = any(
        p.kind == inspect.Parameter.VAR_KEYWORD 
        for p in sig.parameters.values()
    )
    
    # If has **kwargs, accept all
    if has_var_keyword:
        return kwargs
    
    # Otherwise, filter to valid parameter names
    valid_keys = set(sig.parameters.keys())
    return {k: v for k, v in kwargs.items() if k in valid_keys}

def split_kwargs(funcs: list[Callable], **kwargs: Any) -> list[dict[str, Any]]:
    """
    Split kwargs among multiple functions by their signatures.
    Earlier functions have priority for overlapping params.
    
    Returns:
        List of filtered kwargs dicts, one per function
    """
    results = []
    remaining = kwargs.copy()
    
    for func in funcs:
        sig = inspect.signature(func)
        has_var_keyword = any(
            p.kind == inspect.Parameter.VAR_KEYWORD 
            for p in sig.parameters.values()
        )
        
        if has_var_keyword:
            results.append(remaining.copy())
        else:
            valid_keys = set(sig.parameters.keys()) - {'self', 'args', 'kwargs'}
            matched = {k: v for k, v in remaining.items() if k in valid_keys}
            results.append(matched)
            # Remove matched keys so they don't go to later functions
            for k in matched:
                remaining.pop(k, None)
    
    return results

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

# ----------------------------------- Hash Utils --------------------------------------

def hash_pil_image(image: Image.Image, size: Optional[int] = None) -> str:
    """
    Generate a hash string for a PIL Image.
    Args:
        image: PIL Image object
        size: Optional thumbnail size for faster hashing. None uses full image.
    Returns:
        str: MD5 hash hex string
    """
    if size is not None:
        image = image.copy()
        image.thumbnail((size, size))
    return hashlib.md5(image.tobytes()).hexdigest()

def hash_tensor(tensor: torch.Tensor, max_elements: int = 1024) -> str:
    """
    Generate a hash string for a torch Tensor.
    Args:
        tensor: Input tensor
        max_elements: Max elements to hash (for efficiency)
    Returns:
        str: MD5 hash hex string
    """
    flat = tensor.detach().flatten()
    if flat.numel() > max_elements:
        # Sample evenly across tensor
        indices = torch.linspace(0, flat.numel() - 1, max_elements).long()
        flat = flat[indices]
    return hashlib.md5(flat.cpu().numpy().tobytes()).hexdigest()

def hash_pil_image_list(images: List[Image.Image], size: int = 32) -> str:
    """
    Generate a combined hash for a list of PIL Images.
    Args:
        images: List of PIL Image objects
        size: Thumbnail size per image
    Returns:
        str: Combined MD5 hash hex string
    """
    hasher = hashlib.md5()
    for img in images:
        hasher.update(hash_pil_image(img, size=size).encode())
    return hasher.hexdigest()

# -------------------------------------Image Utils-------------------------------------

def is_valid_image(image: Union[Image.Image, torch.Tensor, np.ndarray]) -> bool:
    """
    Check if the input is a valid image type (PIL Image, torch Tensor, or NumPy array).
    Args:
        image: Input image
    Returns:
        bool: True if valid image type:
            - A valid PIL.Image
            - A torch.Tensor with shape (C, H, W) or (B, C, H, W)
            - A np.ndarray with shape (H, W, C) or (B, H, W, C)
    """
    # PIL Image
    if isinstance(image, Image.Image):
        return image.size[0] > 0 and image.size[1] > 0
    
    # Torch Tensor: (C, H, W) or (B, C, H, W)
    if isinstance(image, torch.Tensor):
        if image.ndim == 3:
            c, h, w = image.shape
            return c in (1, 3, 4) and h > 0 and w > 0
        elif image.ndim == 4:
            b, c, h, w = image.shape
            return b > 0 and c in (1, 3, 4) and h > 0 and w > 0
        return False
    
    # NumPy array: (H, W, C) or (B, H, W, C)
    if isinstance(image, np.ndarray):
        if image.ndim == 3:
            h, w, c = image.shape
            return h > 0 and w > 0 and c in (1, 3, 4)
        elif image.ndim == 4:
            b, h, w, c = image.shape
            return b > 0 and h > 0 and w > 0 and c in (1, 3, 4)
        return False
    
    return False

def is_valid_image_list(images: Union[List[Image.Image], List[torch.Tensor], List[np.ndarray]]) -> bool:
    """
    Check if the input is a valid list of images.
    Args:
        images: Input image list
    Returns:
        bool: True if valid image list:
            - A non-empty list
            - All elements are valid images (PIL, Tensor, or ndarray)
            - All elements are of the same type
    """
    if not isinstance(images, list):
        return False
    
    if len(images) == 0:
        return False
    
    # Check type consistency
    first_type = type(images[0])
    if not all(isinstance(img, first_type) for img in images):
        return False
    
    # Check each image is valid
    return all(is_valid_image(img) for img in images)


def is_valid_image_batch(
    images: Union[List[Image.Image], List[torch.Tensor], List[np.ndarray], torch.Tensor, np.ndarray]
) -> bool:
    """
    Check if the input is a valid batch of images.
    Args:
        images: Input image batch
    Returns:
        bool: True if valid image batch:
            - A List[PIL.Image]
            - A List[torch.Tensor] where each tensor is (C, H, W)
            - A List[np.ndarray] where each array is (H, W, C)
            - A torch.Tensor with shape (B, C, H, W)
            - A np.ndarray with shape (B, H, W, C)
    """
    # Case 1: List of images
    if isinstance(images, list):
        return is_valid_image_list(images)
    
    # Case 2: Batched torch.Tensor (B, C, H, W)
    if isinstance(images, torch.Tensor):
        if images.ndim != 4:
            return False
        b, c, h, w = images.shape
        return b > 0 and c in (1, 3, 4) and h > 0 and w > 0
    
    # Case 3: Batched np.ndarray (B, H, W, C)
    if isinstance(images, np.ndarray):
        if images.ndim != 4:
            return False
        b, h, w, c = images.shape
        return b > 0 and h > 0 and w > 0 and c in (1, 3, 4)
    
    return False

def is_valid_image_batch_list(image_batches: List[List[Union[Image.Image, torch.Tensor, np.ndarray]]]) -> bool:
    """
    Check if the input is a valid batch of image lists (List[List[Image]]).
    Args:
        image_batches: Batch of image lists, e.g., [[img1, img2], [img3], [img4, img5, img6]]
    Returns:
        bool: True if valid:
            - Outer list is non-empty
            - Each inner element is either a valid image list or an empty list
    """
    if not isinstance(image_batches, list):
        return False
    
    if len(image_batches) == 0:
        return False
    
    for batch in image_batches:
        if not isinstance(batch, list):
            return False
        # Allow empty lists (some samples may have no images)
        if len(batch) > 0 and not is_valid_image_list(batch):
            return False
    
    return True

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
            torch.Tensor: Image tensor of shape (N, C, H, W), where N is the number of images. N=1 if input is a single image.
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
    
    return torch.stack(tensors, dim=0) # Stack to (N, C, H, W)


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



def is_tensor_list(tensor_list: List[torch.Tensor]) -> bool:
    """
    Check if the input is a list of torch Tensors.
    Args:
        tensor_list (List[torch.Tensor]): list to check
    Returns:
        bool: True if all elements are torch Tensors, False otherwise
    """
    return isinstance(tensor_list, list) and all(isinstance(t, torch.Tensor) for t in tensor_list)