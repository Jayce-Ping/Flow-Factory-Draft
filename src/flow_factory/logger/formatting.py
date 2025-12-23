# src/flow_factory/logger/formatting.py
import os
import tempfile
import torch
import numpy as np
from PIL import Image
from typing import Any, Dict, List, Union, Optional
from dataclasses import dataclass, is_dataclass, asdict, field
from ..models.adapter import BaseSample
from ..utils.base import numpy_to_pil_image, tensor_to_pil_image

@dataclass
class LogImage:
    """Intermediate representation for an Image with compression support."""
    _value: Union[str, Image.Image, np.ndarray, torch.Tensor] = field(repr=False)
    _img: Optional[Image.Image] = field(default=None, init=False, repr=False)
    caption: Optional[str] = None
    compress: bool = True
    quality: int = 85
    _temp_path: Optional[str] = field(default=None, init=False, repr=False)
    
    @classmethod
    def to_pil(cls, value: Union[str, Image.Image, np.ndarray, torch.Tensor]) -> Image.Image:
        """Convert various input types to PIL Image."""
        if isinstance(value, Image.Image):
            return value
        elif isinstance(value, torch.Tensor):
            return tensor_to_pil_image(value)[0]
        elif isinstance(value, np.ndarray):
            return numpy_to_pil_image(value)[0]
        elif isinstance(value, str) and os.path.exists(value):
            return Image.open(value).convert('RGB')
        else:
            raise ValueError(f"Unsupported image type: {type(value)}")

    @property
    def value(self) -> Union[str, Image.Image]:
        """Get compressed .jpg file path or original value."""
        if self._temp_path:
            return self._temp_path
            
        # If already a path, return as-is
        if isinstance(self._value, str):
            return self._value
        
        # Convert to PIL Image
        if self._img is None:
            self._img = LogImage.to_pil(self._value)

        # Save to temporary file if compression enabled
        if self.compress:
            # Using mkstemp ensures the file exists and gives us control over closing it
            fd, path = tempfile.mkstemp(suffix='.jpg')
            try:
                with os.fdopen(fd, 'wb') as f:
                    self._img.convert('RGB').save(f, format='JPEG', quality=self.quality)
                self._temp_path = path
            except Exception as e:
                if os.path.exists(path):
                    os.unlink(path)
                raise e
            return self._temp_path

        return self._img
    
    @value.setter
    def value(self, val: Union[str, Image.Image, np.ndarray, torch.Tensor]):
        """Set the value and reset all cached state."""
        self.cleanup()  # Clean up existing temp files before replacing
        self._value = val
        self._img = None
        self._temp_path = None
    
    def cleanup(self):
        """Remove temporary file if created."""
        if self._temp_path and os.path.exists(self._temp_path):
            try:
                os.unlink(self._temp_path)
            finally:
                self._temp_path = None

    def __del__(self):
        """Destructor to prevent storage leaks if cleanup is forgotten."""
        self.cleanup()

    def __exit__(self, exc_type, exc_value, traceback):
        """Context manager exit to ensure cleanup."""
        self.cleanup()

    def __enter__(self):
        """Context manager enter."""
        return self


@dataclass
class LogVideo:
    """Intermediate representation for a Video."""
    _value: Union[str, np.ndarray, torch.Tensor] = field(repr=False)
    caption: Optional[str] = None
    
    def __init__(
        self,
        value: Union[str, np.ndarray, torch.Tensor],
        caption: Optional[str] = None
    ):
        self._value = value
        self.caption = caption
    
    @property
    def value(self) -> Union[str, np.ndarray, torch.Tensor]:
        """Get the original value."""
        return self._value
    
    @value.setter
    def value(self, val: Union[str, np.ndarray, torch.Tensor]):
        """Set the value."""
        self._value = val

class LogFormatter:
    """
    Standardizes input dictionaries for logging.
    Rules:
    1. Strings -> Check path extension -> LogImage/LogVideo
    2. List[Number/Tensor/Array] -> Mean value (float)
    3. PIL Image -> LogImage
    """
    
    IMG_EXTENSIONS = ('.png', '.jpg', '.jpeg', '.bmp', '.tiff', '.webp')
    VID_EXTENSIONS = ('.mp4', '.gif', '.mov', '.avi', '.webm')

    @classmethod
    def format_dict(cls, data: Union[Dict, Any]) -> Dict[str, Any]:
        """Entry point: Converts a Dict or Dataclass (BaseSample) into a clean loggable dict."""
        if is_dataclass(data):
            # Shallow conversion is usually enough, but deep conversion ensures lists are accessible
            data = asdict(data)
            
        if not isinstance(data, dict):
            raise ValueError(f"LogFormatter expects a dict or dataclass, got {type(data)}")

        clean_data = {}
        for k, v in data.items():
            clean_data[k] = cls._process_value(v)
        
        return clean_data

    @classmethod
    def _process_base_sample(cls, samples: List[BaseSample]) -> List[Union[LogImage, LogVideo]]:
        results = []
        for sample in samples:
            caption_parts = []
            if 'reward' in sample.extra_kwargs:
                caption_parts.append(f"{sample.extra_kwargs['reward']:.2f}")
            if sample.prompt:
                caption_parts.append(sample.prompt[:50] + "..." if len(sample.prompt) > 50 else sample.prompt)
            
            final_caption = " | ".join(caption_parts)

            if hasattr(sample, 'image') and sample.image is not None:
                results.append(LogImage(sample.image, caption=final_caption))
            
            # For future extension to videos
            # if hasattr(sample, 'video') and sample.video is not None:
            #     results.append(LogVideo(sample.video, caption=final_caption))

        return results
        
    @classmethod
    def _process_value(cls, value: Any) -> Any:
        """Processes a single value according to the formatting rules."""
        # Rule 0: BaseSample or List of BaseSample
        if isinstance(value, BaseSample):
            value = [value]
        if cls._is_base_sample_collection(value):
            return cls._process_base_sample(value)

        # Rule 1: PIL Image
        if isinstance(value, Image.Image):
            return LogImage(value)

        # Rule 2: String paths
        if isinstance(value, str):
            if os.path.exists(value):
                ext = os.path.splitext(value)[1].lower()
                file_name = os.path.basename(value)
                if ext in cls.IMG_EXTENSIONS:
                    return LogImage(value, caption=file_name)
                if ext in cls.VID_EXTENSIONS:
                    return LogVideo(value, caption=file_name)
            # If string is not a path or file doesn't exist, log as string text
            return value

        # Rule 3: Lists / Arrays / Tensors (Aggregations)
        if cls._is_numerical_collection(value):
            return cls._compute_mean(value)

        # Handle single Tensors/Numpy arrays that aren't images
        if isinstance(value, (torch.Tensor, np.ndarray)):
             if value.ndim == 0 or (value.ndim == 1 and value.shape[0] == 1):
                 return cls._compute_mean(value)

        return value

    @classmethod
    def _is_base_sample_collection(cls, value: Any) -> bool:
        """Checks if value is a list/tuple of BaseSample."""
        if isinstance(value, (list, tuple)):
            if len(value) == 0: return False
            first = value[0]
            return isinstance(first, BaseSample)
        return False

    @classmethod
    def _is_numerical_collection(cls, value: Any) -> bool:
        """Checks if value is a list/tuple of numbers, arrays, or tensors."""
        if isinstance(value, (list, tuple)):
            if len(value) == 0: return False
            first = value[0]
            return isinstance(first, (int, float, complex, np.number, torch.Tensor, np.ndarray))
        return False

    @classmethod
    def _compute_mean(cls, value: Union[List, torch.Tensor, np.ndarray]) -> float:
        """Detaches tensors, converts to float, and computes mean."""
        try:
            # Handle List of Tensors / Arrays
            if isinstance(value, (list, tuple)):
                if isinstance(value[0], torch.Tensor):
                    # Stack and mean
                    return torch.stack([v.detach().cpu().float() for v in value]).mean().item()
                elif isinstance(value[0], (np.ndarray, np.number)):
                    return float(np.mean(value))
                else:
                    # Simple python numbers
                    return float(sum(value) / len(value))
            
            # Handle Direct Tensor
            if isinstance(value, torch.Tensor):
                return value.detach().cpu().float().mean().item()
            
            # Handle Direct Numpy
            if isinstance(value, np.ndarray):
                return float(value.mean())
                
        except Exception as e:
            # Fallback if computation fails
            print(f"Warning: Failed to compute mean for value. Error: {e}")
            return 0.0
            
        return float(value)