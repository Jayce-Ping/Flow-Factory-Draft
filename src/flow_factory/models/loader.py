# src/flow_factory/models/loader.py
"""
Model Adapter Loader
Factory function using registry pattern for extensibility.
"""
import logging
from typing import Tuple
from accelerate import Accelerator
from .adapter import BaseAdapter
from .registry import get_model_adapter_class, list_registered_models
from ..hparams import Arguments

logging.basicConfig(level=logging.INFO, format='[%(asctime)s] [%(levelname)s] [%(name)s]: %(message)s')
logger = logging.getLogger(__name__)


def load_model(config : Arguments) -> BaseAdapter:
    """
    Factory function to instantiate the correct model adapter based on configuration.
    
    Uses a registry pattern for automatic model discovery and loading.
    Supports both built-in models and custom adapters via python paths.
    
    Args:
        config: Arguments object containing model_args with 'model_type'
    
    Returns:
        An instance of a subclass of BaseAdapter.
    """
    model_args = config.model_args
    model_type = model_args.model_type.lower()
    
    logger.info(f"Loading model architecture: {model_type}...")
    
    if model_type == "flux1":
        from .flux1 import Flux1Adapter
        return Flux1Adapter(config=config)
    elif model_type == 'z-image':
        from .z_image import ZImageAdapter
        return ZImageAdapter(config=config)
    else:
        raise NotImplementedError(f"Model type '{model_type}' is not supported yet.")