# src/flow_factory/model/loader.py
import logging
from typing import Tuple
from accelerate import Accelerator
from .adapter import BaseAdapter
from .flux1 import Flux1Adapter
from ..hparams import *

logging.basicConfig(level=logging.INFO, format='[%(asctime)s] [%(levelname)s] [%(name)s]: %(message)s')
logger = logging.getLogger(__name__)


def load_model(config : Arguments) -> BaseAdapter:
    """
    Factory function to instantiate the correct model adapter based on configuration.
    
    Args:
        model_args: DataClass containing 'model_type', 'model_name_or_path', etc.
        training_args: DataClass containing bf16/fp16 settings.
    
    Returns:
        An instance of a subclass of BaseAdapter.
    """
    model_args = config.model_args
    model_type = model_args.model_type.lower()
    
    logger.info(f"Loading model architecture: {model_type}...")
    
    if model_type == "flux1":
        return Flux1Adapter(config=config)
    else:
        raise ValueError(f"Unknown model type: {model_type}. Supported: ['flux', 'sd3', 'sd']")