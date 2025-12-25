# src/flow_factory/models/__init__.py
"""
Model Adapters Module

Provides model adapters for different diffusion/flow-matching architectures
with a registry-based loading system for easy extensibility.
"""

from .adapter import BaseAdapter, BaseSample
from .registry import (
    get_model_adapter_class,
    list_registered_models,
)
from .loader import load_model

__all__ = [
    # Core classes
    "BaseAdapter",
    "BaseSample",
    
    # Registry functions
    "get_model_adapter_class",
    "list_registered_models",
    
    # Factory function
    "load_model",
]