# src/flow_factory/logger/__init__.py
"""
Logging Module

Provides logging backends for experiment tracking with a registry-based
loading system for easy extensibility.

Supported backends:
- WandB (Weights & Biases)
- SwanLab
"""

from .abc import Logger, LogImage, LogVideo
from .registry import (
    get_logger_class,
    list_registered_loggers,
)
from .loader import load_logger

__all__ = [
    # Core classes
    "Logger",
    "LogImage",
    "LogVideo",
    
    # Registry functions
    "get_logger_class",
    "list_registered_loggers",
    
    # Factory function
    "load_logger",
]