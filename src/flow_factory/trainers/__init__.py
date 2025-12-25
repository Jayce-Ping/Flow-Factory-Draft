# src/flow_factory/trainers/__init__.py
"""
Trainers module for various RL algorithms.
"""
from .trainer import BaseTrainer
from .registry import get_trainer_class, list_registered_trainers
from .loader import load_trainer

# Built-in Trainers
# from .grpo import GRPOTrainer

__all__ = [
    'BaseTrainer',
    'get_trainer_class',
    'list_registered_trainers',
    'load_trainer',
]