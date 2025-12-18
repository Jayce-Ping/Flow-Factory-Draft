# src/flow_factory/hparams/args.py
"""
Main arguments class that encapsulates all configurations.
Supports loading from YAML files with nested structure.
"""
from __future__ import annotations
from dataclasses import asdict, dataclass, field
from typing import Any, Literal
import yaml

from .data_args import DataArguments
from .model_args import ModelArguments
from .training_args import TrainingArguments
from .reward_args import RewardArguments


@dataclass
class Arguments:
    """Main arguments class encapsulating all configurations."""
    launcher : Literal['accelerate', 'torchrun'] = field(
        default='accelerate',
        metadata={"help": "Distributed launcher to use."},
    )
    config_path: str | None = field(
        default=None,
        metadata={"help": "Path to distributed configuration file (e.g., deepspeed config)."},
    )
    data_args: DataArguments = field(
        default_factory=DataArguments,
        metadata={"help": "Arguments for data configuration."},
    )
    model_args: ModelArguments = field(
        default_factory=ModelArguments,
        metadata={"help": "Arguments for model configuration."},
    )
    training_args: TrainingArguments = field(
        default_factory=TrainingArguments,
        metadata={"help": "Arguments for training configuration."},
    )
    reward_args: RewardArguments = field(
        default_factory=RewardArguments,
        metadata={"help": "Arguments for reward model configuration."},
    )
    
    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary."""
        return asdict(self)

    @classmethod
    def from_dict(cls, args_dict: dict[str, Any]) -> Arguments:
        """
        Create Arguments instance from dictionary.
        This is a Factory Method.
        """
        data_args = DataArguments(**args_dict.get('data', {}))
        model_args = ModelArguments(**args_dict.get('model', {}))
        training_args = TrainingArguments(**args_dict.get('train', {}))
        reward_args = RewardArguments(**args_dict.get('reward', {}))
        
        return cls(
            data_args=data_args,
            model_args=model_args,
            training_args=training_args,
            reward_args=reward_args,
        )

    @classmethod
    def load_from_yaml(cls, yaml_file: str) -> Arguments:
        """
        Load Arguments from a YAML configuration file.
        Example: args = Arguments.load_from_yaml("config.yaml")
        """
        with open(yaml_file, 'r', encoding='utf-8') as f:
            args_dict = yaml.safe_load(f)
        
        return cls.from_dict(args_dict)