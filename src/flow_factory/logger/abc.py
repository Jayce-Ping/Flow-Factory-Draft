# src/flow_factory/logger/abc.py
from abc import ABC, abstractmethod
from typing import Union, Optional, Dict, Any, List

from ..hparams import *
from ..models.adapter import BaseSample
from .formatting import LogFormatter, LogImage, LogVideo

class Logger(ABC):
    platform: Any

    def __init__(self, config: Arguments):
        self.config = config
        self._init_platform()

    @abstractmethod
    def _init_platform(self):
        pass

    def log_data(
        self,
        data: Dict[str, Any],
        step: int,
        keys: Optional[str] = None,
    ):
        # 1. Process rules (Mean, Paths, wrappers) into IR
        formatted_dict = LogFormatter.format_dict(data)
        
        # 2. Filter keys if requested
        if keys:
            valid_keys = keys.split(',')
            formatted_dict = {k: v for k, v in formatted_dict.items() if k in valid_keys}

        # 3. Convert IR to Platform Objects
        final_dict = {}
        for k, v in formatted_dict.items():
            final_dict[k] = self._recursive_convert(v)

        # 4. Actual Logging
        if final_dict:
            self._log_impl(final_dict, step)

    def _recursive_convert(self, value: Any) -> Any:
        """Helper to handle lists recursively."""
        if isinstance(value, (list, tuple)):
            return [self._recursive_convert(v) for v in value]
        return self._convert_to_platform(value)

    @abstractmethod
    def _convert_to_platform(self, value: Any) -> Any:
        """Convert SINGLE LogImage/LogVideo to wandb.Image/swanlab.Image etc."""
        pass

    @abstractmethod
    def _log_impl(self, data: Dict, step: int):
        pass