# src/flow_factory/logger/swanlab.py
from typing import Any, Dict
import swanlab
from .abc import Logger
from .formatting import LogImage, LogVideo


class SwanlabLogger(Logger):
    def _init_platform(self):
        swanlab.init(
            project=self.config.project,
            name=self.config.run_name,
            config=self.config.to_dict()
        )
        self.platform = swanlab

    def _convert_to_platform(self, value: Any) -> Any:
        if isinstance(value, LogImage):
            return swanlab.Image(value.value, caption=value.caption)
        elif isinstance(value, LogVideo):
            return swanlab.Video(value.value, caption=value.caption)
        return value

    def _log_impl(self, data: Dict, step: int):
        self.platform.log(data, step=step)