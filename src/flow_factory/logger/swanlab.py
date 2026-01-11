# src/flow_factory/logger/swanlab.py
from typing import Any, Dict
import swanlab
from .abc import Logger
from .formatting import LogImage, LogVideo, LogTable
from ..utils.logger_utils import setup_logger

logger = setup_logger(__name__)

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
            return swanlab.Video(value.get_value(format='gif'), caption=value.caption)
        elif isinstance(value, LogTable):
            # logger.warning("SwanLab does not support LogTable natively. Skip conversion.")
            # Convert a table to a dictionary of columns to lists
            table_dict = {col: [] for col in value.columns}
            for row in value.rows:
                for col, item in zip(value.columns, row):
                    table_dict[col].append(self._convert_to_platform(item))
            return table_dict
        return value

    def _log_impl(self, data: Dict, step: int):
        self.platform.log(data, step=step)