# src/flow_factory/scheduler/loader.py
"""
Scheduler Loader
Factory function to instantiate SDE schedulers from pipeline schedulers.
"""
from typing import Union
from diffusers.schedulers.scheduling_utils import SchedulerMixin

from .registry import get_sde_scheduler_class
from ..hparams import SchedulerArguments
from ..utils.logger_utils import setup_logger

logger = setup_logger(__name__)


def load_scheduler(
    pipeline_scheduler: SchedulerMixin,
    scheduler_args: SchedulerArguments,
) -> SchedulerMixin:
    """
    Create an SDE scheduler from a pipeline scheduler and scheduler args.
    
    Merges the original scheduler config with SDE-specific args.
    
    Args:
        pipeline_scheduler: Scheduler from pipeline.from_pretrained()
        scheduler_args: SchedulerArguments with SDE config
    
    Returns:
        Custom SDE scheduler instance
    
    Example:
        >>> pipe = DiffusionPipeline.from_pretrained("...")
        >>> sde_scheduler = load_scheduler(pipe.scheduler, scheduler_args)
    """
    sde_class = get_sde_scheduler_class(pipeline_scheduler)
    
    # Merge base config with SDE args
    base_config = dict(pipeline_scheduler.config)
    base_config.update(scheduler_args.to_dict())
    
    scheduler = sde_class(**base_config)
    logger.info(f"Loaded SDE scheduler: {sde_class.__name__}")
    return scheduler