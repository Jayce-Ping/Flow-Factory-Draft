# src/flow_factory/hparams/scheduler_args.py
import yaml
from dataclasses import dataclass, field
from typing import Any, Literal, Optional, List

from .abc import ArgABC


@dataclass
class SchedulerArguments(ArgABC):
    r"""Arguments pertaining to scheduler configuration."""

    dynamics_type: Literal["Flow-SDE", 'Dance-SDE', 'CPS', 'ODE'] = field(
        default="Flow-SDE",
        metadata={"help": "Type of SDE dynamics to use."},
    )
    num_inference_steps: int = field(
        default=10,
        metadata={"help": "Number of timesteps for SDE."},
    )
    noise_level: float = field(
        default=0.7,
        metadata={"help": "Noise level for SDE sampling."},
    )
    num_train_steps: int = field(
        default=1,
        metadata={"help": "Number of train steps to sample per rollout."},
    )
    train_steps: Optional[List[int]] = field(
        default=None,
        metadata={"help": (
            "Training step indices for optimization. "
            "`num_train_steps` will be randomly sampled from this list. "
            "If None, uses the first 1/3 of timesteps."
        )},
    )
    seed: int = field(
        default=42,
        metadata={"help": "Random seed for selecting train steps."},
    )

    def __post_init__(self):
        if self.train_steps is None:
            first_n_steps = max(1, self.num_inference_steps // 3)
            self.train_steps = list(range(first_n_steps))

    def to_dict(self) -> dict[str, Any]:
        return super().to_dict()

    def __str__(self) -> str:
        return yaml.dump(self.to_dict(), default_flow_style=False, sort_keys=False, indent=2)

    def __repr__(self) -> str:
        return self.__str__()