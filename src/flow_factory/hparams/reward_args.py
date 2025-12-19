# src/flow_factory/hparams/reward_args.py
import os
import math
from dataclasses import asdict, dataclass, field
from typing import Any, Literal, Optional, Type
import logging
import torch

from ..rewards.registry import get_reward_model_class

logging.basicConfig(level=logging.INFO, format='[%(asctime)s] [%(levelname)s] [%(name)s]: %(message)s')
logger = logging.getLogger(__name__)

def get_world_size() -> int:
    # Standard PyTorch/Accelerate/DDP variable
    if "WORLD_SIZE" in os.environ:
        return int(os.environ["WORLD_SIZE"])
    
    # OpenMPI / Horovod
    if "OMPI_COMM_WORLD_SIZE" in os.environ:
        return int(os.environ["OMPI_COMM_WORLD_SIZE"])
    
    # Intel MPI / Slurm (sometimes)
    if "PMI_SIZE" in os.environ:
        return int(os.environ["PMI_SIZE"])
    
    return 1

@dataclass
class RewardArguments:
    r"""Arguments pertaining to reward configuration."""

    reward_model : str = field(
        default='PickScore',
        metadata={"help": "The path or name of the reward model to use. You can specify 'PickScore' to use the default PickScore model. Or /path/to/your/model:class_name to use your own reward model."},
    )

    dtype: Literal['float16', 'bfloat16', 'float32'] = field(
        default='float16',
        metadata={"help": "The data type for the reward model."},
    )

    device: Literal['cpu', 'cuda'] = field(
        default='cuda',
        metadata={"help": "The device to load the reward model on."},
    )

    reward_model_kwargs: Optional[dict[str, Any]] = field(
        default=None,
        metadata={"help": "Additional keyword arguments for the reward model."},
    )

    batch_size: int = field(
        default=16,
        metadata={"help": "Batch size for reward model inference."},
    )

    _reward_model_cls: Optional[Type] = field(init=False, repr=False, default=None)

    def __post_init__(self):
        # Convert dtype string to torch dtype
        dtype_map = {
            'float16': torch.float16,
            'bfloat16': torch.bfloat16,
            'float32': torch.float32,
        }
        self.torch_dtype = dtype_map[self.dtype]

        # Convert device string to torch device
        self.torch_device = torch.device(self.device)

        # Parse reward model name/path
        self._reward_model_cls = get_reward_model_class(self.reward_model)

    @property
    def reward_model_cls(self) -> Type:
        """Access the loaded class safely."""
        return self._reward_model_cls
        

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)