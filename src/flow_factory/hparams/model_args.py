# src/flow_factory/hparams/model_args.py
import os
import math
from dataclasses import asdict, dataclass, field
from typing import Any, Literal, Optional
import logging

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
class ModelArguments:
    r"""Arguments pertaining to model configuration."""

    model_name_or_path: str = field(
        default="black-forest-labs/FLUX.1-dev",
        metadata={"help": "Path to pre-trained model or model identifier from huggingface.co/models"},
    )

    finetune_type : Literal['full', 'lora'] = field(
        default='full',
        metadata={"help": "Fine-tuning type. Options are ['full', 'lora']"}
    )

    resume_path : Optional[str] = field(
        default=None,
        metadata={"help": "Resume from checkpoint directory."}
    )

    model_type: Literal["sd3", "flux1", "flux1-kontext", 'flux2', 'qwenimage', 'qwenimage-edit'] = field(
        default="flux1",
        metadata={"help": "Type of model to use."},
    )

    lora_rank : int = field(
        default=8,
        metadata={"help": "Rank for LoRA adapters."},
    )

    lora_alpha : int = field(
        default=16,
        metadata={"help": "Alpha scaling factor for LoRA adapters."},
    )

    def __post_init__(self):
        pass

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)