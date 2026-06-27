# Copyright 2026 Jayce-Ping
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

# src/flow_factory/hparams/acceleration_args.py
import yaml
from dataclasses import dataclass, field
from typing import Any, Dict, Optional

from .abc import ArgABC


@dataclass
class AccelerationArguments(ArgABC):
    r"""Arguments for the model-agnostic acceleration plugin layer.

    Two independent slots, both off by default (so existing configs are
    unaffected):

    * ``shared_*`` — a lossless accelerator applied to BOTH rollout and the
      training forward (e.g. ``torch_compile``).
    * ``rollout_*`` — an accelerator applied only during Stage-3 rollout. May be
      lossy (e.g. feature caching), in which case the trainer paradigm validator
      restricts it to decoupled / distillation algorithms.

    Example YAML::

        acceleration:
          shared_accelerator: torch_compile
          shared_params: { mode: regional }
          rollout_accelerator: diffusers_cache
          rollout_params: { policy: first_block, threshold: 0.08 }
    """

    shared_accelerator: Optional[str] = field(
        default=None,
        metadata={
            "help": (
                "Lossless accelerator id (or python path) applied to both rollout and training. "
                "Options: 'torch_compile'. None disables it. "
                "(Attention backend has its own knob, model.attn_backend, applied "
                "automatically before this.)"
            )
        },
    )
    shared_params: Dict[str, Any] = field(
        default_factory=dict,
        metadata={"help": "Keyword parameters forwarded to the shared accelerator constructor."},
    )
    rollout_accelerator: Optional[str] = field(
        default=None,
        metadata={
            "help": (
                "Accelerator id (or python path) applied only during Stage-3 rollout. "
                "Options: 'diffusers_cache', 'cache_dit' (lossy, decoupled/distillation only), or any "
                "lossless accelerator. None disables it."
            )
        },
    )
    rollout_params: Dict[str, Any] = field(
        default_factory=dict,
        metadata={"help": "Keyword parameters forwarded to the rollout accelerator constructor."},
    )

    def to_dict(self) -> dict[str, Any]:
        return super().to_dict()

    def __str__(self) -> str:
        return yaml.dump(self.to_dict(), default_flow_style=False, sort_keys=False, indent=2)

    def __repr__(self) -> str:
        return self.__str__()
