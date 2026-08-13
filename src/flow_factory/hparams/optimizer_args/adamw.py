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

"""AdamW-specific optimizer arguments."""

from dataclasses import dataclass
from typing import Tuple

from ._base import OptimizerArguments


@dataclass
class AdamWOptimizerArguments(OptimizerArguments):
    """Configure ``torch.optim.AdamW`` for one trainable variant.

    Attributes:
        betas: Exponential decay rates for the first and second moment estimates.
        eps: Term added to the denominator for numerical stability.
    """

    optimizer: str = "adamw"
    betas: Tuple[float, float] = (0.9, 0.999)
    eps: float = 1e-8

    def __post_init__(self) -> None:
        """Validate the AdamW moment parameters."""
        super().__post_init__()
        if len(tuple(self.betas)) != 2:
            raise ValueError(
                f"expected two betas for optimizer {self.name!r}, received {self.betas!r}"
            )
        self.betas = (float(self.betas[0]), float(self.betas[1]))
        for index, beta in enumerate(self.betas):
            if not 0.0 <= beta < 1.0:
                raise ValueError(
                    f"expected betas[{index}] in [0, 1) for optimizer {self.name!r}, "
                    f"received {beta}"
                )
        self.eps = float(self.eps)
        if self.eps <= 0:
            raise ValueError(
                f"expected a positive eps for optimizer {self.name!r}, received {self.eps}"
            )
