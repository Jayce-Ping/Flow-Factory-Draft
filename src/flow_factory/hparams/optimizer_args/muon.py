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

"""Muon-specific optimizer arguments."""

from dataclasses import dataclass
from typing import Literal, Tuple

from ._base import OptimizerArguments


@dataclass
class MuonOptimizerArguments(OptimizerArguments):
    """Configure ``torch.optim.Muon`` for one trainable variant.

    Muon orthogonalizes the momentum-smoothed gradient of 2D hidden-layer weights and
    rejects everything else, so a variant optimized with Muon is really optimized by
    two algorithms: Muon for its matrices and AdamW for its biases, normalization
    parameters and embeddings. The ``fallback_`` fields configure that second half,
    which is why they are here rather than in a separate configuration entry.

    Attributes:
        momentum: Momentum factor for the Muon update.
        nesterov: Whether the momentum is Nesterov-style.
        ns_coefficients: Newton-Schulz polynomial coefficients ``(a, b, c)``.
        ns_steps: Newton-Schulz iteration count.
        eps: Numerical stabilization inside the orthogonalization.
        adjust_lr_fn: Learning-rate adjustment that keeps the orthogonalized update's
            RMS consistent across rectangular matrices. ``original`` follows Keller
            Jordan's scaling; ``match_rms_adamw`` follows the scalable-Muon paper and
            lets a learning rate tuned for AdamW be reused directly.
        fallback_betas: AdamW betas for the parameters Muon cannot take.
        fallback_eps: AdamW epsilon for the parameters Muon cannot take.
    """

    optimizer: str = "muon"
    momentum: float = 0.95
    nesterov: bool = True
    ns_coefficients: Tuple[float, float, float] = (3.4445, -4.7750, 2.0315)
    ns_steps: int = 5
    eps: float = 1e-7
    adjust_lr_fn: Literal["original", "match_rms_adamw"] = "match_rms_adamw"
    fallback_betas: Tuple[float, float] = (0.9, 0.999)
    fallback_eps: float = 1e-8

    def __post_init__(self) -> None:
        """Validate the Muon and fallback parameters."""
        super().__post_init__()
        self.momentum = float(self.momentum)
        if not 0.0 <= self.momentum < 1.0:
            raise ValueError(
                f"expected momentum in [0, 1) for optimizer {self.name!r}, "
                f"received {self.momentum}"
            )
        if not isinstance(self.nesterov, bool):
            raise TypeError(
                f"expected bool nesterov for optimizer {self.name!r}, received "
                f"{type(self.nesterov).__name__}: {self.nesterov!r}"
            )
        if len(tuple(self.ns_coefficients)) != 3:
            raise ValueError(
                f"expected three Newton-Schulz coefficients for optimizer {self.name!r}, "
                f"received {self.ns_coefficients!r}"
            )
        self.ns_coefficients = tuple(float(value) for value in self.ns_coefficients)
        if not isinstance(self.ns_steps, int) or isinstance(self.ns_steps, bool):
            raise TypeError(
                f"expected int ns_steps for optimizer {self.name!r}, received "
                f"{type(self.ns_steps).__name__}: {self.ns_steps!r}"
            )
        if self.ns_steps < 1:
            raise ValueError(
                f"expected ns_steps of at least 1 for optimizer {self.name!r}, "
                f"received {self.ns_steps}"
            )
        if self.adjust_lr_fn not in ("original", "match_rms_adamw"):
            raise ValueError(
                f"expected adjust_lr_fn in ('original', 'match_rms_adamw') for optimizer "
                f"{self.name!r}, received {self.adjust_lr_fn!r}"
            )
        self.eps = float(self.eps)
        self.fallback_eps = float(self.fallback_eps)
        if self.eps <= 0 or self.fallback_eps <= 0:
            raise ValueError(
                f"expected positive eps and fallback_eps for optimizer {self.name!r}, "
                f"received eps={self.eps} and fallback_eps={self.fallback_eps}"
            )
        if len(tuple(self.fallback_betas)) != 2:
            raise ValueError(
                f"expected two fallback_betas for optimizer {self.name!r}, "
                f"received {self.fallback_betas!r}"
            )
        self.fallback_betas = (float(self.fallback_betas[0]), float(self.fallback_betas[1]))
