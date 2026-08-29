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

"""Runtime condition-prefix preparation for conditioned MiniMax H3 workflows."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any, ClassVar, Optional, Tuple

import torch

from ..condition_state import PreparedConditionState
from .blocks import prepare_h3_condition_prefixes
from .workflow import _normalize_layout

_RUNTIME_OWNED_FIELDS = frozenset(
    {
        "condition_latents",
        "audio_condition_latents",
        "condition_prefixes",
        "layout",
        "position_ids",
        "token_tags",
        "video_indices",
        "audio_indices",
        "text_indices",
        "num_condition_video_rows",
        "num_condition_audio_rows",
    }
)


@dataclass(frozen=True, slots=True)
class MiniMaxH3ConditionStatePreparer:
    """Realize one FL2VA/Ref2VA prefix and share it across offline arms."""

    adapter: Any
    required_components: ClassVar[Tuple[str, ...]] = ("scheduler",)

    def prepare_condition_state(
        self,
        condition: Mapping[str, Any],
        generator: Optional[torch.Generator] = None,
    ) -> PreparedConditionState:
        """Noise cached visual conditions once and bind canonical H3 layout.

        Args:
            condition: B=1 cached H3 condition with clean condition latents.
            generator: Optional generator consumed in official packed condition order.

        Returns:
            Input-owned condition realization shared by every offline target arm.
        """
        workflow = getattr(self.adapter, "workflow", None)
        if workflow not in ("fl2va", "ref2va"):
            raise ValueError(
                "MiniMax H3 condition preparer requires workflow 'fl2va' or 'ref2va', "
                f"received {workflow!r}"
            )
        if "condition_prefixes" in condition:
            raise ValueError(
                f"MiniMax H3 workflow={workflow!r} cached condition must not contain "
                "already-realized condition_prefixes"
            )

        layout = _normalize_layout(condition)
        prefixes = prepare_h3_condition_prefixes(
            self.adapter.pipeline,
            condition,
            workflow=workflow,
            generator=generator,
        )
        static_condition = {
            key: value for key, value in condition.items() if key not in _RUNTIME_OWNED_FIELDS
        }
        return PreparedConditionState(
            condition=static_condition,
            forward_context={
                "condition_prefixes": prefixes,
                "layout": layout,
            },
            output_context={"layout": layout},
        )


__all__ = ["MiniMaxH3ConditionStatePreparer"]
