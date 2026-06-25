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

# src/flow_factory/models/qwen_image/_utils.py
"""Shared helpers for the Qwen-Image adapter family."""
import torch


def _pad_seq_dim(x: torch.Tensor, target_len: int, value: float) -> torch.Tensor:
    """Right-pad a 2-D mask ``(B, L)`` or 3-D embedding ``(B, L, D)`` along dim=1.

    Pads up to ``target_len`` (no-op if already that long), so cond/uncond text
    streams can be concatenated along the batch dim for a single CFG forward.
    """
    pad = target_len - x.shape[1]
    if pad <= 0:
        return x
    if x.dim() == 2:
        return torch.nn.functional.pad(x, (0, pad), value=value)
    return torch.nn.functional.pad(x, (0, 0, 0, pad), value=value)
