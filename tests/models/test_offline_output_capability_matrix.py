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

import pytest

from flow_factory.models.ltx2.ltx2_i2av import LTX2_I2AV_Adapter
from flow_factory.models.ltx2.ltx2_t2av import LTX2_T2AV_Adapter
from flow_factory.models.minimax_h3.adapters import (
    MiniMaxH3FL2VAAdapter,
    MiniMaxH3Ref2VAAdapter,
    MiniMaxH3T2VAAdapter,
)
from flow_factory.models.wan.wan2_i2v import Wan2_I2V_Adapter


@pytest.mark.parametrize(
    ("adapter_type", "reason_fragment"),
    [
        (Wan2_I2V_Adapter, "first-frame VAE condition"),
        (LTX2_T2AV_Adapter, "paired video/audio decoding"),
        (LTX2_I2AV_Adapter, "active mask"),
        (MiniMaxH3FL2VAAdapter, "conditioned-prefix binder"),
        (MiniMaxH3Ref2VAAdapter, "conditioned-prefix binder"),
    ],
)
def test_unimplemented_offline_media_semantics_fail_before_model_loading(
    adapter_type: type,
    reason_fragment: str,
) -> None:
    """Expose actionable blockers instead of silently guessing target encoding."""
    with pytest.raises(NotImplementedError, match=reason_fragment):
        adapter_type.validate_offline_output_capability()


def test_minimax_h3_t2va_declares_complete_offline_output_semantics() -> None:
    """T2VA has no conditioned prefix and can encode paired AV targets on demand."""
    MiniMaxH3T2VAAdapter.validate_offline_output_capability()
