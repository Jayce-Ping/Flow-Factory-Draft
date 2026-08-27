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

"""Static offline-output capability declarations for online model adapters."""

import pytest

from flow_factory.models.flux.flux1 import Flux1Adapter
from flow_factory.models.flux.flux1_kontext import Flux1KontextAdapter
from flow_factory.models.flux.flux2 import Flux2Adapter
from flow_factory.models.flux.flux2_klein import Flux2KleinAdapter
from flow_factory.models.ltx2.ltx2_i2av import LTX2_I2AV_Adapter
from flow_factory.models.ltx2.ltx2_t2av import LTX2_T2AV_Adapter
from flow_factory.models.minimax_h3.adapters import (
    MiniMaxH3FL2VAAdapter,
    MiniMaxH3Ref2VAAdapter,
    MiniMaxH3T2VAAdapter,
)
from flow_factory.models.qwen_image.qwen_image import QwenImageAdapter
from flow_factory.models.qwen_image.qwen_image_edit_plus import QwenImageEditPlusAdapter
from flow_factory.models.sensenova.sensenova import SenseNovaAdapter
from flow_factory.models.stable_diffusion.sd3_5 import SD3_5Adapter
from flow_factory.models.wan.wan2_i2v import Wan2_I2V_Adapter
from flow_factory.models.wan.wan2_t2v import Wan2_T2V_Adapter
from flow_factory.models.z_image.z_image import ZImageAdapter


@pytest.mark.parametrize(
    "adapter_cls",
    [
        Flux1Adapter,
        Flux1KontextAdapter,
        Flux2Adapter,
        Flux2KleinAdapter,
        QwenImageAdapter,
        QwenImageEditPlusAdapter,
        ZImageAdapter,
        SD3_5Adapter,
        SenseNovaAdapter,
        Wan2_T2V_Adapter,
    ],
)
def test_supported_adapters_pass_weight_free_offline_codec_preflight(
    adapter_cls: type,
) -> None:
    adapter_cls.validate_offline_output_capability()


@pytest.mark.parametrize(
    ("adapter_cls", "actionable_details"),
    [
        (LTX2_T2AV_Adapter, ("target-audio decoder", "waveform-to-training-mel")),
        (LTX2_I2AV_Adapter, ("target-audio decoder", "waveform-to-training-mel")),
        (MiniMaxH3T2VAAdapter, ("target-audio decoder", "posterior selection")),
        (MiniMaxH3FL2VAAdapter, ("target-audio decoder", "posterior selection")),
        (MiniMaxH3Ref2VAAdapter, ("target-audio decoder", "posterior selection")),
        (Wan2_I2V_Adapter, ("VAE condition tensor", "do not infer")),
    ],
)
def test_known_offline_output_blockers_fail_with_actionable_reason(
    adapter_cls: type,
    actionable_details: tuple[str, ...],
) -> None:
    reason = adapter_cls.output_state_codec_unavailable_reason

    assert isinstance(reason, str)
    for detail in actionable_details:
        assert detail in reason
    with pytest.raises(NotImplementedError) as exc_info:
        adapter_cls.validate_offline_output_capability()
    for detail in actionable_details:
        assert detail in str(exc_info.value)
