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

from flow_factory.models.ltx2.ltx2_i2av import LTX2_I2AV_Adapter
from flow_factory.models.ltx2.ltx2_t2av import LTX2_T2AV_Adapter
from flow_factory.models.minimax_h3.adapters import (
    MiniMaxH3FL2VAAdapter,
    MiniMaxH3Ref2VAAdapter,
    MiniMaxH3T2VAAdapter,
)
from flow_factory.models.wan.wan2_i2v import Wan2_I2V_Adapter
from flow_factory.models.wan.wan2_t2v import Wan2_T2V_Adapter


def test_video_and_av_adapters_declare_complete_offline_capability() -> None:
    """Every implemented video/AV workflow exposes its complete offline codec."""
    adapter_types = (
        Wan2_T2V_Adapter,
        Wan2_I2V_Adapter,
        LTX2_T2AV_Adapter,
        LTX2_I2AV_Adapter,
        MiniMaxH3T2VAAdapter,
        MiniMaxH3FL2VAAdapter,
        MiniMaxH3Ref2VAAdapter,
    )

    for adapter_type in adapter_types:
        adapter_type.validate_offline_output_capability()
