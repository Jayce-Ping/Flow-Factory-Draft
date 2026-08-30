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

from types import SimpleNamespace

import torch

from flow_factory.rewards import GroupwiseRewardModel, RewardModelOutput
from flow_factory.rewards.reward_processor import RewardProcessor
from flow_factory.samples import MiniMaxH3Ref2VASample


class PromptOnlyGroupReward(GroupwiseRewardModel):
    required_fields = ("prompt",)

    def __init__(self) -> None:
        pass

    def __call__(self, prompt: list[str]) -> RewardModelOutput:
        return RewardModelOutput(rewards=torch.zeros(len(prompt)))


def test_distributed_group_reward_preserves_sample_reconstruction_fields() -> None:
    accelerator = SimpleNamespace(
        device=torch.device("cpu"),
        process_index=0,
        num_processes=1,
        is_local_main_process=True,
        wait_for_everyone=lambda: None,
        reduce=lambda tensor, reduction: tensor,
    )
    model = PromptOnlyGroupReward()
    processor = RewardProcessor(
        accelerator=accelerator,
        reward_models={"prompt_only": model},
        group_on_same_rank=False,
        verbose=False,
    )
    sample = MiniMaxH3Ref2VASample(
        prompt="A reference-conditioned prompt",
        reference_manifest='[{"kind":"image","path":"condition.png"}]',
    )

    rewards = processor.compute_rewards([sample], store_to_samples=False)

    torch.testing.assert_close(rewards["prompt_only"], torch.zeros(1))
