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

"""Offline algorithm configuration stays orthogonal to online rollout geometry."""

from pathlib import Path
from unittest.mock import patch

import pytest

from flow_factory.contracts.execution import (
    OFFLINE_EXECUTION_CONTRACT,
    ONLINE_EXECUTION_CONTRACT,
    CycleUnit,
)
from flow_factory.hparams import (
    Arguments,
    DataArguments,
    DatasetArguments,
    DatasetEvalSpec,
    DatasetTrainSpec,
    DiffusionOPDTrainingArguments,
    DPOTrainingArguments,
    MultiRewardArguments,
    OfflineDPOTrainingArguments,
    OfflineFlowMatchingTrainingArguments,
    OfflineTrainingArguments,
    RewardArguments,
    SFTTrainingArguments,
    get_training_args_class,
)


def _data(*, weight: float = 1.0, with_eval: bool = False) -> DataArguments:
    return DataArguments(
        datasets=[
            DatasetArguments(
                name="offline_source",
                dataset_dir="dataset/offline_source",
                train=DatasetTrainSpec(weight=weight),
                eval=DatasetEvalSpec() if with_eval else None,
            )
        ]
    )


@pytest.mark.parametrize(
    ("arguments_class", "trainer_type"),
    [
        (SFTTrainingArguments, "sft"),
        (OfflineDPOTrainingArguments, "offline-dpo"),
    ],
)
def test_offline_algorithm_defaults_are_finite_data_epochs(
    arguments_class: type[OfflineTrainingArguments],
    trainer_type: str,
) -> None:
    training_args = arguments_class()

    assert training_args.execution_contract is OFFLINE_EXECUTION_CONTRACT
    assert training_args.execution_contract.cycle_unit is CycleUnit.DATA_EPOCH
    assert training_args.trainer_type == trainer_type
    assert training_args.max_epochs == 1
    assert training_args.gradient_accumulation_steps == 1
    assert training_args._manual_gradient_accumulation_steps is True
    assert training_args.num_batches_per_epoch == 0
    assert "execution_contract" not in training_args.to_dict()


@pytest.mark.parametrize("arguments_class", [SFTTrainingArguments, OfflineDPOTrainingArguments])
def test_offline_algorithms_share_flow_matching_timestep_contract(
    arguments_class: type[OfflineFlowMatchingTrainingArguments],
) -> None:
    training_args = arguments_class()

    assert isinstance(training_args, OfflineFlowMatchingTrainingArguments)
    assert training_args.weighting_scheme == "logit_normal"
    assert training_args.logit_mean == 0.0
    assert training_args.logit_std == 1.0
    assert training_args.num_train_timesteps == 1
    assert training_args.time_shift == 1.0
    assert training_args.timestep_range == (0.0, 0.99)
    assert training_args.get_num_train_timesteps(None) == 1


@pytest.mark.parametrize(
    "invalid_gas",
    ["auto", "2", 1.0, True, None],
)
def test_offline_gradient_accumulation_requires_an_explicit_integer(invalid_gas: object) -> None:
    with pytest.raises(TypeError, match="explicit int"):
        SFTTrainingArguments(gradient_accumulation_steps=invalid_gas)  # type: ignore[arg-type]


@pytest.mark.parametrize("invalid_gas", [0, -1])
def test_offline_gradient_accumulation_requires_a_positive_integer(invalid_gas: int) -> None:
    with pytest.raises(ValueError, match="must be >= 1"):
        SFTTrainingArguments(gradient_accumulation_steps=invalid_gas)


def test_offline_configuration_never_enters_online_geometry_helpers() -> None:
    training_args = OfflineDPOTrainingArguments(
        gradient_accumulation_steps=3,
        gradient_step_per_epoch=97,
        group_size=7,
        unique_sample_num_per_epoch=13,
        per_device_batch_size=5,
        num_train_timesteps=4,
    )
    data_args = _data()

    with (
        patch(
            "flow_factory.hparams.training_args._base.get_world_size",
            side_effect=AssertionError("offline TrainingArguments used online geometry"),
        ),
        patch(
            "flow_factory.hparams.args.get_world_size",
            side_effect=AssertionError("offline Arguments used online geometry"),
        ),
    ):
        config = Arguments(training_args=training_args, data_args=data_args)

    assert config.data_args.sampler_type == "auto"
    assert config.training_args.gradient_accumulation_steps == 3
    assert config.training_args.gradient_step_per_epoch == 97
    assert config.training_args.group_size == 7
    assert config.training_args.unique_sample_num_per_epoch == 13
    assert config.training_args.num_batches_per_epoch == 0
    assert config.data_args.training_datasets[0].train is not None
    assert config.data_args.training_datasets[0].train.unique_sample_num_per_epoch is None
    assert config.data_args.training_datasets[0].train.num_batches_per_epoch is None


@pytest.mark.parametrize(
    "sampler_type",
    ["distributed_k_repeat", "group_contiguous", "group_distributed"],
)
def test_offline_configuration_rejects_online_sampler_selection(sampler_type: str) -> None:
    with pytest.raises(ValueError, match="DistributedSampler.*must remain 'auto'"):
        Arguments(
            training_args=SFTTrainingArguments(),
            data_args=DataArguments(
                sampler_type=sampler_type,  # type: ignore[arg-type]
                datasets=_data().datasets,
            ),
        )


def test_offline_sources_require_unit_weights() -> None:
    with pytest.raises(ValueError, match="train.weight=1"):
        Arguments(training_args=SFTTrainingArguments(), data_args=_data(weight=2))

    config = Arguments(training_args=SFTTrainingArguments(), data_args=_data(weight=1.0))

    assert config.data_args.training_datasets[0].train is not None
    assert config.data_args.training_datasets[0].train.weight == 1


@pytest.mark.parametrize(
    "arguments_class",
    [SFTTrainingArguments, OfflineDPOTrainingArguments],
)
def test_offline_training_rejects_runtime_rewards(
    arguments_class: type[OfflineTrainingArguments],
) -> None:
    rewards = MultiRewardArguments(
        reward_configs=[RewardArguments(name="train_score", reward_model="clip")]
    )

    with pytest.raises(ValueError, match="does not accept training rewards.*eval_rewards"):
        Arguments(training_args=arguments_class(), data_args=_data(), reward_args=rewards)


def test_reward_free_online_training_also_rejects_runtime_rewards() -> None:
    rewards = MultiRewardArguments(
        reward_configs=[RewardArguments(name="train_score", reward_model="clip")]
    )

    with pytest.raises(ValueError, match="does not accept training rewards.*eval_rewards"):
        Arguments(
            training_args=DiffusionOPDTrainingArguments(
                trainer_type="diffusion-opd",
                teachers=[
                    {
                        "path": "teacher.safetensors",
                        "applicable_datasets": ["offline_source"],
                    }
                ],
            ),
            data_args=_data(),
            reward_args=rewards,
        )


def test_offline_configuration_keeps_eval_only_rewards() -> None:
    eval_reward = RewardArguments(name="eval_score", reward_model="clip")
    config = Arguments(
        training_args=SFTTrainingArguments(),
        data_args=_data(with_eval=True),
        eval_reward_args=MultiRewardArguments(reward_configs=[eval_reward]),
    )

    assert len(config.reward_args) == 0
    assert config.eval_reward_args is not None
    assert config.eval_reward_args[0].applicable_datasets == ["offline_source"]
    assert config.eval_reward_args[0]._datasets_resolved == frozenset({0})


@pytest.mark.parametrize("arguments_class", [SFTTrainingArguments, OfflineDPOTrainingArguments])
def test_offline_flow_matching_normalizes_timesteps_without_multiplying_gas(
    arguments_class: type[OfflineFlowMatchingTrainingArguments],
) -> None:
    training_args = arguments_class(
        gradient_accumulation_steps=2,
        num_inference_steps=10,
        num_train_timesteps=0,
        timestep_range=(0.2, 0.8),
    )
    config = Arguments(training_args=training_args, data_args=_data())

    assert config.training_args.timestep_range == (0.2, 0.8)
    assert config.training_args.num_train_timesteps == 6
    assert config.training_args.get_num_train_timesteps(config) == 6
    assert config.training_args.gradient_accumulation_steps == 2
    assert config.training_args.requires_ref_model is (
        arguments_class is OfflineDPOTrainingArguments
    )


def test_existing_dpo_remains_online_and_keeps_online_gas_derivation() -> None:
    training_args = DPOTrainingArguments(
        unique_sample_num_per_epoch=8,
        group_size=4,
        per_device_batch_size=2,
        gradient_step_per_epoch=2,
        num_train_timesteps=3,
    )
    config = Arguments(training_args=training_args)

    assert get_training_args_class("dpo") is DPOTrainingArguments
    assert config.training_args.execution_contract is ONLINE_EXECUTION_CONTRACT
    assert config.data_args.sampler_type == "group_contiguous"
    assert config.training_args.num_batches_per_epoch == 16
    assert config.training_args.gradient_accumulation_steps == 6


@pytest.mark.parametrize(
    "relative_path",
    [
        "examples/opd/lora/sd3_5/DiffusionOPD_aligned.yaml",
        "examples/opd/lora/sd3_5/geneval_pickscore_ocr.yaml",
        "examples/opd/lora/sd3_5/geneval_pickscore_ocr_x0_norm.yaml",
    ],
)
def test_opd_examples_keep_monitoring_under_eval_rewards(relative_path: str) -> None:
    config = Arguments.load_from_yaml(str(Path(__file__).parents[2] / relative_path))

    assert len(config.reward_args) == 0
    assert config.eval_reward_args is not None
    assert len(config.eval_reward_args) == 3
