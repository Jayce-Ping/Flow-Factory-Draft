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

"""Export-path guarantees for ``Arguments.to_dict``.

These guard the wandb / SwanLab config sink: the exported dict must not leak
internal ``_``-prefixed runtime caches (e.g. ``RewardArguments._datasets_resolved``,
a ``frozenset``) and must be JSON-serializable even when user-supplied
``extra_kwargs`` contain a ``set``.
"""

from __future__ import annotations

import json
from typing import Any, Iterator

from flow_factory.hparams.args import Arguments


def _all_keys(obj: Any) -> Iterator[str]:
    """Yield every dict key found anywhere in a nested structure."""
    if isinstance(obj, dict):
        for key, value in obj.items():
            yield key
            yield from _all_keys(value)
    elif isinstance(obj, (list, tuple)):
        for value in obj:
            yield from _all_keys(value)


def _multi_source_dict() -> dict[str, Any]:
    """Minimal multi-source config that populates ``_datasets_resolved``."""
    return {
        "data": {
            "datasets": [
                {"name": "src_a", "dataset_dir": "dataset/a", "train": {"weight": 1}},
                {"name": "src_b", "dataset_dir": "dataset/b", "train": {"weight": 1}},
            ],
        },
        "model": {
            "model_type": "sd3-5",
            "model_name_or_path": "stabilityai/stable-diffusion-3.5-medium",
            "finetune_type": "lora",
        },
        "log": {"logging_backend": "none", "project": "ff-test"},
        "train": {
            "trainer_type": "grpo",
            "ref_param_device": "cpu",
            "ema_device": "cpu",
        },
        "rewards": [
            {
                "name": "r0",
                "reward_model": "PickScore",
                "device": "cpu",
                "applicable_datasets": ["src_a"],
            },
            {"name": "r1", "reward_model": "PickScore", "device": "cpu"},
        ],
    }


def test_reward_to_dict_omits_internal_datasets_resolved() -> None:
    cfg = Arguments.from_dict(_multi_source_dict())
    # Sanity: the cache is actually populated on the live object.
    assert cfg.reward_args.reward_configs[0]._datasets_resolved is not None

    exported = cfg.to_dict()
    assert "_datasets_resolved" not in set(_all_keys(exported))
    # No internal underscore-prefixed key should survive anywhere.
    assert not any(k.startswith("_") for k in _all_keys(exported))


def test_arguments_to_dict_is_json_serializable() -> None:
    cfg = Arguments.from_dict(_multi_source_dict())
    # Must not raise TypeError("Object of type frozenset is not JSON serializable").
    json.dumps(cfg.to_dict())


def test_set_in_extra_kwargs_is_coerced_to_list() -> None:
    cfg_dict = _multi_source_dict()
    # `aspects` is unknown to RewardArguments -> lands in extra_kwargs as a set.
    cfg_dict["rewards"][0]["aspects"] = {"a", "b"}

    cfg = Arguments.from_dict(cfg_dict)
    exported = cfg.to_dict()

    # The coerced value is JSON-safe and exported as a (sorted) list.
    json.dumps(exported)
    assert exported["reward"]["reward_0"]["aspects"] == ["a", "b"]


def test_single_source_legacy_to_dict_is_json_serializable() -> None:
    cfg_dict = {
        "data": {"dataset_dir": "dataset/pickscore"},
        "model": {
            "model_type": "sd3-5",
            "model_name_or_path": "stabilityai/stable-diffusion-3.5-medium",
            "finetune_type": "lora",
        },
        "log": {"logging_backend": "none"},
        "train": {"trainer_type": "grpo", "ref_param_device": "cpu", "ema_device": "cpu"},
        "rewards": [{"name": "r0", "reward_model": "PickScore", "device": "cpu"}],
    }
    cfg = Arguments.from_dict(cfg_dict)
    json.dumps(cfg.to_dict())
