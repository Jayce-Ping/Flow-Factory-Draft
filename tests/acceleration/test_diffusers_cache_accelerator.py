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

import pytest

from flow_factory.acceleration.diffusers_cache import DiffusersCacheAccelerator


class RecordingTransformer:
    def __init__(self, events):
        self.events = events
        self.is_cache_enabled = False

    def enable_cache(self, config):
        self.events.append(("enable", type(config).__name__))
        self.is_cache_enabled = True

    def disable_cache(self):
        self.events.append(("disable",))
        self.is_cache_enabled = False


class RecordingAdapter:
    supports_diffusers_cache = True

    def __init__(self, supported_policies, component_name="transformer"):
        self.events = []
        self.supported_diffusers_cache_policies = supported_policies
        self.component_name = component_name
        self.transformer_names = [component_name]
        self.transformer = RecordingTransformer(self.events)
        self.accelerator = SimpleNamespace(is_main_process=False)

    def get_component(self, name):
        assert name == self.component_name
        return self.transformer

    def get_component_unwrapped(self, name):
        assert name == self.component_name
        return self.transformer

    def prepare_diffusers_cache(self, policy, component_name, transformer):
        assert transformer is self.transformer
        self.events.append(("prepare", policy, component_name))


def test_policy_capability_rejects_before_preparation_or_enablement():
    adapter = RecordingAdapter(frozenset({"first_block"}))
    cache = DiffusersCacheAccelerator(policy="taylorseer")

    with pytest.raises(
        ValueError,
        match=r"RecordingAdapter does not support policy='taylorseer'.*\['first_block'\]",
    ):
        with cache.rollout_context(adapter):
            pass

    assert adapter.events == []


def test_invalid_config_rejects_before_preparation_or_enablement():
    adapter = RecordingAdapter(frozenset({"first_block"}))
    cache = DiffusersCacheAccelerator(policy="first_block", unknown_parameter=True)

    with pytest.raises(ValueError, match="invalid parameters.*unknown_parameter"):
        with cache.rollout_context(adapter):
            pass

    assert adapter.events == []


@pytest.mark.parametrize("threshold", ["bad", True, float("nan"), float("inf"), -0.01])
def test_invalid_first_block_threshold_rejects_before_cache_preparation(threshold):
    adapter = RecordingAdapter(frozenset({"first_block"}))
    cache = DiffusersCacheAccelerator(policy="first_block", threshold=threshold)

    with pytest.raises(ValueError, match="finite non-negative real number"):
        with cache.rollout_context(adapter):
            pass

    assert adapter.events == []


def test_all_components_validate_before_any_cache_preparation():
    adapter = RecordingAdapter(frozenset({"first_block"}))
    adapter.transformer_names = ["transformer", "transformer_2"]
    adapter.get_component = lambda name: (
        adapter.transformer if name == "transformer" else object()
    )
    cache = DiffusersCacheAccelerator(policy="first_block")

    with pytest.raises(ValueError, match="component 'transformer_2'.*CacheMixin"):
        with cache.rollout_context(adapter):
            pass

    assert adapter.events == []


def test_supported_policy_prepares_before_enable_and_disables_on_exit():
    adapter = RecordingAdapter(frozenset({"first_block"}))
    cache = DiffusersCacheAccelerator(threshold=0.05)

    with cache.rollout_context(adapter):
        assert adapter.transformer.is_cache_enabled
        assert adapter.events == [
            ("prepare", "first_block", "transformer"),
            ("enable", "FirstBlockCacheConfig"),
        ]

    assert adapter.events[-1] == ("disable",)
    assert not adapter.transformer.is_cache_enabled


def test_unrestricted_cache_ready_adapter_preserves_existing_policy_behavior():
    adapter = RecordingAdapter(None)
    cache = DiffusersCacheAccelerator(policy="taylorseer", cache_interval=2)

    with cache.rollout_context(adapter):
        assert adapter.events[:2] == [
            ("prepare", "taylorseer", "transformer"),
            ("enable", "TaylorSeerCacheConfig"),
        ]

    assert adapter.events[-1] == ("disable",)


def test_cache_targets_declared_reference_transformer_name():
    adapter = RecordingAdapter(frozenset({"first_block"}), component_name="transformer_ref")
    cache = DiffusersCacheAccelerator(policy="first_block")

    with cache.rollout_context(adapter):
        assert adapter.events[:2] == [
            ("prepare", "first_block", "transformer_ref"),
            ("enable", "FirstBlockCacheConfig"),
        ]

    assert adapter.events[-1] == ("disable",)
