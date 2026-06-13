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

"""Regression tests for the OpenAI-compatible (API-based) reward models.

Guards against the loop-binding bug where ``asyncio.Semaphore`` / ``AsyncOpenAI``
were created once in ``__init__`` and then reused across the fresh event loop that
``asyncio.run`` spins up on every ``__call__`` (and across ``ThreadPoolExecutor``
workers in the async-reward path), which raised
``RuntimeError: ... is bound to a different event loop`` on the second loop.

No network is used: the OpenAI async client is replaced with a fake that returns
empty content, so each reward falls back to its documented 0.0 reward. The tests
only assert shapes and that no loop-binding error is raised.
"""

from __future__ import annotations

import asyncio
import sys
import types
from concurrent.futures import ThreadPoolExecutor

import pytest
import torch
from PIL import Image

from flow_factory.hparams import RewardArguments


class _FakeChoice:
    def __init__(self, content: str):
        self.message = types.SimpleNamespace(content=content)
        self.logprobs = None


class _FakeCompletion:
    def __init__(self, content: str):
        self.choices = [_FakeChoice(content)]


class _FakeCompletions:
    async def create(self, **kwargs):
        # Yield to the running loop so the semaphore is actually acquired here.
        await asyncio.sleep(0)
        # Empty content -> every text reward takes its "empty reply -> 0.0" path,
        # and the vllm logprob parse fails fast into its 0.0 fallback. This keeps
        # the test independent of each judge's response schema.
        return _FakeCompletion("")


class FakeAsyncOpenAI:
    """Stand-in supporting ``async with`` and ``chat.completions.create``."""

    def __init__(self, *args, **kwargs):
        self.chat = types.SimpleNamespace(completions=_FakeCompletions())

    async def __aenter__(self):
        return self

    async def __aexit__(self, *exc):
        return False


class _FakeTransportError(Exception):
    pass


@pytest.fixture
def fake_openai(monkeypatch):
    """Install a fake ``openai`` module and patch the symbol qwen bound at import."""
    fake_module = types.ModuleType("openai")
    fake_module.AsyncOpenAI = FakeAsyncOpenAI
    fake_module.APIConnectionError = _FakeTransportError
    fake_module.APITimeoutError = _FakeTransportError
    fake_module.RateLimitError = _FakeTransportError
    monkeypatch.setitem(sys.modules, "openai", fake_module)

    # qwen imports ``AsyncOpenAI`` at module load, so patch the bound name too.
    from flow_factory.rewards.qwen_image_bench import reward as qwen_reward

    monkeypatch.setattr(qwen_reward, "AsyncOpenAI", FakeAsyncOpenAI)
    return fake_module


def _accelerator_stub():
    return types.SimpleNamespace(device=torch.device("cpu"))


def _config(reward_model: str, **extra) -> RewardArguments:
    base = {
        "name": "t",
        "reward_model": reward_model,
        "device": "cpu",
        "dtype": "float32",
        "batch_size": 4,
        "max_retries": 1,
        "max_concurrent": 2,
        "timeout": 1,
    }
    base.update(extra)
    return RewardArguments.from_dict(base)


def _make_qwen():
    from flow_factory.rewards.qwen_image_bench.reward import QwenImageBenchRewardModel

    return QwenImageBenchRewardModel(_config("qwen_image_bench"), _accelerator_stub())


def _make_t2i():
    from flow_factory.rewards.rational_rewards_t2i import RationalRewardsT2IRewardModel

    return RationalRewardsT2IRewardModel(_config("rational_rewards_t2i"), _accelerator_stub())


def _make_edit():
    from flow_factory.rewards.rational_rewards_edit import RationalRewardsEditRewardModel

    return RationalRewardsEditRewardModel(_config("rational_rewards_edit"), _accelerator_stub())


def _make_vllm():
    from flow_factory.rewards.vllm_evaluate import VLMEvaluateRewardModel

    return VLMEvaluateRewardModel(_config("vllm_evaluate"), _accelerator_stub())


_FACTORIES = {
    "qwen_image_bench": _make_qwen,
    "rational_rewards_t2i": _make_t2i,
    "rational_rewards_edit": _make_edit,
    "vllm_evaluate": _make_vllm,
}


def _img() -> Image.Image:
    return Image.new("RGB", (8, 8), color=(127, 127, 127))


def _call(reward, n: int):
    """Invoke ``reward`` on a batch of ``n`` (prompt, image) pairs."""
    prompts = [f"a photo {i}" for i in range(n)]
    images = [_img() for _ in range(n)]
    kwargs = {}
    # rational_rewards_edit additionally requires a source (condition) image.
    if reward.__class__.__name__ == "RationalRewardsEditRewardModel":
        kwargs["condition_images"] = [[_img()] for _ in range(n)]
    return reward(prompt=prompts, image=images, **kwargs)


@pytest.mark.parametrize("name", list(_FACTORIES))
def test_no_loop_bound_state_cached_on_instance(name, fake_openai):
    reward = _FACTORIES[name]()
    # The fix: the loop-bound client/semaphore must not be cached on the
    # instance (they are created per call, inside the event loop, instead).
    assert not hasattr(reward, "client")
    assert not hasattr(reward, "semaphore")


@pytest.mark.parametrize("name", list(_FACTORIES))
def test_sequential_calls_use_independent_loops(name, fake_openai):
    reward = _FACTORIES[name]()
    # Two consecutive __call__s each run their own asyncio.run() loop. The old
    # code raised "bound to a different event loop" on the second one.
    out1 = _call(reward, 2)
    out2 = _call(reward, 3)
    assert tuple(out1.rewards.shape) == (2,)
    assert tuple(out2.rewards.shape) == (3,)


@pytest.mark.parametrize("name", list(_FACTORIES))
def test_concurrent_thread_pool_calls(name, fake_openai):
    reward = _FACTORIES[name]()
    # Mirrors the async-reward ThreadPoolExecutor path: several __call__s, each
    # in its own worker thread + event loop, sharing one reward instance.
    with ThreadPoolExecutor(max_workers=4) as pool:
        futures = [pool.submit(_call, reward, 2) for _ in range(6)]
        outputs = [f.result() for f in futures]
    assert all(tuple(o.rewards.shape) == (2,) for o in outputs)
