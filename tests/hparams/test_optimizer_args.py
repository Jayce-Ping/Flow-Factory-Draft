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
import torch

from flow_factory.hparams.optimizer_args import (
    AdamWOptimizerArguments,
    MultiOptimizerArguments,
    MuonOptimizerArguments,
)
from flow_factory.optimizer import CompositeOptimizer, build_optimizer, split_muon_parameters

requires_muon_api = pytest.mark.skipif(
    not hasattr(torch.optim, "Muon"),
    reason="functional Muon tests require torch.optim.Muon",
)


def test_the_optimizer_key_selects_the_arguments_subclass() -> None:
    """AdamW and Muon hyperparameters live on separate classes, chosen by one key."""
    args = MultiOptimizerArguments.from_dict(
        [
            {"name": "generator", "optimizer": "muon", "learning_rate": 2e-5},
            {"name": "fake", "learning_rate": 1e-5, "update_frequency": 5},
        ]
    )

    generator = args.get_by_name("generator")
    fake = args.get_by_name("fake")
    assert isinstance(generator, MuonOptimizerArguments)
    assert isinstance(fake, AdamWOptimizerArguments)
    assert not hasattr(fake, "ns_steps")
    assert not hasattr(generator, "betas")
    assert fake.update_frequency == 5


def test_duplicate_names_are_rejected_because_lookup_would_be_ambiguous() -> None:
    """Two configurations for one variant have no defined precedence."""
    with pytest.raises(ValueError, match="unique optimizer names"):
        MultiOptimizerArguments.from_dict(
            [{"name": "generator"}, {"name": "generator", "learning_rate": 1e-4}]
        )


def test_muon_arguments_reject_an_unknown_learning_rate_adjustment() -> None:
    """The adjustment keeps the orthogonalized update's RMS consistent; typos matter."""
    with pytest.raises(ValueError, match="adjust_lr_fn"):
        MuonOptimizerArguments(name="generator", adjust_lr_fn="whatever")


def test_all_adamw_configurations_build_one_plain_adamw() -> None:
    """The common case must stay exactly what it was before optimizer selection."""
    parameters = {"base": [torch.nn.Parameter(torch.randn(4, 3))]}
    configs = (AdamWOptimizerArguments(name="base", learning_rate=1e-4),)

    optimizer = build_optimizer(configs, parameters)

    assert type(optimizer) is torch.optim.AdamW
    assert [group["role_name"] for group in optimizer.param_groups] == ["base"]


@requires_muon_api
def test_muon_splits_a_variant_into_its_matrices_and_the_adamw_remainder() -> None:
    """torch.optim.Muon rejects non-matrix parameters, so the rest needs AdamW."""
    matrix = torch.nn.Parameter(torch.randn(4, 3))
    bias = torch.nn.Parameter(torch.randn(4))
    matrices, remainder = split_muon_parameters([matrix, bias])
    assert matrices == [matrix]
    assert remainder == [bias]

    optimizer = build_optimizer(
        (MuonOptimizerArguments(name="base", learning_rate=1e-3),),
        {"base": [matrix, bias]},
    )

    assert isinstance(optimizer, CompositeOptimizer)
    assert [type(child).__name__ for child in optimizer.optimizers] == ["Muon", "AdamW"]
    assert [group["role_name"] for group in optimizer.param_groups] == ["base", "base"]


@requires_muon_api
def test_a_composite_steps_every_child_and_round_trips_its_state() -> None:
    """The framework prepares one optimizer, so the composite must behave like one."""
    matrix = torch.nn.Parameter(torch.randn(4, 3))
    bias = torch.nn.Parameter(torch.randn(4))
    optimizer = build_optimizer(
        (MuonOptimizerArguments(name="base", learning_rate=1e-2),),
        {"base": [matrix, bias]},
    )
    before = (matrix.detach().clone(), bias.detach().clone())

    (matrix.square().sum() + bias.square().sum()).backward()
    optimizer.step()

    assert not torch.equal(matrix, before[0])
    assert not torch.equal(bias, before[1])

    restored = build_optimizer(
        (MuonOptimizerArguments(name="base", learning_rate=1e-2),),
        {"base": [matrix, bias]},
    )
    restored.load_state_dict(optimizer.state_dict())
    assert len(restored.state_dict()["composite"]) == 2

    optimizer.zero_grad()
    assert matrix.grad is None


@requires_muon_api
def test_mixed_optimizers_across_variants_share_one_root() -> None:
    """One variant on Muon and another on AdamW still yields a single optimizer."""
    generator = torch.nn.Parameter(torch.randn(4, 3))
    fake = torch.nn.Parameter(torch.randn(3, 2))

    optimizer = build_optimizer(
        (
            MuonOptimizerArguments(name="generator", learning_rate=1e-3),
            AdamWOptimizerArguments(name="fake", learning_rate=1e-4),
        ),
        {"generator": [generator], "fake": [fake]},
    )

    assert isinstance(optimizer, CompositeOptimizer)
    assert sorted(group["role_name"] for group in optimizer.param_groups) == ["fake", "generator"]


def test_muon_requires_at_least_one_matrix_parameter() -> None:
    """A variant of scalars has nothing for Muon to orthogonalize."""
    with pytest.raises(ValueError, match="at least one matrix parameter"):
        build_optimizer(
            (MuonOptimizerArguments(name="base"),),
            {"base": [torch.nn.Parameter(torch.randn(4))]},
        )


def test_muon_requires_the_optional_pytorch_api(monkeypatch: pytest.MonkeyPatch) -> None:
    """Direct optimizer construction must share the trainer's capability error."""
    monkeypatch.delattr(torch.optim, "Muon", raising=False)

    with pytest.raises(ValueError, match="torch.optim.Muon is unavailable"):
        build_optimizer(
            (MuonOptimizerArguments(name="base"),),
            {"base": [torch.nn.Parameter(torch.randn(4, 3))]},
        )


@requires_muon_api
def test_a_composite_refuses_to_have_its_groups_reassigned() -> None:
    """The group list is a view; reassigning it would detach the children silently."""
    optimizer = build_optimizer(
        (MuonOptimizerArguments(name="base"),),
        {"base": [torch.nn.Parameter(torch.randn(4, 3))]},
    )

    with pytest.raises(AttributeError, match="cannot be reassigned"):
        optimizer.param_groups = []


def test_the_optimizer_entry_owns_the_clip_norm_and_mirrors_it_for_the_shared_step() -> None:
    """One authoring location: the shared gradient step must not disagree with it."""
    from flow_factory.hparams import Arguments

    config = Arguments.from_dict(
        {
            "train": {"trainer_type": "grpo", "max_grad_norm": 9.0},
            "optimizers": [{"name": "default", "learning_rate": 2e-5, "max_grad_norm": 0.5}],
        }
    )

    assert config.optimizer_args[0].max_grad_norm == 0.5
    assert config.training_args.max_grad_norm == 0.5


def test_the_flat_clip_norm_still_feeds_the_synthesized_entry() -> None:
    """A config without an `optimizers` list keeps configuring the clip under train."""
    from flow_factory.hparams import Arguments

    config = Arguments.from_dict(
        {"train": {"trainer_type": "grpo", "learning_rate": 1e-5, "max_grad_norm": 0.25}}
    )

    assert config.optimizer_args[0].max_grad_norm == 0.25
    assert config.training_args.max_grad_norm == 0.25
