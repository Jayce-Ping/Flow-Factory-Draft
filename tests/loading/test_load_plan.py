from dataclasses import FrozenInstanceError

import pytest

from flow_factory.loading import (
    ComponentDescriptor,
    ComponentRole,
    ComponentStage,
    LoadPlanner,
    MaterializationMode,
)


def _component(
    name: str,
    root: str,
    *,
    path: str | tuple[str, ...] = (),
    role: ComponentRole = ComponentRole.AUXILIARY,
    stages: set[ComponentStage] | frozenset[ComponentStage],
    mode: MaterializationMode = MaterializationMode.FULL,
) -> ComponentDescriptor:
    return ComponentDescriptor(
        name=name,
        root=root,
        path=path,
        role=role,
        stages=stages,
        mode=mode,
    )


def test_planner_emits_each_physical_root_exactly_once() -> None:
    plan = LoadPlanner().build(
        [
            _component(
                "encoder",
                "model",
                path="encoder",
                stages={ComponentStage.PREPROCESS},
            ),
            _component(
                "decoder",
                "model",
                path="decoder",
                stages={ComponentStage.ROLLOUT},
            ),
            _component(
                "vae",
                "vae",
                stages={ComponentStage.PREPROCESS, ComponentStage.ROLLOUT},
            ),
        ]
    )

    visited_roots = [request.root for request in plan]

    assert visited_roots == ["model", "vae"]
    assert len(visited_roots) == len(set(visited_roots))
    assert plan.requests["model"].logical_names == ("encoder", "decoder")


def test_bagel_transformer_alias_promotes_physical_root_to_target() -> None:
    plan = LoadPlanner().plan(
        [
            _component(
                "bagel",
                "bagel",
                role=ComponentRole.AUXILIARY,
                stages={ComponentStage.ROLLOUT},
            ),
            _component(
                "transformer",
                "bagel",
                path="language_model",
                role=ComponentRole.TARGET,
                stages={ComponentStage.OPTIMIZE, ComponentStage.ROLLOUT},
            ),
        ]
    )

    request = plan.request_for_root("bagel")

    assert len(plan) == 1
    assert request.role is ComponentRole.TARGET
    assert request.routes == {
        "bagel": (),
        "transformer": ("language_model",),
    }
    assert plan.routes["transformer"] == "bagel.language_model"
    assert plan.request_for_component("transformer") is request


@pytest.mark.parametrize(
    ("descriptors", "message"),
    [
        (
            [
                _component(
                    "transformer",
                    "first",
                    stages={ComponentStage.OPTIMIZE},
                ),
                _component(
                    "transformer",
                    "second",
                    stages={ComponentStage.ROLLOUT},
                ),
            ],
            r"logical component 'transformer'.*conflicts",
        ),
        (
            [
                _component(
                    "shared",
                    "shared",
                    role=ComponentRole.AUXILIARY,
                    stages={ComponentStage.ROLLOUT},
                ),
                _component(
                    "reward_alias",
                    "shared",
                    path="scorer",
                    role=ComponentRole.REWARD,
                    stages={ComponentStage.REWARD},
                ),
            ],
            r"one role per physical root.*root='shared'",
        ),
        (
            [
                _component(
                    "model",
                    "model",
                    stages={ComponentStage.ROLLOUT},
                ),
                _component(
                    "model_config",
                    "model",
                    path="config",
                    stages={ComponentStage.PREPROCESS},
                    mode=MaterializationMode.CONFIG_ONLY,
                ),
            ],
            r"one materialization mode.*root 'model'",
        ),
    ],
)
def test_planner_fails_fast_on_conflicting_declarations(
    descriptors: list[ComponentDescriptor],
    message: str,
) -> None:
    with pytest.raises(ValueError, match=message):
        LoadPlanner().build(descriptors)


def test_planner_merges_repeated_and_root_stage_requirements() -> None:
    plan = LoadPlanner().build(
        [
            _component(
                "vae",
                "vae",
                stages={ComponentStage.PREPROCESS},
            ),
            _component(
                "vae",
                "vae",
                stages={ComponentStage.EVALUATE},
            ),
            _component(
                "vae_decoder",
                "vae",
                path="decoder",
                stages={ComponentStage.ROLLOUT},
            ),
            _component(
                "transformer",
                "transformer",
                role=ComponentRole.TARGET,
                stages={ComponentStage.OPTIMIZE},
            ),
        ]
    )

    assert plan.descriptors["vae"].stages == {
        ComponentStage.PREPROCESS,
        ComponentStage.EVALUATE,
    }
    assert plan.requests["vae"].stages == {
        ComponentStage.PREPROCESS,
        ComponentStage.ROLLOUT,
        ComponentStage.EVALUATE,
    }
    assert [request.root for request in plan.requests_for_stage(ComponentStage.ROLLOUT)] == ["vae"]
    assert [request.root for request in plan.requests_for_stage(ComponentStage.OPTIMIZE)] == [
        "transformer"
    ]


def test_plan_dataclasses_and_mappings_are_immutable() -> None:
    mutable_stages = {ComponentStage.ROLLOUT}
    descriptor = _component("vae", "vae", stages=mutable_stages)
    plan = LoadPlanner().build([descriptor])
    mutable_stages.add(ComponentStage.EVALUATE)

    assert descriptor.stages == {ComponentStage.ROLLOUT}
    with pytest.raises(FrozenInstanceError):
        setattr(descriptor, "root", "other")
    with pytest.raises(TypeError):
        plan.requests["other"] = plan.requests["vae"]
    with pytest.raises(TypeError):
        plan.requests["vae"].descriptors["other"] = descriptor
    with pytest.raises(TypeError):
        plan.requests["vae"].routes["other"] = ()
