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

"""Helper-level parity for the migrated DiffusionOPD target and loss math.

Every structured helper is pinned twice: against the pre-migration single-tensor
formula for the legacy one-component replay, and against a hand-written
element-weighted oracle for heterogeneous component groups.
"""

from types import SimpleNamespace
from typing import Any, Dict, List, Mapping, Optional

import pytest
import torch

from flow_factory.models.abc import BaseAdapter
from flow_factory.samples import (
    BaseSample,
    ComponentTimes,
    LatentState,
    MultiModalStepOutput,
)
from flow_factory.scheduler import SchedulerGroup, SDESchedulerMixin, SDESchedulerOutput
from flow_factory.trainers.opd.common import (
    compute_per_sample_distillation_loss,
    compute_structured_distillation_loss,
    project_distillation_target,
    project_distillation_target_state,
    resolve_scheduler_group_dynamics,
    validate_loss_target_for_dynamics,
)
from flow_factory.utils.base import to_broadcast_tensor
from flow_factory.utils.noise_schedule import flow_match_sigma


class SchedulerFake:
    """Scheduler stub exposing the dynamics surface DiffusionOPD reads."""

    def __init__(self, dynamics_type: str = "ODE", noise_level: float = 0.7) -> None:
        self.dynamics_type = dynamics_type
        self.noise_level = noise_level
        self.train_timesteps = torch.tensor([0, 1])
        self.seeds: List[int] = []

    def step(self) -> None:
        """Provide scheduler compatibility."""

    def set_seed(self, seed: int) -> None:
        """Record the dispatched seed."""
        self.seeds.append(seed)

    # The KL denominator is pure dynamics arithmetic, so the real implementation
    # is reused rather than duplicated into the fake.
    get_kl_divergence_denominator = SDESchedulerMixin.get_kl_divergence_denominator


class AdapterFake(BaseAdapter):
    """Minimal single-component adapter recording forward arguments."""

    def load_pipeline(self) -> Any:
        """Return an unused pipeline fake."""
        raise NotImplementedError

    def decode_latents(self, latents: torch.Tensor, **kwargs: Any) -> torch.Tensor:
        """Return latents unchanged."""
        return latents

    def inference(self, **kwargs: Any) -> List[BaseSample]:
        """Return no samples."""
        return []

    def forward(self, **kwargs: Any) -> SDESchedulerOutput:
        """Record kwargs and return a deterministic scheduler output."""
        self.forward_kwargs = kwargs
        latents = kwargs["latents"]
        broadcast_shape = (latents.shape[0],) + (1,) * (latents.ndim - 1)
        return SDESchedulerOutput(
            next_latents=latents + 1,
            next_latents_mean=latents + 2,
            std_dev_t=torch.full(broadcast_shape, 0.25),
            dt=torch.full(broadcast_shape, -0.5),
            velocity=latents + 3,
        )


class StructuredAdapterFake(AdapterFake):
    """Adapter fake declaring a heterogeneous video/audio component contract."""

    trajectory_component_order = ("video", "audio")


class DynamicMaskAdapterFake(AdapterFake):
    """Adapter reducing only the positions the current state marks active."""

    def _reduce_latent_values(
        self,
        values: Mapping[str, torch.Tensor],
        *,
        active_numel: Optional[Mapping[str, int]] = None,
        state: Optional[LatentState] = None,
    ) -> torch.Tensor:
        """Average the elements the per-sample state mask selects."""
        if state is None:
            raise ValueError("expected state context for DynamicMaskAdapterFake reduction")
        mask = state.components["latent"]
        flattened = values["latent"].reshape(mask.shape[0], -1)
        return (flattened * mask).sum(dim=1) / mask.sum(dim=1)


def _adapter(dynamics_type: str = "ODE") -> AdapterFake:
    adapter = object.__new__(AdapterFake)
    adapter.pipeline = SimpleNamespace(scheduler=SchedulerFake(dynamics_type))
    adapter.scheduler_group = adapter.build_scheduler_group()
    return adapter


def _dynamic_mask_adapter() -> DynamicMaskAdapterFake:
    adapter = object.__new__(DynamicMaskAdapterFake)
    adapter.pipeline = SimpleNamespace(scheduler=SchedulerFake())
    adapter.scheduler_group = adapter.build_scheduler_group()
    return adapter


def _structured_adapter(
    video_dynamics: str = "ODE", audio_dynamics: str = "ODE"
) -> StructuredAdapterFake:
    adapter = object.__new__(StructuredAdapterFake)
    video = SchedulerFake(video_dynamics)
    adapter.pipeline = SimpleNamespace(scheduler=video)
    adapter.scheduler_group = SchedulerGroup(
        {"video": video, "audio": SchedulerFake(audio_dynamics)},
        primary_name="video",
    )
    return adapter


def _legacy_times(timestep: torch.Tensor) -> ComponentTimes:
    """Component times for the legacy single-``latent`` replay (no stored sigma)."""
    return ComponentTimes(
        timestep={"latent": timestep},
        next_timestep={"latent": torch.zeros_like(timestep)},
    )


# ============================ Target projection ============================


@pytest.mark.parametrize("loss_target", ["xt", "v", "x0"])
def test_structured_projection_matches_the_legacy_single_tensor_helper(
    loss_target: str,
) -> None:
    torch.manual_seed(0)
    latents = torch.randn(2, 3, 4)
    next_mean = torch.randn(2, 3, 4)
    velocity = torch.randn(2, 3, 4)
    timestep = torch.tensor([700.0, 300.0])
    adapter = _adapter()

    projected = project_distillation_target_state(
        adapter,
        loss_target=loss_target,
        state=LatentState({"latent": latents}),
        output=MultiModalStepOutput(
            next_state_mean=LatentState({"latent": next_mean}),
            velocity=LatentState({"latent": velocity}),
        ),
        times=_legacy_times(timestep),
    )

    legacy = project_distillation_target(
        loss_target=loss_target,
        latents=latents,
        timestep=timestep,
        next_latents_mean=next_mean,
        velocity=velocity,
    )
    assert projected.component_names == ("latent",)
    assert torch.equal(projected.components["latent"], legacy)


def test_legacy_helper_accepts_an_explicit_sigma_instead_of_the_flow_match_schedule() -> None:
    torch.manual_seed(1)
    latents = torch.randn(2, 5)
    velocity = torch.randn(2, 5)
    sigma = torch.tensor([0.125, 0.875])

    projected = project_distillation_target(
        loss_target="x0",
        latents=latents,
        timestep=torch.tensor([700.0, 300.0]),
        next_latents_mean=None,
        velocity=velocity,
        sigma=sigma,
    )

    expected = latents.float() - to_broadcast_tensor(sigma, latents.float()) * velocity.float()
    assert torch.equal(projected, expected)


def test_structured_x0_projection_uses_each_component_stored_sigma() -> None:
    torch.manual_seed(2)
    video, audio = torch.randn(2, 3, 4), torch.randn(2, 5)
    video_v, audio_v = torch.randn(2, 3, 4), torch.randn(2, 5)
    video_sigma, audio_sigma = torch.tensor([0.25, 0.75]), torch.tensor([0.5, 0.125])
    adapter = _structured_adapter()

    projected = project_distillation_target_state(
        adapter,
        loss_target="x0",
        state=LatentState({"video": video, "audio": audio}),
        output=MultiModalStepOutput(velocity=LatentState({"video": video_v, "audio": audio_v})),
        times=ComponentTimes(
            timestep={"video": torch.tensor([700.0, 300.0]), "audio": torch.tensor([650.0, 250.0])},
            next_timestep={"video": torch.zeros(2), "audio": torch.zeros(2)},
            sigma={"video": video_sigma, "audio": audio_sigma},
            next_sigma={"video": torch.zeros(2), "audio": torch.zeros(2)},
        ),
    )

    assert projected.component_names == ("video", "audio")
    assert torch.equal(
        projected.components["video"],
        video.float() - to_broadcast_tensor(video_sigma, video.float()) * video_v.float(),
    )
    assert torch.equal(
        projected.components["audio"],
        audio.float() - to_broadcast_tensor(audio_sigma, audio.float()) * audio_v.float(),
    )


def test_structured_xt_projection_reads_the_per_component_transition_mean() -> None:
    torch.manual_seed(3)
    means = {"video": torch.randn(2, 3, 4), "audio": torch.randn(2, 5)}
    adapter = _structured_adapter()

    projected = project_distillation_target_state(
        adapter,
        loss_target="xt",
        state=LatentState({"video": torch.zeros(2, 3, 4), "audio": torch.zeros(2, 5)}),
        output=MultiModalStepOutput(next_state_mean=LatentState(dict(means))),
        times=ComponentTimes(
            timestep={"video": torch.full((2,), 700.0), "audio": torch.full((2,), 650.0)},
            next_timestep={"video": torch.zeros(2), "audio": torch.zeros(2)},
        ),
    )

    for name, value in means.items():
        assert torch.equal(projected.components[name], value)


def test_structured_projection_rejects_a_multi_component_x0_without_stored_sigma() -> None:
    adapter = _structured_adapter()

    with pytest.raises(
        ValueError,
        match=r"loss_target='x0'.*component order \('video', 'audio'\).*stored sigma",
    ):
        project_distillation_target_state(
            adapter,
            loss_target="x0",
            state=LatentState({"video": torch.zeros(2, 3), "audio": torch.zeros(2, 5)}),
            output=MultiModalStepOutput(
                velocity=LatentState({"video": torch.zeros(2, 3), "audio": torch.zeros(2, 5)})
            ),
            times=ComponentTimes(
                timestep={"video": torch.zeros(2), "audio": torch.zeros(2)},
                next_timestep={"video": torch.zeros(2), "audio": torch.zeros(2)},
            ),
        )


def test_structured_projection_rejects_a_state_in_the_wrong_component_order() -> None:
    adapter = _structured_adapter()

    with pytest.raises(
        ValueError,
        match=r"state.*component order \('video', 'audio'\), received \('audio', 'video'\)",
    ):
        project_distillation_target_state(
            adapter,
            loss_target="xt",
            state=LatentState({"audio": torch.zeros(2, 5), "video": torch.zeros(2, 3)}),
            output=MultiModalStepOutput(
                next_state_mean=LatentState(
                    {"video": torch.zeros(2, 3), "audio": torch.zeros(2, 5)}
                )
            ),
            times=ComponentTimes(
                timestep={"video": torch.zeros(2), "audio": torch.zeros(2)},
                next_timestep={"video": torch.zeros(2), "audio": torch.zeros(2)},
            ),
        )


def test_structured_projection_rejects_a_missing_output_field() -> None:
    adapter = _adapter()

    with pytest.raises(
        ValueError,
        match=r"next_state_mean.*loss_target='xt'.*component order \('latent',\).*received None",
    ):
        project_distillation_target_state(
            adapter,
            loss_target="xt",
            state=LatentState({"latent": torch.zeros(2, 3)}),
            output=MultiModalStepOutput(velocity=LatentState({"latent": torch.zeros(2, 3)})),
            times=_legacy_times(torch.zeros(2)),
        )


def test_structured_projection_rejects_an_output_shape_that_differs_from_the_state() -> None:
    adapter = _adapter()

    with pytest.raises(
        ValueError,
        match=r"component 'latent'.*shape \(2, 3\).*received.*\(2, 4\)",
    ):
        project_distillation_target_state(
            adapter,
            loss_target="v",
            state=LatentState({"latent": torch.zeros(2, 3)}),
            output=MultiModalStepOutput(velocity=LatentState({"latent": torch.zeros(2, 4)})),
            times=_legacy_times(torch.zeros(2)),
        )


def test_structured_projection_rejects_a_sigma_that_is_not_one_value_per_sample() -> None:
    adapter = _adapter()

    with pytest.raises(
        ValueError,
        match=r"sigma.*component 'latent'.*one value per sample.*\(2, 3\)",
    ):
        project_distillation_target_state(
            adapter,
            loss_target="x0",
            state=LatentState({"latent": torch.zeros(2, 3)}),
            output=MultiModalStepOutput(velocity=LatentState({"latent": torch.zeros(2, 3)})),
            times=ComponentTimes(
                timestep={"latent": torch.zeros(2)},
                next_timestep={"latent": torch.zeros(2)},
                sigma={"latent": torch.zeros(2, 3)},
                next_sigma={"latent": torch.zeros(2, 3)},
            ),
        )


def test_structured_projection_rejects_an_unknown_target_space() -> None:
    adapter = _adapter()

    with pytest.raises(ValueError, match=r"\('xt', 'v', 'x0'\), got 'mean'"):
        project_distillation_target_state(
            adapter,
            loss_target="mean",
            state=LatentState({"latent": torch.zeros(2, 3)}),
            output=MultiModalStepOutput(velocity=LatentState({"latent": torch.zeros(2, 3)})),
            times=_legacy_times(torch.zeros(2)),
        )


# ======================= Replay time contract validation =======================


def _structured_state() -> LatentState:
    return LatentState({"video": torch.zeros(2, 3), "audio": torch.zeros(2, 5)})


def _structured_output() -> MultiModalStepOutput:
    return MultiModalStepOutput(
        next_state_mean=LatentState({"video": torch.zeros(2, 3), "audio": torch.zeros(2, 5)}),
        velocity=LatentState({"video": torch.zeros(2, 3), "audio": torch.zeros(2, 5)}),
    )


def _structured_times(**overrides: Any) -> ComponentTimes:
    fields: Dict[str, Any] = {
        "timestep": {"video": torch.zeros(2), "audio": torch.zeros(2)},
        "next_timestep": {"video": torch.zeros(2), "audio": torch.zeros(2)},
        "sigma": {"video": torch.zeros(2), "audio": torch.zeros(2)},
        "next_sigma": {"video": torch.zeros(2), "audio": torch.zeros(2)},
    }
    fields.update(overrides)
    return ComponentTimes(**fields)


@pytest.mark.parametrize("loss_target", ["xt", "v", "x0"])
def test_structured_projection_validates_the_times_contract_for_every_target(
    loss_target: str,
) -> None:
    """`xt` and `v` never read a sigma, but a drifted replay is still a bug.

    The whole `ComponentTimes` contract is checked before the target branch, so
    a corrupted sigma cannot slip through the two targets that ignore it.
    """
    adapter = _structured_adapter()

    with pytest.raises(ValueError, match=r"sigma.*component 'video'.*one value per sample"):
        project_distillation_target_state(
            adapter,
            loss_target=loss_target,
            state=_structured_state(),
            output=_structured_output(),
            times=_structured_times(
                sigma={"video": torch.zeros(2, 3), "audio": torch.zeros(2)},
            ),
        )


def test_structured_projection_rejects_a_timestep_component_order_mismatch() -> None:
    adapter = _structured_adapter()

    with pytest.raises(
        ValueError,
        match=r"timestep.*component order \('video', 'audio'\), received \('audio', 'video'\)",
    ):
        project_distillation_target_state(
            adapter,
            loss_target="xt",
            state=_structured_state(),
            output=_structured_output(),
            times=ComponentTimes(
                timestep={"audio": torch.zeros(2), "video": torch.zeros(2)},
                next_timestep={"audio": torch.zeros(2), "video": torch.zeros(2)},
            ),
        )


@pytest.mark.parametrize("field", ["timestep", "next_timestep", "sigma", "next_sigma"])
def test_structured_projection_rejects_a_time_field_that_is_not_one_value_per_sample(
    field: str,
) -> None:
    adapter = _structured_adapter()
    drifted = {"video": torch.zeros(2, 3), "audio": torch.zeros(2)}

    with pytest.raises(
        ValueError,
        match=rf"{field}.*component 'video'.*one value per sample.*\(2,\).*\(2, 3\)",
    ):
        project_distillation_target_state(
            adapter,
            loss_target="xt",
            state=_structured_state(),
            output=_structured_output(),
            times=_structured_times(**{field: drifted}),
        )


@pytest.mark.parametrize("field", ["timestep", "next_timestep", "sigma", "next_sigma"])
def test_structured_projection_rejects_a_time_field_on_another_device(field: str) -> None:
    adapter = _structured_adapter()
    elsewhere = {"video": torch.zeros(2, device="meta"), "audio": torch.zeros(2)}

    with pytest.raises(ValueError, match=rf"{field}.*component 'video'.*device"):
        project_distillation_target_state(
            adapter,
            loss_target="xt",
            state=_structured_state(),
            output=_structured_output(),
            times=_structured_times(**{field: elsewhere}),
        )


def test_structured_projection_accepts_the_legacy_terminal_scalar_zero_next_timestep() -> None:
    """The legacy single-``latent`` replay stores the terminal ``t_next`` as scalar 0."""
    adapter = _adapter()

    projected = project_distillation_target_state(
        adapter,
        loss_target="v",
        state=LatentState({"latent": torch.zeros(2, 3)}),
        output=MultiModalStepOutput(velocity=LatentState({"latent": torch.ones(2, 3)})),
        times=ComponentTimes(
            timestep={"latent": torch.zeros(2)},
            next_timestep={"latent": torch.tensor(0)},
        ),
    )

    assert torch.equal(projected.components["latent"], torch.ones(2, 3))


def test_structured_projection_rejects_a_scalar_next_timestep_for_a_component_group() -> None:
    """The terminal fallback is documented for the legacy single component only."""
    adapter = _structured_adapter()

    with pytest.raises(
        ValueError,
        match=r"next_timestep.*component 'video'.*one value per sample.*\(2,\).*\(\)",
    ):
        project_distillation_target_state(
            adapter,
            loss_target="xt",
            state=_structured_state(),
            output=_structured_output(),
            times=ComponentTimes(
                timestep={"video": torch.zeros(2), "audio": torch.zeros(2)},
                next_timestep={"video": torch.tensor(0), "audio": torch.zeros(2)},
            ),
        )


def test_structured_projection_rejects_a_non_zero_scalar_next_timestep() -> None:
    adapter = _adapter()

    with pytest.raises(
        ValueError,
        match=r"next_timestep.*component 'latent'.*terminal.*scalar 0.*received 250",
    ):
        project_distillation_target_state(
            adapter,
            loss_target="v",
            state=LatentState({"latent": torch.zeros(2, 3)}),
            output=MultiModalStepOutput(velocity=LatentState({"latent": torch.zeros(2, 3)})),
            times=ComponentTimes(
                timestep={"latent": torch.zeros(2)},
                next_timestep={"latent": torch.tensor(250)},
            ),
        )


@pytest.mark.parametrize("role", ["state", "student target", "teacher target"])
def test_structured_helpers_reject_a_component_without_a_batch_axis(role: str) -> None:
    """A 0-D component must raise a named contract error, not ``IndexError``."""
    adapter = _structured_adapter()
    batched = {"video": torch.zeros(2, 3), "audio": torch.zeros(2, 5)}
    malformed = {"video": torch.zeros(2, 3), "audio": torch.tensor(0.0)}

    if role == "state":
        with pytest.raises(
            ValueError,
            match=r"replay state.*component 'audio'.*batched tensor.*received shape \(\)",
        ):
            project_distillation_target_state(
                adapter,
                loss_target="xt",
                state=LatentState(dict(malformed)),
                output=_structured_output(),
                times=_structured_times(),
            )
        return

    targets = {"student target": malformed, "teacher target": batched}
    if role == "teacher target":
        targets = {"student target": batched, "teacher target": malformed}
    with pytest.raises(
        ValueError,
        match=rf"{role}.*component 'audio'.*batched tensor.*received shape \(\)",
    ):
        compute_structured_distillation_loss(
            adapter,
            student_target=LatentState(dict(targets["student target"])),
            teacher_target=LatentState(dict(targets["teacher target"])),
            state=LatentState(dict(batched)),
            self_normalize=False,
        )


def test_structured_helpers_reject_components_with_different_batch_sizes() -> None:
    adapter = _structured_adapter()

    with pytest.raises(
        ValueError,
        match=r"replay state.*component 'audio'.*batch size 2.*received \(3, 5\)",
    ):
        project_distillation_target_state(
            adapter,
            loss_target="xt",
            state=LatentState({"video": torch.zeros(2, 3), "audio": torch.zeros(3, 5)}),
            output=_structured_output(),
            times=_structured_times(),
        )


def test_structured_projection_errors_carry_the_caller_context() -> None:
    adapter = _adapter()

    with pytest.raises(ValueError, match=r"teacher 'teacher_a' pass at step_index=3.*velocity"):
        project_distillation_target_state(
            adapter,
            loss_target="v",
            state=LatentState({"latent": torch.zeros(2, 3)}),
            output=MultiModalStepOutput(),
            times=_legacy_times(torch.zeros(2)),
            context="teacher 'teacher_a' pass at step_index=3",
        )


# ============================ Distillation loss ============================


@pytest.mark.parametrize("self_normalize", [False, True])
def test_one_component_structured_loss_is_bit_identical_to_the_legacy_formula(
    self_normalize: bool,
) -> None:
    torch.manual_seed(4)
    student, teacher = torch.randn(3, 4, 5), torch.randn(3, 4, 5)
    adapter = _adapter()

    loss = compute_structured_distillation_loss(
        adapter,
        student_target=LatentState({"latent": student}),
        teacher_target=LatentState({"latent": teacher}),
        state=LatentState({"latent": torch.zeros(3, 4, 5)}),
        self_normalize=self_normalize,
    )

    legacy = compute_per_sample_distillation_loss(student, teacher, self_normalize=self_normalize)
    assert torch.equal(loss, legacy)


def test_structured_loss_weights_components_by_their_active_degrees_of_freedom() -> None:
    torch.manual_seed(5)
    student = {"video": torch.randn(2, 3, 4), "audio": torch.randn(2, 5)}
    teacher = {"video": torch.randn(2, 3, 4), "audio": torch.randn(2, 5)}
    adapter = _structured_adapter()

    loss = compute_structured_distillation_loss(
        adapter,
        student_target=LatentState(dict(student)),
        teacher_target=LatentState(dict(teacher)),
        state=LatentState({"video": torch.zeros(2, 3, 4), "audio": torch.zeros(2, 5)}),
        self_normalize=False,
    )

    video_sum = (student["video"] - teacher["video"]).float().square().flatten(1).sum(dim=1)
    audio_sum = (student["audio"] - teacher["audio"]).float().square().flatten(1).sum(dim=1)
    assert torch.equal(loss, (video_sum + audio_sum) / 17)


def test_structured_self_normalization_uses_one_global_scale_for_every_component() -> None:
    torch.manual_seed(6)
    student = {"video": torch.randn(2, 3, 4), "audio": torch.randn(2, 5)}
    teacher = {"video": torch.randn(2, 3, 4), "audio": torch.randn(2, 5)}
    adapter = _structured_adapter()

    loss = compute_structured_distillation_loss(
        adapter,
        student_target=LatentState(dict(student)),
        teacher_target=LatentState(dict(teacher)),
        state=LatentState({"video": torch.zeros(2, 3, 4), "audio": torch.zeros(2, 5)}),
        self_normalize=True,
    )

    video_error = (student["video"] - teacher["video"]).float()
    audio_error = (student["audio"] - teacher["audio"]).float()
    scale = (video_error.abs().flatten(1).sum(dim=1) + audio_error.abs().flatten(1).sum(dim=1)) / 17
    squared = (
        video_error.square().flatten(1).sum(dim=1) + audio_error.square().flatten(1).sum(dim=1)
    ) / 17
    assert torch.equal(loss, squared / (scale + 1e-8))


def test_structured_loss_passes_the_current_state_to_a_dynamic_mask_reducer() -> None:
    torch.manual_seed(7)
    student, teacher = torch.randn(2, 4), torch.randn(2, 4)
    adapter = _dynamic_mask_adapter()
    mask = torch.tensor([[1.0, 1.0, 0.0, 0.0], [1.0, 0.0, 0.0, 0.0]])

    loss = compute_structured_distillation_loss(
        adapter,
        student_target=LatentState({"latent": student}),
        teacher_target=LatentState({"latent": teacher}),
        state=LatentState({"latent": mask}),
        self_normalize=False,
    )

    squared = (student - teacher).float().square()
    assert torch.equal(loss, (squared * mask).sum(dim=1) / mask.sum(dim=1))


def test_structured_loss_divides_each_component_by_its_own_denominator() -> None:
    torch.manual_seed(8)
    student = {"video": torch.randn(2, 3, 4), "audio": torch.randn(2, 5)}
    teacher = {"video": torch.randn(2, 3, 4), "audio": torch.randn(2, 5)}
    # Deliberately not powers of two: dividing before or after the reduction is
    # only bit-identical for exactly representable scalings.
    denominators = {
        "video": torch.tensor([0.3, 1.7]),
        "audio": torch.tensor([2.5, 0.7]),
    }
    adapter = _structured_adapter("Flow-SDE", "CPS")

    loss = compute_structured_distillation_loss(
        adapter,
        student_target=LatentState(dict(student)),
        teacher_target=LatentState(dict(teacher)),
        state=LatentState({"video": torch.zeros(2, 3, 4), "audio": torch.zeros(2, 5)}),
        self_normalize=False,
        denominators=denominators,
    )

    # Raw-element oracle: the denominator divides every element of its own
    # component, and exactly one active-DOF weighted reduction follows.
    video_error = (student["video"] - teacher["video"]).float().square()
    audio_error = (student["audio"] - teacher["audio"]).float().square()
    video_sum = (video_error / denominators["video"].reshape(2, 1, 1)).flatten(1).sum(dim=1)
    audio_sum = (audio_error / denominators["audio"].reshape(2, 1)).flatten(1).sum(dim=1)
    assert torch.equal(loss, (video_sum + audio_sum) / 17)


def test_masked_one_component_stochastic_loss_divides_the_masked_reduction() -> None:
    """A per-sample denominator divides the masked reduction, not the raw elements.

    The denominator is a per-sample scalar, so dividing the reduced value is
    mathematically identical to scaling the raw elements while keeping the
    legacy operation order. The samples below (3 and 2 active positions, with
    non-binary denominators) pin both halves: the reduction still sees raw
    latent-shaped values — feeding a pre-reduced ``(B,)`` tensor to a masked
    reducer produces a different number — and the division happens after it.
    """
    torch.manual_seed(10)
    student, teacher = torch.randn(2, 4), torch.randn(2, 4)
    denominator = torch.tensor([0.7, 3.0])
    mask = torch.tensor([[1.0, 1.0, 1.0, 0.0], [1.0, 1.0, 0.0, 0.0]])
    adapter = _dynamic_mask_adapter()

    loss = compute_structured_distillation_loss(
        adapter,
        student_target=LatentState({"latent": student}),
        teacher_target=LatentState({"latent": teacher}),
        state=LatentState({"latent": mask}),
        self_normalize=False,
        denominators={"latent": denominator},
    )

    squared = (student - teacher).float().square()
    masked_mean = (squared * mask).sum(dim=1) / mask.sum(dim=1)
    assert torch.equal(loss, masked_mean / denominator)


@pytest.mark.parametrize(
    "denominator",
    [
        pytest.param([0.25, 0.125], id="binary"),
        pytest.param([0.3, 1.7], id="non-binary"),
    ],
)
@pytest.mark.parametrize("self_normalize", [False, True])
def test_one_component_stochastic_loss_is_the_legacy_loss_over_the_denominator(
    denominator: List[float], self_normalize: bool
) -> None:
    """One component keeps the legacy float order: reduce, normalize, then divide.

    Dividing raw elements instead would only be bit-identical for exactly
    representable denominators, and the real schedulers produce arbitrary
    positive transition variances.
    """
    torch.manual_seed(9)
    student, teacher = torch.randn(2, 3, 4), torch.randn(2, 3, 4)
    denominators = {"latent": torch.tensor(denominator)}
    adapter = _adapter("Flow-SDE")

    loss = compute_structured_distillation_loss(
        adapter,
        student_target=LatentState({"latent": student}),
        teacher_target=LatentState({"latent": teacher}),
        state=LatentState({"latent": torch.zeros(2, 3, 4)}),
        self_normalize=self_normalize,
        denominators=denominators,
    )

    legacy = compute_per_sample_distillation_loss(student, teacher, self_normalize=self_normalize)
    assert torch.equal(loss, legacy / denominators["latent"])


@pytest.mark.parametrize("self_normalize", [False, True])
def test_one_component_stochastic_gradient_matches_the_legacy_formula(
    self_normalize: bool,
) -> None:
    """Loss and gradient stay bit-identical for an arbitrary positive denominator."""
    torch.manual_seed(12)
    teacher = torch.randn(2, 3, 4)
    denominator = torch.tensor([0.3, 1.7])
    latents = torch.randn(2, 3, 4)
    adapter = _adapter("Flow-SDE")

    weight = torch.nn.Parameter(torch.tensor(0.6))
    loss = compute_structured_distillation_loss(
        adapter,
        student_target=LatentState({"latent": latents * weight}),
        teacher_target=LatentState({"latent": teacher}),
        state=LatentState({"latent": torch.zeros(2, 3, 4)}),
        self_normalize=self_normalize,
        denominators={"latent": denominator},
    ).mean()
    loss.backward()

    legacy_weight = torch.nn.Parameter(torch.tensor(0.6))
    legacy_loss = (
        compute_per_sample_distillation_loss(
            latents * legacy_weight, teacher, self_normalize=self_normalize
        )
        / denominator
    ).mean()
    legacy_loss.backward()

    assert torch.equal(loss.detach(), legacy_loss.detach())
    assert torch.equal(weight.grad, legacy_weight.grad)


def test_structured_gradient_matches_the_non_binary_raw_element_oracle() -> None:
    """Two components with different denominators scale raw elements, loss and grad."""
    torch.manual_seed(13)
    teacher = {"video": torch.randn(2, 3, 4), "audio": torch.randn(2, 5)}
    latents = {"video": torch.randn(2, 3, 4), "audio": torch.randn(2, 5)}
    denominators = {"video": torch.tensor([0.3, 1.7]), "audio": torch.tensor([2.5, 0.7])}
    state = LatentState({"video": torch.zeros(2, 3, 4), "audio": torch.zeros(2, 5)})
    adapter = _structured_adapter("Flow-SDE", "Flow-SDE")

    weight = torch.nn.Parameter(torch.tensor(0.6))
    loss = compute_structured_distillation_loss(
        adapter,
        student_target=LatentState({name: latents[name] * weight for name in latents}),
        teacher_target=LatentState(dict(teacher)),
        state=state,
        self_normalize=False,
        denominators=denominators,
    ).mean()
    loss.backward()

    oracle_weight = torch.nn.Parameter(torch.tensor(0.6))
    video = (latents["video"] * oracle_weight - teacher["video"]).square()
    audio = (latents["audio"] * oracle_weight - teacher["audio"]).square()
    oracle_loss = (
        (
            (video / denominators["video"].reshape(2, 1, 1)).flatten(1).sum(dim=1)
            + (audio / denominators["audio"].reshape(2, 1)).flatten(1).sum(dim=1)
        )
        / 17
    ).mean()
    oracle_loss.backward()

    assert torch.equal(loss.detach(), oracle_loss.detach())
    assert torch.equal(weight.grad, oracle_weight.grad)


@pytest.mark.parametrize(
    "denominator, message",
    [
        (torch.zeros(2), r"positive"),
        (torch.tensor([1.0, float("nan")]), r"finite"),
        (torch.zeros(2, 1), r"shape \(2,\)"),
    ],
)
def test_structured_loss_rejects_an_invalid_denominator(
    denominator: torch.Tensor, message: str
) -> None:
    adapter = _adapter("Flow-SDE")

    with pytest.raises(ValueError, match=rf"denominator.*'latent'.*{message}"):
        compute_structured_distillation_loss(
            adapter,
            student_target=LatentState({"latent": torch.zeros(2, 3)}),
            teacher_target=LatentState({"latent": torch.zeros(2, 3)}),
            state=LatentState({"latent": torch.zeros(2, 3)}),
            self_normalize=False,
            denominators={"latent": denominator},
        )


def test_structured_loss_rejects_a_denominator_mapping_in_the_wrong_order() -> None:
    adapter = _structured_adapter("Flow-SDE", "Flow-SDE")

    with pytest.raises(
        ValueError,
        match=r"denominator.*component order \('video', 'audio'\), received \('audio', 'video'\)",
    ):
        compute_structured_distillation_loss(
            adapter,
            student_target=LatentState({"video": torch.zeros(2, 3), "audio": torch.zeros(2, 5)}),
            teacher_target=LatentState({"video": torch.zeros(2, 3), "audio": torch.zeros(2, 5)}),
            state=LatentState({"video": torch.zeros(2, 3), "audio": torch.zeros(2, 5)}),
            self_normalize=False,
            denominators={"audio": torch.ones(2), "video": torch.ones(2)},
        )


def test_structured_loss_rejects_a_teacher_target_shape_mismatch() -> None:
    adapter = _adapter()

    with pytest.raises(
        ValueError,
        match=r"teacher target.*component 'latent'.*\(2, 3\).*\(2, 4\)",
    ):
        compute_structured_distillation_loss(
            adapter,
            student_target=LatentState({"latent": torch.zeros(2, 3)}),
            teacher_target=LatentState({"latent": torch.zeros(2, 4)}),
            state=LatentState({"latent": torch.zeros(2, 3)}),
            self_normalize=False,
        )


def test_structured_loss_keeps_the_gradient_on_the_student_only() -> None:
    student = torch.ones(2, 3, requires_grad=True)
    teacher = torch.zeros(2, 3, requires_grad=True)
    adapter = _adapter()

    loss = compute_structured_distillation_loss(
        adapter,
        student_target=LatentState({"latent": student}),
        teacher_target=LatentState({"latent": teacher.detach()}),
        state=LatentState({"latent": torch.zeros(2, 3)}),
        self_normalize=True,
    )
    loss.sum().backward()

    assert student.grad is not None
    assert teacher.grad is None


# ============================ Scheduler-group dynamics ============================


def test_scheduler_group_dynamics_reports_a_homogeneous_ode_group() -> None:
    adapter = _structured_adapter("ODE", "ODE")

    assert resolve_scheduler_group_dynamics(adapter, "x0") is False


def test_scheduler_group_dynamics_reports_a_homogeneous_stochastic_group() -> None:
    adapter = _structured_adapter("Flow-SDE", "CPS")

    assert resolve_scheduler_group_dynamics(adapter, "xt") is True


def test_scheduler_group_dynamics_rejects_a_mixed_ode_and_sde_group() -> None:
    adapter = _structured_adapter("ODE", "Flow-SDE")

    with pytest.raises(
        ValueError,
        match=r"mixed ODE/SDE.*'video'.*'ODE'.*'audio'.*'Flow-SDE'",
    ):
        resolve_scheduler_group_dynamics(adapter, "xt")


@pytest.mark.parametrize("loss_target", ["v", "x0"])
def test_scheduler_group_dynamics_rejects_velocity_targets_on_a_stochastic_component(
    loss_target: str,
) -> None:
    adapter = _structured_adapter("Flow-SDE", "Flow-SDE")

    with pytest.raises(
        ValueError,
        match=rf"loss_target={loss_target!r}.*dynamics_type='Flow-SDE'.*component 'video'",
    ):
        resolve_scheduler_group_dynamics(adapter, loss_target)


def test_loss_target_validation_keeps_its_legacy_component_free_signature() -> None:
    validate_loss_target_for_dynamics("xt", "Flow-SDE")

    with pytest.raises(ValueError, match=r"loss_target='v'.*dynamics_type='CPS'"):
        validate_loss_target_for_dynamics("v", "CPS")


# ============================ KL denominator ============================


def test_component_denominators_follow_each_component_scheduler_dynamics() -> None:
    adapter = _structured_adapter("Flow-SDE", "CPS")
    std_dev_t = {"video": torch.tensor([[0.5], [0.25]]), "audio": torch.tensor([[2.0], [4.0]])}
    dt = {"video": torch.tensor([[-0.5], [-2.0]]), "audio": torch.tensor([[-1.0], [-1.0]])}

    video = adapter.scheduler_group["video"].get_kl_divergence_denominator(
        std_dev_t["video"], dt["video"]
    )
    audio = adapter.scheduler_group["audio"].get_kl_divergence_denominator(
        std_dev_t["audio"], dt["audio"]
    )

    assert torch.equal(video, torch.tensor([[0.125], [0.125]]))
    assert torch.equal(audio, torch.tensor([[4.0], [16.0]]))


# ============================ Replay contract ============================


def _sparse_legacy_batch() -> Any:
    """Legacy rollout storing only the last two of three denoising transitions."""
    samples = [
        BaseSample(
            timesteps=torch.tensor([1000.0, 700.0, 300.0]),
            all_latents=torch.tensor([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]]) + offset,
            latent_index_map=torch.tensor([-1, 0, 1, 2]),
            prompt_embeds=torch.tensor([offset]),
        )
        for offset in (0.0, 100.0)
    ]
    return BaseSample.stack(samples)


@pytest.mark.parametrize("step_index", [1, 2])
def test_legacy_sparse_replay_reproduces_the_stored_index_expressions(step_index: int) -> None:
    adapter = _adapter()
    batch = _sparse_legacy_batch()
    latent_index_map = batch["latent_index_map"]

    replay = adapter.get_replay_step(batch, step_index)

    assert torch.equal(
        replay.state.components["latent"],
        batch["all_latents"][:, latent_index_map[step_index]],
    )
    assert torch.equal(
        replay.next_state.components["latent"],
        batch["all_latents"][:, latent_index_map[step_index + 1]],
    )
    assert torch.equal(replay.times.timestep["latent"], batch["timesteps"][:, step_index])
    expected_next = (
        batch["timesteps"][:, step_index + 1]
        if step_index + 1 < batch["timesteps"].shape[1]
        else torch.tensor(0)
    )
    assert torch.equal(replay.times.next_timestep["latent"], expected_next)


def test_legacy_terminal_replay_projects_x0_from_the_flow_match_sigma_fallback() -> None:
    """The legacy stored trajectory carries no sigma, so the schedule supplies it."""
    adapter = _adapter()
    batch = _sparse_legacy_batch()
    replay = adapter.get_replay_step(batch, 2)
    velocity = torch.randn(2, 2)

    projected = project_distillation_target_state(
        adapter,
        loss_target="x0",
        state=replay.state,
        output=MultiModalStepOutput(velocity=LatentState({"latent": velocity})),
        times=replay.times,
    )

    latents = replay.state.components["latent"].float()
    sigma = to_broadcast_tensor(flow_match_sigma(replay.times.timestep["latent"].float()), latents)
    assert torch.equal(projected.components["latent"], latents - sigma * velocity.float())


def test_trajectory_seed_dispatch_reaches_every_component_scheduler() -> None:
    adapter = _structured_adapter()

    adapter.set_trajectory_seed(17)

    assert adapter.scheduler_group["video"].seeds == [17]
    assert adapter.scheduler_group["audio"].seeds == [17]


def test_state_active_numel_counts_every_component_element() -> None:
    adapter = _structured_adapter()

    active: Dict[str, int] = dict(
        adapter.get_state_active_numel(
            LatentState({"video": torch.zeros(2, 3, 4), "audio": torch.zeros(2, 5)})
        )
    )

    assert active == {"video": 12, "audio": 5}
