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

"""Bridge stored rollout trajectories to the adapter training boundary."""

from .dispatch import (
    build_forward_state_kwargs,
    default_forward_state,
)
from .noising import (
    add_forward_process_noise,
    apply_forward_process_noise,
    build_training_component_times,
    project_velocity_to_clean_state,
)
from .reduction import (
    default_reduce_component_latent_values,
    default_reduce_latent_values,
    resolve_component_latent_axes,
    validate_reduced_component_values,
    validate_reduced_latent_values,
    validate_reduction_inputs,
)
from .replay import (
    get_replay_callback,
    get_replay_step,
    get_state_active_numel,
    get_terminal_state,
    get_train_step_indices,
)

__all__ = [
    "add_forward_process_noise",
    "apply_forward_process_noise",
    "build_forward_state_kwargs",
    "build_training_component_times",
    "default_forward_state",
    "default_reduce_component_latent_values",
    "default_reduce_latent_values",
    "get_replay_callback",
    "get_replay_step",
    "get_state_active_numel",
    "get_terminal_state",
    "get_train_step_indices",
    "project_velocity_to_clean_state",
    "resolve_component_latent_axes",
    "validate_reduced_component_values",
    "validate_reduced_latent_values",
    "validate_reduction_inputs",
]
