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


"""Bridge stored rollout trajectories to the adapter training boundary.

The bridge is split by responsibility: reading stored trajectories (``replay``),
the forward noising process (``noising``), replayed-step dispatch (``dispatch``),
active-element reduction (``reduction``), and the active-mask geometry they
share (``masks``).
"""

from .dispatch import _build_forward_state_kwargs, _default_forward_state
from .masks import _active_element_counts, _expand_active_mask
from .noising import (
    _add_forward_process_noise,
    _apply_forward_process_noise,
    _build_training_component_times,
    _project_velocity_to_clean_state,
)
from .reduction import (
    _default_reduce_component_latent_values,
    _default_reduce_latent_values,
    _resolve_component_latent_axes,
    _validate_reduced_component_values,
    _validate_reduced_latent_values,
    _validate_reduction_inputs,
)
from .replay import (
    _get_replay_callback,
    _get_replay_step,
    _get_state_active_numel,
    _get_terminal_state,
    _get_train_step_indices,
)

__all__ = [
    "_active_element_counts",
    "_add_forward_process_noise",
    "_apply_forward_process_noise",
    "_build_forward_state_kwargs",
    "_build_training_component_times",
    "_default_forward_state",
    "_default_reduce_component_latent_values",
    "_default_reduce_latent_values",
    "_expand_active_mask",
    "_get_replay_callback",
    "_get_replay_step",
    "_get_state_active_numel",
    "_get_terminal_state",
    "_get_train_step_indices",
    "_project_velocity_to_clean_state",
    "_resolve_component_latent_axes",
    "_validate_reduced_component_values",
    "_validate_reduced_latent_values",
    "_validate_reduction_inputs",
]
