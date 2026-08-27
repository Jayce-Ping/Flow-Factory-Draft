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

"""Reserved model-condition field ownership shared across runtime layers.

Only model-conditioning fields may cross the adapter forward boundary.  The
constants in this module are the single authority used by both online replay
dispatch and offline output-context binding.
"""

FORWARD_STATE_BOUNDARY_KEYS = frozenset(
    {
        "batch",
        "state",
        "times",
        "next_state",
        "return_fields",
        "forward_kwargs",
    }
)

FORWARD_STATE_OWNED_KEYS = frozenset(
    {
        "t",
        "t_next",
        "latents",
        "next_latents",
        "compute_log_prob",
        "return_kwargs",
        "noise_level",
    }
)

ROLLOUT_STORAGE_KEYS = frozenset(
    {
        "trajectory",
        "timesteps",
        "all_latents",
        "latent_index_map",
        "log_probs",
        "log_prob_index_map",
    }
)

TRAINER_METADATA_KEYS = frozenset({"advantage"})

ALGORITHM_STATE_KEYS = frozenset(
    {
        "_unique_id",
        "active_masks",
        "advantages",
        "applicable_rewards",
        "callbacks",
        "clean_state",
        "component_log_probs",
        "decode_context",
        "dt",
        "forward_context",
        "generator",
        "geometry_signature",
        "geometry_signatures",
        "loss",
        "loss_weight",
        "next_sigma",
        "next_state_mean",
        "next_timestep",
        "noise",
        "reward",
        "rewards",
        "schema_version",
        "sigma",
        "state_index_map",
        "std_dev_t",
        "supervision",
        "target_velocity",
        "timestep",
        "velocity",
    }
)

OFFLINE_PROVENANCE_KEYS = frozenset(
    {
        "__offline_condition_id__",
        "condition",
        "condition_id",
        "condition_ids",
        "record_id",
        "record_ids",
        "source",
        "sources",
        "source_id",
        "source_ids",
        "model_input",
        "model_inputs",
        "supervision_type",
        "output",
        "target_media",
        "chosen_media",
        "rejected_media",
        "metadata",
        "metadata_json",
    }
)

NON_MODEL_CONDITION_KEYS = (
    FORWARD_STATE_BOUNDARY_KEYS
    | FORWARD_STATE_OWNED_KEYS
    | ROLLOUT_STORAGE_KEYS
    | TRAINER_METADATA_KEYS
    | ALGORITHM_STATE_KEYS
    | OFFLINE_PROVENANCE_KEYS
)

__all__ = [
    "ALGORITHM_STATE_KEYS",
    "FORWARD_STATE_BOUNDARY_KEYS",
    "FORWARD_STATE_OWNED_KEYS",
    "NON_MODEL_CONDITION_KEYS",
    "OFFLINE_PROVENANCE_KEYS",
    "ROLLOUT_STORAGE_KEYS",
    "TRAINER_METADATA_KEYS",
]
