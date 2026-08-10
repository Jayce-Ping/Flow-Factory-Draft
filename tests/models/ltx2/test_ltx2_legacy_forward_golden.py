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

"""Baseline oracle for the unchanged public concatenated LTX2 ``forward``.

``legacy_forward_golden.json`` was captured by running the pre-Task-4A
implementation against the shared fakes; see ``generate_legacy_forward_golden.py``
in this directory. The expectations therefore never come from the new
component-return branch, which this module never calls. The recorded commit and
adapter blob hashes are re-verified against this repository's Git objects, so the
provenance is auditable rather than a handwritten label.
"""

import json
import subprocess
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Dict, List, Optional, Tuple

import pytest
import torch
from ltx2_fakes import (
    AUDIO_SCHEDULER_OFFSET,
    BATCH_SIZE,
    TIMESTEP,
    VIDEO_SCHEDULER_OFFSET,
    VIDEO_SEQ_LEN,
    PipelineFake,
    SchedulerFake,
    TransformerFake,
    audio_latents,
    conditioning_mask,
    forward_conditioning_kwargs,
    video_latents,
)

from flow_factory.models.ltx2.ltx2_i2av import LTX2_I2AV_Adapter
from flow_factory.models.ltx2.ltx2_t2av import LTX2_T2AV_Adapter
from flow_factory.scheduler import SDESchedulerOutput

TESTS_DIR = Path(__file__).resolve().parent
REPO = TESTS_DIR.parents[2]
GOLDEN = json.loads((TESTS_DIR / "legacy_forward_golden.json").read_text())
ADAPTERS = {"t2av": LTX2_T2AV_Adapter, "i2av": LTX2_I2AV_Adapter}
OUTPUT_FIELDS = (
    "next_latents",
    "next_latents_mean",
    "std_dev_t",
    "dt",
    "log_prob",
    "velocity",
)


def _adapter(cls: type) -> Tuple[Any, List[Tuple[str, Any]]]:
    log: List[Tuple[str, Any]] = []
    transformer = TransformerFake(noise_scale=GOLDEN["noise_scale"])
    adapter = object.__new__(cls)
    adapter.pipeline = PipelineFake(SchedulerFake(VIDEO_SCHEDULER_OFFSET, log), transformer)
    adapter.component_runtime = SimpleNamespace(get_component=lambda name: transformer)
    adapter.load_scheduler = lambda: SchedulerFake(AUDIO_SCHEDULER_OFFSET, log)
    adapter.scheduler_group = adapter.build_scheduler_group()
    return adapter, log


def _expected(entry: Optional[Dict[str, Any]]) -> Optional[torch.Tensor]:
    if entry is None:
        return None
    return torch.tensor(entry["values"], dtype=torch.float32).reshape(entry["shape"])


def _run(name: str, *, compute_log_prob: bool, fields: Tuple[str, ...]) -> Dict[str, Any]:
    cls = ADAPTERS[name]
    adapter, log = _adapter(cls)
    extra = {"conditioning_mask": conditioning_mask()} if cls is LTX2_I2AV_Adapter else {}
    torch.manual_seed(GOLDEN["seed"])
    output = adapter.forward(
        t=torch.full((BATCH_SIZE,), TIMESTEP),
        t_next=torch.zeros(BATCH_SIZE),
        latents=torch.cat([video_latents(), audio_latents()], dim=1),
        video_seq_len=VIDEO_SEQ_LEN,
        compute_log_prob=compute_log_prob,
        return_kwargs=list(fields),
        **extra,
        **forward_conditioning_kwargs(),
    )
    return {
        "output": output,
        "log": log,
        "rng_state_sum": int(torch.get_rng_state().sum().item()),
        "post_draw": torch.randn(4),
    }


def _cases() -> List[Tuple[str, str, bool, Tuple[str, ...]]]:
    cases = []
    for key in sorted(GOLDEN["cases"]):
        name, log_prob_part, fields_part = key.split("|")
        compute_log_prob = log_prob_part.split("=")[1] == "True"
        fields = tuple(fields_part.split("=")[1].split(","))
        cases.append((key, name, compute_log_prob, fields))
    return cases


@pytest.mark.parametrize(
    "key,name,compute_log_prob,fields",
    _cases(),
    ids=[case[0] for case in _cases()],
)
def test_legacy_concatenated_forward_matches_the_pre_task_4a_oracle(
    key: str, name: str, compute_log_prob: bool, fields: Tuple[str, ...]
) -> None:
    golden = GOLDEN["cases"][key]

    result = _run(name, compute_log_prob=compute_log_prob, fields=fields)
    output = result["output"]

    assert isinstance(output, SDESchedulerOutput)
    for field in OUTPUT_FIELDS:
        expected = _expected(golden[field])
        received = getattr(output, field)
        if expected is None:
            assert received is None, f"{key}: expected {field} to stay unset"
            continue
        assert received is not None, f"{key}: expected {field} to be returned"
        assert torch.allclose(received, expected, atol=0, rtol=0), f"{key}: {field} drifted"


@pytest.mark.parametrize(
    "key,name,compute_log_prob,fields",
    _cases(),
    ids=[case[0] for case in _cases()],
)
def test_legacy_forward_keeps_the_scheduler_order_and_rng_position(
    key: str, name: str, compute_log_prob: bool, fields: Tuple[str, ...]
) -> None:
    golden = GOLDEN["cases"][key]

    result = _run(name, compute_log_prob=compute_log_prob, fields=fields)

    assert [list(entry) for entry in result["log"]] == golden["dispatch_log"]
    assert result["rng_state_sum"] == golden["rng_state_sum"]
    assert torch.allclose(
        result["post_draw"],
        torch.tensor(golden["post_forward_draw"], dtype=torch.float32),
        atol=0,
        rtol=0,
    )


def _git(*args: str) -> str:
    result = subprocess.run(["git", *args], cwd=REPO, check=True, capture_output=True, text=True)
    return result.stdout.strip()


def test_the_golden_file_records_a_commit_that_exists_in_this_repository() -> None:
    commit = GOLDEN["oracle_commit"]

    assert len(commit) == 40
    assert _git("rev-parse", f"{commit}^{{commit}}") == commit
    assert _git("rev-parse", f"{GOLDEN['oracle_commit_ref']}^{{commit}}") == commit


def test_the_golden_file_records_the_pre_task_4a_adapter_blobs() -> None:
    commit = GOLDEN["oracle_commit"]
    blobs = GOLDEN["oracle_blobs"]

    assert set(blobs) == {
        "src/flow_factory/models/ltx2/ltx2_t2av.py",
        "src/flow_factory/models/ltx2/ltx2_i2av.py",
    }
    for path, blob in blobs.items():
        assert _git("rev-parse", f"{commit}:{path}") == blob
        # The adapters changed in Task 4A, so the oracle blobs must not be the current ones.
        assert _git("rev-parse", f"HEAD:{path}") != blob


def test_the_generator_pins_the_recorded_identity_and_refuses_other_checkouts(
    tmp_path: Path,
) -> None:
    import generate_legacy_forward_golden as generator

    identity = generator.resolve_oracle_identity()

    assert identity["oracle_commit"] == GOLDEN["oracle_commit"]
    assert identity["oracle_blobs"] == GOLDEN["oracle_blobs"]
    with pytest.raises(ValueError, match=r"oracle worktree.*HEAD.*ee6d247.*received"):
        generator.require_oracle_worktree(REPO, identity["oracle_commit"])
    with pytest.raises(FileNotFoundError, match=r"expected an oracle worktree"):
        generator.require_oracle_worktree(tmp_path / "missing", identity["oracle_commit"])


def test_the_golden_file_covers_every_recorded_case() -> None:
    assert GOLDEN["oracle_commit_ref"] == "ee6d247"
    assert len(GOLDEN["cases"]) == 12
