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

"""Regenerate ``legacy_forward_golden.json`` from the pre-Task-4A LTX2 adapters.

The expectations must come from the old concatenated ``forward``, never from the
new component-return branch, so this script refuses to run unless it is pointed at
a checkout whose ``HEAD`` is exactly the oracle commit. It records that commit plus
both adapter source blob hashes, which the test then re-verifies against this
repository's Git objects.

Usage::

    git worktree add .scratch/oracle-ee6d247 ee6d247 --detach
    python tests/models/ltx2/generate_legacy_forward_golden.py \
        --oracle-worktree .scratch/oracle-ee6d247
    git worktree remove .scratch/oracle-ee6d247
"""

import argparse
import json
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Dict, List, Optional, Tuple

ORACLE_COMMIT = "ee6d247"
ORACLE_SOURCES = (
    "src/flow_factory/models/ltx2/ltx2_t2av.py",
    "src/flow_factory/models/ltx2/ltx2_i2av.py",
)
TESTS_DIR = Path(__file__).resolve().parent
REPO = TESTS_DIR.parents[2]
GOLDEN_PATH = TESTS_DIR / "legacy_forward_golden.json"

SEED = 20260810
NOISE_SCALE = 0.125
TIMESTEP = 500.0
FIELD_SETS: Tuple[Tuple[str, ...], ...] = (
    ("next_latents", "next_latents_mean", "std_dev_t", "dt", "log_prob", "velocity"),
    ("next_latents", "log_prob"),
    ("next_latents", "velocity"),
)


def git(*args: str, cwd: Path) -> str:
    """Return the stripped stdout of one Git command."""
    result = subprocess.run(["git", *args], cwd=cwd, check=True, capture_output=True, text=True)
    return result.stdout.strip()


def resolve_oracle_identity() -> Dict[str, Any]:
    """Return the oracle commit hash and both adapter source blob hashes."""
    commit = git("rev-parse", f"{ORACLE_COMMIT}^{{commit}}", cwd=REPO)
    return {
        "oracle_commit": commit,
        "oracle_commit_ref": ORACLE_COMMIT,
        "oracle_blobs": {
            path: git("rev-parse", f"{commit}:{path}", cwd=REPO) for path in ORACLE_SOURCES
        },
    }


def require_oracle_worktree(worktree: Path, commit: str) -> Path:
    """Return the oracle ``src`` directory after checking the worktree is at ``commit``."""
    if not worktree.is_dir():
        raise FileNotFoundError(
            f"expected an oracle worktree at {worktree}; create it with "
            f"'git worktree add {worktree} {ORACLE_COMMIT} --detach'"
        )
    head = git("rev-parse", "HEAD", cwd=worktree)
    if head != commit:
        raise ValueError(
            f"expected the oracle worktree {worktree} at HEAD {commit} ({ORACLE_COMMIT}), "
            f"received {head}"
        )
    source = worktree / "src"
    if not (source / "flow_factory").is_dir():
        raise FileNotFoundError(f"expected the oracle package under {source}, received nothing")
    return source


def main() -> None:
    """Write the golden file for every adapter/flag/field combination."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--oracle-worktree",
        type=Path,
        default=REPO / ".scratch" / f"oracle-{ORACLE_COMMIT}",
        help="Checkout of the oracle commit to import the old adapters from.",
    )
    arguments = parser.parse_args()

    identity = resolve_oracle_identity()
    oracle_src = require_oracle_worktree(
        arguments.oracle_worktree.resolve(), identity["oracle_commit"]
    )

    sys.path.insert(0, str(oracle_src))
    sys.path.insert(0, str(TESTS_DIR))

    import torch

    import flow_factory

    if Path(flow_factory.__file__).resolve().parents[1] != oracle_src:
        raise RuntimeError(
            f"expected the oracle flow_factory under {oracle_src}, "
            f"imported {flow_factory.__file__}"
        )

    from ltx2_fakes import (
        AUDIO_SCHEDULER_OFFSET,
        BATCH_SIZE,
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

    def build_adapter(cls: type) -> Tuple[Any, List[Tuple[str, Any]]]:
        log: List[Tuple[str, Any]] = []
        transformer = TransformerFake(noise_scale=NOISE_SCALE)
        adapter = object.__new__(cls)
        adapter.pipeline = PipelineFake(SchedulerFake(VIDEO_SCHEDULER_OFFSET, log), transformer)
        adapter.component_runtime = SimpleNamespace(get_component=lambda name: transformer)
        # The oracle commit predates the scheduler group, so the twin is assigned directly.
        adapter.audio_scheduler = SchedulerFake(AUDIO_SCHEDULER_OFFSET, log)
        return adapter, log

    def serialize(value: Optional[Any]) -> Optional[Dict[str, Any]]:
        if value is None:
            return None
        return {"shape": list(value.shape), "values": value.reshape(-1).tolist()}

    def run_case(cls: type, *, compute_log_prob: bool, fields: Tuple[str, ...]) -> Dict[str, Any]:
        adapter, log = build_adapter(cls)
        extra = {"conditioning_mask": conditioning_mask()} if cls is LTX2_I2AV_Adapter else {}
        torch.manual_seed(SEED)
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
            "next_latents": serialize(output.next_latents),
            "next_latents_mean": serialize(output.next_latents_mean),
            "std_dev_t": serialize(output.std_dev_t),
            "dt": serialize(output.dt),
            "log_prob": serialize(output.log_prob),
            "velocity": serialize(output.velocity),
            "dispatch_log": [[name, offset] for name, offset in log],
            "rng_state_sum": int(torch.get_rng_state().sum().item()),
            # Drawing after the call pins the exact RNG position the forward left behind.
            "post_forward_draw": torch.randn(4).tolist(),
        }

    cases: Dict[str, Any] = {}
    for cls, name in ((LTX2_T2AV_Adapter, "t2av"), (LTX2_I2AV_Adapter, "i2av")):
        for compute_log_prob in (True, False):
            for fields in FIELD_SETS:
                key = f"{name}|log_prob={compute_log_prob}|fields={','.join(fields)}"
                cases[key] = run_case(cls, compute_log_prob=compute_log_prob, fields=fields)

    payload = {**identity, "seed": SEED, "noise_scale": NOISE_SCALE, "cases": cases}
    GOLDEN_PATH.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    print(f"wrote {GOLDEN_PATH} with {len(cases)} cases from {identity['oracle_commit']}")


if __name__ == "__main__":
    main()
