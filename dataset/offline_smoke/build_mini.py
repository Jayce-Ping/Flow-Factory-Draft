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

"""Build the two deterministic, procedural offline-smoke HF staging trees."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import shutil
import tempfile
import wave
from dataclasses import dataclass
from fractions import Fraction
from pathlib import Path
from typing import Any, Iterable, Literal, Mapping, Sequence

import av
import numpy as np
from PIL import Image, ImageDraw

from .profiles import (
    GPU_ALIAS_TO_PROFILE,
    MAIN_GPU_ALIASES,
    OFFLINE_DPO_REPO_ID,
    SFT_REPO_ID,
)

RUNTIME_ALIASES = MAIN_GPU_ALIASES + ("image-i2i",)
RECORDS_PER_ALIAS = 32
DEFAULT_SEED = 20260830
FPS = 24
CC0 = "CC0-1.0"
COLORS = (
    ("coral", (224, 83, 74)),
    ("amber", (234, 166, 52)),
    ("teal", (42, 157, 143)),
    ("blue", (65, 105, 225)),
    ("violet", (139, 92, 246)),
)
SHAPES = ("circle", "square", "triangle", "diamond", "star")


@dataclass(frozen=True, slots=True)
class _Asset:
    """One repo-root-relative generated media asset."""

    type: Literal["image", "video", "audio"]
    path: str
    fps: int | None = None
    sample_rate: int | None = None


class _Writer:
    """Write one self-contained repository and its provenance index."""

    def __init__(
        self,
        root: Path,
        repo_id: str,
        supervision: Literal["demonstration", "preference"],
        seed: int,
        records_per_alias: int,
    ) -> None:
        self.root = root
        self.repo_id = repo_id
        self.supervision = supervision
        self.seed = seed
        self.records_per_alias = records_per_alias
        self.provenance: list[dict[str, Any]] = []
        self.script_sha = _sha256(Path(__file__))

    def image(self, relative: str, value: Image.Image, seed: int) -> _Asset:
        path = self._new_path(relative)
        value.convert("RGB").save(path, format="PNG", compress_level=9)
        self._record(relative, "image", seed, width=value.width, height=value.height)
        return _Asset("image", relative)

    def video(self, relative: str, frames: Sequence[np.ndarray], seed: int) -> _Asset:
        path = self._new_path(relative)
        height, width, channels = frames[0].shape
        if channels != 3 or any(frame.shape != frames[0].shape for frame in frames):
            raise ValueError("video frames must share one RGB geometry")
        with av.open(str(path), "w", format="avi") as container:
            stream = container.add_stream("ffv1", rate=Fraction(FPS, 1))
            stream.width, stream.height, stream.pix_fmt = width, height, "bgr0"
            for index, array in enumerate(frames):
                frame = av.VideoFrame.from_ndarray(array, format="rgb24")
                frame.pts = index
                for packet in stream.encode(frame):
                    container.mux(packet)
            for packet in stream.encode():
                container.mux(packet)
        decoded = _decode(path)
        if len(decoded) != len(frames) or decoded[0].shape != frames[0].shape:
            raise RuntimeError(f"generated video failed decode probe: {relative}")
        self._record(
            relative,
            "video",
            seed,
            width=width,
            height=height,
            num_frames=len(frames),
            fps=FPS,
            codec="ffv1",
        )
        return _Asset("video", relative, fps=FPS)

    def audio(self, relative: str, samples: np.ndarray, rate: int, seed: int) -> _Asset:
        path = self._new_path(relative)
        with wave.open(str(path), "wb") as output:
            output.setparams((2, 2, rate, samples.shape[0], "NONE", "not compressed"))
            output.writeframes(samples.astype("<i2", copy=False).tobytes())
        self._record(
            relative,
            "audio",
            seed,
            channels=2,
            sample_rate=rate,
            num_samples=samples.shape[0],
        )
        return _Asset("audio", relative, sample_rate=rate)

    def finish(self, records: Mapping[str, Sequence[Mapping[str, Any]]]) -> None:
        for alias in RUNTIME_ALIASES:
            alias_dir = self.root / "profiles" / alias
            alias_dir.mkdir(parents=True)
            _write_jsonl(alias_dir / "train.jsonl", records[alias])
        _write_jsonl(
            self.root / "provenance.jsonl", sorted(self.provenance, key=lambda x: x["path"])
        )
        manifest = {
            "schema_version": 1,
            "repository_id": self.repo_id,
            "flow_factory_schema_version": 2,
            "supervision_type": self.supervision,
            "records_per_alias": self.records_per_alias,
            "runtime_aliases": list(RUNTIME_ALIASES),
            "license": CC0,
            "condition_endpoint_check": {
                "metric": "decoded_rgb_max_absolute_difference",
                "tolerance": 0,
            },
            "generator": {
                "script": "dataset/offline_smoke/build_mini.py",
                "script_sha256": self.script_sha,
                "seed": self.seed,
            },
        }
        _write_json(self.root / "dataset_manifest.json", manifest)
        (self.root / "README.md").write_text(_card(self), encoding="utf-8")
        (self.root / "LICENSE").write_text(_cc0_notice(), encoding="utf-8")

    def metadata(self, alias: str, index: int, seed: int) -> dict[str, Any]:
        value: dict[str, Any] = {
            "sample_id": f"{alias}-{index:04d}",
            "profile": GPU_ALIAS_TO_PROFILE[alias].name,
            "gpu_alias": alias,
            "usage_tier": "smoke_only",
            "source": {
                "origin": "flow_factory_procedural",
                "license": CC0,
                "generator_seed": seed,
                "generator_script_sha256": self.script_sha,
            },
        }
        if self.supervision == "preference":
            value.update(
                preference_origin="deterministic_corruption",
                semantic_preference_claim=False,
            )
        return value

    def _new_path(self, relative: str) -> Path:
        path = self.root / relative
        if not path.resolve().is_relative_to(self.root.resolve()) or path.exists():
            raise ValueError(f"invalid or duplicate asset path: {relative!r}")
        path.parent.mkdir(parents=True, exist_ok=True)
        return path

    def _record(self, relative: str, media_type: str, seed: int, **media: Any) -> None:
        path = self.root / relative
        self.provenance.append(
            {
                "path": relative,
                "type": media_type,
                "sha256": _sha256(path),
                "size_bytes": path.stat().st_size,
                "origin": "flow_factory_procedural",
                "license": CC0,
                "generator_seed": seed,
                "media": media,
            }
        )


def build(
    staging_root: Path,
    seed: int,
    replace: bool,
    records_per_alias: int = RECORDS_PER_ALIAS,
) -> tuple[Path, Path]:
    """Build independent SFT and offline-DPO staging trees atomically.

    Args:
        staging_root: Directory that receives the two repository trees.
        seed: Base seed for deterministic procedural records.
        replace: Whether to replace the two exact destination directories.
        records_per_alias: Number of records generated for every runtime alias.

    Returns:
        Paths to the SFT and offline-DPO staging trees, in that order.

    Raises:
        ValueError: If ``records_per_alias`` is not positive.
        FileExistsError: If a destination exists and ``replace`` is false.
    """
    if records_per_alias < 1:
        raise ValueError("records_per_alias must be positive")
    staging_root.mkdir(parents=True, exist_ok=True)
    destinations = tuple(
        staging_root / repo_id.rsplit("/", 1)[1] for repo_id in (SFT_REPO_ID, OFFLINE_DPO_REPO_ID)
    )
    if not replace and any(path.exists() for path in destinations):
        raise FileExistsError("staging repo exists; pass --replace for the two exact targets")
    temporary = Path(tempfile.mkdtemp(prefix=".build-mini-", dir=staging_root))
    try:
        for repo_id, supervision, destination in zip(
            (SFT_REPO_ID, OFFLINE_DPO_REPO_ID),
            ("demonstration", "preference"),
            destinations,
        ):
            writer = _Writer(
                temporary / destination.name,
                repo_id,
                supervision,
                seed,
                records_per_alias,
            )
            writer.root.mkdir()
            records = {alias: [] for alias in RUNTIME_ALIASES}
            for index in range(records_per_alias):
                row_seed = _seed(seed, supervision, index)
                pools = _assets(writer, index, row_seed)
                for alias in RUNTIME_ALIASES:
                    records[alias].append(_record(writer, alias, index, row_seed, pools))
            writer.finish(records)
        if replace:
            for destination in destinations:
                if destination.exists():
                    shutil.rmtree(destination)
        for destination in destinations:
            os.replace(temporary / destination.name, destination)
    finally:
        shutil.rmtree(temporary, ignore_errors=True)
    return destinations


def _assets(writer: _Writer, index: int, seed: int) -> dict[str, dict[str, _Asset]]:
    """Generate shared image, Wan, LTX, and H3 pools for one logical row."""
    first = (SHAPES[index % len(SHAPES)], COLORS[index % len(COLORS)])
    second = (SHAPES[(index + 2) % len(SHAPES)], COLORS[(index + 2) % len(COLORS)])
    prefix = f"media/{index:04d}"
    image_pool = {
        "input": writer.image(f"{prefix}/image/input.png", _scene(first, "center"), seed),
        "ref1": writer.image(f"{prefix}/image/ref1.png", _scene(first, "center"), seed),
        "ref2": writer.image(f"{prefix}/image/ref2.png", _scene(second, "center"), seed),
        "t2i_chosen": writer.image(f"{prefix}/image/t2i_chosen.png", _scene(first, "center"), seed),
        "i2i_chosen": writer.image(
            f"{prefix}/image/i2i_chosen.png", _scene(second, "upper_left"), seed
        ),
        "multi_chosen": writer.image(
            f"{prefix}/image/multi_chosen.png", _composition(first, second, False), seed
        ),
    }
    if writer.supervision == "preference":
        image_pool.update(
            t2i_rejected=writer.image(
                f"{prefix}/image/t2i_rejected.png", _scene(second, "lower_right"), seed
            ),
            i2i_rejected=writer.image(
                f"{prefix}/image/i2i_rejected.png", _scene(first, "lower_right"), seed
            ),
            multi_rejected=writer.image(
                f"{prefix}/image/multi_rejected.png",
                _composition(first, second, True),
                seed,
            ),
        )
    pools = {"image": image_pool}
    for family, alias in (("wan", "wan-t2v"), ("ltx", "ltx2-t2av"), ("h3", "h3-t2va")):
        geometry = _geometry(alias)
        frames = _motion(geometry.width, geometry.height, geometry.num_frames, first, second)
        pool: dict[str, _Asset] = {
            "chosen_video": writer.video(f"{prefix}/{family}/chosen.avi", frames, seed)
        }
        decoded = _decode(writer.root / pool["chosen_video"].path)
        pool["first"] = writer.image(
            f"{prefix}/{family}/first.png", Image.fromarray(decoded[0]), seed
        )
        pool["last"] = writer.image(
            f"{prefix}/{family}/last.png", Image.fromarray(decoded[-1]), seed
        )
        if geometry.sample_rate is not None:
            samples = _audio(geometry.sample_rate, geometry.num_frames / geometry.frame_rate, index)
            pool["chosen_audio"] = writer.audio(
                f"{prefix}/{family}/chosen.wav", samples, geometry.sample_rate, seed
            )
        if writer.supervision == "preference":
            rejected_frames = _corrupt_frames(frames, second[1][1])
            pool["rejected_video"] = writer.video(
                f"{prefix}/{family}/rejected.avi", rejected_frames, seed
            )
            rejected_decoded = _decode(writer.root / pool["rejected_video"].path)
            if not (
                np.array_equal(decoded[0], rejected_decoded[0])
                and np.array_equal(decoded[-1], rejected_decoded[-1])
            ):
                raise RuntimeError(f"{family} DPO corruption changed a decoded endpoint")
            if geometry.sample_rate is not None:
                pool["rejected_audio"] = writer.audio(
                    f"{prefix}/{family}/rejected.wav",
                    _corrupt_audio(samples),
                    geometry.sample_rate,
                    seed,
                )
        pools[family] = pool
    return pools


def _record(
    writer: _Writer,
    alias: str,
    index: int,
    seed: int,
    pools: Mapping[str, Mapping[str, _Asset]],
) -> dict[str, Any]:
    profile = GPU_ALIAS_TO_PROFILE[alias]
    first = (SHAPES[index % len(SHAPES)], COLORS[index % len(COLORS)][0])
    second = (SHAPES[(index + 2) % len(SHAPES)], COLORS[(index + 2) % len(COLORS)][0])
    prompt = f"A {first[1]} {first[0]} moves left to right beside a {second[1]} {second[0]}."
    family = _family(alias)
    input_media: list[dict[str, Any]] = []
    if alias == "image-i2i":
        prompt = f"Move the {first[1]} {first[0]} to the upper left and recolor it {second[1]}."
        input_media = [_media(pools["image"]["input"])]
    elif alias == "bagel-mri2i":
        prompt = "Place reference 1 on the left and reference 2 on the right."
        input_media = [_media(pools["image"]["ref1"]), _media(pools["image"]["ref2"])]
    elif alias in ("wan-i2v-first", "ltx2-i2av"):
        input_media = [_media(pools[family]["first"], slot="first_frame")]
    elif alias == "wan-flf2v":
        input_media = [
            _media(pools[family]["first"], slot="first_frame"),
            _media(pools[family]["last"], slot="last_frame"),
        ]
    elif alias == "h3-fl2va":
        cases = (
            [_media(pools[family]["first"], slot="first_frame")],
            [_media(pools[family]["last"], slot="last_frame")],
            [
                _media(pools[family]["first"], slot="first_frame"),
                _media(pools[family]["last"], slot="last_frame"),
            ],
        )
        input_media = cases[index % len(cases)]
    elif alias == "h3-ref2va":
        refs = [
            _media(pools["wan"]["first"]),
            _media(pools["wan"]["chosen_video"]),
            _media(pools["h3"]["chosen_audio"]),
        ]
        offset = index % len(refs)
        input_media = refs[offset:] + refs[:offset]

    output_types = tuple(item.type.value for item in profile.contract.output_media.items)
    candidates = _candidate_assets(alias, family, pools)
    chosen = _candidate(output_types, candidates, "chosen")
    if writer.supervision == "demonstration":
        supervision: dict[str, Any] = {"type": "demonstration", "target": chosen}
    else:
        supervision = {
            "type": "preference",
            "chosen": chosen,
            "rejected": _candidate(output_types, candidates, "rejected"),
        }
    return {
        "schema_version": 2,
        "input": {"prompt": prompt, "media": input_media},
        "supervision": supervision,
        "metadata": writer.metadata(alias, index, seed),
    }


def _candidate_assets(
    alias: str,
    family: str,
    pools: Mapping[str, Mapping[str, _Asset]],
) -> Mapping[str, Mapping[str, _Asset]]:
    if alias == "sd35-t2i":
        names = ("t2i_chosen", "t2i_rejected")
    elif alias == "image-i2i":
        names = ("i2i_chosen", "i2i_rejected")
    elif alias == "bagel-mri2i":
        names = ("multi_chosen", "multi_rejected")
    else:
        pool = pools[family]
        return {
            "chosen": {
                media_type: pool[f"chosen_{media_type}"]
                for media_type in ("video", "audio")
                if f"chosen_{media_type}" in pool
            },
            "rejected": {
                media_type: pool[f"rejected_{media_type}"]
                for media_type in ("video", "audio")
                if f"rejected_{media_type}" in pool
            },
        }
    pool = pools["image"]
    return {
        "chosen": {"image": pool[names[0]]},
        "rejected": {"image": pool[names[1]]} if names[1] in pool else {},
    }


def _candidate(
    output_types: Sequence[str],
    candidates: Mapping[str, Mapping[str, _Asset]],
    side: str,
) -> dict[str, Any]:
    """Project any declared output sequence without assuming AV-only candidates."""
    available = candidates[side]
    media = [_media(available[media_type]) for media_type in output_types]
    if tuple(item["type"] for item in media) != tuple(output_types):
        raise RuntimeError("candidate media order diverged from the profile contract")
    return {"media": media}


def _geometry(alias: str) -> Any:
    profile = GPU_ALIAS_TO_PROFILE[alias]
    return next(case.geometry for case in profile.gpu_cases if case.alias == alias)


def _family(alias: str) -> str:
    if alias.startswith("wan-"):
        return "wan"
    if alias.startswith("ltx2-"):
        return "ltx"
    if alias.startswith("h3-"):
        return "h3"
    return "image"


def _media(asset: _Asset, slot: str | None = None) -> dict[str, Any]:
    value: dict[str, Any] = {"type": asset.type, "path": f"../../{asset.path}"}
    if asset.fps is not None:
        value["fps"] = asset.fps
    if asset.sample_rate is not None:
        value["sample_rate"] = asset.sample_rate
    if slot is not None:
        value["slot"] = slot
    return value


def _scene(item: tuple[str, tuple[str, tuple[int, int, int]]], position: str) -> Image.Image:
    shape, (_, color) = item
    image = Image.new("RGB", (256, 256), (235, 239, 242))
    centers = {"center": (128, 128), "upper_left": (80, 80), "lower_right": (176, 176)}
    _draw_shape(ImageDraw.Draw(image), shape, centers[position], 42, color)
    return image


def _composition(
    first: tuple[str, tuple[str, tuple[int, int, int]]],
    second: tuple[str, tuple[str, tuple[int, int, int]]],
    swap: bool,
) -> Image.Image:
    image = Image.new("RGB", (256, 256), (235, 239, 242))
    positions = ((76, 128), (180, 128)) if not swap else ((180, 128), (76, 128))
    draw = ImageDraw.Draw(image)
    _draw_shape(draw, first[0], positions[0], 36, first[1][1])
    _draw_shape(draw, second[0], positions[1], 36, second[1][1])
    return image


def _motion(
    width: int,
    height: int,
    count: int,
    first: tuple[str, tuple[str, tuple[int, int, int]]],
    second: tuple[str, tuple[str, tuple[int, int, int]]],
) -> list[np.ndarray]:
    frames = []
    radius = max(min(width, height) // 12, 4)
    for index in range(count):
        phase = index / max(count - 1, 1)
        image = Image.new("RGB", (width, height), (232, 238, 242))
        draw = ImageDraw.Draw(image)
        x = round(radius * 2 + phase * (width - radius * 4))
        y = round(height * (0.55 + 0.1 * math.sin(phase * math.tau)))
        _draw_shape(draw, first[0], (x, y), radius, first[1][1])
        _draw_shape(draw, second[0], (width * 3 // 4, height // 3), radius, second[1][1])
        frames.append(np.asarray(image, dtype=np.uint8).copy())
    return frames


def _draw_shape(
    draw: ImageDraw.ImageDraw,
    shape: str,
    center: tuple[int, int],
    radius: int,
    color: tuple[int, int, int],
) -> None:
    x, y = center
    if shape == "circle":
        draw.ellipse((x - radius, y - radius, x + radius, y + radius), fill=color)
    elif shape == "square":
        draw.rectangle((x - radius, y - radius, x + radius, y + radius), fill=color)
    elif shape == "triangle":
        draw.polygon(
            ((x, y - radius), (x - radius, y + radius), (x + radius, y + radius)), fill=color
        )
    elif shape == "diamond":
        draw.polygon(
            ((x, y - radius), (x - radius, y), (x, y + radius), (x + radius, y)), fill=color
        )
    else:
        points = []
        for point in range(10):
            angle = -math.pi / 2 + point * math.pi / 5
            distance = radius if point % 2 == 0 else radius * 0.45
            points.append(
                (round(x + math.cos(angle) * distance), round(y + math.sin(angle) * distance))
            )
        draw.polygon(points, fill=color)


def _audio(rate: int, duration: float, index: int) -> np.ndarray:
    timeline = np.arange(round(rate * duration), dtype=np.float64) / rate
    frequency = 180 + 17 * (index % 7)
    phase = math.tau * (frequency * timeline + 20 * timeline**2)
    envelope = np.sin(np.pi * timeline / duration) ** 0.5
    stereo = np.stack((np.sin(phase), np.sin(phase + math.pi / 3)), axis=-1)
    return np.rint(stereo * envelope[:, None] * 9000).astype(np.int16)


def _corrupt_frames(frames: Sequence[np.ndarray], accent: tuple[int, int, int]) -> list[np.ndarray]:
    output = [frames[0].copy()]
    for index, frame in enumerate(reversed(frames[1:-1]), start=1):
        changed = frame.copy()
        width = changed.shape[1]
        x = (index * 7) % max(width - 2, 1)
        changed[:, x : x + 2] = accent
        output.append(changed)
    output.append(frames[-1].copy())
    return output


def _corrupt_audio(samples: np.ndarray) -> np.ndarray:
    output = np.roll(samples, max(samples.shape[0] // 9, 1), axis=0).copy()
    output[:, 1] *= -1
    return output


def _decode(path: Path) -> list[np.ndarray]:
    with av.open(str(path)) as container:
        frames = [frame.to_ndarray(format="rgb24") for frame in container.decode(video=0)]
    if not frames:
        raise RuntimeError(f"generated video decoded no frames: {path}")
    return frames


def _seed(base: int, supervision: str, index: int) -> int:
    digest = hashlib.sha256(f"{base}:{supervision}:{index}".encode()).digest()
    return int.from_bytes(digest[:8], "big")


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _write_json(path: Path, value: Mapping[str, Any]) -> None:
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _write_jsonl(path: Path, values: Iterable[Mapping[str, Any]]) -> None:
    with path.open("w", encoding="utf-8") as output:
        for value in values:
            output.write(json.dumps(value, separators=(",", ":"), sort_keys=True) + "\n")


def _card(writer: _Writer) -> str:
    return f"""---
license: cc0-1.0
pretty_name: Flow-Factory {writer.supervision} Smoke Fixtures
---

# {writer.repo_id}

Deterministic procedural correctness fixtures for Flow-Factory's strict V2
offline schema. Each runtime alias under `profiles/` has {writer.records_per_alias} records and resolves
its `../../media/...` paths inside this repository. Joint outputs are ordered
`[video, audio]`. Preference pairs are synthetic smoke-only corruptions, not
human labels. See `dataset_manifest.json` and `provenance.jsonl` for identities.
"""


def _cc0_notice() -> str:
    return """CC0 1.0 Universal

To the extent possible under law, the authors waive copyright and related
rights in these generated dataset assets.
https://creativecommons.org/publicdomain/zero/1.0/legalcode
"""


def main(argv: Sequence[str] | None = None) -> int:
    """Build both public dataset staging trees from command-line arguments.

    Args:
        argv: Optional argument sequence. Uses ``sys.argv`` when omitted.

    Returns:
        Process exit code zero after a successful build.
    """
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--staging-root",
        type=Path,
        default=Path(__file__).resolve().parent / "_staging",
    )
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    parser.add_argument("--records-per-alias", type=int, default=RECORDS_PER_ALIAS)
    parser.add_argument("--replace", action="store_true")
    args = parser.parse_args(argv)
    paths = build(
        args.staging_root.expanduser().resolve(),
        args.seed,
        args.replace,
        args.records_per_alias,
    )
    print(json.dumps({"sft": str(paths[0]), "offline_dpo": str(paths[1])}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
