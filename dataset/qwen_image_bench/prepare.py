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

"""Build a local Flow-Factory dataset folder from Qwen/Qwen-Image-Bench.

Flow-Factory loads each ``dataset_dir`` from local ``train.jsonl`` / ``test.jsonl``
(it does not resolve HF repo ids), so this script downloads the benchmark and
writes those files. Each row keeps ``prompt`` plus ``dims_en`` (the per-prompt
facet checklist) and ``ID``; non-prompt columns are packed into the reward's
``metadata`` JSON, which switches the ``qwen_image_bench`` reward into faithful
per-prompt scoring.

Usage:
    python dataset/qwen_image_bench/prepare.py                  # English prompts
    python dataset/qwen_image_bench/prepare.py --lang cn        # Chinese prompts
    python dataset/qwen_image_bench/prepare.py --test-size 64   # eval slice size
    python dataset/qwen_image_bench/prepare.py --seed 0         # change eval sampling

Note: these are the benchmark's own 1000 prompts. Training on them and then
evaluating on Qwen-Image-Bench is train/test contamination -- fine for a demo,
but do not report it as a held-out benchmark result.
"""

import argparse
import json
import os
import random

from datasets import load_dataset

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo", default="Qwen/Qwen-Image-Bench", help="HF dataset repo id")
    parser.add_argument(
        "--lang",
        choices=["en", "cn"],
        default="en",
        help="Prompt language column to use (prompt_en / prompt_cn). Default: en.",
    )
    parser.add_argument(
        "--test-size",
        type=int,
        default=128,
        help="Number of rows randomly sampled into test.jsonl for eval. Default: 128.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for reproducible test-set sampling. Default: 42.",
    )
    args = parser.parse_args()

    dataset = load_dataset(args.repo)
    split_name = next(iter(dataset.keys()))
    split = dataset[split_name]
    prompt_col = f"prompt_{args.lang}"

    if prompt_col not in split.column_names:
        raise ValueError(
            f"column {prompt_col!r} not found in {args.repo} (columns: "
            f"{split.column_names}); pass --lang en or --lang cn accordingly"
        )
    if "dims_en" not in split.column_names:
        raise ValueError(
            f"column 'dims_en' not found in {args.repo} (columns: {split.column_names})"
        )

    rows = []
    for record in split:
        prompt = record[prompt_col]
        dims_en = record["dims_en"]
        if not prompt or not dims_en:
            continue
        rows.append({"prompt": prompt, "dims_en": dims_en, "ID": record["ID"]})

    if not rows:
        raise ValueError(f"no usable rows extracted from {args.repo}")

    if args.test_size < 0:
        raise ValueError(f"expected --test-size >= 0, got {args.test_size}")

    # Reproducible random eval slice (seed-controlled) instead of a sequential head.
    rng = random.Random(args.seed)
    test_count = min(args.test_size, len(rows))
    test_indices = sorted(rng.sample(range(len(rows)), test_count))
    test_rows = [rows[i] for i in test_indices]

    splits = {"train": rows, "test": test_rows}
    for name, data in splits.items():
        out_path = os.path.join(SCRIPT_DIR, f"{name}.jsonl")
        with open(out_path, "w", encoding="utf-8") as f:
            for row in data:
                f.write(json.dumps(row, ensure_ascii=False) + "\n")
        print(f"Wrote {len(data)} rows -> {out_path}")


if __name__ == "__main__":
    main()
