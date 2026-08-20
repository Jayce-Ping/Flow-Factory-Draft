#!/usr/bin/env python3
"""Convert Flow-Factory's plain-text OCR prompts into reward-ready JSONL.

The bundled ``dataset/ocr/{train,test}.txt`` files contain one prompt per line and
put the text to render inside straight or curly double quotes. Training can consume
the text files directly, but the OCR reward requires a ``visible_texts`` metadata
field. This converter extracts that target without changing the prompt.

Usage:
    python scripts/convert_ocr_prompts.py \
        --input-dir dataset/ocr \
        --output-dir dataset/ocr_metadata
"""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

_QUOTED_TEXT = re.compile(r'["“](.*?)["”]')


def convert_split(input_path: Path, output_path: Path) -> int:
    """Convert one split and fail on any prompt without an explicit target."""
    prompts = input_path.read_text(encoding="utf-8").splitlines()
    if not prompts:
        raise ValueError(f"expected non-empty OCR prompt file, received {input_path}")

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as handle:
        for line_number, prompt in enumerate(prompts, 1):
            targets = [target.strip() for target in _QUOTED_TEXT.findall(prompt) if target.strip()]
            if not targets:
                raise ValueError(
                    "expected at least one straight/curly double-quoted OCR target, "
                    f"received none at {input_path}:{line_number}: {prompt!r}"
                )
            record = {
                "prompt": prompt,
                # Complex source fields are JSON strings so Arrow metadata stays stable.
                "visible_texts": json.dumps(targets, ensure_ascii=False),
                "id": f"ocr-{input_path.stem}-{line_number:06d}",
            }
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")
    return len(prompts)


def main() -> None:
    """Convert train and test splits."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()

    for split in ("train", "test"):
        count = convert_split(
            args.input_dir / f"{split}.txt",
            args.output_dir / f"{split}.jsonl",
        )
        print(f"converted {count} {split} prompts")


if __name__ == "__main__":
    main()
