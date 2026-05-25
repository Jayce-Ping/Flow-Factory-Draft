#!/usr/bin/env python3
"""
Convert DiffusionNFT GenEval dataset to Flow-Factory format.

Input format (DiffusionNFT):
    {"tag": "...", "include": [...], "prompt": "...", "exclude": [...]}

Output format (Flow-Factory):
    {"prompt": "...", "geneval_metadata": "<JSON string of full metadata>"}

Usage:
    python scripts/convert_geneval_dataset.py
"""
import json
import os

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT = os.path.dirname(SCRIPT_DIR)
SCRATCH_DIR = os.path.join(REPO_ROOT, ".scratch")
OUTPUT_DIR = os.path.join(REPO_ROOT, "dataset", "geneval")


def process_geneval(input_path: str, output_path: str) -> tuple:
    """Deduplicate by prompt and convert to Flow-Factory format."""
    seen_prompts = set()
    unique_records = []
    duplicates = 0

    with open(input_path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            record = json.loads(line)
            prompt = record["prompt"]
            if prompt in seen_prompts:
                duplicates += 1
                continue
            seen_prompts.add(prompt)
            # Convert: store full metadata as JSON string to avoid Arrow issues
            output_record = {
                "prompt": prompt,
                "geneval_metadata": json.dumps(record, ensure_ascii=False),
            }
            unique_records.append(output_record)

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, "w") as f:
        for rec in unique_records:
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")

    return len(unique_records), duplicates


def main():
    train_input = os.path.join(SCRATCH_DIR, "geneval_train_raw.jsonl")
    test_input = os.path.join(SCRATCH_DIR, "geneval_test_raw.jsonl")
    train_output = os.path.join(OUTPUT_DIR, "train.jsonl")
    test_output = os.path.join(OUTPUT_DIR, "test.jsonl")

    if not os.path.exists(train_input):
        print(f"ERROR: {train_input} not found.")
        print("Download first with:")
        print(
            "  curl -sL https://raw.githubusercontent.com/NVlabs/DiffusionNFT/main/dataset/geneval/train_metadata.jsonl"
            f" -o {train_input}"
        )
        return

    print("Processing GenEval dataset...")
    print(f"  Input:  {train_input}")
    print(f"  Output: {train_output}")

    train_unique, train_dups = process_geneval(train_input, train_output)
    print(f"  Train: {train_unique} unique prompts, {train_dups} duplicates removed")

    if os.path.exists(test_input):
        test_unique, test_dups = process_geneval(test_input, test_output)
        print(f"  Test:  {test_unique} unique prompts, {test_dups} duplicates removed")
    else:
        print(f"  WARNING: {test_input} not found, skipping test set.")

    print("Done!")


if __name__ == "__main__":
    main()
