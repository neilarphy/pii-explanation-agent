"""Run with python -m data; no dataset text is printed in error reports."""

import argparse
import hashlib
import json
import re
import urllib.request
from pathlib import Path

from data.loader import DATASET, REVISION, convert_row, read_csv, read_jsonl
from data.schema import Example


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    schema = commands.add_parser("schema")
    schema.add_argument("output", type=Path)
    validate = commands.add_parser("validate")
    validate.add_argument("input", type=Path)
    load = commands.add_parser("load-redmadrobot")
    load.add_argument("--revision", default=REVISION)
    load.add_argument("--input", type=Path, help="Offline CSV instead of download")
    load.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    if args.command == "schema":
        args.output.write_text(
            json.dumps(Example.model_json_schema(), ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )
        return
    if args.command == "validate":
        print(json.dumps({"valid_records": sum(1 for _ in read_jsonl(args.input))}))
        return
    if not re.fullmatch(r"[0-9a-f]{40}", args.revision):
        parser.error("revision must be an immutable 40-character commit SHA")
    # A fresh directory prevents accidental overwrite of an earlier experiment.
    args.output_dir.mkdir(parents=True, exist_ok=False)
    csv_path = args.input
    url = f"https://huggingface.co/datasets/{DATASET}/resolve/{args.revision}/test.csv"
    if csv_path is None:
        csv_path = args.output_dir / "source.csv"
        with urllib.request.urlopen(url, timeout=60) as response:
            csv_path.write_bytes(response.read())
    digest = hashlib.sha256(csv_path.read_bytes()).hexdigest()
    counts = {"rows": 0, "accepted": 0, "rejected": 0, "duplicates": 0, "entities": 0}
    seen = set()
    labels: dict[str, int] = {}
    with (
        (args.output_dir / "examples.jsonl").open("w", encoding="utf-8") as output,
        (args.output_dir / "rejections.jsonl").open("w", encoding="utf-8") as rejected,
    ):
        for index, row in enumerate(read_csv(csv_path)):
            counts["rows"] += 1
            try:
                example = convert_row(row, index, args.revision)
            except (ValueError, KeyError, TypeError):
                counts["rejected"] += 1
                # Keep row references, never PII or validation exceptions containing input.
                rejected.write(
                    json.dumps({"source_id": str(index), "reason": "invalid_bio_or_alignment"})
                    + "\n"
                )
                continue
            if example.source.group_id in seen:
                counts["duplicates"] += 1
                continue
            seen.add(example.source.group_id)
            output.write(example.model_dump_json() + "\n")
            counts["accepted"] += 1
            counts["entities"] += len(example.gold_entities)
            for entity in example.gold_entities:
                labels[entity.entity_type] = labels.get(entity.entity_type, 0) + 1
    manifest = {
        "dataset": DATASET,
        "revision": args.revision,
        "license": "mit",
        "url": url,
        "input_mode": "local_unverified" if args.input else "download_pinned",
        "sha256": digest,
        "adapter_version": "1.0",
        "counts": counts,
        "entity_counts": labels,
        "split": "test",
        "filters": [
            "strict BIO",
            "anchored case-insensitive token alignment",
            "exact text deduplication",
        ],
        "transformations": ["BIO to Unicode character spans", "one sentence to one user message"],
        "split_policy": "original test only; no train or calibration reuse; group by text SHA256",
    }
    (args.output_dir / "manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(counts))


if __name__ == "__main__":
    main()
