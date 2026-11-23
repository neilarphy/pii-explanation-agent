"""Convert the pinned redmadrobot BIO benchmark to the shared evaluation format."""

import argparse
import csv
import hashlib
import io
import json
import logging
import re
import shutil
import tempfile
import urllib.request
from collections.abc import Callable
from pathlib import Path
from typing import TextIO

from backend.schemas import Message, Session
from evaluation.pii_bench import BenchItem, GoldSpan

DATASET = "redmadrobot-rnd/pii_benchmark"
REVISION = "f77ea831274daf980cc45c61a93c226be9d978d6"
SHA256 = "6bf544a380a3ee5bec94b946124bea3afaecce49e734679ad0f0c0e7c12977bb"
URL = f"https://huggingface.co/datasets/{DATASET}/resolve/{REVISION}/test.csv"
LOGGER = logging.getLogger(__name__)

TYPE: dict[str, tuple[str, str]] = {  # тип redmadrobot -> (тип pii-bench, категория)
    "FIRST_NAME": ("NAME", "pii"),
    "LAST_NAME": ("NAME", "pii"),
    "MIDDLE_NAME": ("NAME", "pii"),
    "CITY": ("ADDRESS", "pii"),
    "STREET": ("ADDRESS", "pii"),
    "HOUSE": ("ADDRESS", "pii"),
    "COUNTRY": ("COUNTRY", "none"),
    "REGION": ("REGION", "none"),
    "DISTRICT": ("DISTRICT", "none"),
    "EMAIL": ("EMAIL", "pii"),
    "PHONE": ("PHONE_NUMBER", "pii"),
    "URL": ("URL", "none"),
    "IP_ADDRESS": ("IP_ADDRESS", "pii"),
    "PASSPORT": ("PASSPORT_NUMBER", "pii"),
    "INN": ("INN", "pii"),
    "SNILS": ("SNILS", "pii"),
    "OMS": ("OMS", "pii"),
    "CREDIT_CARD": ("BANK_CARD_NUMBER", "pii"),
    "DRIVER_LICENSE": ("DRIVER_LICENSE", "pii"),
    "MILITARY_ID": ("MILITARY_ID", "pii"),
    "BIRTH_CERTIFICATE": ("BIRTH_CERTIFICATE", "pii"),
}


class RowError(ValueError):
    """A row rejection with a fixed, safe reason code."""


def _item(index: int, text: str, tokens: list[str], tags: list[str]) -> BenchItem:
    if (
        not isinstance(text, str)
        or not isinstance(tokens, list)
        or not isinstance(tags, list)
        or not tokens
        or len(tokens) != len(tags)
    ):
        raise RowError("invalid_token_arrays")

    if not text.strip():
        raise RowError("empty_text")

    cursor = 0
    spans: list[tuple[str, int, int]] = []
    active: str | None = None
    for token, tag in zip(tokens, tags, strict=True):
        if not isinstance(tag, str):
            raise RowError("invalid_bio_tag")
        prefix, _, label = tag.partition("-")
        if tag != "O" and (prefix not in {"B", "I"} or label not in TYPE):
            raise RowError("invalid_bio_tag")
        if (
            not isinstance(token, str)
            or (not token and tag != "O")
            or (token and not token.strip())
        ):
            raise RowError("invalid_token")
        if tag == "O":
            active = None
        if not token:
            continue

        # Match on the original string to preserve Unicode code-point offsets.
        match = re.compile(r"\s*(" + re.escape(token) + ")", re.IGNORECASE).match(text, cursor)
        if match is None:
            raise RowError("unaligned_token")
        start, end = match.span(1)
        cursor = end

        if prefix == "B":
            active = label
            spans.append((label, start, end))
        elif prefix == "I":
            if active != label:
                raise RowError("invalid_bio_transition")
            spans[-1] = (label, spans[-1][1], end)
    if text[cursor:].strip():
        raise RowError("unaligned_tail")

    gold = []
    for label, start, end in spans:
        if not 0 <= start < end <= len(text):
            raise RowError("invalid_span")
        bench_type, category = TYPE[label]
        gold.append(GoldSpan(0, start, end, bench_type, text[start:end], category))
    session = Session(
        id=f"rmr-{index}",
        messages=[Message(turn=0, direction="in", text=text)],
        meta={"source": "redmadrobot"},
    )
    return BenchItem(domain="redmadrobot", session=session, gold=tuple(gold))


def to_item(index: int, text: str, tokens: list[str], tags: list[str]) -> BenchItem | None:
    try:
        return _item(index, text, tokens, tags)
    except ValueError:
        return None


def _scan(handle: TextIO, reject: Callable[[int, str], None]) -> list[BenchItem]:
    items = []
    reader = csv.DictReader(handle, strict=True)
    try:
        if reader.fieldnames != ["text", "tokens", "ner_tags"]:
            raise ValueError("invalid_csv_header")
        for index, row in enumerate(reader):
            if None in row or any(value is None for value in row.values()):
                reject(index, "invalid_csv_row")
                continue
            try:
                tokens = json.loads(row["tokens"])
                tags = json.loads(row["ner_tags"])
            except (ValueError, RecursionError):
                reject(index, "invalid_json")
                continue
            try:
                items.append(_item(index, row["text"], tokens, tags))
            except RowError as exc:
                reject(index, str(exc))
            except ValueError:
                reject(index, "invalid_record")
    except (csv.Error, UnicodeError):
        pass
    else:
        return items
    raise ValueError("invalid_csv_encoding_or_structure")


def load(path: Path) -> list[BenchItem]:
    with path.open(encoding="utf-8-sig", newline="") as handle:
        return _scan(handle, lambda i, reason: LOGGER.warning("redmadrobot row %d: %s", i, reason))


def _verify(path: Path) -> None:
    with path.open("rb") as handle:
        if hashlib.file_digest(handle, "sha256").hexdigest() != SHA256:
            raise ValueError("sha256_mismatch")


def download(path: Path) -> None:
    """Publish the pinned CSV only after verifying its original bytes."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(dir=path.parent) as directory:
        pending = Path(directory) / "test.csv"
        with urllib.request.urlopen(URL, timeout=60) as response, pending.open("wb") as output:
            shutil.copyfileobj(response, output)
        _verify(pending)
        pending.replace(path)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, default=Path("data/raw/redmadrobot/test.csv"))
    parser.add_argument("--download", action="store_true")
    parser.add_argument("--output-dir", type=Path, default=Path("data/raw/redmadrobot/audit"))
    args = parser.parse_args()
    rejected = []

    def reject(index: int, reason: str) -> None:
        rejected.append({"source_index": index, "reason": reason})

    try:
        if args.download:
            download(args.input)
        # Hash an immutable snapshot so a concurrent replacement cannot change the audit input.
        snapshot = args.input.read_bytes()
        if hashlib.sha256(snapshot).hexdigest() != SHA256:
            raise ValueError("sha256_mismatch")
        with io.TextIOWrapper(io.BytesIO(snapshot), encoding="utf-8-sig", newline="") as handle:
            items = _scan(handle, reject)
        counts = {
            "rows": len(items) + len(rejected),
            "accepted": len(items),
            "rejected": len(rejected),
            "entities": sum(len(item.gold) for item in items),
            "duplicate_texts": len(items) - len({item.session.messages[0].text for item in items}),
        }
        manifest = {
            "dataset": DATASET,
            "revision": REVISION,
            "license": "MIT",
            "url": URL,
            "sha256": SHA256,
            "adapter_version": 2,
            "split": "test",
            "counts": counts,
        }
        args.output_dir.mkdir(parents=True, exist_ok=True)
        (args.output_dir / "manifest.json").write_text(
            json.dumps(manifest, indent=2) + "\n", encoding="utf-8"
        )
        (args.output_dir / "rejections.jsonl").write_text(
            "".join(json.dumps(row) + "\n" for row in rejected), encoding="utf-8"
        )
    except ValueError as exc:
        parser.exit(1, f"redmadrobot: {exc}\n")
    except OSError:
        parser.exit(1, "redmadrobot: input, download or output unavailable\n")
    print(json.dumps(counts))
    return int(bool(rejected))


if __name__ == "__main__":
    raise SystemExit(main())
