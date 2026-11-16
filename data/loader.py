"""Strict CSV/BIO adapter with explicit, auditable rejection of ambiguous rows."""

import csv
import hashlib
import json
import re
from collections.abc import Iterator
from pathlib import Path

from data.schema import Example, Message, Source, Span

DATASET = "redmadrobot-rnd/pii_benchmark"
REVISION = "f77ea831274daf980cc45c61a93c226be9d978d6"
LABELS = frozenset(
    [
        "FIRST_NAME",
        "LAST_NAME",
        "MIDDLE_NAME",
        "COUNTRY",
        "REGION",
        "DISTRICT",
        "CITY",
        "STREET",
        "HOUSE",
        "EMAIL",
        "PHONE",
        "URL",
        "IP_ADDRESS",
        "PASSPORT",
        "INN",
        "SNILS",
        "OMS",
        "CREDIT_CARD",
        "DRIVER_LICENSE",
        "MILITARY_ID",
        "BIRTH_CERTIFICATE",
    ]
)


def convert_row(row: dict[str, str], index: int, revision: str = REVISION) -> Example:
    text = row["text"]
    tokens, tags = json.loads(row["tokens"]), json.loads(row["ner_tags"])
    if (
        not isinstance(tokens, list)
        or not isinstance(tags, list)
        or len(tokens) != len(tags)
        or not tokens
    ):
        raise ValueError("tokens and tags must be nonempty equally sized arrays")
    offsets = []
    cursor = 0
    for token in tokens:
        if not isinstance(token, str) or not token:
            raise ValueError("invalid token")
        # Anchored alignment: never skip non-whitespace or normalize the source text.
        match = re.compile(r"\s*" + re.escape(token), re.IGNORECASE).match(text, cursor)
        if match is None:
            raise ValueError("token cannot be aligned without skipping text")
        start = match.end() - len(token)
        offsets.append((start, match.end()))
        cursor = match.end()
    if text[cursor:].strip():
        raise ValueError("uncovered trailing text")
    entities: list[Span] = []
    active: tuple[str, int, int] | None = None

    def flush() -> None:
        if active is not None:
            label, start, end = active
            entities.append(
                Span(
                    id=f"e{len(entities)}",
                    message_id="m0",
                    entity_type=label,
                    start=start,
                    end=end,
                    text=text[start:end],
                )
            )

    for tag, (start, end) in zip(tags, offsets, strict=True):
        if not isinstance(tag, str) or (
            tag != "O" and (tag[:2] not in {"B-", "I-"} or tag[2:] not in LABELS)
        ):
            raise ValueError("unknown BIO label")
        if tag.startswith("I-"):
            if active is None or active[0] != tag[2:]:
                raise ValueError("orphan or mismatched I tag")
            active = (active[0], active[1], end)
        else:
            flush()
            active = (tag[2:], start, end) if tag.startswith("B-") else None
    flush()
    digest = hashlib.sha256(text.encode()).hexdigest()
    return Example(
        session_id=f"redmadrobot:{revision}:{index}",
        messages=[Message(id="m0", role="user", text=text)],
        gold_entities=entities,
        source=Source(
            dataset=DATASET,
            revision=revision,
            license="mit",
            source_id=str(index),
            group_id=digest,
            split="test",
            url=f"https://huggingface.co/datasets/{DATASET}",
        ),
        metadata={
            "adapter_version": "1.0",
            "alignment": "anchored_case_insensitive",
            "original_split": "test",
        },
    )


def read_csv(path: Path) -> Iterator[dict[str, str]]:
    with path.open(encoding="utf-8-sig", newline="") as stream:
        reader = csv.DictReader(stream)
        if reader.fieldnames != ["text", "tokens", "ner_tags"]:
            raise ValueError("unexpected CSV header")
        yield from reader


def read_jsonl(path: Path) -> Iterator[Example]:
    seen = set()
    with path.open(encoding="utf-8") as stream:
        for line_number, line in enumerate(stream, 1):
            try:
                example = Example.model_validate_json(line)
                if example.session_id in seen:
                    raise ValueError("duplicate session_id")
                seen.add(example.session_id)
                yield example
            except ValueError as exc:
                raise ValueError(f"invalid JSONL record at line {line_number}") from exc
