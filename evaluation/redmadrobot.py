"""redmadrobot-rnd/pii_benchmark (MIT) → наши сессии и эталонные спаны.

Разметка в бенчмарке — BIO по токенам, а токены местами нормализованы (регистр,
пробелы вокруг пунктуации). Ищем каждый токен в исходном тексте по порядку без учёта
регистра; если токен не нашёлся — пример пропускаем целиком.

Типы приводим к словарю pii-bench, чтобы работали те же метрики. Части адреса без человека
(страна, регион, район) и URL — не ПДн по 152-ФЗ: категория none.
"""

import csv
import json
from pathlib import Path

from backend.schemas import Message, Session
from evaluation.pii_bench import BenchItem, GoldSpan

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


def _token_offsets(text: str, tokens: list[str]) -> list[tuple[int, int]] | None:
    lower, cursor, offsets = text.lower(), 0, []
    for token in tokens:
        pos = lower.find(token.lower(), cursor)
        if pos < 0:
            return None
        offsets.append((pos, pos + len(token)))
        cursor = pos + len(token)
    return offsets


def to_item(index: int, text: str, tokens: list[str], tags: list[str]) -> BenchItem | None:
    offsets = _token_offsets(text, tokens)
    if offsets is None or len(tokens) != len(tags):
        return None
    spans: list[list] = []  # [label, start, end]
    for (start, end), tag in zip(offsets, tags, strict=True):
        if tag == "O":
            continue
        prefix, label = tag.split("-", 1)
        if prefix == "I" and spans and spans[-1][0] == label:
            spans[-1][2] = end
        else:
            spans.append([label, start, end])
    gold = []
    for label, start, end in spans:
        bench_type, category = TYPE.get(label, (label, "pii"))
        gold.append(GoldSpan(0, start, end, bench_type, text[start:end], category))
    session = Session(
        id=f"rmr-{index}",
        messages=[Message(turn=0, direction="in", text=text)],
        meta={"source": "redmadrobot"},
    )
    return BenchItem(domain="redmadrobot", session=session, gold=tuple(gold))


def load(path: Path) -> list[BenchItem]:
    items = []
    with path.open(encoding="utf-8") as handle:
        for i, row in enumerate(csv.DictReader(handle)):
            item = to_item(i, row["text"], json.loads(row["tokens"]), json.loads(row["ner_tags"]))
            if item is not None:
                items.append(item)
    return items
