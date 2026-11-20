"""pii-bench (hivetrace/pii-bench) → наши сессии + эталонные спаны по репликам.

Обычный пример — сессия из одной реплики пользователя. L-DIALOG хранит диалог как
JSON-массив сообщений, а разметку — офсетами по сырой строке; режем на реплики
(без system) и переносим офсеты внутрь реплик.
"""

import json
from dataclasses import dataclass
from pathlib import Path

from backend.schemas import Message, Session

# Черновой словарь типов: pii-bench → публичный тип детектора (профиль pii-ru-en-v1).
# None — тип, которого детектор не выдаёт; такие спаны в метриках считаются отдельно.
TYPE_MAP: dict[str, str | None] = {
    "NAME": "PERSON",
    "PHONE_NUMBER": "PHONE",
    "EMAIL": "EMAIL",
    "ADDRESS": "ADDRESS",
    "BANK_CARD_NUMBER": "PAYMENT_CARD",
    "PASSPORT_NUMBER": "PASSPORT",
    "INN": "NATIONAL_ID",
    "SNILS": "NATIONAL_ID",
    "CVC": None,
    "KPP": None,
    "OGRN": None,
    "OGRNIP": None,
    "TOKEN": None,
    # документов ниже в профиле детектора пока нет (задача в PR #5): до правки — полнота 0,
    # после — прирост виден в той же таблице
    "DRIVER_LICENSE": "DRIVER_LICENSE",
    "BIRTH_CERTIFICATE": "BIRTH_CERTIFICATE",
    "MILITARY_ID": "MILITARY_ID",
    "OMS": "OMS",
    # типы профиля, которых нет в pii-bench и redmadrobot, — для своих наборов (test-v1)
    "DATE_OF_BIRTH": "DATE_OF_BIRTH",
    "BANK_ACCOUNT": "BANK_ACCOUNT",
}

# Категория данных по 152-ФЗ: pii — ПДн человека, sensitive — не ПДн, но маскируем,
# none — реквизиты организаций. ИНН решается по длине: 12 цифр — физлицо, 10 — юрлицо.
CATEGORY: dict[str, str] = {
    "NAME": "pii",
    "PHONE_NUMBER": "pii",
    "EMAIL": "pii",
    "ADDRESS": "pii",
    "BANK_CARD_NUMBER": "pii",
    "PASSPORT_NUMBER": "pii",
    "SNILS": "pii",
    "OGRNIP": "pii",
    "INN": "pii",
    "CVC": "sensitive",
    "TOKEN": "sensitive",
    "KPP": "none",
    "OGRN": "none",
}

_DIRECTION = {"user": "in", "assistant": "out"}


@dataclass(frozen=True)
class GoldSpan:
    turn: int
    start: int
    end: int
    type: str  # тип pii-bench
    text: str
    cat: str | None = None  # явная категория (в своих сценариях: тестовая карта — none)

    @property
    def detector_type(self) -> str | None:
        return TYPE_MAP.get(self.type)

    @property
    def category(self) -> str:
        # ИНН из 10 цифр — организация при любой разметке: redmadrobot ставит ИНН категорию pii явно,
        # и до итерации 7 правило по длине для него не срабатывало
        if self.type == "INN" and len("".join(ch for ch in self.text if ch.isdigit())) == 10:
            return "none"
        if self.cat is not None:
            return self.cat
        return CATEGORY.get(self.type, "pii")


@dataclass(frozen=True)
class BenchItem:
    domain: str
    session: Session
    gold: tuple[GoldSpan, ...]


def _dialog_turns(raw: str) -> list[tuple[str, str, int]] | None:
    """(role, content, офсет content в raw) для user/assistant; None — если не разобрать."""
    try:
        messages = json.loads(raw)
    except json.JSONDecodeError:
        return None
    turns, cursor = [], 0
    for m in messages:
        content = m.get("content", "")
        escaped = json.dumps(content, ensure_ascii=False)[1:-1]
        pos = raw.find(escaped, cursor)
        if pos < 0:
            return None
        cursor = pos + len(escaped)
        if m.get("role") not in _DIRECTION:
            continue
        if escaped != content:  # внутри есть экранирование — офсеты разъедутся
            return None
        turns.append((m["role"], content, pos))
    return turns


def to_item(row: dict) -> BenchItem | None:
    """Строка pii-bench → BenchItem; None для диалога, который не удалось разобрать."""
    raw, sid = row["text"], f"pb-{row['id']}"
    if row["domain"] == "L-DIALOG":
        turns = _dialog_turns(raw)
        if not turns:
            return None
        messages = [
            Message(turn=i, direction=_DIRECTION[role], text=content)
            for i, (role, content, _) in enumerate(turns)
        ]
        gold = []
        for e in row["entities"]:
            for i, (_, content, pos) in enumerate(turns):
                if pos <= e["start"] and e["end"] <= pos + len(content):
                    s, t = e["start"] - pos, e["end"] - pos
                    gold.append(GoldSpan(i, s, t, e["type"], content[s:t]))
                    break
            else:
                return None  # разметка вне реплик — пропускаем пример целиком
    else:
        messages = [Message(turn=0, direction="in", text=raw)]
        gold = [GoldSpan(0, e["start"], e["end"], e["type"], e["text"]) for e in row["entities"]]

    session = Session(
        id=sid, messages=messages, meta={"source": "pii-bench", "domain": row["domain"]}
    )
    return BenchItem(domain=row["domain"], session=session, gold=tuple(gold))


def load(data_dir: Path, split: str = "domain") -> list[BenchItem]:
    import pyarrow.parquet as pq  # только для загрузки; в тестах не нужен

    rows = pq.read_table(data_dir / f"{split}.parquet").to_pylist()
    return [item for row in rows if (item := to_item(row)) is not None]


def gold_to_json(gold: tuple[GoldSpan, ...]) -> list[dict]:
    return [g.__dict__ for g in gold]


def gold_from_json(items: list[dict]) -> tuple[GoldSpan, ...]:
    return tuple(GoldSpan(**g) for g in items)
