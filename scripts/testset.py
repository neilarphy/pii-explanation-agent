"""Замороженный тестовый набор: проверка черновика и сборка файла с офсетами.

    # отпечаток — до того, как набор увидит кто-то, кроме автора
    uv run python -m scripts.testset hash --in test-v1.draft.jsonl
    # проверка и сборка: офсеты считает скрипт, руками их писать не надо
    uv run python -m scripts.testset build --in test-v1.draft.jsonl --out data/scenarios/test-v1.jsonl

Строка черновика — как в data/scenarios/*.jsonl, только в эталоне нет start/end:
{"domain": "...", "session": {"id": "ts-...", "messages": [{"turn", "direction", "text"}]},
 "gold": [{"turn": 0, "text": "дословно из реплики", "type": "PHONE_NUMBER", "cat": "pii",
           "occurrence": 1}]}
occurrence — какое по счёту вхождение текста в реплике (по умолчанию первое).
"""

import argparse
import hashlib
import json
import sys
from collections import Counter
from pathlib import Path

from backend.schemas import Session

TYPES = {
    "NAME",
    "PHONE_NUMBER",
    "EMAIL",
    "ADDRESS",
    "BANK_CARD_NUMBER",
    "PASSPORT_NUMBER",
    "INN",
    "SNILS",
    "OGRN",
    "OGRNIP",
    "KPP",
    "CVC",
    "TOKEN",
    "DATE_OF_BIRTH",
    "DRIVER_LICENSE",
    "BIRTH_CERTIFICATE",
    "MILITARY_ID",
    "OMS",
    "IP_ADDRESS",
    "SMS_CODE",
    "BANK_ACCOUNT",
    "OTHER",
}
CATS = {"pii", "sensitive", "none", "ambiguous"}


def _locate(text: str, needle: str, occurrence: int) -> int | None:
    start = -1
    for _ in range(occurrence):
        start = text.find(needle, start + 1)
        if start < 0:
            return None
    return start


def build_row(raw: dict, n: int) -> tuple[dict | None, list[str]]:
    errors = []
    where = f"строка {n}"
    try:
        session = Session.model_validate(raw["session"])
    except Exception as exc:  # pydantic объяснит, что не так со схемой
        return None, [f"{where}: сессия не по схеме — {exc}"]
    where = f"{where} ({session.id})"
    if not session.id.startswith("ts-"):
        errors.append(f"{where}: id сессии должен начинаться с ts-")
    if not raw.get("domain"):
        errors.append(f"{where}: нет domain (тема диалога)")
    turns = {m.turn: m.text for m in session.messages}
    gold = []
    for g in raw.get("gold", []):
        if g.get("type") not in TYPES:
            errors.append(f"{where}: тип {g.get('type')!r} не из списка")
        if g.get("cat") not in CATS:
            errors.append(f"{where}: категория {g.get('cat')!r} не из {sorted(CATS)}")
        text = turns.get(g.get("turn"))
        start = None if text is None else _locate(text, g.get("text", ""), g.get("occurrence", 1))
        if not g.get("text") or start is None:
            errors.append(f"{where}: «{g.get('text')}» нет в реплике {g.get('turn')} дословно")
            continue
        gold.append(
            {
                "turn": g["turn"],
                "start": start,
                "end": start + len(g["text"]),
                "type": g["type"],
                "text": g["text"],
                "cat": g["cat"],
            }
        )
    row = {"domain": raw.get("domain"), "session": session.model_dump(mode="json"), "gold": gold}
    return row, errors


def main() -> int:
    sys.stdout.reconfigure(encoding="utf-8")
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    sub = parser.add_subparsers(dest="command", required=True)
    h = sub.add_parser("hash")
    h.add_argument("--in", dest="src", type=Path, required=True)
    b = sub.add_parser("build")
    b.add_argument("--in", dest="src", type=Path, required=True)
    b.add_argument("--out", type=Path)
    args = parser.parse_args()
    if args.command == "hash":
        print(f"sha256 {hashlib.sha256(args.src.read_bytes()).hexdigest()}  {args.src.name}")
        return 0
    rows, errors, ids = [], [], Counter()
    for n, line in enumerate(args.src.read_text("utf-8").splitlines(), 1):
        if not line.strip():
            continue
        row, errs = build_row(json.loads(line), n)
        errors += errs
        if row:
            rows.append(row)
            ids[row["session"]["id"]] += 1
    errors += [f"id {i} повторяется {c} раз" for i, c in ids.items() if c > 1]
    cats = Counter(g["cat"] for r in rows for g in r["gold"])
    domains = Counter(r["domain"] for r in rows)
    print(f"Диалогов: {len(rows)}, эталонных фрагментов: {sum(cats.values())} {dict(cats)}")
    print(f"Тем: {len(domains)}; без эталона (только ловушки): {sum(not r['gold'] for r in rows)}")
    if errors:
        print(f"\nОшибок: {len(errors)}")
        print("\n".join(f"- {e}" for e in errors))
        return 1
    if args.out:
        args.out.write_text(
            "".join(json.dumps(r, ensure_ascii=False) + "\n" for r in rows), "utf-8"
        )
        print(f"Собрано: {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
