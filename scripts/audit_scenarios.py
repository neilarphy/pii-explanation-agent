"""Проверка сгенерированного набора: где эталон может быть неполным.

    uv run python -m scripts.audit_scenarios --logs runs/detector/scenarios-v1.jsonl
    uv run python -m scripts.audit_scenarios --apply data/scenarios/gold-fixes-v1.json \
        --to data/scenarios/scenarios-v1.jsonl runs/detector/scenarios-v1.jsonl

Генератор пишет текст сам и иногда вписывает ПДн сверх заглушек («меня зовут Алексей»).
Таких значений нет в эталоне, и верный ответ агента засчитывается как ошибка. Скрипт
выводит всё, что нашли детектор (принятые сущности) или поиск кодом и что не пересекается
ни с одним эталонным спаном, — это список для ручного просмотра.

Подтверждённые дыры записываются в gold-fixes-*.json (с причиной) и дописываются в эталон
через --apply. Офсеты считает код по тексту реплики; повторный запуск ничего не меняет.
"""

import argparse
import json
import sys
from pathlib import Path

from backend.agent.checks import scan
from backend.schemas import DetectionLog, Session
from evaluation.pii_bench import GoldSpan, gold_from_json
from evaluation.scenarios import apply_fixes


def _outside(turn: int, start: int, end: int, gold: tuple[GoldSpan, ...]) -> bool:
    return not any(g.turn == turn and g.start < end and start < g.end for g in gold)


def suspects(session: Session, log: DetectionLog, gold: tuple[GoldSpan, ...]) -> list[dict]:
    texts = {m.turn: m.text for m in session.messages}
    found = [
        {"turn": e.turn, "text": e.text, "by": f"детектор {e.type}"}
        for e in log.entities
        if _outside(e.turn, e.start, e.end, gold)
    ]
    for m in session.messages:
        for start, end, check in scan(m.text):
            if _outside(m.turn, start, end, gold):
                found.append(
                    {"turn": m.turn, "text": m.text[start:end], "by": f"код {check.label}"}
                )
    for f in found:
        f["context"] = texts[f["turn"]]
    return found


def _apply(fixes_path: Path, targets: list[Path]) -> None:
    fixes = json.loads(fixes_path.read_text("utf-8"))
    for path in targets:
        records = [json.loads(line) for line in path.read_text("utf-8").splitlines()]
        added = sum(apply_fixes(r, fixes) for r in records)
        lines = (json.dumps(r, ensure_ascii=False) + "\n" for r in records)
        path.write_text("".join(lines), "utf-8", newline="\n")
        print(f"{path}: добавлено спанов {added}")


def main() -> int:
    sys.stdout.reconfigure(encoding="utf-8")
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument(
        "--logs", type=Path, help="логи детектора по набору — вывести подозрительное"
    )
    parser.add_argument("--apply", type=Path, help="gold-fixes-*.json — дописать в эталон")
    parser.add_argument("--to", type=Path, nargs="+", default=[], help="куда дописывать")
    args = parser.parse_args()
    if args.apply:
        _apply(args.apply, args.to)
        return 0
    total = 0
    for line in args.logs.read_text("utf-8").splitlines():
        record = json.loads(line)
        session = Session.model_validate(record["session"])
        log = DetectionLog.model_validate(record["log"])
        items = suspects(session, log, gold_from_json(record["gold"]))
        if items:
            print(f"\n## {session.id}")
            for f in items:
                print(f"- [{f['turn']}] «{f['text']}» — {f['by']}\n  {f['context']}")
            total += len(items)
    print(f"\nНа просмотр: {total}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
