"""Расход на LLM по сохранённым прогонам — без вызовов модели.

    uv run python -m scripts.costs --out docs/costs.md

Каждый ответ агента пишет телеметрию: вызовы, токены, а с 19.11 (итерация 6б) — точную цену от
провайдера (`usage.cost`, USD). Где цены нет, она оценивается по тарифам bothub × токены; где нет
и токенов (итерации 1–4, первые прогоны судьи), — «нет данных». Снимки баланса bothub — в `BALANCE`.
"""

import argparse
import json
import re
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path

CAPS_PER_USD = 700_000  # курс по usage.cost в итерациях 6–7

# Капсов за токен (вход, выход), https://bothub.chat/models?currency=caps на 19.11.2026.
# deepseek-v4-pro списывается по цене провайдера, днём ~2.4× тарифа — оценка по нему занижена.
TARIFFS = {
    "qwen3.8-27b": (0.04, 1.55),
    "deepseek-v4-pro": (0.39, 1.17),
    "qwen3.5-9b": (0.12, 0.18),
    "qwen3-30b-a3b": (0.08, 0.35),
    "gemma-3-12b-it": (0.04, 0.11),
}
DEFAULT_MODEL = "qwen3.8-27b"  # основной агент итераций 1–5: файлы agent_answers.jsonl

# Снимки баланса bothub (капсы) — со слов пользователя и по выгрузке bothub.
BALANCE = [
    ("итерация 5", "пополнение на 8 млн"),
    ("пилот итерации 6", "≈ 6 млн — с пополнения списано ≈ 2.5 млн"),
    ("19.11, после итерации 6", "≈ 0.6 млн; выгрузка bothub за 19.11 — 5.49 млн"),
    ("19.11, итерация 6б", "пополнено до ≈ 9 млн"),
    ("20.11, старт итерации 7", "≈ 6.4 млн"),
    ("20.11, старт итерации 8", "9.8 млн после пополнения"),
]

# Итерация 8: оценка до шага и факт после (капсы; факт — по usage.cost в прогонах шага).
PLAN = [
    ("0. Сводка затрат", "0", "0"),
    ("1. Невидимые ПДн по политике", "0", "0"),
    ("2. Детектор с низким порогом", "0 (локальная GPU)", "0 (3 набора, ~6 мин GPU)"),
    ("3. Поиск кодом шире", "0", "0"),
    ("3а. Агент на находках поиска кодом (64, вне плана)", "—", "0.25 млн"),
    ("4. Набор hidden-v1: генерация", "0.1–0.2 млн", "0.07 млн"),
    ("5. Чтение диалога LLM: код и тесты", "0", "0"),
    ("6. Пилот чтения: 27b против qwen3-30b", "≈ 0.3 млн", "0.37 млн (чтение 0.15, разбор 0.23)"),
    ("7. Полный прогон лучшего варианта", "1–2 млн", "0 — отменён: пилот ответил на вопрос"),
    ("8. Промпт v2 на маленьких моделях и каскаде", "≈ 0.3 млн", "0.36 млн"),
    ("9. Сводка, отчёт, PR", "0", "0"),
]


@dataclass
class Run:
    iteration: str
    step: str
    model: str
    rows: int = 0
    calls: int = 0
    prompt_tokens: int = 0
    completion_tokens: int = 0
    usd: float = 0.0  # точная цена по usage.cost
    priced_rows: int = 0  # строк с usage.cost
    estimated_caps: float = 0.0  # по тарифам для строк без usage.cost
    unknown_rows: int = 0  # ни цены, ни токенов (или модель без тарифа)

    @property
    def caps(self) -> float:
        return self.usd * CAPS_PER_USD + self.estimated_caps

    @property
    def basis(self) -> str:
        if self.unknown_rows == self.rows:
            return "нет данных"
        if self.priced_rows == self.rows:
            return "usage.cost"
        if self.priced_rows == 0 and self.unknown_rows == 0:
            return "тарифы × токены"
        return "смешанно"


def iteration_of(path: Path) -> str:
    top = path.parts[path.parts.index("runs") + 1] if "runs" in path.parts else path.parts[0]
    if top.startswith("judge-iter7"):
        return "7"
    if top.startswith("judge"):
        return "6"
    if m := re.match(r"iter(\d+)", top):
        return m.group(1)
    return "1"  # eval-* — первый сквозной прогон


def model_of(path: Path) -> str:
    return path.stem.split("__")[0] if "__" in path.stem else DEFAULT_MODEL


def read_run(path: Path, root: Path) -> Run:
    rel = path.relative_to(root)
    judge = path.name == "judge.jsonl"
    run = Run(
        iteration=iteration_of(rel),
        step=(rel.parent if judge or path.name == "agent_answers.jsonl" else rel)
        .relative_to("runs")
        .as_posix(),
        model="судья" if judge else model_of(path),
    )
    for line in path.read_text("utf-8").splitlines():
        if not line.strip():
            continue
        r = json.loads(line)
        run.rows += 1
        t = r.get("telemetry") or {}
        cost = t.get("cost", r.get("cost"))
        run.calls += t.get("llm_calls", 1 if judge else 0)
        run.prompt_tokens += t.get("prompt_tokens", 0)
        run.completion_tokens += t.get("completion_tokens", 0)
        if cost is not None:
            run.usd += cost
            run.priced_rows += 1
        elif t.get("prompt_tokens") and run.model in TARIFFS:
            tin, tout = TARIFFS[run.model]
            run.estimated_caps += t["prompt_tokens"] * tin + t.get("completion_tokens", 0) * tout
        else:
            run.unknown_rows += 1
    return run


def collect(root: Path) -> list[Run]:
    files = [
        p
        for p in sorted(root.rglob("*.jsonl"))
        if p.name == "judge.jsonl" or p.name == "agent_answers.jsonl" or "__" in p.stem
    ]
    return [read_run(p, root.parent) for p in files]


def _mln(caps: float) -> str:
    return f"{caps / 1e6:.2f}"


def _n(x: float) -> str:
    return f"{x:,.0f}".replace(",", " ")


def report(runs: list[Run]) -> str:
    by_iter: dict[str, list[Run]] = defaultdict(list)
    for r in runs:
        by_iter[r.iteration].append(r)
    lines = [
        "# Расход на LLM по итерациям",
        "",
        "Собрано `scripts/costs.py` из сохранённых прогонов (`runs/`), без вызовов модели. Цена — по",
        f"`usage.cost` от провайдера (курс ≈ {_n(CAPS_PER_USD)} капсов за $) или по тарифам bothub × токены,",
        "где цены нет. В итерациях 1–4 телеметрии не было — «нет данных». Прогоны, которые не сохранили",
        "ответов (генерация сценариев, прерванные команды), сюда не попадают — они в журнале итераций.",
        "",
        "## Итого по итерациям",
        "",
        "| Итерация | Прогонов | Ответов | Вызовов LLM | Токенов вход / выход | Капсов, млн | Основа |",
        "|---|---|---|---|---|---|---|",
    ]
    for it in sorted(by_iter, key=int):
        rs = by_iter[it]
        rows = sum(r.rows for r in rs)
        known = [r for r in rs if r.basis != "нет данных"]
        basis = sorted({r.basis for r in rs})
        lines.append(
            f"| {it} | {len(rs)} | {rows} | {sum(r.calls for r in rs) or '—'} | "
            f"{_n(sum(r.prompt_tokens for r in rs))} / {_n(sum(r.completion_tokens for r in rs))} | "
            f"{_mln(sum(r.caps for r in known)) if known else '—'} | {', '.join(basis)} |"
        )
    lines += ["", "## По шагам", ""]
    for it in sorted(by_iter, key=int):
        lines += [
            f"### Итерация {it}",
            "",
            "| Прогон | Модель | Ответов | Капсов на ответ | Капсов, тыс. | Основа |",
            "|---|---|---|---|---|---|",
        ]
        for r in sorted(by_iter[it], key=lambda r: r.step):
            known = r.basis != "нет данных"
            per = _n(r.caps / r.rows) if known and r.rows else "—"
            total = _n(r.caps / 1e3) if known else "—"
            lines.append(f"| `{r.step}` | {r.model} | {r.rows} | {per} | {total} | {r.basis} |")
        lines.append("")
    lines += [
        "## Итерация 8: оценка и факт по шагам",
        "",
        "| Шаг | Оценка | Факт |",
        "|---|---|---|",
        *(f"| {step} | {plan} | {fact or '…'} |" for step, plan, fact in PLAN),
        "",
        "## Баланс bothub",
        "",
        "| Когда | Баланс |",
        "|---|---|",
        *(f"| {when} | {what} |" for when, what in BALANCE),
        "",
        "Расход по счёту больше суммы по прогонам: deepseek-v4-pro днём списывается по цене провайдера",
        "(≈ 2.4× тарифа), а генерация сценариев и прерванные запуски ответов не сохраняют.",
        "",
    ]
    return "\n".join(lines)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--runs", type=Path, default=Path("runs"))
    ap.add_argument("--out", type=Path)
    args = ap.parse_args()
    text = report(collect(args.runs))
    if args.out:
        args.out.write_text(text, "utf-8")
    print(text)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
