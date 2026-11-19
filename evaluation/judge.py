"""LLM-судья для ответов агента: проверяет то, что автоэталон не видит.

Автоэталон говорит только, верен ли вердикт. Судья смотрит на объяснение, цитаты и
рекомендацию. Судья — другая модель, чем агент (иначе модель оценивает сама себя), и
сам проверяется на ручной разметке: см. scripts/review.py.
"""

import json
from dataclasses import dataclass

from backend.agent.agent import CODE_EVIDENCE
from backend.agent.cases import Case
from backend.agent.prompt import build_messages
from backend.schemas import AgentAnswer, DetectionLog, Session

JUDGE_PROMPT = """\
Ты проверяешь работу агента, который разбирает срабатывания детектора персональных данных.
Тебе дают инструкцию, по которой работал агент, то, что он видел (диалог, находка, данные
детектора, проверки кодом), и его ответ. Правила из инструкции агента — тоже основание для
утверждений. Строки evidence, которые добавляет код (id кандидатов и т. п.), убраны — их
проверяет код. Оцени ответ строго по трём вопросам.

1. faithful — каждое утверждение в explanation подтверждается диалогом, данными детектора
   или проверками кодом. Ничего не выдумано: ни фактов о диалоге, ни score, ни правил.
2. citations_ok — цитаты относятся к делу: первая — сама находка, остальные — контекст,
   на который опирается вывод.
3. recommendation_ok — "n/a", если рекомендации нет. Иначе 1, если правка адресована
   существующему правилу или модели из профиля, исправляет именно эту ошибку и не ломает
   очевидно другие случаи; 0 — если нет.

Отдельно — verdict_ok: согласен ли ты с вердиктом и категорией агента (1/0).
Политика — та же, что в инструкции агента: маскировать ПДн по 152-ФЗ (pii) и секреты
(sensitive: CVC, коды, токены); реквизиты организаций, номера заказов, служебные контакты — none.

Ответ — один JSON-объект без текста вокруг:
{"faithful": 0|1, "citations_ok": 0|1, "recommendation_ok": 0|1|"n/a", "verdict_ok": 0|1,
 "unsupported_claims": ["..."], "comment": "одно-два предложения"}
"""


@dataclass(frozen=True)
class Verdicts:
    faithful: int
    citations_ok: int
    recommendation_ok: int | None  # None — рекомендации нет
    verdict_ok: int
    unsupported_claims: tuple[str, ...]
    comment: str


def judge_messages(
    case: Case, session: Session, log: DetectionLog, answer: AgentAnswer, facts: bool = True
) -> list[dict[str, str]]:
    # ровно то, что видел агент: голая модель (facts=False) фактов кода не видела
    instruction, seen = (m["content"] for m in build_messages(case, session, log, facts=facts))
    shown = answer.model_dump(mode="json", exclude={"entity_id"})
    shown["evidence"] = [e for e in shown["evidence"] if not e.startswith(CODE_EVIDENCE)]
    body = (
        f"Инструкция, по которой работал агент:\n{instruction}\n\n"
        f"Что видел агент:\n{seen}\n\n"
        f"Ответ агента:\n{json.dumps(shown, ensure_ascii=False, indent=1)}"
    )
    return [{"role": "system", "content": JUDGE_PROMPT}, {"role": "user", "content": body}]


def _bit(value) -> int:
    return 1 if value in (1, "1", True, "true") else 0


def parse_verdicts(data: dict) -> Verdicts:
    rec = data.get("recommendation_ok")
    return Verdicts(
        faithful=_bit(data.get("faithful")),
        citations_ok=_bit(data.get("citations_ok")),
        recommendation_ok=None if rec in (None, "n/a", "na") else _bit(rec),
        verdict_ok=_bit(data.get("verdict_ok")),
        unsupported_claims=tuple(data.get("unsupported_claims") or ()),
        comment=data.get("comment") or "",
    )


def cohen_kappa(pairs: list[tuple[int, int]]) -> float:
    """Согласие двух оценщиков (человек и судья) с поправкой на случайное совпадение."""
    n = len(pairs)
    if not n:
        return 0.0
    observed = sum(a == b for a, b in pairs) / n
    p_a, p_b = sum(a for a, _ in pairs) / n, sum(b for _, b in pairs) / n
    expected = p_a * p_b + (1 - p_a) * (1 - p_b)
    return 1.0 if expected == 1 else (observed - expected) / (1 - expected)
