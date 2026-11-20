"""Харнесс на фейковой модели: склейка по репликам, возврат ошибки модели, инструменты."""

import json

import pytest

from backend.agent.agent import explain_case
from backend.agent.cases import build_cases
from backend.agent.harness import (
    MAX_RETRIES,
    MODES,
    Cascade,
    Flags,
    Harness,
    check_value,
    search_session,
)
from backend.agent.llm import Reply, ToolCall
from backend.schemas import (
    Candidate,
    DetectionLog,
    Entity,
    Message,
    ModelConfig,
    Session,
    Verdict,
)

SESSION = Session(
    id="split",
    messages=[
        Message(turn=0, direction="in", text="Номер начинается с 89164."),
        Message(turn=1, direction="out", text="Подскажите оставшиеся цифры."),
        Message(turn=2, direction="in", text="Да, дальше 855707."),
    ],
)
LOG = DetectionLog(
    session_id="split",
    profile="test",
    policy_version="1",
    model=ModelConfig(id="ner", emit_threshold=0.3, accept_threshold=0.6, labels=["PHONE"]),
    rules=[],
    entities=[],
    candidates=[
        Candidate(
            id="c1",
            turn=2,
            start=11,
            end=17,
            text="855707",
            type="PHONE",
            source="model",
            decision="rejected",
            reason="below_model_threshold",
            entity_id=None,
            score=0.4,
        )
    ],
)
GOOD = {
    "category": "pii",
    "verdict": "missed_pii",
    "confidence": 0.8,
    "explanation": "Вторая часть телефона, начало в реплике 0.",
    "citations": [{"turn": 2, "quote": "855707"}, {"turn": 0, "quote": "89164"}],
    "evidence": [],
    "note": "Маскировать обе части.",
    "recommendation": None,
}


def case():
    (only,) = build_cases(LOG, SESSION)
    return only


def scripted(*replies):
    """Фейковая модель: отдаёт ответы по очереди и запоминает, что ей прислали."""
    seen: list[list[dict]] = []
    queue = list(replies)

    def chat(messages, tools=None):
        seen.append(list(messages))
        reply = queue.pop(0)
        if isinstance(reply, Reply):
            return reply
        text = reply if isinstance(reply, str) else json.dumps(reply, ensure_ascii=False)
        return Reply(text=text, prompt_tokens=100, completion_tokens=20, cost=0.001)

    return chat, seen


def test_split_value_is_confirmed_and_not_downgraded():
    chat, seen = scripted(GOOD)
    answer, telemetry = Harness(chat).explain(case(), SESSION, LOG)
    assert answer.verdict == Verdict.MISSED_PII
    assert any("склейка по репликам 89164855707" in e for e in answer.evidence)
    assert "продиктованного по частям" in seen[0][-1]["content"]
    assert (telemetry.llm_calls, telemetry.retries, telemetry.prompt_tokens) == (1, 0, 100)
    assert telemetry.cost == pytest.approx(0.001)


def test_pipeline_still_downgrades_the_fragment():
    """Конвейер итерации 4 видит только фрагмент «855707» — с ним и сравниваем харнесс."""
    answer = explain_case(case(), SESSION, LOG, lambda messages: json.dumps(GOOD))
    assert answer.verdict == Verdict.UNCERTAIN


def test_bare_model_gets_no_code_and_keeps_its_verdict():
    """Голая модель: ни фактов в промпте, ни понижения — вердикт модели как есть."""
    chat, seen = scripted(GOOD)
    answer, _ = Harness(chat, MODES["bare"]).explain(case(), SESSION, LOG)
    assert answer.verdict == Verdict.MISSED_PII
    assert "проверки кодом" not in seen[0][-1]["content"]
    assert not any("код подтвердил" in e for e in answer.evidence)


def test_harness_does_not_downgrade_unconfirmed_miss():
    """Склейки нет, формат фрагмента не сходится — харнесс всё равно не понижает (итерация 7):
    понижение кодом испортило 16 верных ответов и исправило 3. Конвейер итерации 4 — понижает."""
    lone = SESSION.model_copy(update={"messages": [m for m in SESSION.messages if m.turn != 0]})
    chat, _ = scripted({**GOOD, "citations": [{"turn": 2, "quote": "855707"}]})
    answer, _ = Harness(chat, MODES["facts"]).explain(case(), lone, LOG)
    assert answer.verdict == Verdict.MISSED_PII


def test_wrong_quote_goes_back_to_the_model():
    bad = {**GOOD, "citations": [{"turn": 2, "quote": "дальше 855 707"}]}
    chat, seen = scripted(bad, GOOD)
    answer, telemetry = Harness(chat).explain(case(), SESSION, LOG)
    assert (telemetry.llm_calls, telemetry.retries) == (2, 1)
    assert telemetry.cost == pytest.approx(0.002)  # стоимость повтора тоже учтена
    feedback = seen[1][-1]["content"]
    assert "не дословные" in feedback and "дальше 855 707" in feedback
    assert [c.quote for c in answer.citations] == ["855707", "89164"]


def test_broken_json_is_retried_then_fails():
    chat, _ = scripted(*["не json"] * (MAX_RETRIES + 1))
    with pytest.raises(ValueError, match="JSON"):
        Harness(chat).explain(case(), SESSION, LOG)


def test_without_verify_bad_answer_is_used_as_is():
    bad = {**GOOD, "citations": [{"turn": 2, "quote": "дальше 855 707"}]}
    chat, _ = scripted(bad)
    answer, telemetry = Harness(chat, Flags(verify=False)).explain(case(), SESSION, LOG)
    assert telemetry.llm_calls == 1
    assert answer.citations == []


def test_model_can_call_tools():
    call = ToolCall("t1", "check_value", json.dumps({"text": "89164855707"}))
    chat, seen = scripted(Reply(text="", tool_calls=(call,)), GOOD)
    answer, telemetry = Harness(chat, Flags(tools=True)).explain(case(), SESSION, LOG)
    assert (telemetry.llm_calls, telemetry.tool_calls) == (2, 1)
    tool_message = seen[1][-1]
    assert tool_message["role"] == "tool" and "телефон" in tool_message["content"]
    assert answer.verdict == Verdict.MISSED_PII


def test_tools():
    assert "телефон" in check_value("89164855707")
    assert check_value("855707").startswith("проверки не пройдены")
    assert search_session(SESSION, "855707") == "[2] Да, дальше 855707."
    assert search_session(SESSION, "паспорт") == "не найдено"


def test_graph_is_drawable():
    chart = Harness(scripted()[0]).mermaid()
    assert all(node in chart for node in ("gather", "decide", "verify", "finish"))


NONE = {**GOOD, "category": "none", "verdict": "no_issue"}


def test_contradiction_with_code_goes_back_once():
    """Код склеил телефон, а модель сказала none — ей возвращают противоречие."""
    chat, seen = scripted(NONE, GOOD)
    answer, telemetry = Harness(chat).explain(case(), SESSION, LOG)
    assert "код нашёл значение целиком" in seen[1][-1]["content"]
    assert (telemetry.llm_calls, telemetry.retries, answer.verdict) == (2, 1, Verdict.MISSED_PII)


def test_model_may_insist_after_contradiction():
    """Противоречие возвращается один раз: настаивает — решение за моделью, повторы не тратятся."""
    chat, _ = scripted(NONE, NONE)
    answer, telemetry = Harness(chat).explain(case(), SESSION, LOG)
    assert answer is None and telemetry.llm_calls == 2


def test_type_name_without_digits_is_questioned():
    word = LOG.candidates[0].model_copy(update={"turn": 0, "start": 0, "end": 5, "text": "Номер"})
    log = LOG.model_copy(update={"candidates": [word]})
    (only,) = build_cases(log, SESSION)
    said_pii = {**GOOD, "citations": [{"turn": 0, "quote": "Номер"}]}
    chat, seen = scripted(said_pii, {**said_pii, "category": "none", "verdict": "no_issue"})
    answer, _ = Harness(chat).explain(only, SESSION, log)
    assert "название типа данных" in seen[1][-1]["content"]
    assert answer is None


def test_bare_mode_does_not_check_contradictions():
    chat, _ = scripted(NONE)
    answer, telemetry = Harness(chat, MODES["bare"]).explain(case(), SESSION, LOG)
    assert answer is None and telemetry.llm_calls == 1


def test_broken_recommendation_goes_back_then_is_dropped():
    broken = {**GOOD, "recommendation": {"kind": "retrain", "target": "ner", "change": "дообучить"}}
    chat, seen = scripted(*[broken] * (MAX_RETRIES + 1))
    answer, telemetry = Harness(chat).explain(case(), SESSION, LOG)
    assert "recommendation не по схеме (rationale)" in seen[1][-1]["content"]
    assert telemetry.retries == MAX_RETRIES
    assert answer.verdict == Verdict.MISSED_PII and answer.recommendation is None


def test_llm_failure_fails_open_to_a_human():
    def broken(messages, tools=None):
        raise TimeoutError("bothub не ответил")

    answer, telemetry = Harness(broken).explain_or_fail_open(case(), SESSION, LOG)
    assert answer.verdict == Verdict.UNCERTAIN and answer.category.value == "pii"
    assert "TimeoutError" in answer.explanation and "проверьте вручную" in answer.explanation
    assert telemetry.llm_calls == 0


def test_session_survives_a_broken_answer():
    from backend.agent.agent import explain_session

    ((only, answer),) = explain_session(SESSION, LOG, lambda messages: "не json")
    assert only.id == case().id and answer.verdict == Verdict.UNCERTAIN
    assert "сбой агента, fail-open" in answer.evidence


def test_recommendation_must_address_the_one_who_erred():
    """Пропуск отклонила модель ner — threshold для правила не годится, ответ возвращается."""
    wrong = {
        **GOOD,
        "recommendation": {
            "kind": "threshold",
            "target": "phone_ru_v1",
            "change": "снизить порог",
            "rationale": "ниже порога",
        },
    }
    right = {**wrong, "recommendation": {**wrong["recommendation"], "target": "ner"}}
    chat, seen = scripted(wrong, right)
    answer, telemetry = Harness(chat).explain(case(), SESSION, LOG)
    assert "threshold бывает только у модели ner" in seen[1][-1]["content"]
    assert telemetry.retries == 1 and answer.recommendation.target == "ner"


def test_recommendation_is_dropped_when_detector_was_right():
    """Детектор принял фрагмент верно — рекомендации быть не должно, код её убирает без вызова."""
    accepted = LOG.candidates[0].model_copy(update={"decision": "accepted", "entity_id": "e1"})
    entity = Entity(
        id="e1", turn=2, start=11, end=17, text="855707", type="PHONE", candidate_ids=["c1"]
    )
    log = LOG.model_copy(update={"candidates": [accepted], "entities": [entity]})
    (only,) = build_cases(log, SESSION)
    rec = {
        "kind": "retrain",
        "target": "ner",
        "change": "дообучить",
        "rationale": "на всякий случай",
    }
    chat, _ = scripted({**GOOD, "verdict": "true_pii", "recommendation": rec})
    answer, telemetry = Harness(chat).explain(only, SESSION, log)
    assert answer.verdict == Verdict.TRUE_PII and answer.recommendation is None
    assert telemetry.llm_calls == 1


def test_cascade_takes_small_model_when_answers_agree():
    small, _ = scripted(GOOD, GOOD)
    big, big_seen = scripted()
    answer, telemetry = Cascade(Harness(small), Harness(big)).explain(case(), SESSION, LOG)
    assert answer.verdict == Verdict.MISSED_PII and big_seen == []
    assert (telemetry.llm_calls, telemetry.escalated) == (2, 0)


def test_cascade_escalates_disagreement_to_big_model():
    # второй ответ — none: харнесс возвращает противоречие, модель настаивает (2 вызова)
    small, _ = scripted(GOOD, NONE, NONE)
    big, _ = scripted(GOOD)
    answer, telemetry = Cascade(Harness(small), Harness(big)).explain(case(), SESSION, LOG)
    assert answer.verdict == Verdict.MISSED_PII and telemetry.escalated == 1
    assert telemetry.llm_calls == 4
    assert any(
        e.startswith("каскад: маленькая модель ответила missed_pii / no_issue")
        for e in answer.evidence
    )


def test_cascade_without_big_model_sends_disagreement_to_human():
    small, _ = scripted(GOOD, NONE, NONE)
    answer, telemetry = Cascade(Harness(small)).explain(case(), SESSION, LOG)
    assert answer.verdict == Verdict.UNCERTAIN and answer.category.value == "pii"
    assert "по-разному" in answer.explanation and telemetry.escalated == 1
