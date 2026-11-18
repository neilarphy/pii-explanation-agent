"""Харнесс на фейковой модели: склейка по репликам, возврат ошибки модели, инструменты."""

import json

import pytest

from backend.agent.agent import explain_case
from backend.agent.cases import build_cases
from backend.agent.harness import MAX_RETRIES, Flags, Harness, check_value, search_session
from backend.agent.llm import Reply, ToolCall
from backend.schemas import Candidate, DetectionLog, Message, ModelConfig, Session, Verdict

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
        return Reply(text=text, prompt_tokens=100, completion_tokens=20)

    return chat, seen


def test_split_value_is_confirmed_and_not_downgraded():
    chat, seen = scripted(GOOD)
    answer, telemetry = Harness(chat).explain(case(), SESSION, LOG)
    assert answer.verdict == Verdict.MISSED_PII
    assert any("склейка по репликам 89164855707" in e for e in answer.evidence)
    assert "продиктованного по частям" in seen[0][-1]["content"]
    assert (telemetry.llm_calls, telemetry.retries, telemetry.prompt_tokens) == (1, 0, 100)


def test_pipeline_still_downgrades_the_fragment():
    """Конвейер итерации 4 видит только фрагмент «855707» — с ним и сравниваем харнесс."""
    answer = explain_case(case(), SESSION, LOG, lambda messages: json.dumps(GOOD))
    assert answer.verdict == Verdict.UNCERTAIN


def test_without_facts_harness_behaves_like_pipeline():
    chat, seen = scripted(GOOD)
    answer, _ = Harness(chat, Flags(facts=False)).explain(case(), SESSION, LOG)
    assert answer.verdict == Verdict.UNCERTAIN
    assert "проверки кодом" not in seen[0][-1]["content"]


def test_wrong_quote_goes_back_to_the_model():
    bad = {**GOOD, "citations": [{"turn": 2, "quote": "дальше 855 707"}]}
    chat, seen = scripted(bad, GOOD)
    answer, telemetry = Harness(chat).explain(case(), SESSION, LOG)
    assert (telemetry.llm_calls, telemetry.retries) == (2, 1)
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
