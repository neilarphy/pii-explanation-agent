"""Харнесс агента на LangGraph: код собирает факты и проверяет ответ, модель судит по смыслу.

    gather (код) → decide (LLM) → verify (код) → finish (код)
                      ▲              │
                      └── ошибка ────┘  (не больше MAX_RETRIES раз)

- gather — промпт и «досье»: проверки кодом, целое значение вокруг фрагмента, склейка по
  репликам. Всё, что можно посчитать всегда, считает код: модели не нужно догадываться
  вызвать инструмент (маленькие модели делают это плохо или не умеют вовсе).
- decide — вызов модели; с флагом tools модель может сама поискать по диалогу и проверить
  значение, не больше MAX_TOOL_STEPS шагов.
- verify — ответ разбирается и проверяется: JSON, допустимые значения, дословность цитат, а с
  фактами — и противоречие им (код нашёл значение целиком, а ответ none; название типа без цифр,
  а ответ pii). Найденная ошибка возвращается модели конкретным текстом, и она отвечает заново.
- finish — AgentAnswer: вердикт по категории, офсеты цитат, понижение пропуска, который
  код не подтвердил (с учётом целого значения и склейки — в отличие от конвейера).

Флаги включают части харнесса по отдельности — так видно вклад каждой (абляция). Понижение
пропуска опирается на факты кода, поэтому без facts его нет: голая модель — без кода вообще.
"""

import json
import time
from dataclasses import dataclass, replace
from typing import TypedDict

from langgraph.graph import END, START, StateGraph
from pydantic import ValidationError

from backend.agent.agent import fail_open, failure_reason, finalize, parse_json
from backend.agent.cases import Case
from backend.agent.checks import format_check, identify, names_a_type, resolve_citations
from backend.agent.llm import Chat, Reply
from backend.agent.prompt import build_messages, enclosing_value, joined_value
from backend.schemas import (
    AgentAnswer,
    Category,
    DetectionLog,
    Recommendation,
    Session,
    Verdict,
)

MAX_RETRIES = 2
MAX_TOOL_STEPS = 3

VERDICTS = {v.value for v in Verdict} | {"no_issue"}
CATEGORIES = {c.value for c in Category}


@dataclass(frozen=True)
class Flags:
    facts: bool = True  # досье от кода в промпте
    verify: bool = True  # проверка ответа и возврат модели с ошибкой
    tools: bool = False  # инструменты search_session и check_value


DEFAULT = Flags()

# Лестница режимов для сравнения моделей: каждая ступень добавляет одну часть харнесса.
MODES = {
    "bare": Flags(facts=False, verify=False),  # голая модель: без фактов, проверки и понижения
    "facts": Flags(verify=False),  # + факты кода и понижение неподтверждённого пропуска
    "harness": DEFAULT,  # + проверка ответа с возвратом ошибки модели (итерация 5)
    "tools": Flags(tools=True),  # + инструменты (только для моделей, которые их умеют)
}


@dataclass(frozen=True)
class Telemetry:
    llm_calls: int = 0
    tool_calls: int = 0
    retries: int = 0
    prompt_tokens: int = 0
    completion_tokens: int = 0
    cost: float = 0.0  # USD по данным провайдера
    seconds: float = 0.0

    def add(self, reply: Reply) -> "Telemetry":
        return replace(
            self,
            llm_calls=self.llm_calls + 1,
            prompt_tokens=self.prompt_tokens + reply.prompt_tokens,
            completion_tokens=self.completion_tokens + reply.completion_tokens,
            cost=self.cost + reply.cost,
        )


class State(TypedDict, total=False):
    case: Case
    session: Session
    log: DetectionLog
    messages: list[dict]
    raw: str
    data: dict | None
    problems: list[str]
    confirmed: str | None
    answer: AgentAnswer | None
    telemetry: Telemetry


TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "search_session",
            "description": "Найти реплики диалога, где встречается текст или цифры (без учёта регистра).",
            "parameters": {
                "type": "object",
                "properties": {"query": {"type": "string"}},
                "required": ["query"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "check_value",
            "description": "Проверить значение кодом: контрольные суммы и форматы (карта, ИНН, "
            "СНИЛС, ОГРН, паспорт, телефон, email, адрес, токен).",
            "parameters": {
                "type": "object",
                "properties": {"text": {"type": "string"}},
                "required": ["text"],
            },
        },
    },
]
TOOLS_HINT = (
    "\n\nИнструменты: search_session — найти текст или цифры в диалоге; check_value — проверить "
    "значение кодом (например, склеенное из нескольких реплик). Вызывай, только если фактов выше "
    "не хватает."
)


def search_session(session: Session, query: str) -> str:
    q = query.strip().casefold()
    digits = "".join(ch for ch in q if ch.isdigit())
    found = [
        f"[{m.turn}] {m.text}"
        for m in session.messages
        if q
        and (q in m.text.casefold() or (len(digits) >= 3 and digits in m.text.replace(" ", "")))
    ]
    return "\n".join(found[:5]) or "не найдено"


def check_value(text: str) -> str:
    found = [c.description for c in identify(text)]
    if found:
        return "; ".join(found)
    checks = {t: format_check(t, text) for t in ("PHONE", "EMAIL", "PAYMENT_CARD")}
    passed = [t for t, ok in checks.items() if ok]
    return "проверки не пройдены" + (f"; формат {', '.join(passed)} подходит" if passed else "")


def code_confirmation(case: Case, session: Session) -> str | None:
    """Что код подтвердил о значении: сам фрагмент, целое значение вокруг или склейка."""
    if found := identify(case.text):
        return found[0].description
    if whole := enclosing_value(case, session):
        return f"целое значение «{whole[0]}»: {whole[1].description}"
    if joined := joined_value(case, session):
        return f"склейка по репликам {joined.value}: {joined.check.description}"
    return None


def problems_in(raw: str, session: Session) -> tuple[dict | None, list[str]]:
    """Ответ модели → данные и список конкретных ошибок (пустой — ответ годится)."""
    try:
        data = parse_json(raw)
    except ValueError as exc:
        return None, [
            f"ответ не разобран как JSON ({exc}). Пришли один JSON-объект без текста вокруг"
        ]
    if not isinstance(data, dict):
        return None, ["ответ должен быть JSON-объектом"]
    found = []
    if data.get("category") not in CATEGORIES:
        found.append(f"category «{data.get('category')}» — допустимо: pii, sensitive, none")
    if data.get("verdict") not in VERDICTS:
        found.append(f"verdict «{data.get('verdict')}» — допустимо: {', '.join(sorted(VERDICTS))}")
    citations = data.get("citations") or []
    _, dropped = resolve_citations(citations, session)
    if dropped:
        found.append(
            "цитаты не дословные или номер реплики неверный: "
            + "; ".join(dropped)
            + ". Копируй фрагмент реплики символ в символ"
        )
    elif not citations:
        found.append("нет цитат: первая цитата — сама находка с номером реплики")
    if data.get("recommendation") is not None:
        try:
            Recommendation.model_validate(data["recommendation"])
        except ValidationError as exc:
            fields = ", ".join(".".join(map(str, e["loc"])) or "объект" for e in exc.errors())
            found.append(
                f"recommendation не по схеме ({fields}): нужны kind, target, change, rationale "
                "или null"
            )
    return data, found


def contradictions(data: dict, case: Case, confirmed: str | None) -> list[str]:
    """Ответ против фактов кода. Оба правила проверены на эталонах (итерация 6).

    - Код нашёл значение целиком (находка кода, часть целого, склейка), а модель говорит «none»:
      на трёх наборах такие находки — ПДн в 137 случаях из 137.
    - Фрагмент числового типа без цифр («номер телефона», «карту»), а модель говорит «pii»:
      на тех же наборах это ни разу не ПДн (290 из 290); на redmadrobot — 631 из 632, исключение
      «СНИЛС» похоже на ошибку разметки.
    """
    category = data.get("category")
    found = []
    whole = case.kind == "scan" or (confirmed is not None and confirmed != _fragment(case))
    if category == "none" and whole:
        found.append(
            f"код нашёл значение целиком ({confirmed or 'находка кода'}), а category none. Часть "
            "телефона, email или адреса человека — тоже pii. Пересмотри; если оставляешь none, "
            "объясни в explanation, почему это не данные человека"
        )
    if category in ("pii", "sensitive") and names_a_type(case.type, case.text):
        found.append(
            f"во фрагменте «{case.text}» нет ни цифр, ни серии — это название типа данных, а не "
            "само значение, маскировать во фрагменте нечего. Пересмотри; если оставляешь "
            f"{category}, объясни в explanation, где здесь значение"
        )
    return found


def build_graph(chat: Chat, flags: Flags = DEFAULT):
    def gather(state: State) -> State:
        case, session = state["case"], state["session"]
        messages = build_messages(case, session, state["log"], facts=flags.facts)
        if flags.tools:
            messages[-1] = {**messages[-1], "content": messages[-1]["content"] + TOOLS_HINT}
        return {
            "messages": messages,
            "confirmed": code_confirmation(case, session) if flags.facts else None,
            "telemetry": Telemetry(),
        }

    def decide(state: State) -> State:
        messages, telemetry = list(state["messages"]), state["telemetry"]
        tools = TOOLS if flags.tools else None
        for _ in range(MAX_TOOL_STEPS + 1):
            reply = chat(messages, tools)
            telemetry = telemetry.add(reply)
            if not reply.tool_calls:
                break
            messages.append(_assistant_with_calls(reply))
            for call in reply.tool_calls:
                messages.append(
                    {"role": "tool", "tool_call_id": call.id, "content": _run_tool(call, state)}
                )
                telemetry = replace(telemetry, tool_calls=telemetry.tool_calls + 1)
            if len([m for m in messages if m["role"] == "tool"]) >= MAX_TOOL_STEPS:
                tools = None  # шаги кончились — дальше только ответ
        messages.append({"role": "assistant", "content": reply.text})
        return {"messages": messages, "raw": reply.text, "telemetry": telemetry}

    def verify(state: State) -> State:
        data, found = problems_in(state["raw"], state["session"])
        if not flags.verify:
            return {"data": data, "problems": []}
        if flags.facts and data is not None and state["telemetry"].retries == 0:
            # противоречие фактам — один раз: настаивает — решение за моделью
            found += contradictions(data, state["case"], state["confirmed"])
        if found:
            feedback = (
                "Ответ не прошёл проверку кодом:\n- "
                + "\n- ".join(found)
                + "\nИсправь и пришли JSON целиком."
            )
            return {
                "data": data,
                "problems": found,
                "messages": [*state["messages"], {"role": "user", "content": feedback}],
            }
        return {"data": data, "problems": []}

    def route(state: State) -> str:
        telemetry = state["telemetry"]
        if state["problems"] and telemetry.retries < MAX_RETRIES:
            return "retry"
        return "finish"

    def retry(state: State) -> State:
        telemetry = state["telemetry"]
        return {"telemetry": replace(telemetry, retries=telemetry.retries + 1)}

    def finish(state: State) -> State:
        data = state.get("data")
        if data is None:
            raise ValueError(f"В ответе LLM нет разбираемого JSON: {state['raw'][:200]!r}")
        if flags.verify and any(p.startswith("recommendation") for p in state["problems"]):
            data = {**data, "recommendation": None}  # повторы кончились — ответ важнее рекомендации
        answer = finalize(
            state["case"], state["session"], data, state["confirmed"], guard=flags.facts
        )
        confirmed = state["confirmed"]
        if answer and confirmed and confirmed != _fragment(state["case"]):
            answer = answer.model_copy(
                update={"evidence": [*answer.evidence, f"код подтвердил: {confirmed}"]}
            )
        return {"answer": answer}

    graph = StateGraph(State)
    graph.add_node("gather", gather)
    graph.add_node("decide", decide)
    graph.add_node("verify", verify)
    graph.add_node("retry", retry)
    graph.add_node("finish", finish)
    graph.add_edge(START, "gather")
    graph.add_edge("gather", "decide")
    graph.add_edge("decide", "verify")
    graph.add_conditional_edges("verify", route, {"retry": "retry", "finish": "finish"})
    graph.add_edge("retry", "decide")
    graph.add_edge("finish", END)
    return graph.compile()


def _fragment(case: Case) -> str | None:
    found = identify(case.text)
    return found[0].description if found else None


def _assistant_with_calls(reply: Reply) -> dict:
    return {
        "role": "assistant",
        "content": reply.text or None,
        "tool_calls": [
            {"id": c.id, "type": "function", "function": {"name": c.name, "arguments": c.arguments}}
            for c in reply.tool_calls
        ],
    }


def _run_tool(call, state: State) -> str:
    try:
        args = json.loads(call.arguments)
    except json.JSONDecodeError:
        return "аргументы не разобраны как JSON"
    if call.name == "search_session":
        return search_session(state["session"], str(args.get("query", "")))
    if call.name == "check_value":
        return check_value(str(args.get("text", "")))
    return f"нет инструмента {call.name}"


class Harness:
    """Агент на графе: explain() по одной находке, mermaid() — схема для документации."""

    def __init__(self, chat: Chat, flags: Flags = DEFAULT):
        self.flags = flags
        self.graph = build_graph(chat, flags)

    def explain(
        self, case: Case, session: Session, log: DetectionLog
    ) -> tuple[AgentAnswer | None, Telemetry]:
        started = time.perf_counter()
        state = self.graph.invoke({"case": case, "session": session, "log": log})
        telemetry = replace(state["telemetry"], seconds=round(time.perf_counter() - started, 2))
        return state["answer"], telemetry

    def explain_or_fail_open(
        self, case: Case, session: Session, log: DetectionLog
    ) -> tuple[AgentAnswer | None, Telemetry]:
        """Для сервиса: при сбое LLM — fail-open вместо исключения (оценка зовёт explain)."""
        try:
            return self.explain(case, session, log)
        except Exception as exc:
            return fail_open(case, failure_reason(exc)), Telemetry()

    def mermaid(self) -> str:
        return self.graph.get_graph().draw_mermaid()
