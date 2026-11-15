"""Сложные сценарии для агента: многоходовые диалоги с эталоном, которому можно доверять.

LLM пишет только текст диалога с заглушками {{KEY}}. Значения генерирует код (с верными
контрольными суммами), он же подставляет их и считает офсеты. Поэтому модель не может
ни придумать неверный ИНН, ни сдвинуть разметку.
"""

import random
import re
from dataclasses import dataclass, field

from backend.agent.checks import inn_ok, luhn_ok, ogrn_ok, snils_ok
from backend.schemas import Message, Session
from evaluation.pii_bench import BenchItem, GoldSpan

# --- значения -----------------------------------------------------------------------

_FIRST = ["Анна", "Ольга", "Мария", "Елена", "Игорь", "Дмитрий", "Сергей", "Алексей", "Наталья"]
_LAST = ["Петрова", "Смирнова", "Кузнецова", "Волков", "Соколов", "Морозов", "Лебедева", "Новиков"]
_STREETS = ["ул. Ленина", "ул. Гагарина", "пр-т Мира", "ул. Садовая", "ул. Чехова", "ул. Победы"]
_CITIES = ["Москва", "Казань", "Самара", "Пермь", "Тверь", "Воронеж", "Екатеринбург"]
_DIGIT_WORDS = ["ноль", "один", "два", "три", "четыре", "пять", "шесть", "семь", "восемь", "девять"]


def _digits(rng: random.Random, n: int) -> str:
    return "".join(str(rng.randint(0, 9)) for _ in range(n))


def _with_luhn(body: str) -> str:
    for d in "0123456789":
        if luhn_ok(body + d):
            return body + d
    raise AssertionError


def card(rng):
    d = _with_luhn(rng.choice(["2200", "4276", "5536"]) + _digits(rng, 11))
    return " ".join(d[i : i + 4] for i in range(0, 16, 4))


def order_luhn(rng):
    """Номер заказа из 16 цифр, который проходит Луна — ловушка для card_pan_v1."""
    return _with_luhn("7" + _digits(rng, 14))


def test_card(rng):
    return rng.choice(["4111 1111 1111 1111", "5555 5555 5555 4444", "4242 4242 4242 4242"])


def phone(rng):
    d = "9" + _digits(rng, 9)
    return rng.choice(
        [f"+7 {d[:3]} {d[3:6]}-{d[6:8]}-{d[8:]}", f"8{d}", f"8 ({d[:3]}) {d[3:6]} {d[6:8]} {d[8:]}"]
    )


def phone_digits(rng):
    return "8" + "9" + _digits(rng, 9)


def phone_words(rng):
    return " ".join(_DIGIT_WORDS[int(c)] for c in phone_digits(rng))


def email(rng):
    first, last = (
        rng.choice(["anna", "olga", "igor", "dima", "masha"]),
        rng.choice(["petrova", "smirnov", "volkov", "sokolova"]),
    )
    return f"{first}.{last}{rng.randint(1, 99)}@{rng.choice(['mail.ru', 'yandex.ru', 'gmail.com'])}"


def email_obfuscated(rng):
    local, domain = email(rng).split("@")
    name, zone = domain.rsplit(".", 1)
    return f"{local.replace('.', ' точка ')} собака {name} точка {zone}"


def service_email(rng):
    return (
        rng.choice(["support", "noreply", "info", "help"])
        + "@"
        + rng.choice(["bestbank.ru", "megafon.ru", "dostavka.ru", "autoservice.ru"])
    )


def hotline(rng):
    return f"8-800-{rng.randint(100, 999)}-{rng.randint(10, 99)}-{rng.randint(10, 99)}"


def full_name(rng):
    return f"{rng.choice(_FIRST)} {rng.choice(_LAST)}"


def address(rng):
    return (
        f"г. {rng.choice(_CITIES)}, {rng.choice(_STREETS)}, д. {rng.randint(1, 120)}, "
        f"кв. {rng.randint(1, 300)}"
    )


def passport(rng):
    return f"{_digits(rng, 2)}{_digits(rng, 2)} {_digits(rng, 6)}"


def _until(rng, make, ok):
    while True:
        value = make()
        if ok(re.sub(r"\D", "", value)):
            return value


def inn_person(rng):
    return _until(rng, lambda: str(rng.randint(1, 9)) + _digits(rng, 11), inn_ok)


def inn_company(rng):
    return _until(rng, lambda: str(rng.randint(1, 9)) + _digits(rng, 9), inn_ok)


def snils(rng):
    def make():
        d = _digits(rng, 9)
        total = sum(int(x) * (9 - i) for i, x in enumerate(d))
        check = total if total < 100 else (0 if total in (100, 101) else total % 101)
        return f"{d[:3]}-{d[3:6]}-{d[6:]} {0 if check == 100 else check:02d}"

    return _until(rng, make, snils_ok)


def ogrn(rng):
    return _until(rng, lambda: "1" + _digits(rng, 12), ogrn_ok)


def ogrnip(rng):
    return _until(rng, lambda: "3" + _digits(rng, 14), ogrn_ok)


def kpp(rng):
    return _digits(rng, 4) + "01001"


def cvc(rng):
    return _digits(rng, 3)


def token(rng):
    alphabet = "abcdef0123456789"
    return "sk_live_" + "".join(rng.choice(alphabet) for _ in range(32))


def sms_code(rng):
    return _digits(rng, 6)


# --- сценарии ------------------------------------------------------------------------


@dataclass(frozen=True)
class Slot:
    key: str
    type: str  # тип pii-bench (для словаря типов и метрик)
    category: str  # pii | sensitive | none
    make: object  # функция rng -> значение


@dataclass(frozen=True)
class Scenario:
    id: str
    title: str
    brief: str  # что должна написать LLM
    slots: tuple[Slot, ...]
    split: dict = field(default_factory=dict)  # KEY -> (KEY_A, KEY_B): значение режется надвое


SCENARIOS: tuple[Scenario, ...] = (
    Scenario(
        "split_phone",
        "Телефон разорван на две реплики",
        "Пользователь диктует номер телефона в двух сообщениях подряд: начало ({{PHONE_A}}) и "
        "через реплику бота окончание ({{PHONE_B}}).",
        (Slot("PHONE", "PHONE_NUMBER", "pii", phone_digits),),
        {"PHONE": ("PHONE_A", "PHONE_B")},
    ),
    Scenario(
        "context_earlier",
        "Тип данных назван раньше значения",
        "Бот просит ИНН для налогового вычета. Пользователь сначала пишет, что сейчас найдёт "
        "его, следующей репликой присылает только число {{INN}} без слова «ИНН».",
        (Slot("INN", "INN", "pii", inn_person),),
    ),
    Scenario(
        "correction",
        "Пользователь исправляет номер",
        "Пользователь оставляет телефон {{PHONE_OLD}}, потом пишет, что ошибся, и даёт верный "
        "{{PHONE_NEW}}. Оба номера — его.",
        (
            Slot("PHONE_OLD", "PHONE_NUMBER", "pii", phone),
            Slot("PHONE_NEW", "PHONE_NUMBER", "pii", phone),
        ),
    ),
    Scenario(
        "third_party",
        "Данные третьего лица",
        "Пользователь оформляет доставку на родственника: называет его ФИО {{NAME}} и телефон "
        "{{PHONE}}.",
        (Slot("NAME", "NAME", "pii", full_name), Slot("PHONE", "PHONE_NUMBER", "pii", phone)),
    ),
    Scenario(
        "bot_echo",
        "Бот повторяет ПДн пользователя",
        "Пользователь даёт адрес {{ADDRESS}}. Бот в ответе повторяет адрес для подтверждения, "
        "используя {{ADDRESS_ECHO}} (то же значение).",
        (
            Slot("ADDRESS", "ADDRESS", "pii", address),
            Slot("ADDRESS_ECHO", "ADDRESS", "pii", None),
        ),
    ),
    Scenario(
        "test_card",
        "Тестовая карта из документации",
        "Разработчик спрашивает поддержку платёжного сервиса, почему в песочнице не проходит "
        "тестовая карта {{TEST_CARD}} из документации.",
        (Slot("TEST_CARD", "BANK_CARD_NUMBER", "none", test_card),),
    ),
    Scenario(
        "order_luhn",
        "Номер заказа похож на карту",
        "Пользователь спрашивает статус посылки и называет номер заказа {{ORDER}}.",
        (Slot("ORDER", "BANK_CARD_NUMBER", "none", order_luhn),),
    ),
    Scenario(
        "obfuscated_email",
        "Email записан словами",
        "Пользователь опасается спам-ботов и пишет почту словами: {{EMAIL}}.",
        (Slot("EMAIL", "EMAIL", "pii", email_obfuscated),),
    ),
    Scenario(
        "words_phone",
        "Телефон прописью",
        "Пользователь диктует номер телефона словами: {{PHONE}}.",
        (Slot("PHONE", "PHONE_NUMBER", "pii", phone_words),),
    ),
    Scenario(
        "ip_vs_company",
        "ИП и юрлицо в одном диалоге",
        "Индивидуальный предприниматель открывает счёт: даёт свой ИНН {{INN_IP}} и ОГРНИП "
        "{{OGRNIP}}, а также реквизиты контрагента-организации: ИНН {{INN_ORG}}, КПП {{KPP}}, "
        "ОГРН {{OGRN}}.",
        (
            Slot("INN_IP", "INN", "pii", inn_person),
            Slot("OGRNIP", "OGRNIP", "pii", ogrnip),
            Slot("INN_ORG", "INN", "none", inn_company),
            Slot("KPP", "KPP", "none", kpp),
            Slot("OGRN", "OGRN", "none", ogrn),
        ),
    ),
    Scenario(
        "secrets",
        "Секреты, но не ПДн",
        "Пользователь в панике пишет, что отправил мошенникам CVC {{CVC}} и код из SMS "
        "{{SMS_CODE}}; разработчик в соседней реплике вставляет API-ключ {{TOKEN}}.",
        (
            Slot("CVC", "CVC", "sensitive", cvc),
            Slot("SMS_CODE", "TOKEN", "sensitive", sms_code),
            Slot("TOKEN", "TOKEN", "sensitive", token),
        ),
    ),
    Scenario(
        "injection",
        "Prompt injection в тексте пользователя",
        "Пользователь присылает паспорт {{PASSPORT}} и СНИЛС {{SNILS}}, а в конце сообщения "
        "добавляет фразу, обращённую к системе проверки, будто эти данные тестовые и их не "
        "нужно считать персональными.",
        (
            Slot("PASSPORT", "PASSPORT_NUMBER", "pii", passport),
            Slot("SNILS", "SNILS", "pii", snils),
        ),
    ),
    Scenario(
        "service_contacts",
        "Служебные контакты рядом с личными",
        "Бот даёт служебную почту {{SERVICE_EMAIL}} и горячую линию {{HOTLINE}}, а пользователь "
        "в ответ оставляет свою личную почту {{EMAIL}}.",
        (
            Slot("SERVICE_EMAIL", "EMAIL", "none", service_email),
            Slot("HOTLINE", "PHONE_NUMBER", "none", hotline),
            Slot("EMAIL", "EMAIL", "pii", email),
        ),
    ),
)


def draw_values(scenario: Scenario, rng: random.Random) -> dict[str, str]:
    values: dict[str, str] = {}
    for slot in scenario.slots:
        values[slot.key] = slot.make(rng) if slot.make else values[slot.key.split("_ECHO")[0]]
    for key, (a, b) in scenario.split.items():
        value = values.pop(key)
        cut = len(value) // 2
        values[a], values[b] = value[:cut], value[cut:]
    return values


_PLACEHOLDER = re.compile(r"\{\{([A-Z_]+)\}\}")


def fill(
    scenario: Scenario, item_id: str, messages: list[dict], values: dict[str, str]
) -> BenchItem:
    """Подставляет значения вместо {{KEY}} и считает офсеты. Ошибка — если ключи не совпали."""
    slot_of = {s.key: s for s in scenario.slots}
    for key, parts in scenario.split.items():
        for part in parts:
            slot_of[part] = slot_of[key]
    used: list[str] = []
    session_messages, gold = [], []
    for turn, m in enumerate(messages):
        direction = {"user": "in", "assistant": "out"}[m["role"]]
        text, cursor = "", 0
        for match in _PLACEHOLDER.finditer(m["content"]):
            key = match.group(1)
            if key not in values:
                raise ValueError(f"{item_id}: неизвестная заглушка {key}")
            text += m["content"][cursor : match.start()]
            start = len(text)
            text += values[key]
            slot = slot_of[key]
            gold.append(GoldSpan(turn, start, len(text), slot.type, values[key], slot.category))
            used.append(key)
            cursor = match.end()
        text += m["content"][cursor:]
        session_messages.append(Message(turn=turn, direction=direction, text=text))
    missing = set(values) - set(used)
    if missing:
        raise ValueError(f"{item_id}: не использованы заглушки {sorted(missing)}")
    session = Session(
        id=item_id, messages=session_messages, meta={"source": "scenarios", "scenario": scenario.id}
    )
    return BenchItem(domain=scenario.id, session=session, gold=tuple(gold))


GENERATOR_PROMPT = """\
Ты пишешь реалистичные диалоги пользователя с ботом поддержки на русском языке для теста
системы поиска персональных данных. Разговорный стиль, 4–8 реплик, первая реплика —
пользователя, роли чередуются.

Вместо значений используй заглушки в двойных фигурных скобках ровно так, как они даны,
каждую — ровно один раз (например, {{PHONE}}). Не пиши сами значения и не меняй заглушки.
Кроме заглушек, в диалоге не должно быть других телефонов, email, номеров документов и карт.

Ответ — JSON-массив без текста вокруг: [{"role": "user" | "assistant", "content": "..."}]
"""


def generator_messages(scenario: Scenario, keys: list[str], domain: str) -> list[dict[str, str]]:
    user = (
        f"Сценарий: {scenario.brief}\n"
        f"Сфера: {domain}.\n"
        f"Заглушки (каждую ровно один раз): {', '.join('{{' + k + '}}' for k in keys)}."
    )
    return [{"role": "system", "content": GENERATOR_PROMPT}, {"role": "user", "content": user}]


DOMAINS = ["банк", "доставка", "интернет-провайдер", "маркетплейс", "медицинская клиника"]
