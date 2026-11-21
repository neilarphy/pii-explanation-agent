"""Standard-library validation of structured identifiers; no network or model calls."""

import ipaddress
import re
from datetime import date

H = r"[^\S\r\n]"
SEP = rf"{H}*(?:[-–—]{H}*)?"
DOC_NUMBER = rf"{H}*(?:(?:,{H}*)?(?:(?i:номер)|№){H}*(?::{H}*)?|[-–—]{H}*)?"
DOC10 = re.compile(rf"[0-9]{{2}}{H}*[0-9]{{2}}{DOC_NUMBER}[0-9]{{6}}", re.I)
BIRTH = re.compile(rf"([IVXLCDM]{{1,12}}){SEP}([А-ЯЁ]{{2}}){DOC_NUMBER}([0-9]{{6}})")
MILITARY = re.compile(rf"[А-ЯЁ]{{2}}{DOC_NUMBER}[0-9]{{7}}", re.I)
ROMAN = re.compile(r"M{0,3}(?:CM|CD|D?C{0,3})(?:XC|XL|L?X{0,3})(?:IX|IV|V?I{0,3})")
NON_PAYMENT = re.compile(
    rf"(?i)(?<!\w)(?:номер{H}+(?:заказа|договора|накладной|сч[её]та)|заказ(?:а)?|"
    rf"договор(?:а)?|инн|кпп|огрн|бик|(?:расч[её]тный|корреспондентский){H}+сч[её]т)"
    rf"(?!\w){H}*(?:[:№=–—-]{H}*)?$"
)
PASSPORT_CONTEXT = re.compile(r"(?i)(?<!\w)паспорт(?:а|у|е|ом|ы|ов|ами|ные|ных|ным|ными)?(?!\w)")
OTHER_DOCUMENT_FIELD = re.compile(
    rf"(?i)(?<!\w)(?:инн|снилс|огрн|кпп|телефон(?:а)?|тел\.?|мобильный|"
    rf"номер{H}+(?:заказа|договора|телефона|сч[её]та|заявки|накладной)|"
    rf"заказ(?:а)?|договор(?:а)?|сч[её]т(?:а)?|карт[аы]|полис(?:а)?|омс|"
    rf"водительск(?:ое|ого){H}+удостоверени(?:е|я)|права|"
    rf"(?:расч[её]тный|корреспондентский){H}+сч[её]т)"
    rf"(?!\w){H}*(?:[:№=–—-]{H}*)?(?:номер{H}*[:№]?{H}*)?$"
)
CONTEXT_BREAK = re.compile(r'[\r\n";]')
PAYMENT_CONTEXT = re.compile(
    rf"(?i)(?<!\w)(?:карта|номер{H}+(?:банковской{H}+)?карты|card(?:{H}+number)?)(?!\w){H}*(?:[:№=–—-]{H}*)?$"
)


def payment_context(text: str, start: int) -> bool:
    return PAYMENT_CONTEXT.search(text, max(0, start - 100), start) is not None


def ascii_digits(value: str) -> str:
    return "".join(char for char in value if "0" <= char <= "9")


def non_payment_context(text: str, start: int) -> bool:
    return NON_PAYMENT.search(text, max(0, start - 100), start) is not None


def passport_context(text: str, start: int, end: int) -> bool:
    """Require nearby passport wording without crossing fields or dialog strings."""
    before = CONTEXT_BREAK.split(text[max(0, start - 140) : start])[-1]
    after = CONTEXT_BREAK.split(text[end : end + 100])[0]
    if (start and text[start - 1] in "+@") or (end < len(text) and text[end] == "@"):
        return False
    if OTHER_DOCUMENT_FIELD.search(before):
        return False
    return PASSPORT_CONTEXT.search(before) is not None or PASSPORT_CONTEXT.search(after) is not None


def inn_checksum(digits: str) -> bool:
    if len(digits) not in (10, 12) or not digits.isascii() or not digits.isdigit():
        return False

    def check(weights):
        return (
            sum(
                int(digit) * weight
                for digit, weight in zip(digits[: len(weights)], weights, strict=True)
            )
            % 11
            % 10
        )

    if len(digits) == 10:
        return check((2, 4, 10, 3, 5, 9, 4, 6, 8)) == int(digits[9])
    return check((7, 2, 4, 10, 3, 5, 9, 4, 6, 8)) == int(digits[10]) and check(
        (3, 7, 2, 4, 10, 3, 5, 9, 4, 6, 8)
    ) == int(digits[11])


def snils_checksum(digits: str) -> bool:
    if len(digits) != 11 or not digits.isascii() or not digits.isdigit():
        return False
    total = sum(
        int(digit) * weight for digit, weight in zip(digits[:9], range(9, 0, -1), strict=True)
    )
    return total % 101 % 100 == int(digits[9:])


def valid_phone_ru(value: str) -> bool:
    return (
        re.fullmatch(
            rf"(?:\+7|8){SEP}(?:\([0-9]{{3}}\)|[0-9]{{3}}){SEP}"
            rf"[0-9]{{3}}{SEP}[0-9]{{2}}{SEP}[0-9]{{2}}",
            value,
        )
        is not None
    )


def structured_reason(value: str, validator: str) -> str:
    digits = ascii_digits(value)
    if validator in {"inn", "snils", "oms_policy", "phone_context"}:
        pattern = (
            rf"[0-9]{{4}}(?:{H}*(?:[-.–—]{H}*)?[0-9]{{4}}){{3}}"
            if validator == "oms_policy"
            else r"\+?[0-9](?:[^\S\r\n]|[-–—]|[0-9])*"
        )
        if re.fullmatch(pattern, value) is None:
            return "invalid_format"
        if validator == "phone_context":
            good = digits[0] != "0" and (
                (value.startswith("+") and 8 <= len(digits) <= 15)
                or (
                    not value.startswith("+")
                    and (len(digits) in (7, 10) or (len(digits) == 11 and digits[0] in "78"))
                )
            )
            return "rule_passed" if good else "invalid_format"
        lengths = {"inn": (10, 12), "snils": (11,), "oms_policy": (16,)}[validator]
        if value.startswith("+") or len(digits) not in lengths or len(set(digits)) == 1:
            return "invalid_format"
        if validator == "inn" and not inn_checksum(digits):
            return "checksum_failed"
        if validator == "snils" and not snils_checksum(digits):
            return "checksum_failed"
        return "rule_passed"
    if validator == "passport_context_ru":
        # Keep source spelling, including a paired series/number designation.
        value = re.sub(rf"(?i)^сер(?:ия|ии|\.){H}*[:№]?{H}*", "", value)
        good = (
            DOC10.fullmatch(value) is not None
            or re.fullmatch(rf"(?i)(?:номер|№){H}*:?{H}*[0-9]{{6}}", value) is not None
        ) and len(set(digits)) > 1
    elif validator in {"passport_ru", "driver_license_ru"}:
        good = DOC10.fullmatch(value) is not None and len(set(digits)) > 1
    elif validator == "birth_certificate":
        match = BIRTH.fullmatch(value)
        good = match is not None and ROMAN.fullmatch(match[1]) is not None
    elif validator == "military_id":
        good = MILITARY.fullmatch(value) is not None
    elif validator == "date_of_birth":
        try:
            if re.fullmatch(r"[0-9]{4}-[0-9]{2}-[0-9]{2}", value):
                date.fromisoformat(value)
            else:
                match = re.fullmatch(r"([0-9]{1,2})([./-])([0-9]{1,2})\2([0-9]{4})", value)
                if match is None:
                    return "invalid_format"
                date(int(match[4]), int(match[3]), int(match[1]))
            good = True
        except ValueError:
            good = False
    elif validator == "ip_address":
        try:
            ipaddress.ip_address(value)
            good = True
        except ValueError:
            good = False
    else:
        # The address rule itself requires a street marker, name and house number.
        good = validator == "address_ru" and len(value) <= 300
    return "rule_passed" if good else "invalid_format"


STRUCTURED_VALIDATORS = frozenset(
    {
        "inn",
        "snils",
        "oms_policy",
        "passport_ru",
        "passport_context_ru",
        "driver_license_ru",
        "birth_certificate",
        "military_id",
        "date_of_birth",
        "address_ru",
        "phone_context",
        "ip_address",
    }
)
