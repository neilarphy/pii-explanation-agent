"""Explain narrow conflicts between identifiers while keeping unrelated overlaps."""

from .structured import ascii_digits, non_payment_context, payment_context, structured_reason

DOCUMENT_MODELS = {
    "DRIVER_LICENSE": "driver_license_ru",
    "BIRTH_CERTIFICATE": "birth_certificate",
    "MILITARY_ID": "military_id",
    "OMS": "oms_policy",
}

SPECIFIC_TYPES = frozenset(
    {
        "INN",
        "SNILS",
        "OMS",
        "OMS_POLICY",
        "PASSPORT",
        "DRIVER_LICENSE",
        "BIRTH_CERTIFICATE",
        "MILITARY_ID",
        "DATE_OF_BIRTH",
    }
)
COMPETING_TYPES = SPECIFIC_TYPES | {"NATIONAL_ID", "BANK_ACCOUNT", "PAYMENT_CARD", "PHONE"}


def resolve_identifier_conflicts(text, candidates, contextual_ids):
    """Use accepted regex evidence only; equal-strength disagreements remain visible."""
    supports = {}
    priorities = {}

    def value_key(candidate):
        value = text[candidate.start : candidate.end]
        digits = ascii_digits(value)
        return ("digits", digits) if digits else ("text", value.casefold())

    for candidate in candidates:
        priority = 0
        supporting = False
        if candidate.decision == "accepted" and candidate.source == "regex":
            if candidate.type in SPECIFIC_TYPES:
                priority = 2 if candidate.id in contextual_ids else 1
                supporting = True
            elif candidate.type in {"PHONE", "PAYMENT_CARD", "BANK_ACCOUNT"}:
                priority = 2 if candidate.id in contextual_ids else 1
                supporting = candidate.type == "PHONE" and candidate.id in contextual_ids
                if candidate.type == "PAYMENT_CARD" and payment_context(text, candidate.start):
                    priority, supporting = 2, True
        priorities[candidate.id] = priority
        if supporting:
            supports.setdefault(value_key(candidate), []).append(candidate)
    # Remove card-shaped order/requisite values independently of any other detector.
    for candidate in candidates:
        validator = DOCUMENT_MODELS.get(candidate.type)
        if candidate.source == "model" and candidate.decision == "accepted" and validator:
            valid_shape = (
                structured_reason(text[candidate.start : candidate.end], validator) == "rule_passed"
            )
            same_rule = any(
                support.type == candidate.type
                and candidate.start <= support.start < support.end <= candidate.end
                for support in supports.get(value_key(candidate), ())
            )
            if not valid_shape and not same_rule:
                candidate.decision = "rejected"
                candidate.reason = "invalid_format"
            elif not same_rule and non_payment_context(text, candidate.start):
                candidate.decision = "rejected"
                candidate.reason = "context_mismatch"
        if (
            candidate.decision == "accepted"
            and candidate.type == "PAYMENT_CARD"
            and non_payment_context(text, candidate.start)
        ):
            candidate.decision = "rejected"
            candidate.reason = "context_mismatch"
    for candidate in candidates:
        if candidate.decision != "accepted":
            continue
        if candidate.type not in COMPETING_TYPES and candidate.type != "PERSON":
            continue
        key = value_key(candidate)
        for support in supports.get(key, ()):
            if (
                support.type == candidate.type
                or priorities[support.id] <= priorities[candidate.id]
                or not support.start <= candidate.start < candidate.end <= support.end
                or (
                    candidate.type == "PERSON"
                    and support.type not in {"BIRTH_CERTIFICATE", "MILITARY_ID"}
                )
            ):
                continue
            # A support is active unless a stronger accepted rule also contains it.
            if any(
                other.type != support.type
                and priorities[other.id] > priorities[support.id]
                and other.start <= support.start < support.end <= other.end
                for other in supports[key]
            ):
                continue
            candidate.decision = "rejected"
            candidate.reason = "superseded_by_specific_rule"
            break
