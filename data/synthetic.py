"""Offline synthetic dialogues: python -m data.synthetic --output-dir PATH."""

import argparse
import hashlib
import json
import random
from collections import Counter
from pathlib import Path
from typing import Literal

from data.schema import Example, Message, Source, Span

VERSION = "1.0"
DATASET = "synthetic-dialogues"
SPLITS = ("train", "validation", "calibration", "test")
Split = Literal["train", "validation", "calibration", "test"]
Mode = Literal["educational", "realistic"]
# Fixed allowlist from https://docs.stripe.com/testing (Visa and Mastercard).
TEST_CARDS = ("4242424242424242", "5555555555554444")
REALISTIC_VERSION = "1.0"


def _realistic_values(identity: str) -> dict[str, str]:
    rng = random.Random(f"realistic:{REALISTIC_VERSION}:{identity}")
    # Algorithmically generated; checksum validity does not imply issued identity.
    inn = "77" + "".join(str(rng.randrange(10)) for _ in range(8))
    for weights in ((7, 2, 4, 10, 3, 5, 9, 4, 6, 8), (3, 7, 2, 4, 10, 3, 5, 9, 4, 6, 8)):
        inn += str(sum(int(d) * w for d, w in zip(inn, weights, strict=True)) % 11 % 10)
    card = rng.choice(TEST_CARDS)
    separator = rng.choice(("", " ", "-"))
    return {
        # Ofcom drama-only mobile range 07700 900000–900999; international form.
        "PHONE": f"+44 7700 900{rng.randrange(1000):03d}",
        "PASSPORT": f"{rng.randrange(1, 100):02d} {rng.randrange(100):02d} {rng.randrange(1, 1000000):06d}",
        "INN": inn,
        "CREDIT_CARD": separator.join(card[i : i + 4] for i in range(0, 16, 4)),
    }


def _digest(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def group_split(group_id: str) -> Split:
    """Stable 60/20/10/10 hash allocation, independent of corpus size and ordering."""
    bucket = int(_digest("split-v1:" + group_id), 16) % 10
    if bucket < 6:
        return "train"
    if bucket < 8:
        return "validation"
    return "calibration" if bucket == 8 else "test"


def generate(
    seed: int = 42, groups: int = 40, variants: int = 2, *, mode: Mode = "educational"
) -> list[Example]:
    """Each parent persona (or clean scenario) owns all its dialogue variants.

    Every fourth parent is clean. Parameters are implementation defaults rather
    than team requirements. No files, network, detector logs or source datasets.
    """
    if type(seed) is not int or type(groups) is not int or type(variants) is not int:
        raise ValueError("seed, groups and variants must be integers")
    if groups < 4 or variants < 1:
        raise ValueError("groups must be >= 4 and variants >= 1")
    if mode not in ("educational", "realistic"):
        raise ValueError("mode must be educational or realistic")
    examples = []
    for index in range(groups):
        parent = f"{DATASET}:{VERSION}:{seed}:{index}"
        identity = _digest(parent)
        rng = random.Random(identity)
        clean = index % 4 == 0
        profile = {
            "FIRST_NAME": rng.choice(["Люмослав", "Звездолика", "Облакосвет"]),
            "LAST_NAME": f"Вымышленный-{identity}",
            "EMAIL": f"persona-{identity}@example.invalid",
            "PHONE": f"+0 (000) {identity[:12]}",
            "CITY": f"Тестоград-{identity[:12]}",
            "URL": f"https://example.invalid/persona/{identity}",
            "IP_ADDRESS": f"192.0.2.{1 + rng.randrange(254)}",
            "PASSPORT": f"TEST-PASSPORT-{identity}",
            "INN": f"TEST-INN-{identity}",
            "CREDIT_CARD": f"TEST-CARD-{identity}",
        }
        # The immutable parent includes its full persona; variants inherit its ID.
        group_id = _digest(json.dumps([parent, clean, profile], ensure_ascii=False))
        split = group_split(group_id)
        # Both renderings of the same persona retain the original group and split.
        if mode == "realistic":
            profile.update(_realistic_values(identity))
        for variant in range(variants):
            messages: list[Message] = []
            entities: list[Span] = []

            def message(
                role: str,
                *parts: str | tuple[str, str],
                messages: list[Message] = messages,
                entities: list[Span] = entities,
            ) -> None:
                message_id = f"m{len(messages)}"
                text = ""
                for part in parts:
                    if isinstance(part, tuple):
                        label, value = part
                        start = len(text)
                        text += value
                        entities.append(
                            Span(
                                id=f"e{len(entities)}",
                                message_id=message_id,
                                entity_type=label,
                                start=start,
                                end=len(text),
                                text=value,
                            )
                        )
                    else:
                        text += part
                messages.append(Message(id=message_id, role=role, text=text))

            message("system", "Учебный диалог: все сведения вымышлены.")
            if clean:
                message(
                    "user", f"🌤️ Учебный сценарий {identity}, вариант {variant}: как читать прогноз?"
                )
                message("assistant", "Сравните температуру и вероятность осадков.")
                message("user", "Спасибо! Личных сведений не передаю.")
            else:
                greeting = ("🙂 Здравствуйте! Я ", "🧪 Помогите с учебной анкетой: ")[variant % 2]
                message(
                    "user",
                    greeting,
                    ("FIRST_NAME", profile["FIRST_NAME"]),
                    " ",
                    ("LAST_NAME", profile["LAST_NAME"]),
                    f". Вариант {variant}.",
                )
                message("assistant", "Укажите учебные контакты и город.")
                message(
                    "user",
                    "Почта: ",
                    ("EMAIL", profile["EMAIL"]),
                    "; телефон: ",
                    ("PHONE", profile["PHONE"]),
                    "; город: ",
                    ("CITY", profile["CITY"]),
                    ".",
                )
                message(
                    "tool",
                    "Тестовый профиль: ",
                    ("URL", profile["URL"]),
                    "; IP: ",
                    ("IP_ADDRESS", profile["IP_ADDRESS"]),
                    ".",
                )
                message(
                    "user",
                    "Учебные реквизиты: паспорт ",
                    ("PASSPORT", profile["PASSPORT"]),
                    "; ИНН ",
                    ("INN", profile["INN"]),
                    "; карта ",
                    ("CREDIT_CARD", profile["CREDIT_CARD"]),
                    ".",
                )
                message("assistant", "Подтверждаю адрес: ", ("EMAIL", profile["EMAIL"]), ".")
            source_id = f"{identity}:{variant}"
            if mode == "realistic":
                source_id += f":realistic-{REALISTIC_VERSION}"
            examples.append(
                Example(
                    session_id=f"{DATASET}:{VERSION}:{source_id}",
                    messages=messages,
                    gold_entities=entities,
                    source=Source(
                        dataset=DATASET,
                        revision=VERSION,
                        license="CC0-1.0",
                        source_id=source_id,
                        group_id=group_id,
                        split=split,
                        url="local:data/synthetic.py",
                    ),
                    metadata={
                        "synthetic": True,
                        "generator_version": VERSION,
                        "seed": seed,
                        "parent_id": parent,
                        "variant": variant,
                        "clean": clean,
                        "identifier_policy": "explicit_test_markers; not checksum-valid",
                    },
                )
            )
            if mode == "realistic":
                examples[-1].metadata.update(
                    mode=mode,
                    realistic_version=REALISTIC_VERSION,
                    identifier_policy="synthetic_formats; INN checksum; official Stripe test cards; Ofcom drama phones",
                    card_source="https://docs.stripe.com/testing",
                    phone_source="https://www.ofcom.org.uk/phones-and-broadband/phone-numbers/numbers-for-drama",
                )
    return examples


def write_dataset(
    output_dir: Path,
    *,
    seed: int = 42,
    groups: int = 40,
    variants: int = 2,
    mode: Mode = "educational",
) -> dict:
    examples = generate(seed, groups, variants, mode=mode)
    output_dir.mkdir(parents=True, exist_ok=False)
    files = {}
    for split in SPLITS:
        records = [e for e in examples if e.source.split == split]
        content = "".join(e.model_dump_json() + "\n" for e in records).encode("utf-8")
        filename = f"{split}.jsonl"
        (output_dir / filename).write_bytes(content)
        files[filename] = {
            "sha256": hashlib.sha256(content).hexdigest(),
            "examples": len(records),
            "groups": len({e.source.group_id for e in records}),
            "clean": sum(not e.gold_entities for e in records),
        }
    manifest = {
        "dataset": DATASET,
        "generator_version": VERSION,
        "seed": seed,
        "groups": groups,
        "variants": variants,
        "files": files,
        "entity_counts": dict(
            sorted(
                Counter(entity.entity_type for e in examples for entity in e.gold_entities).items()
            )
        ),
        "split_policy": "parent SHA256 buckets 60/20/10/10 before variants; no redmadrobot",
        "offset_unit": "Python Unicode code points [start,end)",
    }
    if mode == "realistic":
        manifest.update(
            mode=mode,
            realistic_version=REALISTIC_VERSION,
            card_source="https://docs.stripe.com/testing",
            phone_source="https://www.ofcom.org.uk/phones-and-broadband/phone-numbers/numbers-for-drama",
        )
    (output_dir / "manifest.json").write_bytes(
        (json.dumps(manifest, ensure_ascii=False, indent=2) + "\n").encode("utf-8")
    )
    return manifest


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--groups", type=int, default=40)
    parser.add_argument("--variants", type=int, default=2)
    parser.add_argument("--mode", choices=("educational", "realistic"), default="educational")
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    if args.groups < 4 or args.variants < 1:
        parser.error("groups must be >= 4 and variants >= 1")
    manifest = write_dataset(
        args.output_dir, seed=args.seed, groups=args.groups, variants=args.variants, mode=args.mode
    )
    print(json.dumps({"files": manifest["files"]}))


if __name__ == "__main__":
    main()
