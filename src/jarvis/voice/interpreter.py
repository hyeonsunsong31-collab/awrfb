"""Interpreter mode (speech to speech): Korean <-> English through the broker's translator role."""

from __future__ import annotations

from jarvis.brain.prompts import TRANSLATE

TARGET = {"ko": ("en", "English"), "en": ("ko", "Korean (한국어)")}


class Interpreter:
    def __init__(self, broker) -> None:
        self.broker = broker

    async def translate(self, text: str, src_lang: str) -> tuple[str, str]:
        """Returns (translation, target language code)."""
        target, name = TARGET.get(src_lang, TARGET["en"])
        res = await self.broker.complete(
            "translator",
            [{"role": "system", "content": TRANSLATE.format(target=name)}, {"role": "user", "content": text}],
        )
        return res.text.strip(), target
