"""Capability scoping of the OWNER's request (trusted channel only), before any untrusted content is
read. Generous by design (JARVIS is fully autonomous), but fixed for the task: what a web page or a
screen says later cannot add capabilities."""

from __future__ import annotations

import asyncio
import logging

from jarvis.brain.broker import LLMBroker
from jarvis.safety.capabilities import CLASSIFIER_PROMPT, Cap, classify_heuristic, parse_classifier_output

log = logging.getLogger(__name__)


async def scope_request(text: str, broker: LLMBroker | None, timeout_s: float = 1.5) -> set[Cap]:
    caps = classify_heuristic(text)
    if broker is None:
        return caps
    try:
        res = await asyncio.wait_for(
            broker.complete("classifier", [{"role": "system", "content": CLASSIFIER_PROMPT}, {"role": "user", "content": text}]),
            timeout_s,
        )
        caps |= parse_classifier_output(res.text)
    except Exception as e:  # slow/unavailable classifier: the heuristic alone is still a valid scope
        log.debug("capability classifier unavailable: %s", e)
    # Fights can only be started by the dedicated owner-verified intent, never via the agent.
    caps.discard(Cap.GAME)
    return caps
