"""When the background learner may do what (pure decision logic, unit-tested).

Fights always win: no GPU job runs while a fight is armed or on, and a running one is killed the moment a
fight starts. Heavy work waits for an idle PC; the nightly retrain runs once a day in its hour (or when the
owner asks), and only when enough new fights were recorded since the last one.
"""

from __future__ import annotations

import time
from dataclasses import asdict, dataclass, field

GPU_JOBS = {"nightly", "video"}


@dataclass
class LearnState:
    processed_sessions: list[str] = field(default_factory=list)
    sessions_at_last_train: int = 0
    last_nightly: float = 0.0
    last_kb: float = 0.0
    last_consolidate: float = 0.0
    videos_today: int = 0
    videos_day: str = ""
    requested: list[str] = field(default_factory=list)  # owner asked: "jarvis learn now nightly"

    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: dict) -> LearnState:
        return cls(**{k: v for k, v in d.items() if k in cls.__dataclass_fields__})


@dataclass
class Status:
    fight_active: bool
    idle_s: float  # since the last fight ended / owner activity
    cpu_pct: float
    gpu_util_pct: float | None
    sessions: list[str]  # fight/shadow/video sessions on disk with frames
    running: str | None = None


def next_job(st: LearnState, s: Status, cfg, now: float | None = None) -> str | None:
    """The single next job to start (or None). ``cfg`` is a LearnConfig."""
    now = time.time() if now is None else now
    if s.fight_active or s.running is not None:
        return None
    if not cfg.enabled:  # learning switched off: only jobs the owner asks for by name
        return next((r for r in st.requested if r not in GPU_JOBS or _gpu_free(s, cfg)), None)
    done = set(st.processed_sessions)
    if any(x not in done for x in s.sessions):
        return "post_fight"  # cheap: labels + drift check, right after a match
    for req in st.requested:
        if req not in GPU_JOBS or _gpu_free(s, cfg):
            return req
    if s.idle_s < cfg.min_idle_s or s.cpu_pct > 60:
        return None
    hour = time.localtime(now).tm_hour
    enough = len(s.sessions) - st.sessions_at_last_train >= cfg.retrain_min_new_sessions
    since = now - st.last_nightly
    in_window = (hour == cfg.nightly_hour and since > 20 * 3600) or since > 44 * 3600  # catch up on missed nights
    if enough and in_window and _gpu_free(s, cfg):
        return "nightly"
    if now - st.last_consolidate > 24 * 3600:
        return "consolidate"
    if now - st.last_kb > 7 * 24 * 3600:
        return "kb"
    vi = cfg.video_ingest
    today = time.strftime("%Y-%m-%d", time.localtime(now))
    used = st.videos_today if st.videos_day == today else 0
    if vi.enabled and used < vi.max_videos_per_day and _gpu_free(s, cfg):
        return "video"
    return None


def _gpu_free(s: Status, cfg) -> bool:
    return s.gpu_util_pct is None or s.gpu_util_pct <= cfg.max_gpu_util_pct
