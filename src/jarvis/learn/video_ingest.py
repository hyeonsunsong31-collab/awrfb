"""Video ingestion: UBG clips from YouTube (yt-dlp) or local/OBS folders become labelled training sessions.

The useful part of a video is its labels: the recording player's HP bar drops exactly when the opponent's
attacks land. A clip is accepted only when our calibrated HUD reader actually sees UBG's combat HUD in it;
it is then re-read frame by frame into the same session format our own fights use (kind "video"), with the
opponent located by the detector + tracker + follow-camera self-identification. Whether video data is kept
at all is decided by the nightly retrain: it must not make the model worse on our own fights."""

from __future__ import annotations

import hashlib
import logging
import shutil
from pathlib import Path

import cv2

from jarvis.fight.events import HudEventTracker
from jarvis.fight.perception.hud_bars import HudReader
from jarvis.fight.perception.self_id import SelfIdentifier
from jarvis.fight.perception.tracker import Track, Tracker
from jarvis.fight.recorder import SessionRecorder

log = logging.getLogger(__name__)

VIDEO_EXTS = (".mp4", ".mkv", ".webm", ".mov", ".avi")


def find_local_videos(folders: list[str]) -> list[Path]:
    out = []
    for f in folders:
        p = Path(f).expanduser()
        if p.is_dir():
            out += sorted(x for x in p.rglob("*") if x.suffix.lower() in VIDEO_EXTS)
    return out


def folder_gb(p: Path) -> float:
    if not p.exists():
        return 0.0
    return sum(f.stat().st_size for f in p.rglob("*") if f.is_file()) / 1e9


def ytdlp_download(query: str, out_dir: Path, max_n: int, max_height: int = 720) -> list[Path]:
    """Public clips only, no sign-in, at most ``max_n`` per query; files land in ``out_dir``."""
    if max_n <= 0:
        return []
    try:
        from yt_dlp import YoutubeDL
    except ImportError as e:
        raise RuntimeError("install the 'web' extra (yt-dlp) for video ingestion") from e
    out_dir.mkdir(parents=True, exist_ok=True)
    before = set(out_dir.iterdir())
    opts = {"format": f"bv*[height<={max_height}]+ba/b[height<={max_height}]", "outtmpl": str(out_dir / "%(id)s.%(ext)s"),
            "noplaylist": True, "quiet": True, "no_warnings": True, "noprogress": True, "max_downloads": max_n,
            "merge_output_format": "mkv", "match_filter": _short_enough}
    with YoutubeDL(opts) as ydl:
        try:
            ydl.download([f"ytsearch{max_n}:{query}"])
        except Exception as e:  # MaxDownloadsReached and network errors end the batch
            log.info("yt-dlp stopped: %s", e)
    return sorted(p for p in set(out_dir.iterdir()) - before if p.suffix.lower() in VIDEO_EXTS)


def _short_enough(info: dict, *, incomplete: bool = False) -> str | None:
    d = info.get("duration")
    if d and d > 40 * 60:
        return "longer than 40 minutes"
    if info.get("is_live"):
        return "live stream"
    return None


def hud_visible_share(video: Path, reader: HudReader, samples: int = 40, size: tuple[int, int] = (960, 540)) -> float:
    cap = cv2.VideoCapture(str(video))
    n = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
    seen = total = 0
    try:
        for k in range(samples):
            if n:
                cap.set(cv2.CAP_PROP_POS_FRAMES, int(n * (k + 0.5) / samples))
            ok, img = cap.read()
            if not ok:
                break
            total += 1
            seen += bool(reader.read(cv2.resize(img, size, interpolation=cv2.INTER_AREA)).combat_hud)
    finally:
        cap.release()
    return seen / total if total else 0.0


def _box(t: Track | None) -> list[float] | None:
    return None if t is None else [round(t.cx, 4), round(t.cy, 4), round(t.w, 4), round(t.h, 4), t.id]


def ingest_video(video: Path, recordings: Path, layout, calibration_dir: Path, *, detector=None, min_share: float = 0.3,
                 out_fps: float = 30.0, size: tuple[int, int] = (960, 540), source: str = "", ffmpeg: str = "ffmpeg",
                 encoder: str = "auto", detect_every: int = 2) -> Path | None:
    """Accepted clip -> recordings/<ts>_video session (frames/events jsonl + video.mkv). None = rejected."""
    reader = HudReader(layout, calibration_dir=calibration_dir)
    share = hud_visible_share(video, reader, size=size)
    if share < min_share:
        log.info("rejected %s: UBG combat HUD visible in %.0f%% of samples", video.name, share * 100)
        return None
    cap = cv2.VideoCapture(str(video))
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    step = max(1, round(fps / out_fps))
    sid = hashlib.sha256((source or str(video)).encode()).hexdigest()[:10]
    rec = SessionRecorder(recordings, "video", width=size[0], height=size[1], ffmpeg=ffmpeg, encoder=encoder,
                          fps_hint=max(1, round(fps / step)), meta={"source_sha": sid, "hud_share": round(share, 3),
                                                                    "has_boxes": detector is not None})
    if rec.stats.encoder is None or rec._ffmpeg is None:
        rec.close()
        shutil.rmtree(rec.dir, ignore_errors=True)
        cap.release()
        raise RuntimeError("ffmpeg is needed to store ingested videos")
    tracker, ident, events = Tracker(), SelfIdentifier(), HudEventTracker()
    k = j = 0
    try:
        while True:
            ok, img = cap.read()
            if not ok:
                break
            k += 1
            if (k - 1) % step:
                continue
            t_ns = int((k - 1) / fps * 1e9)
            small = cv2.resize(img, size, interpolation=cv2.INTER_AREA)
            rd = reader.read(small, t_ns)
            for e in events.update(rd):
                rec.log("events", e.to_row())
            me = opp = None
            if detector is not None:
                tracks = tracker.update(detector.detect(small), t_ns) if j % detect_every == 0 else tracker.predict(t_ns)
                who = ident.update(tracks, t_ns)
                me, opp = who.me, who.opp
            rec.add_frame(small, {"t_ns": t_ns, "hud": {kk: (None if v is None else round(v, 4)) for kk, v in rd.values.items()},
                                  "hud_conf": round(rd.hud_conf, 3), "combat_hud": rd.combat_hud, "me": _box(me),
                                  "opp": _box(opp)}, block=True)
            j += 1
    finally:
        cap.release()
        rec.close({"frames": j, "source_fps": fps, "step": step})
    return rec.dir
