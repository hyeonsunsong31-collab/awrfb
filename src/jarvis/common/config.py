"""Typed configuration.

Load order (later wins):
  1. ``configs/default.yaml`` + ``configs/brain.yaml`` + ``configs/voice.yaml``
  2. ``configs/tiers/<tier>.yaml`` (VRAM tier overlay, chosen by ``jarvis doctor --vram``)
  3. the user's ``<data_dir>/config.yaml``
  4. environment variables ``JARVIS__SECTION__KEY=value`` (values parsed as YAML)

Unknown keys are rejected so a typo in a YAML file fails loudly instead of silently doing nothing.
"""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any, Literal

import yaml
from pydantic import BaseModel, ConfigDict, Field

from jarvis.common.paths import DataPaths, config_dir, data_paths


class _Model(BaseModel):
    model_config = ConfigDict(extra="forbid", validate_assignment=True)


# --------------------------------------------------------------------------------------------- IPC
class IpcConfig(_Model):
    host: str = "127.0.0.1"
    bus_in_port: int = 47810
    bus_out_port: int = 47811
    rpc_ports: dict[str, int] = Field(
        default_factory=lambda: {
            "supervisor": 47820,
            "core": 47821,
            "toolhost": 47822,
            "fightd": 47823,
            "learnerd": 47824,
        }
    )
    hud_ws_port: int = 47830
    request_timeout_s: float = 10.0


# ------------------------------------------------------------------------------------------- brain
class EndpointConfig(_Model):
    kind: Literal["nvidia", "ollama", "openai_compat"] = "openai_compat"
    base_url: str
    api_key_env: str | None = None
    keyring_service: str | None = None
    keyring_user: str | None = None
    timeout_s: float = 60.0
    max_rpm: int = 30
    local: bool = False


class RoleConfig(_Model):
    chain: list[str]
    priority: int = 5
    max_tokens: int = 1024
    temperature: float | None = 0.6
    tools: bool = False
    vision: bool = False
    timeout_s: float = 60.0
    # Provider-specific request fields (e.g. reasoning toggles for NVIDIA-hosted models).
    extra_body: dict[str, Any] = Field(default_factory=dict)
    # Drop <think>...</think> / reasoning_content so internal reasoning is never spoken aloud.
    strip_reasoning: bool = True


class BrainConfig(_Model):
    endpoints: dict[str, EndpointConfig]
    roles: dict[str, RoleConfig]
    probe_cache_hours: float = 24.0
    agent_max_steps: int = 16
    agent_max_parallel_tools: int = 4
    persona_name: str = "JARVIS"
    owner_title_en: str = "sir"
    owner_title_ko: str = "주인님"
    memory_summary_every_turns: int = 12
    history_max_messages: int = 40


# ------------------------------------------------------------------------------------------- voice
class WakeConfig(_Model):
    model: str = "hey_jarvis"
    threshold: float = 0.5
    patience_frames: int = 2
    second_stage: bool = True
    second_stage_keywords: list[str] = Field(
        default_factory=lambda: ["jarvis", "jervis", "javis", "자비스", "재비스", "쟈비스"]
    )
    optimistic_listen: bool = True
    debounce_s: float = 1.5


class VadConfig(_Model):
    model_path: str | None = None
    threshold: float = 0.5
    min_speech_ms: int = 250
    min_silence_ms: int = 650
    max_utterance_s: float = 20.0
    pre_roll_ms: int = 300


class SttConfig(_Model):
    model: str = "large-v3-turbo"
    device: Literal["auto", "cuda", "cpu"] = "auto"
    compute_type: str = "int8_float16"
    cpu_compute_type: str = "int8"
    beam_size: int = 3
    languages: list[str] = Field(default_factory=lambda: ["ko", "en"])
    hotwords: list[str] = Field(
        default_factory=lambda: [
            "JARVIS",
            "자비스",
            "Roblox",
            "로블록스",
            "untitled boxing game",
            "멈춰",
            "싸워",
        ]
    )
    initial_prompt: str | None = "JARVIS, 자비스. 한국어와 English를 섞어서 말합니다."


class TtsConfig(_Model):
    engine: Literal["auto", "supertonic", "qwen3", "edge"] = "auto"
    fallback_chain: list[str] = Field(default_factory=lambda: ["supertonic", "edge"])
    fight_engine: Literal["cache_only", "supertonic", "edge"] = "cache_only"
    speed: float = 1.0
    supertonic_voice: str = "M1"
    supertonic_steps: int = 8
    qwen_base_model: str = "Qwen/Qwen3-TTS-12Hz-0.6B-Base"
    qwen_design_model: str = "Qwen/Qwen3-TTS-12Hz-1.7B-VoiceDesign"
    qwen_voice_ref_wav: str | None = None
    qwen_voice_ref_text: str | None = None
    voice_design_prompt: str = (
        "An original, calm, refined, articulate middle-aged male butler voice with a warm baritone, "
        "measured pace and dry wit. Do not imitate any real person or actor."
    )
    edge_voice_en: str = "en-GB-RyanNeural"
    edge_voice_ko: str = "ko-KR-InJoonNeural"


class BargeInConfig(_Model):
    duck_gain: float = 0.25
    stop_after_ms: int = 300
    require_owner_for_stop: bool = True
    backchannel_words: list[str] = Field(
        default_factory=lambda: ["응", "네", "예", "음", "어", "yeah", "yes", "uh-huh", "mm", "ok", "okay", "right"]
    )


class AecConfig(_Model):
    enabled: bool = True
    reference: Literal["loopback", "tts", "none"] = "loopback"
    delay_ms: int = 60
    noise_suppression: bool = True
    auto_gain_control: bool = False
    high_pass_filter: bool = True


class SpeakerConfig(_Model):
    enabled: bool = True
    threshold: float = 0.45
    min_audio_s: float = 1.5
    enrollment_seconds: int = 60
    model: str = "speechbrain/spkrec-ecapa-voxceleb"


class MuteConfig(_Model):
    hotkey: str = "ctrl+alt+m"
    start_muted: bool = False
    soft_mute: bool = False


class VoiceConfig(_Model):
    enabled: bool = True
    sample_rate: int = 16000
    input_device: str | int | None = None
    output_device: str | int | None = None
    followup_window_s: float = 8.0
    wake: WakeConfig = Field(default_factory=WakeConfig)
    vad: VadConfig = Field(default_factory=VadConfig)
    stt: SttConfig = Field(default_factory=SttConfig)
    tts: TtsConfig = Field(default_factory=TtsConfig)
    barge_in: BargeInConfig = Field(default_factory=BargeInConfig)
    aec: AecConfig = Field(default_factory=AecConfig)
    speaker: SpeakerConfig = Field(default_factory=SpeakerConfig)
    mute: MuteConfig = Field(default_factory=MuteConfig)


# --------------------------------------------------------------------------------------------- HUD
class HudConfig(_Model):
    enabled: bool = True
    content_protection: bool = True
    stream_mode: bool = False
    interactive_hotkey: str = "ctrl+alt+j"
    combat_mode: Literal["dim", "hide"] = "dim"
    combat_fps_cap: int = 30
    prefer_secondary_monitor: bool = True
    gestures: bool = False
    electron_dir: str = "hud"


# ------------------------------------------------------------------------------------------ safety
class SafetyConfig(_Model):
    panic_hotkeys: list[str] = Field(default_factory=lambda: ["ctrl+alt+end", "ctrl+shift+f12"])
    kill_words: list[str] = Field(default_factory=lambda: ["stop", "abort", "멈춰", "그만", "정지", "스톱"])
    audit_enabled: bool = True


# ------------------------------------------------------------------------------------------- fight
class GameConfig(_Model):
    place_id: int = 13621938427
    launch_url_template: str = "roblox://experiences/start?placeId={place_id}"
    private_server_url_template: str = "roblox://placeId={place_id}&linkCode={link_code}"
    private_server_link_code: str | None = None
    allowed_servers: Literal["private", "public"] = "private"
    i_accept_ban_risk: bool = False
    window_title: str = "Roblox"
    process_name: str = "RobloxPlayerBeta.exe"
    launch_timeout_s: float = 90.0


class ActuatorConfig(_Model):
    heartbeat_timeout_ms: int = 500
    gate_max_age_ms: int = 150
    default_tap_ms: int = 45
    feint_gap_ms: int = 70
    physical_input_pause_ms: int = 1500


class PerceptionConfig(_Model):
    detector_model: str | None = None
    detector_classes: list[str] = Field(default_factory=lambda: ["person"])
    detector_input: int = 640
    detector_every_n_frames: int = 2
    detector_conf: float = 0.25
    threat_model: str | None = None
    threat_window: int = 8
    crop_size: int = 96
    hud_conf_min: float = 0.6
    max_process_fps: int = 120


class RecordingConfig(_Model):
    enabled: bool = True
    width: int = 960
    height: int = 540
    encoder: Literal["auto", "h264_nvenc", "libx264"] = "auto"
    crf: int = 23
    ffmpeg: str = "ffmpeg"
    max_disk_gb: float = 80.0


class ShadowConfig(_Model):
    record_owner_inputs: bool = False


class FightConfig(_Model):
    game: GameConfig = Field(default_factory=GameConfig)
    actuator: ActuatorConfig = Field(default_factory=ActuatorConfig)
    perception: PerceptionConfig = Field(default_factory=PerceptionConfig)
    recording: RecordingConfig = Field(default_factory=RecordingConfig)
    shadow: ShadowConfig = Field(default_factory=ShadowConfig)
    keymap_file: str = "ubg/keymap.yaml"
    policy_file: str = "ubg/policy_v0.yaml"
    knowledge_file: str = "ubg/knowledge_seed.yaml"
    max_matches_per_arm: int = 1


# ------------------------------------------------------------------------------------------- learn
class VideoIngestConfig(_Model):
    enabled: bool = False
    queries: list[str] = Field(
        default_factory=lambda: ["untitled boxing game pvp", "untitled boxing game ranked", "언타이틀드 복싱 게임"]
    )
    max_videos_per_day: int = 10
    max_gb: float = 20.0
    local_folders: list[str] = Field(default_factory=list)


class LearnConfig(_Model):
    enabled: bool = True
    min_idle_s: int = 300
    max_gpu_util_pct: int = 30
    nightly_hour: int = 4
    retrain_min_new_sessions: int = 3
    video_ingest: VideoIngestConfig = Field(default_factory=VideoIngestConfig)


class TiersConfig(_Model):
    selected: Literal["auto", "t6", "t8", "t12"] = "auto"
    min_free_mb_runtime: int = 400


# -------------------------------------------------------------------------------------------- root
class JarvisConfig(_Model):
    data_dir: str | None = None
    log_level: str = "INFO"
    language: Literal["auto", "ko", "en"] = "auto"
    ipc: IpcConfig = Field(default_factory=IpcConfig)
    brain: BrainConfig
    voice: VoiceConfig = Field(default_factory=VoiceConfig)
    hud: HudConfig = Field(default_factory=HudConfig)
    safety: SafetyConfig = Field(default_factory=SafetyConfig)
    fight: FightConfig = Field(default_factory=FightConfig)
    learn: LearnConfig = Field(default_factory=LearnConfig)
    tiers: TiersConfig = Field(default_factory=TiersConfig)

    # Resolved at load time (not part of the YAML schema).
    tier: str = "t6"

    def paths(self) -> DataPaths:
        return data_paths(Path(self.data_dir) if self.data_dir else None)


# ------------------------------------------------------------------------------------------ loader
def deep_merge(base: dict[str, Any], over: dict[str, Any]) -> dict[str, Any]:
    """Recursively merge ``over`` into a copy of ``base``. Lists and scalars are replaced."""
    out = dict(base)
    for k, v in over.items():
        if isinstance(v, dict) and isinstance(out.get(k), dict):
            out[k] = deep_merge(out[k], v)
        else:
            out[k] = v
    return out


def read_yaml(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    with path.open("r", encoding="utf-8") as f:
        data = yaml.safe_load(f) or {}
    if not isinstance(data, dict):
        raise ValueError(f"{path} must contain a YAML mapping at the top level")
    return data


def env_overrides(environ: dict[str, str] | None = None) -> dict[str, Any]:
    """``JARVIS__VOICE__STT__MODEL=small`` -> ``{"voice": {"stt": {"model": "small"}}}``."""
    environ = dict(os.environ if environ is None else environ)
    out: dict[str, Any] = {}
    for key, raw in environ.items():
        if not key.startswith("JARVIS__"):
            continue
        parts = [p.lower() for p in key[len("JARVIS__") :].split("__") if p]
        if not parts:
            continue
        cur = out
        for p in parts[:-1]:
            cur = cur.setdefault(p, {})
        cur[parts[-1]] = yaml.safe_load(raw)
    return out


def detected_tier(paths: DataPaths) -> str | None:
    f = paths.run / "tier.json"
    if not f.exists():
        return None
    try:
        return str(json.loads(f.read_text(encoding="utf-8"))["tier"])
    except (OSError, ValueError, KeyError):
        return None


def load_config(
    *,
    cfg_dir: Path | None = None,
    user_config: Path | None = None,
    tier: str | None = None,
    environ: dict[str, str] | None = None,
) -> JarvisConfig:
    cdir = cfg_dir or config_dir()
    if not (cdir / "default.yaml").exists():
        raise FileNotFoundError(f"config directory {cdir} has no default.yaml (set JARVIS_CONFIG_DIR?)")
    merged: dict[str, Any] = {}
    for name in ("default.yaml", "brain.yaml", "voice.yaml"):
        merged = deep_merge(merged, read_yaml(cdir / name))

    env = env_overrides(environ)
    # Data dir may itself be overridden, and decides where the user config lives.
    data_dir = env.get("data_dir") or merged.get("data_dir")
    paths = data_paths(Path(data_dir) if data_dir else None)
    user = read_yaml(user_config or paths.user_config)
    pre = deep_merge(deep_merge(merged, user), env)

    selected = tier or pre.get("tiers", {}).get("selected", "auto")
    if selected == "auto":
        selected = detected_tier(paths) or "t6"
    merged = deep_merge(merged, read_yaml(cdir / "tiers" / f"{selected}.yaml"))
    merged = deep_merge(deep_merge(merged, user), env)
    merged["tier"] = selected
    return JarvisConfig.model_validate(merged)


def load_yaml_resource(relative: str, cfg_dir: Path | None = None) -> dict[str, Any]:
    """Read a secondary YAML file (keymap, policy, knowledge) from the config directory."""
    return read_yaml((cfg_dir or config_dir()) / relative)
