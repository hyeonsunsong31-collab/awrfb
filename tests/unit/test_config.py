from __future__ import annotations

import json

import pytest
from pydantic import ValidationError

from jarvis.common.config import deep_merge, env_overrides, load_config


def test_default_config_loads(cfg):
    assert cfg.fight.game.place_id == 13621938427
    assert cfg.fight.game.allowed_servers == "private"
    assert cfg.fight.game.i_accept_ban_risk is False
    assert cfg.brain.endpoints["nvidia"].base_url == "https://integrate.api.nvidia.com/v1"
    assert cfg.brain.roles["chat"].chain[-1].startswith("local:")
    assert cfg.tier == "t6"  # no measured tier yet -> safest tier
    assert cfg.voice.tts.engine == "supertonic"


def test_tier_overlay_and_measured_tier(data_dir):
    (data_dir / "run").mkdir(parents=True)
    (data_dir / "run" / "tier.json").write_text(json.dumps({"tier": "t8"}))
    c = load_config(environ={})
    assert c.tier == "t8"
    assert c.voice.tts.engine == "qwen3"
    # Overlay replaced only the chain, other role fields survive from brain.yaml.
    assert c.brain.roles["chat"].priority == 0
    assert c.brain.roles["chat"].tools is True


def test_env_and_user_overrides(data_dir):
    data_dir.mkdir(parents=True, exist_ok=True)
    (data_dir / "config.yaml").write_text("fight:\n  game:\n    private_server_link_code: abc123\n")
    c = load_config(environ={"JARVIS__VOICE__STT__BEAM_SIZE": "5", "JARVIS__HUD__GESTURES": "true"})
    assert c.fight.game.private_server_link_code == "abc123"
    assert c.voice.stt.beam_size == 5
    assert c.hud.gestures is True


def test_unknown_keys_rejected(data_dir):
    data_dir.mkdir(parents=True, exist_ok=True)
    (data_dir / "config.yaml").write_text("fight:\n  game:\n    plce_id: 1\n")
    with pytest.raises(ValidationError):
        load_config(environ={})


def test_deep_merge_and_env_parse():
    assert deep_merge({"a": {"b": 1, "c": 2}}, {"a": {"b": 3}}) == {"a": {"b": 3, "c": 2}}
    assert env_overrides({"JARVIS__A__B": "[1, 2]", "OTHER": "x"}) == {"a": {"b": [1, 2]}}
