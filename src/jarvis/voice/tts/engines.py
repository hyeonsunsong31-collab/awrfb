"""TTS engines (each imported lazily so missing optional packages only disable that engine).

* Supertonic-3 (ONNX, CPU): the 6 GB-tier voice and the in-fight voice on 12 GB cards.
* Qwen3-TTS Base (GPU) cloning JARVIS's own reference clip, which VoiceDesign creates once from a text
  description of an original butler voice (never a real person's voice).
* Edge-TTS (cloud): fallback when local engines are unavailable.
"""

from __future__ import annotations

import asyncio
import io
import logging
from pathlib import Path

import numpy as np

from jarvis.voice.tts.base import LANG_NAMES, Audio, TtsError

log = logging.getLogger(__name__)

REFERENCE_TEXT = "Good evening. All systems are online, and I am at your service. Shall we begin?"


class SupertonicEngine:
    name = "supertonic"

    def __init__(self, cfg, models_dir: Path, threads: int = 2) -> None:
        from supertonic import TTS

        self._tts = TTS(model=cfg.supertonic_model, model_dir=models_dir / "supertonic" / cfg.supertonic_model,
                        auto_download=True, intra_op_num_threads=threads, inter_op_num_threads=1)
        self._style = self._tts.get_voice_style(cfg.supertonic_voice)
        self.rate = int(self._tts.sample_rate)
        self._steps = cfg.supertonic_steps
        self._speed = 1.05 * cfg.speed  # 1.05 is Supertonic's own default pace

    def synth(self, text: str, lang: str) -> Audio:
        wav, _dur = self._tts.synthesize(text, voice_style=self._style, total_steps=self._steps, speed=self._speed,
                                         lang=lang if lang in LANG_NAMES else "na")
        return Audio(np.asarray(wav, np.float32).reshape(-1), self.rate)


class Qwen3Engine:
    name = "qwen3"

    def __init__(self, cfg, device: str = "cuda:0") -> None:
        ref = cfg.qwen_voice_ref_wav
        if not ref or not Path(ref).exists():
            raise TtsError("no JARVIS reference voice yet - run `jarvis setup-voice`")
        import torch
        from qwen_tts import Qwen3TTSModel

        self._torch = torch
        self._model = Qwen3TTSModel.from_pretrained(cfg.qwen_base_model, device_map=device, dtype=torch.bfloat16)
        self._prompt = self._model.create_voice_clone_prompt(
            ref_audio=str(ref), ref_text=cfg.qwen_voice_ref_text, x_vector_only_mode=not cfg.qwen_voice_ref_text
        )

    def synth(self, text: str, lang: str) -> Audio:
        wavs, sr = self._model.generate_voice_clone(text=text, language=LANG_NAMES.get(lang, "Auto"),
                                                    voice_clone_prompt=self._prompt)
        return Audio(np.asarray(wavs[0], np.float32).reshape(-1), int(sr))

    def close(self) -> None:
        del self._model
        self._torch.cuda.empty_cache()


def design_reference_voice(cfg, out_wav: Path, device: str = "cuda:0") -> tuple[Path, str]:
    """Create JARVIS's original voice once with Qwen3-TTS VoiceDesign, saved as the cloning reference."""
    import soundfile as sf
    import torch
    from qwen_tts import Qwen3TTSModel

    model = Qwen3TTSModel.from_pretrained(cfg.qwen_design_model, device_map=device, dtype=torch.bfloat16)
    try:
        wavs, sr = model.generate_voice_design(text=REFERENCE_TEXT, instruct=cfg.voice_design_prompt, language="English")
    finally:
        del model
        torch.cuda.empty_cache()
    out_wav.parent.mkdir(parents=True, exist_ok=True)
    sf.write(str(out_wav), np.asarray(wavs[0], np.float32).reshape(-1), int(sr))
    return out_wav, REFERENCE_TEXT


def decode_audio_bytes(data: bytes) -> Audio:
    """Decode compressed audio (Edge-TTS mp3) to float32 mono with PyAV."""
    import av

    chunks: list[np.ndarray] = []
    with av.open(io.BytesIO(data)) as container:
        stream = container.streams.audio[0]
        rate = int(stream.rate or 24000)
        rs = av.AudioResampler(format="flt", layout="mono", rate=rate)
        for frame in container.decode(stream):
            chunks.extend(f.to_ndarray().reshape(-1) for f in rs.resample(frame))
        chunks.extend(f.to_ndarray().reshape(-1) for f in rs.resample(None))
    pcm = np.concatenate(chunks).astype(np.float32) if chunks else np.zeros(0, np.float32)
    return Audio(pcm, rate)


class EdgeEngine:
    name = "edge"

    def __init__(self, cfg) -> None:
        import edge_tts  # noqa: F401  (fail early if missing)

        self.voices = {"en": cfg.edge_voice_en, "ko": cfg.edge_voice_ko}
        pct = round((cfg.speed - 1.0) * 100)
        self.rate_arg = f"{pct:+d}%"

    def synth(self, text: str, lang: str) -> Audio:
        import edge_tts

        voice = self.voices.get(lang, self.voices["en"])

        async def fetch() -> bytes:
            buf = bytearray()
            async for chunk in edge_tts.Communicate(text, voice, rate=self.rate_arg).stream():
                if chunk.get("type") == "audio":
                    buf += chunk["data"]
            return bytes(buf)

        data = asyncio.run(fetch())  # called from a worker thread: its own event loop
        if not data:
            raise TtsError("Edge-TTS returned no audio")
        return decode_audio_bytes(data)
