"""Voice and brain commands: probe, api-key, chat, say, listen, enroll, voice, core, voice-models, setup-voice."""

from __future__ import annotations

import asyncio
import time

import typer
from rich.console import Console
from rich.table import Table

console = Console()

ENROLL_LINES = (
    "안녕하세요, 저는 자비스의 주인입니다. 오늘 날씨가 정말 좋네요.",
    "Hello Jarvis, this is my voice. Please remember how I sound.",
    "자비스, 세 판 싸워. 그리고 끝나면 결과를 알려줘.",
    "Jarvis, open the browser and check my calendar for tomorrow.",
    "음악 좀 틀어 줘. 아니, 그만. 볼륨을 조금 줄여 줘.",
    "Stop the fight, mute the microphone, and tell me the time.",
    "로블록스에서 언타이틀드 복싱 게임을 실행해 줘.",
    "Thank you, that's all for now. Talk to you later.",
)


def _cfg():
    from jarvis.common.config import load_config

    return load_config()


def probe() -> None:
    """Check every configured model: reachable, Korean, tool calls, images, speed (cached for the broker)."""
    from jarvis.brain.probe import probe_all, save_cache
    from jarvis.brain.providers import build_providers

    cfg = _cfg()
    providers = build_providers(cfg.brain)
    if not providers:
        console.print("[red]no usable LLM endpoint[/] - set the NVIDIA key with `jarvis api-key` or start Ollama")
        raise typer.Exit(1)
    results = asyncio.run(probe_all(cfg.brain, providers))
    save_cache(cfg.paths().ensure().run / "probe.json", results)
    t = Table(title="model probe")
    for col in ("model", "ok", "korean", "tools", "vision", "first token ms", "tok/s", "error"):
        t.add_column(col, overflow="fold")
    yn = {True: "[green]yes[/]", False: "[red]no[/]", None: "-"}
    for k, r in results.items():
        t.add_row(k, yn[r.ok], yn[r.korean_ok], yn[r.tools], yn[r.vision],
                  f"{r.ttft_ms:.0f}" if r.ttft_ms else "-", f"{r.tokens_per_s:.1f}" if r.tokens_per_s else "-", r.error[:80])
    console.print(t)


def api_key() -> None:
    """Store the NVIDIA API key in Windows Credential Manager (keyring). It is never printed or logged."""
    try:
        import keyring
    except ImportError:
        console.print("[red]keyring is not installed[/] (install the 'win' extra) - or set NVIDIA_API_KEY")
        raise typer.Exit(1) from None
    cfg = _cfg()
    ep = cfg.brain.endpoints.get("nvidia")
    key = typer.prompt("NVIDIA API key (nvapi-...)", hide_input=True).strip()
    if not key.startswith("nvapi-"):
        console.print("[yellow]that does not look like an NVIDIA key (expected nvapi-...); saved anyway[/]")
    keyring.set_password(ep.keyring_service if ep and ep.keyring_service else "jarvis",
                         ep.keyring_user if ep and ep.keyring_user else "nvidia_api_key", key)
    console.print("saved. Run `jarvis probe` to test the models.")


async def _core_chat(cfg, text: str, speak: bool) -> dict | None:
    from jarvis.common.ipc import RpcClient, RpcError, load_or_create_token

    c = RpcClient(cfg.ipc.rpc_ports["core"], load_or_create_token(cfg.paths().ensure().run), cfg.ipc.host, timeout=2.0)
    try:
        await c.call("status", timeout=1.5)
        return await c.call("chat", {"text": text, "speak": speak}, timeout=200.0)
    except RpcError:
        return None
    finally:
        await c.close()


def chat(
    message: str = typer.Argument("", help="One message (omit for an interactive chat)."),
    speak: bool = typer.Option(False, help="Also speak the reply (through the running core)."),
) -> None:
    """Typed chat (TTT) with JARVIS - through the running core, or in-process when core is not running."""
    from jarvis.brain.agent import AgentTurn
    from jarvis.core import build_brain
    from jarvis.voice.textnorm import detect_lang

    cfg = _cfg()
    local = None

    async def one(text: str) -> None:
        nonlocal local
        res = await _core_chat(cfg, text, speak)
        if res is not None:
            console.print(res.get("text", ""))
            return
        if local is None:
            local = build_brain(cfg)
            console.print("[dim](core is not running - answering in-process, without PC tools)[/]")
        _broker, agent, _memory = local

        async def out(d: str) -> None:
            console.print(d, end="", soft_wrap=True)

        await agent.run(AgentTurn(text=text, origin="owner_text", lang=detect_lang(text)), on_text=out)
        console.print()

    async def loop() -> None:
        if message:
            await one(message)
            return
        console.print("[dim]type a message (empty line to quit)[/]")
        while True:
            text = await asyncio.to_thread(input, "> ")
            if not text.strip():
                return
            await one(text)

    asyncio.run(loop())


def say(
    text: str = typer.Argument(..., help="What to say (Korean or English)."),
    lang: str = typer.Option("", help="ko or en (default: detected)."),
    out: str = typer.Option("", help="Write a WAV file instead of playing."),
) -> None:
    """Speak a sentence with the configured voice (tests TTS, number reading and the audio output)."""
    import numpy as np

    from jarvis.voice.runtime import build_router
    from jarvis.voice.textnorm import clean_for_speech, detect_lang

    cfg = _cfg()
    lang = lang or detect_lang(text)
    spoken = clean_for_speech(text, lang)
    router = build_router(cfg)
    t0 = time.perf_counter()
    s = router.synth(spoken, lang)
    if s is None:
        console.print(f"[red]no TTS engine worked[/]: {router.last_error}")
        raise typer.Exit(1)
    console.print(f"engine={s.engine} {s.audio.duration_s:.2f}s audio in {time.perf_counter() - t0:.2f}s  text: {spoken}")
    if out:
        import soundfile as sf

        sf.write(out, s.audio.pcm, s.audio.rate)
        console.print(f"wrote {out}")
        return
    import sounddevice as sd

    sd.play(np.asarray(s.audio.pcm, np.float32), s.audio.rate)
    sd.wait()


def listen(seconds: float = typer.Option(30.0, help="How long to listen.")) -> None:
    """Show what JARVIS hears: VAD segments and Whisper transcripts (no wake word, nothing is executed)."""
    import queue

    from jarvis.voice.aec import build_aec
    from jarvis.voice.audio_io import MicStream
    from jarvis.voice.intents import match_intent
    from jarvis.voice.runtime import build_vad
    from jarvis.voice.stt import WhisperStt
    from jarvis.voice.vad import VadSegmenter

    cfg = _cfg()
    notes: list[str] = []
    seg = VadSegmenter.from_config(build_vad(cfg, notes), cfg.voice.vad)
    aec = build_aec(cfg.voice.aec)
    stt = WhisperStt.from_config(cfg.voice.stt, cfg.paths().ensure().models)
    for n in notes:
        console.print(f"[yellow]{n}[/]")
    q: queue.Queue = queue.Queue()
    mic = MicStream(q.put, device=cfg.voice.input_device)
    mic.start()
    console.print(f"listening for {seconds:.0f}s ... (mic at {mic.device_rate} Hz)")
    end = time.monotonic() + seconds
    try:
        while time.monotonic() < end:
            try:
                frame = q.get(timeout=0.5)
            except queue.Empty:
                continue
            for ev in seg.feed(aec.process(frame)):
                if ev.kind == "end" and ev.audio is not None:
                    r = stt.transcribe(ev.audio)
                    it = match_intent(r.text) if r.ok else None
                    console.print(f"[{ev.voiced_ms / 1000:.1f}s voiced, STT {r.elapsed_s * 1000:.0f} ms, {r.lang}] "
                                  f"{r.text!r}" + (f"  rejected: {r.rejected}" if r.rejected else "")
                                  + (f"  -> intent {it.name} {it.args}" if it else ""))
    finally:
        mic.stop()


def enroll(seconds: float = typer.Option(60.0, help="Recording length (speak the whole time).")) -> None:
    """Record the owner's voiceprint (Korean and English) so only you can start fights and give PC commands."""
    import queue

    import numpy as np

    from jarvis.voice.audio_io import MicStream
    from jarvis.voice.runtime import VOICEPRINT
    from jarvis.voice.speaker_id import EcapaEmbedder
    from jarvis.voice.speaker_id import enroll as do_enroll

    cfg = _cfg()
    q: queue.Queue = queue.Queue()
    mic = MicStream(q.put, device=cfg.voice.input_device)
    console.print("Read these lines aloud, naturally, until the timer ends (repeat them if you finish early):\n")
    for line in ENROLL_LINES:
        console.print(f"  {line}")
    typer.prompt("\nPress Enter to start recording", default="", show_default=False)
    mic.start()
    frames, end = [], time.monotonic() + seconds
    try:
        while time.monotonic() < end:
            try:
                frames.append(q.get(timeout=0.5))
            except queue.Empty:
                continue
            left = int(end - time.monotonic())
            if left % 10 == 0 and len(frames) % 100 == 0:
                console.print(f"  {left}s left")
    finally:
        mic.stop()
    audio = np.concatenate(frames) if frames else np.zeros(0, np.float32)
    models = cfg.paths().ensure().models
    emb = EcapaEmbedder(cfg.voice.speaker.model, savedir=models / "spkrec")
    try:
        embs = do_enroll(emb, audio, models / VOICEPRINT)
    except ValueError as e:
        console.print(f"[red]{e}[/]")
        raise typer.Exit(1) from None
    console.print(f"voiceprint saved ({len(embs)} samples). Only your voice can now start fights, unmute or give PC commands.")


def voice(text_only: bool = typer.Option(False, "--text-only", help="No microphone/speaker; typed chat only.")) -> None:
    """Run JARVIS in this terminal (voice + brain), starting the event bus if no supervisor runs it."""
    from jarvis.common.logging_setup import setup_logging
    from jarvis.core import main as core_main

    cfg = _cfg()
    setup_logging("core", cfg.paths().ensure().logs, cfg.log_level)
    console.print("JARVIS is listening. Say 'Hey Jarvis'. Ctrl+C to quit.")
    core_main(cfg, voice=not text_only, standalone_bus=True)


def core(no_voice: bool = typer.Option(False, "--no-voice", help="Text only (no audio devices).")) -> None:
    """The core process (started by the supervisor)."""
    from jarvis.common.logging_setup import setup_logging
    from jarvis.core import main as core_main

    cfg = _cfg()
    setup_logging("core", cfg.paths().ensure().logs, cfg.log_level)
    core_main(cfg, voice=not no_voice)


def toolhost() -> None:
    """The toolhost process: runs PC tools for the agent under the safety policy (started by the supervisor)."""
    from jarvis.common.logging_setup import setup_logging
    from jarvis.toolhost import main as toolhost_main

    cfg = _cfg()
    setup_logging("toolhost", cfg.paths().ensure().logs, cfg.log_level)
    toolhost_main(cfg)


def voice_models(
    all_models: bool = typer.Option(False, "--all", help="Also pre-download Whisper, Supertonic and the speaker model."),
) -> None:
    """Download the wake-word and VAD models (pinned versions, SHA-256 verified)."""
    from jarvis.voice.models import download, missing

    cfg = _cfg()
    models = cfg.paths().ensure().models
    got = download(models, on_progress=lambda m, s: console.print(f"  {s:12} {m.rel}"))
    left = missing(models)
    console.print(f"downloaded {len(got)} file(s); {'all present' if not left else 'missing: ' + ', '.join(m.rel for m in left)}")
    if all_models:
        from jarvis.voice.runtime import build_router
        from jarvis.voice.speaker_id import EcapaEmbedder
        from jarvis.voice.stt import WhisperStt

        console.print("fetching Whisper (this is large)...")
        WhisperStt.from_config(cfg.voice.stt, models)
        console.print("fetching the voice and pre-rendering phrases...")
        console.print(f"  rendered {build_router(cfg).prerender()} phrase(s)")
        console.print("fetching the speaker-verification model...")
        EcapaEmbedder(cfg.voice.speaker.model, savedir=models / "spkrec")._ensure()
        console.print("done")


def setup_voice(device: str = typer.Option("cuda:0", help="Torch device for Qwen3-TTS.")) -> None:
    """Create JARVIS's own voice once (Qwen3-TTS VoiceDesign, 8 GB+ cards) and make it the cloning reference."""
    from jarvis.common.config import update_user_config
    from jarvis.voice.tts.engines import design_reference_voice

    cfg = _cfg()
    out = cfg.paths().ensure().models / "voice" / "jarvis_reference.wav"
    console.print("designing an original butler voice (not a copy of any real person)...")
    wav, text = design_reference_voice(cfg.voice.tts, out, device=device)
    p = update_user_config(cfg.paths(), {"voice": {"tts": {"qwen_voice_ref_wav": str(wav), "qwen_voice_ref_text": text}}})
    console.print(f"saved {wav}; reference recorded in {p}. Try: jarvis say \"Good evening, sir.\"")


wakeword_app = typer.Typer(help="Your own wake word (e.g. 자비스): record takes, then `jarvis train wakeword`.",
                           no_args_is_help=True)


@wakeword_app.command("record")
def wakeword_record(
    name: str = typer.Option("jarvis_ko", help="Model name (letters, digits, _)."),
    phrases: str = typer.Option("자비스,헤이 자비스", help="Comma-separated wake phrases."),
    takes: int = typer.Option(12, help="Takes per phrase (more = better; 12+ recommended)."),
    sentences: bool = typer.Option(True, help="Also read a few ordinary sentences (things that must NOT wake it)."),
    background: float = typer.Option(60.0, help="Seconds of background audio to record (0 = skip)."),
) -> None:
    """Record your wake-word takes (and things that must not wake JARVIS) for `jarvis train wakeword`."""
    import json as _json
    import time as _time

    from jarvis.voice.wakeword_train import (
        READ_ALOUD,
        beep,
        record_loopback,
        record_mic,
        slug,
        take_problem,
        trim_speech,
        write_wav,
    )

    cfg = _cfg()
    d = cfg.paths().ensure().datasets / "wakeword" / name
    words = [x.strip() for x in phrases.split(",") if x.strip()]
    (d).mkdir(parents=True, exist_ok=True)
    (d / "phrases.json").write_text(_json.dumps(words, ensure_ascii=False), encoding="utf-8")
    dev = cfg.voice.input_device
    console.print(f"Recording into {d}. Speak naturally, at your normal distance from the mic.")
    for w in words:
        got = len(list((d / "pos").glob(f"{slug(w)}_*.wav"))) if (d / "pos").exists() else 0
        while got < takes:
            console.print(f"[cyan][{got + 1}/{takes}][/] after the beep, say: [bold]{w}[/]")
            beep()
            x = trim_speech(record_mic(2.5, dev))
            why = take_problem(x)
            if why:
                console.print(f"[yellow]{why}[/]")
                continue
            write_wav(d / "pos" / f"{slug(w)}_{got:03d}.wav", x)
            got += 1
    if sentences:
        console.print("Now read each sentence aloud (they must NOT wake JARVIS):")
        for i, line in enumerate(READ_ALOUD):
            console.print(f"[cyan]{i + 1}/{len(READ_ALOUD)}[/] {line}")
            beep()
            write_wav(d / "neg" / f"read_{i:02d}_{int(_time.time())}.wav", record_mic(5.0, dev))
    if background > 0:
        console.print(f"Play your game or a video with sound for {background:.0f} s (recording what the speakers play)...")
        bg = record_loopback(background)
        if bg is None:
            console.print("[yellow]no speaker loopback here - recording the room through the mic instead[/]")
            bg = record_mic(min(background, 30.0), dev)
        write_wav(d / "bg" / f"bg_{int(_time.time())}.wav", bg)
    console.print(f"Done. Next: [bold]jarvis train wakeword --name {name}[/]")


def register(app: typer.Typer) -> None:
    app.add_typer(wakeword_app, name="wakeword")
    app.command()(probe)
    app.command("api-key")(api_key)
    app.command()(chat)
    app.command()(say)
    app.command()(listen)
    app.command()(enroll)
    app.command()(voice)
    app.command()(core)
    app.command()(toolhost)
    app.command("voice-models")(voice_models)
    app.command("setup-voice")(setup_voice)
