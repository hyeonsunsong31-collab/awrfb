# JARVIS

A Korean/English voice assistant for your Windows PC that talks, listens, controls the PC, shows a
holographic overlay, keeps learning in the background, and fights in the real Roblox game
**untitled boxing game** (UBG), in the real Roblox app, when you tell it to.

- **Talk:** speech to text, text to speech, voice conversation, typed chat, and a Korean↔English
  interpreter mode. Say "Hey Jarvis" to wake it, interrupt it at any time, mute it with a hotkey.
- **Act:** opens apps, manages files and windows, runs PowerShell, browses, and drives app
  interfaces. It acts on its own without asking, inside a safety floor that it cannot switch off
  ([SAFETY.md](SAFETY.md)).
- **See:** a transparent holographic HUD (an orb that reacts to voice, plus panels for chat,
  transcript, system stats and live fight telemetry). It never takes focus away from your game.
- **Fight:** on your spoken command it plays UBG PvP matches by watching the Roblox window and
  pressing real keys. It never spins and never buys anything.
- **Learn:** after every fight it labels the recording and retrains its threat model overnight.
  A new model is used only if it measures better. It also learns from UBG videos and keeps a
  knowledge base about the game.

> **빠른 시작 (한국어)** — 아래 [한국어 안내](#한국어-안내)를 보세요.

---

## Status: what works and what still needs your PC

This repository holds the complete system: the voice pipeline, LLM brain, PC tools, HUD, fighter,
learner and supervisor. 400+ automated tests run in CI on Linux and Windows. **Live behaviour can
only be checked on your Windows PC with Roblox.** This repository was built in a Linux container
with no Roblox, GPU, microphone or display.

| Area | Checked in development | Needs your PC |
|---|---|---|
| Fighting (capture, HUD reading, actuator, state machine, policy) | Unit tests; a simulated window that runs full matches; replayed recordings | Live private-server matches (sparring tests S1–S4 below) |
| Voice (wake word, VAD, interruption, mute, owner gate) | State machines on a fake clock; real Silero and openWakeWord models | Real microphone and speakers, Korean and English accents, timing on your PC |
| Brain (NVIDIA API, Ollama fallback, tools) | Fake models: rate limits, fallbacks, tool calls, prompt-injection red-team | Your API key and the models your account can use (`jarvis probe`) |
| HUD | Electron end-to-end test against a real hub; unit tests | Staying on top of Roblox without stealing focus; the FPS cost |
| Learning | Real torch → ONNX → onnxruntime training round trip; job kill when a fight starts | At least one model promoted from your own recorded fights |

**Be realistic about the fighter.** On day one it plays with hand-written rules (spacing, blocking,
stamina discipline, punishing whiffed heavies, reacting to heavies when there is time). It gets
better as it records fights. That means your own fights in shadow mode and its own fights, with
YouTube clips as a bonus. It does not start out as a pro, and we report its win rate rather than
promise one.

## ⚠️ Read before fighting

- UBG's rules name **auto-dodging as bannable**, and Roblox's terms prohibit automation that gives
  an unfair advantage. **Use an alt account.** Fight only in **private servers** with people who
  agreed to it. This is the default; public servers need an explicit opt-in that accepts the ban
  risk.
- Set up the bot account with **0 Robux, no payment method and a Parental Controls spending limit
  of 0**.
- **You sign in to Roblox yourself.** JARVIS never asks for, sees or stores your Roblox password.
- **Your NVIDIA API key goes into `jarvis api-key`** (Windows Credential Manager) or the
  `NVIDIA_API_KEY` environment variable. Never paste it into a chat, a file or an issue.

The full list of rules JARVIS enforces on itself is in [SAFETY.md](SAFETY.md).

## Requirements

- Windows 10 version 2004 or newer, or Windows 11 (64-bit)
- An NVIDIA RTX GPU with at least 6 GB of VRAM and a current driver
- The Roblox desktop app, plus an alt account for fighting
- A microphone. Headphones are recommended while gaming: echo cancellation handles speakers, but
  headphones are easier.
- An NVIDIA API key (free developer tier at build.nvidia.com), or [Ollama](https://ollama.com) for
  fully local use. Everything falls back to local models, and the fast commands (stop, mute, fight)
  never need the network.

How the work is split by GPU memory:

| VRAM | Speech to text | Voice | Local LLM fallback |
|---|---|---|---|
| 6 GB | Whisper large-v3-turbo (int8, GPU) | Supertonic (CPU); Edge-TTS cloud fallback | qwen3:4b (CPU) |
| 8 GB | same | Qwen3-TTS 0.6B (GPU, outside fights) | same |
| 12 GB+ | same | Qwen3-TTS 1.7B | qwen3:8b on the GPU outside fights |

`jarvis doctor --vram` measures this with Roblox running and picks the tier for you.

**During a fight, the GPU belongs to Roblox.** Free VRAM can still run short later, for example on a
heavy map or with a browser open, and Roblox's frame rate can drop. When either stays low, JARVIS steps
down one level at a time:

1. It unloads its GPU voice and any local LLM models, and speaks with the CPU voice and pre-rendered
   callouts.
2. It moves speech recognition to the CPU too.

Everything is restored a quiet minute after the fight, and the HUD tells you each step.

## Install

In PowerShell, from the repository folder:

```powershell
powershell -ExecutionPolicy Bypass -File scripts\install.ps1
```

The installer is safe to re-run. It:

1. Installs uv, Node.js LTS and FFmpeg with winget (and Ollama with `-WithOllama`).
2. Creates `.venv` with Python 3.12, PyTorch with CUDA, onnxruntime-gpu and every extra.
3. Builds the HUD.
4. Downloads the pinned voice models (SHA-256 verified) and exports the zero-shot fighter detector.
5. Asks for administrator rights **once**, to write the read-only safety policy to
   `%ProgramData%\Jarvis\safety.yaml`.
6. Registers a logon task (normal rights, not a service) so JARVIS starts when you log on.

Options: `-SkipModels`, `-SkipHud`, `-SkipDetector`, `-NoPolicy`, `-NoAutostart`,
`-CudaIndex <url>`.

## First run

```powershell
.venv\Scripts\jarvis setup
```

The wizard runs these steps one at a time. You can skip any of them and re-run it later on its own.

| Step | Command |
|---|---|
| Check the PC | `jarvis doctor` |
| NVIDIA API key, then test every model | `jarvis api-key`, `jarvis probe` |
| Voice models | `jarvis voice-models --all` |
| Microphone and speaker check | `jarvis listen`, `jarvis say "..."` |
| Your voiceprint (about 60 s, Korean and English), so only you can command fights and the PC | `jarvis enroll` |
| JARVIS's own original voice (8 GB+ cards) | `jarvis setup-voice` |
| Roblox checklist and launch | `jarvis launch` |
| Calibration during a private-server match: HUD bars, popups, keybinds, latency | `jarvis calibrate hud\|popup\|keys\|latency` |
| VRAM tier with Roblox running | `jarvis doctor --vram` |
| Start at logon | `jarvis autostart --enable` |
| Check everything, including your voice through the mic | `jarvis selftest --interactive` |

Then start it:

```powershell
.venv\Scripts\jarvis start      # or log off and on again
.venv\Scripts\jarvis status     # every process: running / restarting / crash-looping
```

## Talking to JARVIS

- **Wake it** with "Hey Jarvis" (헤이 자비스). After it answers you have 8 seconds to keep talking
  without the wake word.
- **Interrupt** by simply speaking. Its voice drops at once. A short "응", "네" or "yeah" lets it
  carry on. "Stop" or "멈춰" stops it. Anything else stops it and becomes your new request. Its
  memory keeps only the words you actually heard.
- **Mute:**
  - <kbd>Ctrl</kbd>+<kbd>Alt</kbd>+<kbd>M</kbd>, the HUD button, the tray icon, or saying "mute" /
    "음소거".
  - A hard mute closes the microphone, so the Windows mic indicator goes off. Undo it with the
    hotkey, HUD or tray.
  - A soft mute (`voice.mute.soft_mute`) keeps listening only for "unmute" in your voice.
- **Type** instead of talking: open the HUD chat with <kbd>Ctrl</kbd>+<kbd>Alt</kbd>+<kbd>J</kbd>, or
  run `jarvis chat "..."`.
- **Interpreter mode:** "통역 모드" / "interpreter mode" translates Korean↔English speech to speech.
  "통역 끝" / "stop translating" ends it.

Common commands. These are handled on the PC without an LLM, so they work offline:

| Intent | English | 한국어 | Who can say it |
|---|---|---|---|
| Stop everything | stop, abort, cancel, be quiet | 멈춰, 그만, 정지, 스톱, 조용히 | anyone (stopping is always safe) |
| Fight | fight, start the fight, fight 3 matches | 싸워, 대전 시작, 세 판 싸워, 알아서 싸워 (auto-queue) | owner only |
| Stop fighting | stop fighting, stand down | 싸움 멈춰, 전투 중지 | anyone |
| Launch UBG | launch UBG, open boxing game | 복싱 게임 켜, 언타이틀드 복싱 실행 | owner only |
| Fight status | fight status, match result | 전적, 경기 결과, 몇 대 몇 | anyone |
| Mute / unmute | mute / unmute | 음소거 / 음소거 해제 | anyone / owner only |
| Volume | volume up/down, louder, quieter | 소리 키워, 볼륨 내려 | anyone |
| Music | play/pause music, next song | 음악 틀어, 일시정지, 다음 곡 | anyone |
| HUD panels | show/hide the fight panel | 전투 패널 보여, 패널 숨겨 | anyone |
| Time | what time is it | 몇 시야 | anyone |

Everything else goes to the agent. For example: "메모장 열고 오늘 할 일 적어줘", "find the PDF I
downloaded yesterday and move it to Documents", "볼륨 30으로", "what's using my GPU?". Agent
requests also need your verified voice, or typed input at your keyboard.

## Fighting in untitled boxing game

1. **Once:**
   - Put your private server's link code in `%LOCALAPPDATA%\Jarvis\config.yaml`:

     ```yaml
     fight:
       game:
         private_server_link_code: "<your private server link code>"
     ```

   - Calibrate during a match: `jarvis calibrate hud`, then `popup`, `keys` and `latency`.
2. **Launch:** say "Hey Jarvis, 복싱 게임 켜" (or run `jarvis launch`). You must already be signed
   in to the Roblox app.
3. **Fight:** stand in a ring's blue box and say **"자비스, 싸워"** or **"Hey Jarvis, fight three
   matches"**. JARVIS arms, waits for the box to turn green and the combat HUD to appear, then
   fights. Add "알아서" / "auto" and it walks to a free ring box itself (WASD only).
4. **Stop at any time:**
   - "멈춰" / "stop" in any voice.
   - Panic hotkeys <kbd>Ctrl</kbd>+<kbd>Alt</kbd>+<kbd>End</kbd> or
     <kbd>Ctrl</kbd>+<kbd>Shift</kbd>+<kbd>F12</kbd>.
   - Or just **touch your keyboard or mouse**: it pauses at once so you can take over.

What it will and won't do in the game:

- **Inputs:** it presses only fight keys, and only while Roblox is in front and the combat HUD is on
  screen. It never presses Esc, Enter, Tab or `/`, never clicks the screen, never opens chat or
  menus, and never touches the reward wheel, the shop or spins.
- **Reacting:** light punches are too fast for anyone to react to on sight. It handles them with
  spacing, blocking up close and stamina discipline. It dodges on reaction only against heavies,
  and only when its measured latency leaves time.
- **Adapting:**
  - It watches each opponent's habits: damage taken in range, how its own punches land, whether
    they punish heavies, whether they retreat to regenerate, and their preferred range.
  - It switches between six tactic bundles with a bandit algorithm.
  - Between matches an LLM coach makes small adjustments within fixed limits.
- **Proof:** every match leaves a proof bundle in `%LOCALAPPDATA%\Jarvis\recordings\`: the captured
  video, every input with timestamps, the HUD readings and a summary.
  - `jarvis overlay <session>` draws what it saw and did over the video.
  - `jarvis report` zips recent sessions so you can send them back for tuning.
- **Teach it:** `jarvis shadow --record-my-keys` records your own matches (it sends no input). It
  learns from them overnight.

### Acceptance tests on your PC

First, with UBG open, run the automatic checks:

```powershell
.venv\Scripts\jarvis selftest --interactive --live
```

It checks the following and writes a report:

- the voice round trip in both languages, the wake word, and that a synthetic voice cannot pass as you;
- the safety policy and the audit log;
- the HUD window: always on top, click-through, never takes focus;
- real capture FPS and HUD reading;
- that a fight key submitted while disarmed is refused and nothing reaches Roblox.

Then run these in a private server with a willing partner. Send the `jarvis report` zip with the results.

| Test | Partner does | Pass |
|---|---|---|
| S1 | stands still | JARVIS takes all 3 stocks in ≥ 4 of 5 matches |
| S2 | holds block | breaks guard and takes ≥ 1 stock in ≥ 3 of 5 |
| S3 | throws only heavies | damage taken ≤ 60% of the no-defence baseline |
| S4 | spars freely, 5 matches | finishes unaided; win rate reported |
| Safety | Alt-Tab, open chat, press a panic key | zero inputs outside a fight or while Roblox is not in front |

### Things to check on your PC first

Each of these can change how the fighter behaves. Note what you find and include it with your
`jarvis report`:

1. Do key presses register in UBG, and does the mouse turn the camera?
2. Is there lock-on, and what do A and D do?
3. Which HUD bars exist (opponent HP, guard, stocks)?
4. The real keybinds for dodge and ultimate.
5. Ping, and the warning time a heavy gives.
6. VRAM use and Roblox FPS with everything running.
7. The popups you see.
8. Whether the cursor is locked during fights.

## Background learning

`learnerd` never touches the PC or the game. Fights always win: a learning job is ended the moment
a fight starts.

- **Right after a match:** hindsight labels (an HP drop marks when the opponent's hit landed) and a
  game-update drift check. If UBG changes its HUD and the reader's confidence collapses, JARVIS
  refuses to fight until you run `jarvis calibrate hud` again, and tells you why.
- **Nightly**, when the PC is idle and there are enough new fights:
  1. Retrain the threat model.
  2. Test it on held-out fights.
  3. Promote it only through the gate: AP@300 ms ≥ 0.75, better than the current model, and a
     warning time longer than your measured latency + 50 ms.
  
  `jarvis models rollback threat` undoes a promotion.
- **Videos** (opt-in: `learn.video_ingest.enabled: true`, plus `queries` and/or `local_folders`):
  - Public clips are downloaded with yt-dlp, with daily and disk limits; your local or OBS folders
    are read in place.
  - A clip is kept only if your calibrated HUD reader actually sees UBG's combat HUD in it.
  - Video labels stay only if they don't make the model worse on your own fights.
- **Knowledge base:** starts from researched notes, all marked unverified, and is refreshed weekly
  from the web. Pages are treated as untrusted data and every fact keeps its source.
- **Housekeeping:** memory consolidation and a skills library of tool sequences that worked before.
- **Controls:** `jarvis learn status`, `jarvis learn now nightly`, `jarvis learn cancel`.

## Commands

| Command | What it does |
|---|---|
| `jarvis start` / `stop` / `status` / `restart <proc>` | The whole system (the supervisor and its processes) |
| `jarvis panic` | Emergency stop from a terminal |
| `jarvis setup` | First-run wizard |
| `jarvis selftest [--interactive] [--live]` | Every check this PC can run, with a report (`--live`: Roblox open; nothing is sent to the game) |
| `jarvis doctor [--vram]` | Check the PC: GPU tier, Roblox, audio, models, API key, safety policy |
| `jarvis chat`, `say`, `listen`, `enroll`, `voice` | Voice and chat tools |
| `jarvis launch`, `fight`, `shadow`, `calibrate …` | Game |
| `jarvis replay`, `overlay`, `report` | Proof and tuning |
| `jarvis learn …`, `train …`, `models …`, `dataset …`, `labels`, `coach` | Learning |
| `jarvis hud [--demo]` | Run the HUD on its own (`--demo` needs no core and no Roblox) |
| `jarvis audit-verify` | Check the tamper-evident audit logs |
| `jarvis autostart --enable/--disable` | The logon task |

## Where things live

- `%LOCALAPPDATA%\Jarvis\` holds your data: `config.yaml` (your overrides), `recordings/`,
  `models/`, `calibration/`, `logs/`, `audit/`, `db/` (memory, skills), `kb/`.
- `configs/` holds the defaults (`default.yaml`, `brain.yaml`, `voice.yaml`, `tiers/`, `ubg/`).
  Override any value in your `config.yaml`, or with an environment variable such as
  `JARVIS__VOICE__STT__MODEL=small`.
- `%ProgramData%\Jarvis\safety.yaml` is the protected safety policy. It is read-only to JARVIS.

## Troubleshooting

| Symptom | Try |
|---|---|
| Anything | `jarvis doctor`, `jarvis status`, then the logs in `%LOCALAPPDATA%\Jarvis\logs\` (`<process>.log` and `<process>.out.log`) |
| A process keeps restarting | The tray and HUD say which one; read its `.out.log`; `jarvis restart <name>` after fixing |
| "I can't start the fight" | It says why: not a private server, HUD not calibrated, game update (drift), a disabled keybind |
| Slow or no replies | NVIDIA's free tier is rate-limited (about 40/min). JARVIS falls back to local models; `jarvis probe` shows what each model can do |
| CUDA or cuDNN DLL errors | Re-run `scripts\install.ps1`. It puts the CUDA runtime in the venv, and JARVIS adds it to the DLL search path at start |
| The HUD is missing | `jarvis hud` shows errors; it needs `npm ci` and `npm run build` in `hud/` (the installer does this) |
| It hears itself | Use headphones, or check that echo cancellation uses the loopback reference (`voice.aec`) |

## How it is built

```
supervisor (logon task, tray, 2nd panic hotkey, mute hotkey, event bus, restarts, GPU lending)
 ├─ core      voice pipeline · LLM broker (NVIDIA → Ollama) · agent · memory · HUD hub
 ├─ toolhost  the only process that runs PC tools; safety policy; Job objects; audit log
 ├─ fightd    the only process that sends input to Roblox; capture → perception → policy → actuator
 ├─ learnerd  background learning; jobs as killable child processes; no tool access
 └─ hud       Electron + Three.js overlay (transparent, click-through, never focused)
```

- **Messages:** processes talk over ZeroMQ with msgspec messages. RPC is token-authenticated;
  events travel on a bus with a `kill` topic every process obeys.
- **The game:**
  - fightd sees the Roblox window through Windows Graphics Capture and presses real keys with
    Win32 `SendInput`.
  - It never injects into Roblox, never reads its memory, never runs scripts, never evades
    anti-cheat and never adds fake "human" noise.

## Development

```bash
uv venv && uv pip install -e ".[dev]"
uv run ruff check src tests
uv run pytest -m "not windows and not gpu and not live"
```

```bash
cd hud && npm ci && npm run typecheck && npm test && npm run build
```

- `tests/fakes` holds the test doubles (simulated game window, fake LLM, fake platform). The runtime
  refuses to load them unless `JARVIS_TEST=1`, so a fake can never stand in for the real game.
- CI runs Linux and Windows Python tests plus the HUD tests on every push.

## Uninstall

```powershell
.venv\Scripts\jarvis stop
powershell -ExecutionPolicy Bypass -File scripts\unregister_autostart.ps1
```

Then delete the repository folder, `%LOCALAPPDATA%\Jarvis` (your recordings, models and memory),
and, as administrator, `%ProgramData%\Jarvis`.

---

## 한국어 안내

**JARVIS**는 한국어와 영어로 대화하고, PC를 제어하고, 홀로그램 HUD를 띄우고, 백그라운드에서
계속 학습하며, 명령하면 실제 로블록스 앱에서 **untitled boxing game**으로 PvP 대전을 하는
음성 비서입니다.

**설치**

```powershell
powershell -ExecutionPolicy Bypass -File scripts\install.ps1
.venv\Scripts\jarvis setup     # API 키, 목소리 등록, 로블록스 점검, 보정
.venv\Scripts\jarvis start
```

**사용법**

- **부르기:** "헤이 자비스". 대답 뒤 8초 동안은 다시 부르지 않고 이어서 말할 수 있습니다.
- **끼어들기:** 그냥 말하면 바로 목소리를 줄이고 들어 줍니다. "멈춰"라고 하면 멈춥니다.
- **음소거:** <kbd>Ctrl</kbd>+<kbd>Alt</kbd>+<kbd>M</kbd>, HUD 버튼, 트레이, "음소거".
  - 하드 음소거는 마이크를 완전히 끄므로 단축키, HUD, 트레이로 해제합니다.
  - 소프트 음소거(`voice.mute.soft_mute`)는 주인 목소리의 "음소거 해제"만 듣습니다.
- **대전:**
  - 비공개 서버에서 링의 파란 칸에 서서 "자비스, 싸워" 또는 "세 판 싸워"라고 하세요.
  - "알아서 싸워"라고 하면 빈 링까지 걸어갑니다.
  - 멈추려면 "멈춰"(누구 목소리든), <kbd>Ctrl</kbd>+<kbd>Alt</kbd>+<kbd>End</kbd>, 또는 키보드나
    마우스를 만지세요. 즉시 멈추고 조작을 넘겨줍니다.
- **채팅:** <kbd>Ctrl</kbd>+<kbd>Alt</kbd>+<kbd>J</kbd>로 HUD 채팅을 열거나 `jarvis chat "..."`.
- **통역:** "통역 모드" / "통역 끝".

**꼭 지켜 주세요**

- **부계정:** 자동 회피는 UBG 규칙상 밴 사유이고, 로블록스도 불공정한 자동화를 금지합니다.
  반드시 부계정으로, 동의한 사람과 비공개 서버에서만 대전하세요. 기본값이 비공개 서버
  전용입니다.
- **계정 보호:** 봇 계정은 로벅스 0, 결제 수단 없음, 자녀 보호 지출 한도 0으로 설정하세요.
- **로그인:** 로블록스 로그인은 직접 하세요. JARVIS는 비밀번호를 묻거나 보거나 저장하지
  않습니다.
- **API 키:** NVIDIA API 키는 `jarvis api-key`로만 입력하세요. 채팅이나 파일에 붙여 넣지 마세요.
- **스핀·구매 금지:** JARVIS는 스핀이나 구매를 절대 하지 않습니다.

**솔직한 현재 상태**

- **검증된 것:** 모든 코드는 자동 테스트와 시뮬레이션으로 검증했습니다.
- **아직 검증되지 않은 것:** 실제 로블록스 대전, 실제 마이크와 음성은 이 PC에서 처음
  확인하게 됩니다.
- **실력:** 첫날의 전투 실력은 규칙 기반이라 평범합니다. 녹화된 대전이 쌓일수록 밤마다
  재학습하며, 측정상 더 나아진 모델만 적용합니다.
- **결과 보내기:** UBG를 켠 채 `jarvis selftest --interactive --live`를 먼저 실행하세요. 그다음 위의 S1–S4 테스트 결과와 `jarvis report` 파일을 보내 주시면 조정하겠습니다.
