# JARVIS safety

JARVIS acts on its own and never asks for confirmation. You chose that. In exchange, a **safety
floor** sits outside the language model, so a confused, jailbroken or prompt-injected model still
cannot cross it. This document lists every rule and where it is enforced, so you can check it.

## 1. Stopping is always possible, and always allowed

**Ways to stop.** Any of these stops everything at once:

- "stop", "멈춰", "그만", "정지" **in any voice**. Stopping is always safe, so it needs no voiceprint.
- The panic hotkeys <kbd>Ctrl</kbd>+<kbd>Alt</kbd>+<kbd>End</kbd> (registered by fightd) and
  <kbd>Ctrl</kbd>+<kbd>Shift</kbd>+<kbd>F12</kbd> (registered by the supervisor). These are two
  independent paths: if one process hangs, the other still works.
- The tray icon's **PANIC STOP**, or `jarvis panic` in a terminal.

**What a stop does.** It reaches every process over a `kill` broadcast:

- It cancels the agent, ends running tool process trees through Windows Job objects, stops speech
  and disarms the fighter.
- If fightd doesn't answer, the supervisor kills it and releases the keys itself.

**No stuck keys.** There are three layers:

1. fightd releases every key it holds whenever it disarms, pauses or exits.
2. The actuator needs a heartbeat. With no heartbeat for 500 ms it releases everything and disarms.
3. If fightd dies, the supervisor sends key-up for every key and mouse button the keymap could hold.

**You take over instantly.** Any physical keyboard or mouse input from you (as opposed to
injected input) pauses the fighter at once.

## 2. Who may command what

- **Your voiceprint gates the dangerous commands:**
  - These need your **verified voice** (ECAPA speaker verification over at least 1.5 s of audio)
    or typed input at your keyboard: starting a fight, launching the game, unmuting, interpreter
    mode, and every PC request.
  - A short command must pass both on its own and together with the wake word you just said. That
    way a TV saying "fight" cannot ride on your earlier speech.
  - TV, YouTube or game audio can never start anything.
- **Harmless commands are open to anyone:** stop, mute, volume, music, HUD panels, time and fight
  status.
- **Starting a fight** comes only from an owner-verified command or your own hotkey or terminal. It
  can never come from an LLM tool call: the agent has no tool that arms the fighter.

## 3. What the PC agent can never do

All tools run in **toolhost**, the only process allowed to run them. toolhost enforces these rules
in code before every call and logs every call to the audit log.

- **Task-scoped capabilities:**
  - Your verified request is sorted into capability classes: files, shell, apps, GUI, web,
    system, and so on.
  - Each tool call must fit that scope.
  - Text that arrives from web pages, files, the clipboard or the screen (OCR) is wrapped and
    marked as untrusted data. It can never widen the scope or issue commands. This blunts prompt
    injection without ever asking you.
- **A protected policy file:**
  - The installer writes `%ProgramData%\Jarvis\safety.yaml` once, with one admin prompt.
  - It is read-only to your user account, and JARVIS runs with your normal rights, so it cannot
    loosen its own rules. `jarvis doctor` shows whether the file is protected.
- **Files:**
  - Deletes go to the Recycle Bin. Nothing is permanently deleted.
  - Deleting, moving or overwriting OS folders, Program Files, your profile root, `ProgramData`,
    or JARVIS's own audit log and policy is refused.
  - Paths are checked as written, as absolute paths and with links resolved, so tricks like `..`
    and symlinks don't work.
- **PowerShell:**
  - Every command is **parsed** with PowerShell's own parser, not pattern-matched.
  - Refused:
    - encoded or Base64-decoded commands, `Invoke-Expression`, and download-and-run;
    - formatting or partitioning disks;
    - changing Defender settings or stopping its service, and turning the firewall off;
    - editing the boot configuration, and deleting shadow copies or backups;
    - deleting machine-wide registry keys, and clearing the Recycle Bin or event logs;
    - shell deletes (deletes go through the Recycle Bin tool instead);
    - anything that mentions Roblox.
- **No money:**
  - The GUI tools refuse buttons and links such as buy, pay, checkout, purchase, subscribe, 결제,
    구매 and 충전.
  - Typed text that looks like a card number is refused.
  - JARVIS never enters payment details anywhere.
- **Protected processes:** the agent cannot kill JARVIS's own processes, Windows' critical
  processes or Roblox.
- **Roblox is off limits to the PC tools:**
  - No tool may type into, click, focus, close, or send media keys to the Roblox window.
  - Only fightd talks to the game, and only under the rules below.

## 4. Rules for the game

**Real inputs only.** JARVIS sees the game through Windows Graphics Capture and presses keys with
Win32 `SendInput`, exactly as a USB keyboard would. It **never**:

- injects code into Roblox, reads or writes its memory, or runs executor scripts;
- bypasses, evades or probes anti-cheat;
- adds fake "human" timing to hide that it is a bot.

**Fight only on command.** fightd arms only on an owner-verified command or your own hotkey or
terminal, and it disarms after the number of matches you asked for.

**Per-state input allowlist.** The actuator enforces it:

| State | Allowed input |
|---|---|
| Disarmed, armed, queued, post-match | nothing (the reward wheel is never touched) |
| Navigating to a ring box | W A S D only |
| Fighting | movement, dash, block, light and heavy punches, feints, and the dodge or ability keys you calibrated |

**Never sent, in any state:** Esc, Enter, Tab, `/`, chat keys, or any click at a screen position.
JARVIS never spins, never buys anything, and never opens menus or the shop.

**Checked before every single input:**

- Roblox is the foreground window.
- The combat HUD is confirmed in the current frame.
- No known popup is on screen: purchase prompt, shop, spin, wheel or menu.
- The cursor is hidden and the chat box is closed.

If any check fails, JARVIS releases all keys and pauses.

**Servers:**

- Private servers only, by default. Public servers need two settings together:
  `fight.game.allowed_servers: public` and `fight.game.i_accept_ban_risk: true`.
- Ranked queues are not supported.
- **Ban risk:** UBG's rules name auto-dodging as bannable, and Roblox prohibits automation that
  gives an unfair advantage. Use an alt account, and fight only with people who agreed to it.
- **Account safety floor (you set this):** 0 Robux, no payment method, and a Parental Controls
  spending limit of 0 on the bot account.

**Stale settings mean no fight.**

- **Game updates:** if UBG updates its interface and HUD-reading confidence collapses, JARVIS
  refuses to fight until you recalibrate, and tells you why.
- **Setup changes:** a changed window size or aspect ratio, HUD layout or keymap since calibration
  also blocks fighting until you recalibrate.

**Proof, not claims:**

- Every match writes a proof bundle: the captured video, a timestamped log of every input sent,
  and the HUD readings.
- Spoken commentary is generated from that telemetry, so JARVIS can only describe what actually
  happened.
- Test stand-ins (simulated game windows) cannot load outside the test suite.

## 5. Background learning

- **No access:** learnerd has no tool access. It cannot touch the PC or the game.
- **Fights win the GPU:** a running training job is ended the moment a fight starts. The supervisor
  enforces this too.
- **New models must earn their place:** a model is used only after it passes a measured gate on
  fights it never trained on. Promotions are recorded and can be rolled back
  (`jarvis models rollback threat`).
- **Untrusted sources stay labelled:**
  - Web pages are read as untrusted data. Extracted facts are marked unverified and keep their
    source.
  - Video ingestion is off by default. When on, it downloads public clips only, with daily and disk
    limits, and signs in to nothing.

## 6. Your data and privacy

**Stays on your PC:**

- Microphone audio is transcribed locally (Whisper). Your voiceprint is stored locally.
- Recordings, memory, the knowledge base, calibration and logs are all under
  `%LOCALAPPDATA%\Jarvis`.
- Recordings capture only the Roblox window, but other players' avatars and names may be visible
  in them.

**Leaves your PC, only when you use these features:**

- **Language model requests** go to the NVIDIA API: what you said, the conversation and tool
  results, and screenshots when the screen tool is used. With Ollama, they stay local.
- **The Edge-TTS fallback voice** sends the reply text to Microsoft's speech service. It is only
  used when the local voices are unavailable.
- **Web search and fetch** contact those websites.

**Secrets:**

- The NVIDIA API key lives in Windows Credential Manager (`jarvis api-key`) or an environment
  variable. It is never printed, logged or stored in a file by JARVIS.
- Your Roblox password is never requested or seen. You sign in to the Roblox app yourself.

**Audit log:** every tool call and every game input is appended to a hash-chained log in
`%LOCALAPPDATA%\Jarvis\audit\`. Secrets are redacted before writing. `jarvis audit-verify` detects
any edit or deletion.

## 7. If something goes wrong

1. **Stop:** say "멈춰" or "stop", or press <kbd>Ctrl</kbd>+<kbd>Alt</kbd>+<kbd>End</kbd>.
2. **Check:** `jarvis status` and `jarvis audit-verify`, then the logs in
   `%LOCALAPPDATA%\Jarvis\logs\`.
3. **Report:** `jarvis report` zips recent proof bundles, calibration and logs. It never includes
   your API key, because the key is not in any file.
