// Pure helpers for the main process (unit-tested without Electron).

export interface HudFileConfig {
  host: string
  port: number
  token_file: string
  hotkey: string
  content_protection: boolean
  stream_mode: boolean
  combat_mode: 'dim' | 'hide'
  combat_fps_cap: number
  prefer_secondary_monitor: boolean
}

export const DEFAULT_CONFIG: HudFileConfig = {
  host: '127.0.0.1',
  port: 47830,
  token_file: '',
  hotkey: 'ctrl+alt+j',
  content_protection: true,
  stream_mode: false,
  combat_mode: 'dim',
  combat_fps_cap: 30,
  prefer_secondary_monitor: true,
}

/** Validate the JSON written by `jarvis hud`; anything malformed falls back to the default. */
export function parseConfig(raw: unknown): HudFileConfig {
  const o = (raw && typeof raw === 'object' ? raw : {}) as Record<string, unknown>
  const pick = <K extends keyof HudFileConfig>(k: K, ok: (v: unknown) => boolean): HudFileConfig[K] =>
    (ok(o[k]) ? o[k] : DEFAULT_CONFIG[k]) as HudFileConfig[K]
  const isStr = (v: unknown) => typeof v === 'string' && v.length > 0
  const isBool = (v: unknown) => typeof v === 'boolean'
  return {
    host: pick('host', (v) => v === '127.0.0.1' || v === 'localhost'), // never a remote hub
    port: pick('port', (v) => Number.isInteger(v) && (v as number) > 0 && (v as number) < 65536),
    token_file: pick('token_file', isStr),
    hotkey: pick('hotkey', isStr),
    content_protection: pick('content_protection', isBool),
    stream_mode: pick('stream_mode', isBool),
    combat_mode: pick('combat_mode', (v) => v === 'dim' || v === 'hide'),
    combat_fps_cap: pick('combat_fps_cap', (v) => typeof v === 'number' && v >= 5 && v <= 144),
    prefer_secondary_monitor: pick('prefer_secondary_monitor', isBool),
  }
}

const KEY_NAMES: Record<string, string> = {
  ctrl: 'Control', control: 'Control', alt: 'Alt', shift: 'Shift', win: 'Super', super: 'Super', meta: 'Super',
  cmd: 'Command', esc: 'Escape', escape: 'Escape', space: 'Space', enter: 'Enter', return: 'Enter', tab: 'Tab',
  up: 'Up', down: 'Down', left: 'Left', right: 'Right', home: 'Home', end: 'End', pgup: 'PageUp', pgdn: 'PageDown',
  delete: 'Delete', del: 'Delete', insert: 'Insert', backspace: 'Backspace',
}

/** "ctrl+alt+j" (the Python config format) -> "Control+Alt+J" (Electron accelerator). */
export function toAccelerator(hotkey: string): string {
  return hotkey
    .split('+')
    .map((p) => p.trim().toLowerCase())
    .filter(Boolean)
    .map((p) => KEY_NAMES[p] ?? (/^f\d{1,2}$/.test(p) ? p.toUpperCase() : p.length === 1 ? p.toUpperCase() : p))
    .join('+')
}

export interface DisplayLike {
  id: number
  workArea: { x: number; y: number; width: number; height: number }
}

/** A secondary monitor keeps the overlay off the game screen entirely, when there is one. */
export function pickDisplay(displays: DisplayLike[], primaryId: number, preferSecondary: boolean): DisplayLike {
  const primary = displays.find((d) => d.id === primaryId) ?? displays[0]
  if (!preferSecondary) return primary
  return displays.find((d) => d.id !== primaryId) ?? primary
}

/** Interactive mode (clickable, focusable) is never allowed while a fight is active. */
export function nextInteractive(current: boolean, fightActive: boolean, requested?: boolean): boolean {
  if (fightActive) return false
  return requested ?? !current
}
