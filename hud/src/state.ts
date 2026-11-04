// HUD state: a pure reducer over hub messages (unit-tested), plus the derived orb mode.

import type { ServerMsg, Stats, Telemetry } from './protocol'

export type OrbMode = 'offline' | 'idle' | 'listening' | 'thinking' | 'speaking' | 'muted' | 'fighting' | 'error'
export const PANEL_IDS = ['stats', 'fight', 'transcript', 'chat', 'pip'] as const
export type PanelId = (typeof PANEL_IDS)[number]

export const ACTIVE_FIGHT = new Set(['armed', 'navigating', 'queued', 'fighting', 'post_match', 'calibrating'])
export const MAX_LINES = 12
const WAKE_FLASH_MS = 2500

export interface Line {
  who: 'you' | 'jarvis' | 'system'
  text: string
  turn?: number
  done: boolean
  interrupted?: boolean
}

export interface HudState {
  connected: boolean
  authFailed: boolean
  voiceMode: string
  muted: string | null
  interpreter: boolean
  lang: string
  wakeUntil: number
  level: { mic: number; out: number }
  fightState: string
  telemetry: Telemetry | null
  frame: { src: string; w: number; h: number } | null
  stats: Stats | null
  lines: Line[]
  notes: string[]
  latencyS: number | null
  lastCallout: string | null
  learning: string | null
  panels: Record<PanelId, boolean>
}

export type ConnectionMsg = { type: 'connection'; connected: boolean; authFailed?: boolean }

export function initialState(): HudState {
  return {
    connected: false,
    authFailed: false,
    voiceMode: 'idle',
    muted: null,
    interpreter: false,
    lang: 'en',
    wakeUntil: 0,
    level: { mic: 0, out: 0 },
    fightState: 'detached',
    telemetry: null,
    frame: null,
    stats: null,
    lines: [],
    notes: [],
    latencyS: null,
    lastCallout: null,
    learning: null,
    panels: { stats: true, fight: true, transcript: true, chat: true, pip: true },
  }
}

type Obj = Record<string, unknown>
const obj = (v: unknown): Obj => (v && typeof v === 'object' ? (v as Obj) : {})
const str = (v: unknown, d = ''): string => (typeof v === 'string' ? v : d)
const num = (v: unknown, d = 0): number => (typeof v === 'number' && Number.isFinite(v) ? v : d)

function pushLine(lines: Line[], line: Line): Line[] {
  const out = [...lines, line]
  return out.length > MAX_LINES ? out.slice(out.length - MAX_LINES) : out
}

export function applyEvent(s: HudState, topic: string, data: unknown, now: number): HudState {
  const d = obj(data)
  switch (topic) {
    case 'voice.state':
      return { ...s, voiceMode: str(d.mode, s.voiceMode), muted: (d.muted as string | null) ?? null, interpreter: !!d.interpreter, lang: str(d.lang, s.lang) }
    case 'voice.wake':
      return { ...s, wakeUntil: now + WAKE_FLASH_MS }
    case 'voice.level':
      return { ...s, level: { mic: num(d.mic), out: num(d.out) } }
    case 'voice.transcript':
      return str(d.text) ? { ...s, lines: pushLine(s.lines, { who: 'you', text: str(d.text), done: true }) } : s
    case 'voice.reply': {
      const turn = num(d.turn, -1)
      const delta = str(d.delta)
      const i = s.lines.findLastIndex((l) => l.who === 'jarvis' && l.turn === turn && !l.done)
      if (i >= 0) {
        const lines = s.lines.slice()
        lines[i] = { ...lines[i], text: lines[i].text + delta }
        return { ...s, lines }
      }
      return { ...s, lines: pushLine(s.lines, { who: 'jarvis', text: delta, turn, done: false }) }
    }
    case 'voice.reply_done': {
      const turn = num(d.turn, -1)
      const heard = typeof d.heard === 'string' && d.heard ? d.heard : null
      const i = s.lines.findLastIndex((l) => l.who === 'jarvis' && l.turn === turn)
      if (i < 0) {
        // No streamed line (history replay after a HUD restart, or a silent reply): show the final text.
        const text = heard ? `${heard} …` : str(d.text)
        return text ? { ...s, lines: pushLine(s.lines, { who: 'jarvis', text, turn, done: true, interrupted: !!heard }) } : s
      }
      const lines = s.lines.slice()
      lines[i] = heard ? { ...lines[i], text: `${heard} …`, done: true, interrupted: true } : { ...lines[i], done: true }
      return { ...s, lines }
    }
    case 'voice.latency':
      return { ...s, latencyS: typeof d.first_audio_s === 'number' ? d.first_audio_s : s.latencyS }
    case 'voice.rejected':
      return { ...s, lines: pushLine(s.lines, { who: 'system', text: s.lang === 'ko' ? '주인님 목소리가 아니라서 무시했습니다.' : "Ignored: not the owner's voice.", done: true }) }
    case 'voice.notes':
      return { ...s, notes: Array.isArray(d.notes) ? d.notes.map(String) : [] }
    case 'fight.state':
      return { ...s, fightState: str(d.new, s.fightState) }
    case 'fight.telemetry':
      return { ...s, telemetry: data as Telemetry, fightState: str(d.state, s.fightState) }
    case 'fight.frame':
      return str(d.jpg) ? { ...s, frame: { src: `data:image/jpeg;base64,${str(d.jpg)}`, w: num(d.w, 320), h: num(d.h, 180) } } : s
    case 'fight.callout': {
      const key = str(d.key)
      if (key === 'match_end') {
        const text = s.lang === 'ko' ? `경기 종료 — 스톡 ${num(d.stocks_won)}:${num(d.stocks_lost)}` : `Match over — stocks ${num(d.stocks_won)}:${num(d.stocks_lost)}`
        return { ...s, lastCallout: key, lines: pushLine(s.lines, { who: 'system', text, done: true }) }
      }
      return { ...s, lastCallout: key }
    }
    case 'fight.panic':
      return { ...s, lastCallout: 'panic', lines: pushLine(s.lines, { who: 'system', text: s.lang === 'ko' ? '긴급 정지' : 'Emergency stop', done: true }) }
    case 'system.stats':
      return { ...s, stats: data as Stats }
    case 'fight.coach':
      return str(d.comment) ? { ...s, lines: pushLine(s.lines, { who: 'system', text: `${s.lang === 'ko' ? '코치' : 'Coach'}: ${str(d.comment)}`, done: true }) } : s
    case 'learn.status':
      return { ...s, learning: str(d.running) || null }
    case 'learn.result': {
      const next = { ...s, learning: null }
      if (d.job === 'nightly' && d.promoted === true) {
        const ap = typeof d.ap_300 === 'number' ? ` (AP@300 ${d.ap_300.toFixed(2)})` : ''
        const text = s.lang === 'ko' ? `새 위협 예측 모델 적용됨${ap}` : `New threat model promoted${ap}`
        return { ...next, lines: pushLine(s.lines, { who: 'system', text, done: true }) }
      }
      return next
    }
    case 'learn.drift': {
      const why = str(d.reason)
      const text = s.lang === 'ko' ? `HUD 재보정 필요 — ${why}` : `HUD needs recalibration — ${why}`
      return { ...s, lines: pushLine(s.lines, { who: 'system', text, done: true }) }
    }
    case 'sys.notice':
    case 'hud.notify':
      return str(d.text) ? { ...s, lines: pushLine(s.lines, { who: 'system', text: str(d.text), done: true }) } : s
    case 'hud.command': {
      const show = d.action === 'show'
      const panel = str(d.panel, 'all')
      const panels = { ...s.panels }
      for (const id of PANEL_IDS) if (panel === 'all' || panel === id) panels[id] = show
      return { ...s, panels }
    }
    default:
      return s
  }
}

export function reduce(s: HudState, msg: ServerMsg | ConnectionMsg, now: number): HudState {
  switch (msg.type) {
    case 'connection':
      return { ...s, connected: msg.connected, authFailed: !!msg.authFailed }
    case 'hello': {
      let next: HudState = { ...s, connected: true, authFailed: false, lines: [] }
      for (const h of msg.history ?? []) next = applyEvent(next, h.topic, h.data, now)
      for (const [topic, data] of Object.entries(msg.snapshot ?? {})) next = applyEvent(next, topic, data, now)
      return next
    }
    case 'event':
      return applyEvent(s, msg.topic, msg.data, now)
    default:
      return s
  }
}

export function inCombat(s: HudState): boolean {
  return ACTIVE_FIGHT.has(s.fightState)
}

export function orbMode(s: HudState, now: number): OrbMode {
  if (s.authFailed) return 'error'
  if (!s.connected) return 'offline'
  if (s.muted === 'hard') return 'muted'
  if (s.voiceMode === 'speaking' || s.voiceMode === 'thinking' || s.voiceMode === 'listening') return s.voiceMode
  if (now < s.wakeUntil) return 'listening'
  if (inCombat(s)) return 'fighting'
  if (s.muted === 'soft') return 'muted'
  return 'idle'
}
