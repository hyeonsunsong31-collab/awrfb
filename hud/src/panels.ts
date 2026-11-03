// DOM panels. Every string from the hub (transcripts, replies, notes) goes through textContent, never
// innerHTML: speech and model output are untrusted text.

import type { Bridge } from './bridge'
import { boxStyle, clamp01, fightStateLabel, pct, radarPoints, statsText, tacticLabel, threatLevel } from './format'
import type { Command } from './protocol'
import { type HudState, inCombat, orbMode, PANEL_IDS, type PanelId } from './state'

export class Panels {
  private interactive = false
  private streamMode = false
  private lastFrame = ''
  private lastLines = ''
  private state: HudState | null = null

  constructor(
    private readonly doc: Document,
    private readonly send: (c: Command) => number | null,
    private readonly bridge: Bridge,
  ) {
    const form = this.q<HTMLFormElement>('[data-k="form"]')
    const input = this.q<HTMLInputElement>('[data-k="input"]')
    form.addEventListener('submit', (e) => {
      e.preventDefault()
      const text = input.value.trim()
      if (text && this.send({ type: 'chat', text }) !== null) input.value = ''
    })
    for (const b of Array.from(doc.querySelectorAll<HTMLButtonElement>('[data-cmd]'))) {
      b.addEventListener('click', () => this.button(b.dataset.cmd ?? ''))
    }
    doc.addEventListener('keydown', (e) => {
      if (e.key === 'Escape') this.bridge.exitInteractive()
    })
  }

  private q<T extends Element>(sel: string): T {
    const el = this.doc.querySelector<T>(sel)
    if (!el) throw new Error(`HUD markup is missing ${sel}`)
    return el
  }

  private set(key: string, text: string): void {
    const el = this.doc.querySelector(`[data-k="${key}"]`)
    if (el && el.textContent !== text) el.textContent = text
  }

  private button(cmd: string): void {
    const s = this.state
    if (cmd === 'stop') this.send({ type: 'stop' })
    else if (cmd === 'disarm') this.send({ type: 'disarm' })
    else if (cmd === 'mute') this.send(s?.muted ? { type: 'unmute' } : { type: 'mute' })
    else if (cmd === 'interpreter') this.send({ type: 'interpreter', on: !s?.interpreter })
    else if (cmd === 'stream') {
      this.streamMode = !this.streamMode
      this.bridge.setStreamMode(this.streamMode)
      this.doc.body.classList.toggle('stream', this.streamMode)
    } else if (cmd === 'close') this.bridge.exitInteractive()
  }

  setInteractive(on: boolean): void {
    this.interactive = on
    this.doc.body.classList.toggle('interactive', on)
    if (on) this.q<HTMLInputElement>('[data-k="input"]').focus()
  }

  private show(id: PanelId, visible: boolean): void {
    this.doc.getElementById(`p-${id}`)?.classList.toggle('hidden', !visible)
  }

  render(s: HudState, now = performance.now()): void {
    this.state = s
    const combat = inCombat(s)
    const ko = s.lang === 'ko'
    for (const id of PANEL_IDS) {
      let visible = s.panels[id]
      if (id === 'chat') visible = visible && this.interactive
      if (id === 'pip') visible = visible && !!s.frame && (combat || s.fightState !== 'detached')
      if (id === 'fight') visible = visible && (combat || !!s.telemetry || s.fightState !== 'detached')
      this.show(id, visible)
    }

    const st = statsText(s.stats)
    this.set('cpu', st.cpu)
    this.set('ram', st.ram)
    this.set('gpu', st.gpu)
    this.set('vram', st.vram)
    this.set('link', s.authFailed ? (ko ? '인증 실패' : 'AUTH FAILED') : s.connected ? (ko ? '연결됨' : 'ONLINE') : ko ? '연결 끊김' : 'OFFLINE')
    this.set('latency', s.latencyS == null ? '—' : `${s.latencyS.toFixed(2)} s`)

    const t = s.telemetry
    this.set('state', fightStateLabel(s.fightState, s.lang))
    const bars: Record<string, number> = {
      own_hp: t?.hud?.own_hp ?? 0,
      opp_hp: t?.hud?.opp_hp ?? 0,
      own_stamina: t?.hud?.own_stamina ?? 0,
      own_guard: t?.hud?.own_guard ?? 0,
      threat: threatLevel(t?.threat),
    }
    for (const [k, v] of Object.entries(bars)) {
      const el = this.doc.querySelector<HTMLElement>(`[data-bar="${k}"]`)
      if (el) el.style.width = `${(clamp01(v) * 100).toFixed(1)}%`
    }
    this.set('own_hp_txt', pct(t?.hud?.own_hp))
    this.set('opp_hp_txt', pct(t?.hud?.opp_hp))
    this.set('tactic', tacticLabel(t?.tactic, s.lang))
    this.set('stocks', t ? `${t.match.won} : ${t.match.lost}` : '—')
    this.set('perf', t ? `${t.fps.toFixed(0)} fps / ${t.perception_ms.toFixed(1)} ms` : '—')
    this.q<SVGPolygonElement>('[data-k="radar"]').setAttribute('points', radarPoints(t?.opponent, 46, 60, 60))
    this.doc.body.classList.toggle('threat-high', threatLevel(t?.threat) > 0.6)

    if (s.frame && s.frame.src !== this.lastFrame) {
      this.lastFrame = s.frame.src
      this.q<HTMLImageElement>('[data-k="pip"]').src = s.frame.src
    }
    for (const who of ['opp', 'me'] as const) {
      const el = this.doc.querySelector<HTMLElement>(`[data-box="${who}"]`)
      const box = t?.[who]
      if (!el) continue
      el.classList.toggle('hidden', !box)
      if (box) Object.assign(el.style, boxStyle(box))
    }

    const sig = s.lines.map((l) => `${l.who}:${l.text}:${l.done}`).join('\n')
    if (sig !== this.lastLines) {
      this.lastLines = sig
      const ol = this.q<HTMLOListElement>('[data-k="lines"]')
      ol.replaceChildren(
        ...s.lines.map((l) => {
          const li = this.doc.createElement('li')
          li.className = `${l.who}${l.done ? '' : ' live'}${l.interrupted ? ' cut' : ''}`
          li.textContent = l.text
          return li
        }),
      )
    }

    const notes = this.q<HTMLElement>('#notes')
    notes.textContent = s.notes.slice(0, 2).join(' · ')
    notes.classList.toggle('hidden', s.notes.length === 0 || combat)

    const mode = orbMode(s, now)
    const label = (ko ? MODE_KO : MODE_EN)[mode] + (s.interpreter ? (ko ? ' · 통역' : ' · INTERPRETER') : '')
    this.set('mode', label)
    this.doc.body.dataset.mode = mode
    this.set('mute-label', s.muted ? (ko ? '음소거 해제' : 'UNMUTE') : ko ? '음소거' : 'MUTE')
  }
}

const MODE_EN = { offline: 'OFFLINE', idle: 'STANDING BY', listening: 'LISTENING', thinking: 'THINKING', speaking: 'SPEAKING', muted: 'MUTED', fighting: 'COMBAT', error: 'AUTH ERROR' }
const MODE_KO = { offline: '연결 끊김', idle: '대기 중', listening: '듣는 중', thinking: '생각 중', speaking: '말하는 중', muted: '음소거', fighting: '전투 모드', error: '인증 오류' }
