// Pure formatting for the panels (unit-tested).

import type { Stats } from './protocol'

export const RADAR_AXES = ['aggression', 'defense', 'punish', 'range_pref', 'regen'] as const

export const clamp01 = (x: number): number => Math.max(0, Math.min(1, x))

export function pct(x: number | null | undefined): string {
  return x == null || !Number.isFinite(x) ? '—' : `${Math.round(clamp01(x) * 100)}%`
}

/** The threat bar shows the most urgent short-horizon risk (200-300 ms ahead). */
export function threatLevel(threat: Record<string, number> | null | undefined): number {
  if (!threat) return 0
  const vals = ['200', '300'].map((k) => threat[k]).filter((v): v is number => typeof v === 'number')
  return clamp01(vals.length ? Math.max(...vals) : Math.max(0, ...Object.values(threat)))
}

export function radarPoints(profile: Record<string, number> | null | undefined, r: number, cx: number, cy: number): string {
  return RADAR_AXES.map((axis, i) => {
    const a = -Math.PI / 2 + (i * 2 * Math.PI) / RADAR_AXES.length
    const v = clamp01(profile?.[axis] ?? 0) * r
    return `${(cx + Math.cos(a) * v).toFixed(1)},${(cy + Math.sin(a) * v).toFixed(1)}`
  }).join(' ')
}

const TACTICS: Record<string, [string, string]> = {
  baseline: ['Baseline', '기본'],
  pressure: ['Pressure', '압박'],
  counter: ['Counter-punch', '카운터'],
  bait_punish: ['Bait & punish', '유인 후 반격'],
  keep_range: ['Keep range', '거리 유지'],
  feint_mix: ['Feint mix-up', '페인트 섞기'],
  regen_retreat: ['Retreat & regen', '후퇴 후 회복'],
}

export function tacticLabel(tactic: string | null | undefined, lang: string): string {
  if (!tactic) return '—'
  const t = TACTICS[tactic]
  return t ? t[lang === 'ko' ? 1 : 0] : tactic
}

export function statsText(s: Stats | null): { cpu: string; ram: string; gpu: string; vram: string } {
  if (!s) return { cpu: '—', ram: '—', gpu: '—', vram: '—' }
  const g = s.gpu
  return {
    cpu: `${Math.round(s.cpu)}%`,
    ram: `${s.ram_used_gb.toFixed(1)} / ${s.ram_total_gb.toFixed(0)} GB`,
    gpu: g ? (g.util == null ? g.name : `${g.util}%`) : 'n/a',
    vram: g ? `${(g.vram_used_mb / 1024).toFixed(1)} / ${(g.vram_total_mb / 1024).toFixed(0)} GB` : 'n/a',
  }
}

/** Normalized [cx, cy, w, h] (0..1 of the frame) -> CSS percentages for an overlay box. */
export function boxStyle(box: [number, number, number, number]): { left: string; top: string; width: string; height: string } {
  const [cx, cy, w, h] = box
  return {
    left: `${((cx - w / 2) * 100).toFixed(2)}%`,
    top: `${((cy - h / 2) * 100).toFixed(2)}%`,
    width: `${(w * 100).toFixed(2)}%`,
    height: `${(h * 100).toFixed(2)}%`,
  }
}

const STATE_LABELS: Record<string, [string, string]> = {
  detached: ['Not connected', '연결 안 됨'],
  disarmed: ['Disarmed', '대기'],
  armed: ['Armed', '전투 준비'],
  navigating: ['To the ring', '링으로 이동'],
  queued: ['Queued', '경기 대기'],
  fighting: ['FIGHTING', '전투 중'],
  post_match: ['Match over', '경기 종료'],
  calibrating: ['Calibrating', '보정 중'],
}

export function fightStateLabel(state: string, lang: string): string {
  const s = STATE_LABELS[state]
  return s ? s[lang === 'ko' ? 1 : 0] : state
}
