// Pure orb parameters per mode (unit-tested; the WebGL side lives in orb.ts).

import type { OrbMode } from './state'

export interface OrbStyle {
  color: [number, number, number]
  glow: number
  speed: number
  disp: number
  ring: number
}

export const STYLES: Record<OrbMode, OrbStyle> = {
  offline: { color: [0.45, 0.5, 0.55], glow: 0.35, speed: 0.05, disp: 0.02, ring: 0.1 },
  idle: { color: [0.21, 0.84, 1.0], glow: 0.8, speed: 0.15, disp: 0.04, ring: 0.3 },
  listening: { color: [0.55, 0.95, 1.0], glow: 1.15, speed: 0.35, disp: 0.07, ring: 0.8 },
  thinking: { color: [1.0, 0.7, 0.28], glow: 1.0, speed: 0.9, disp: 0.09, ring: 1.6 },
  speaking: { color: [0.3, 0.9, 1.0], glow: 1.1, speed: 0.45, disp: 0.06, ring: 0.9 },
  muted: { color: [0.75, 0.3, 0.32], glow: 0.45, speed: 0.08, disp: 0.02, ring: 0.15 },
  fighting: { color: [1.0, 0.33, 0.18], glow: 0.9, speed: 0.6, disp: 0.05, ring: 1.2 },
  error: { color: [1.0, 0.15, 0.2], glow: 0.9, speed: 0.1, disp: 0.03, ring: 0.2 },
}

const lerp = (a: number, b: number, t: number) => a + (b - a) * t

export function lerpStyle(a: OrbStyle, b: OrbStyle, t: number): OrbStyle {
  return {
    color: [lerp(a.color[0], b.color[0], t), lerp(a.color[1], b.color[1], t), lerp(a.color[2], b.color[2], t)],
    glow: lerp(a.glow, b.glow, t),
    speed: lerp(a.speed, b.speed, t),
    disp: lerp(a.disp, b.disp, t),
    ring: lerp(a.ring, b.ring, t),
  }
}

/** How strongly the audio drives the orb: JARVIS's own voice while speaking, the mic while listening. */
export function audioLevel(mode: OrbMode, level: { mic: number; out: number }): number {
  if (mode === 'speaking') return Math.min(1, level.out * 4)
  if (mode === 'listening') return Math.min(1, level.mic * 6)
  return 0
}

export function displacement(style: OrbStyle, level: number): number {
  return style.disp + Math.max(0, Math.min(1, level)) * 0.35
}

/** Idle modes render at 30 fps, active ones at 60; combat uses its own (lower) cap. */
export function fpsFor(mode: OrbMode, combat: boolean, combatCap: number): number {
  if (combat) return combatCap
  return mode === 'speaking' || mode === 'listening' || mode === 'thinking' ? 60 : 30
}

export class FpsLimiter {
  private last = -Infinity
  constructor(public cap: number) {}

  ready(now: number): boolean {
    if (now - this.last >= 1000 / this.cap - 0.5) {
      this.last = now
      return true
    }
    return false
  }
}
