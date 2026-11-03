// Messages exchanged with core's HUD hub (see src/jarvis/hud_bridge/hub.py).

export interface HelloMsg {
  type: 'hello'
  snapshot: Record<string, unknown>
  history?: { topic: string; data: unknown }[]
}

export interface EventMsg {
  type: 'event'
  topic: string
  data: unknown
}

export interface ResultMsg {
  type: 'result'
  id: number
  ok: boolean
  error?: string
}

export type ServerMsg = HelloMsg | EventMsg | ResultMsg

export type Command =
  | { type: 'chat'; text: string; speak?: boolean }
  | { type: 'mute'; soft?: boolean }
  | { type: 'unmute' }
  | { type: 'stop' }
  | { type: 'disarm' }
  | { type: 'interpreter'; on: boolean }

export interface Telemetry {
  state: string
  gate_open: boolean
  gate: string
  hud: Record<string, number | null>
  combat_hud: boolean
  threat: Record<string, number>
  gap: number | null
  opp: [number, number, number, number] | null
  me: [number, number, number, number] | null
  fps: number
  perception_ms: number
  tactic: string | null
  opponent: Record<string, number>
  inputs: Record<string, number>
  match: { won: number; lost: number }
}

export interface Stats {
  cpu: number
  ram_used_gb: number
  ram_total_gb: number
  gpu: { name: string; util: number | null; vram_used_mb: number; vram_total_mb: number } | null
}

export function parseServerMsg(raw: string): ServerMsg | null {
  try {
    const m = JSON.parse(raw) as { type?: unknown }
    if (m && (m.type === 'hello' || m.type === 'event' || m.type === 'result')) return m as ServerMsg
  } catch {
    /* ignore malformed frames */
  }
  return null
}
