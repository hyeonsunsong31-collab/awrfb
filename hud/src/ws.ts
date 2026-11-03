// WebSocket client for core's HUD hub: token auth as the first message, exponential-backoff reconnect,
// and no retries after an authentication failure (close code 4401).

import { parseServerMsg, type Command, type ServerMsg } from './protocol'

export interface SocketLike {
  readyState: number
  send(data: string): void
  close(code?: number): void
  onopen: ((ev: unknown) => void) | null
  onmessage: ((ev: { data: unknown }) => void) | null
  onclose: ((ev: { code: number }) => void) | null
  onerror: ((ev: unknown) => void) | null
}

export type SocketFactory = (url: string) => SocketLike

export interface Timers {
  setTimeout(fn: () => void, ms: number): unknown
  clearTimeout(h: unknown): void
}

export const AUTH_FAILED = 4401
const OPEN = 1

export function backoffMs(attempt: number): number {
  return Math.min(8000, 500 * 2 ** attempt)
}

export class HubClient {
  onMessage: (m: ServerMsg) => void = () => {}
  onConnection: (connected: boolean, authFailed: boolean) => void = () => {}

  private sock: SocketLike | null = null
  private attempt = 0
  private stopped = true
  private timer: unknown = null
  private nextId = 1
  private authed = false

  constructor(
    private readonly url: string,
    private readonly token: string,
    private readonly factory: SocketFactory = (u) => new WebSocket(u) as unknown as SocketLike,
    private readonly timers: Timers = { setTimeout: (f, ms) => setTimeout(f, ms), clearTimeout: (h) => clearTimeout(h as number) },
  ) {}

  get connected(): boolean {
    return this.authed
  }

  start(): void {
    this.stopped = false
    this.connect()
  }

  stop(): void {
    this.stopped = true
    if (this.timer !== null) this.timers.clearTimeout(this.timer)
    this.timer = null
    this.sock?.close(1000)
    this.sock = null
  }

  /** Returns the command id, or null when not connected (commands are never queued: stale clicks are dangerous). */
  send(cmd: Command): number | null {
    if (!this.sock || this.sock.readyState !== OPEN || !this.authed) return null
    const id = this.nextId++
    this.sock.send(JSON.stringify({ ...cmd, id }))
    return id
  }

  private connect(): void {
    const sock = this.factory(this.url)
    this.sock = sock
    sock.onopen = () => sock.send(JSON.stringify({ type: 'auth', token: this.token }))
    sock.onmessage = (ev) => {
      const m = parseServerMsg(String(ev.data))
      if (!m) return
      if (m.type === 'hello') {
        this.authed = true
        this.attempt = 0
        this.onConnection(true, false)
      }
      this.onMessage(m)
    }
    sock.onerror = () => {}
    sock.onclose = (ev) => {
      const wasAuthed = this.authed
      this.authed = false
      this.sock = null
      const authFailed = ev.code === AUTH_FAILED
      if (wasAuthed || authFailed || this.attempt === 0) this.onConnection(false, authFailed)
      if (this.stopped || authFailed) return
      this.timer = this.timers.setTimeout(() => {
        this.timer = null
        this.connect()
      }, backoffMs(this.attempt++))
    }
  }
}
