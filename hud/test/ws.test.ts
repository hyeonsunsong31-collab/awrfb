import { describe, expect, it } from 'vitest'
import { AUTH_FAILED, backoffMs, HubClient, type SocketLike } from '../src/ws'

class FakeSocket implements SocketLike {
  readyState = 0
  sent: string[] = []
  onopen: ((ev: unknown) => void) | null = null
  onmessage: ((ev: { data: unknown }) => void) | null = null
  onclose: ((ev: { code: number }) => void) | null = null
  onerror: ((ev: unknown) => void) | null = null
  send(d: string) {
    this.sent.push(d)
  }
  close() {
    this.readyState = 3
  }
  open() {
    this.readyState = 1
    this.onopen?.({})
  }
  recv(m: unknown) {
    this.onmessage?.({ data: JSON.stringify(m) })
  }
  drop(code = 1006) {
    this.readyState = 3
    this.onclose?.({ code })
  }
}

function rig() {
  const sockets: FakeSocket[] = []
  const timers: { fn: () => void; ms: number }[] = []
  const c = new HubClient('ws://127.0.0.1:1', 'tok', () => {
    const s = new FakeSocket()
    sockets.push(s)
    return s
  }, { setTimeout: (fn, ms) => timers.push({ fn, ms }), clearTimeout: () => {} })
  const conn: [boolean, boolean][] = []
  c.onConnection = (a, b) => conn.push([a, b])
  return { c, sockets, timers, conn }
}

describe('HubClient', () => {
  it('authenticates first and is connected only after hello', () => {
    const { c, sockets, conn } = rig()
    c.start()
    sockets[0].open()
    expect(JSON.parse(sockets[0].sent[0])).toEqual({ type: 'auth', token: 'tok' })
    expect(c.send({ type: 'stop' })).toBeNull() // not authenticated yet
    sockets[0].recv({ type: 'hello', snapshot: {} })
    expect(conn).toEqual([[true, false]])
    expect(c.send({ type: 'chat', text: 'hi' })).toBe(1)
    expect(JSON.parse(sockets[0].sent[1])).toEqual({ type: 'chat', text: 'hi', id: 1 })
  })

  it('reconnects with exponential backoff', () => {
    const { c, sockets, timers } = rig()
    c.start()
    sockets[0].open()
    sockets[0].recv({ type: 'hello', snapshot: {} })
    sockets[0].drop()
    expect(timers.map((t) => t.ms)).toEqual([500])
    timers[0].fn()
    sockets[1].drop()
    timers[1].fn()
    sockets[2].drop()
    expect(timers.map((t) => t.ms)).toEqual([500, 1000, 2000])
    expect(backoffMs(10)).toBe(8000)
  })

  it('stops retrying after an authentication failure', () => {
    const { c, sockets, timers, conn } = rig()
    c.start()
    sockets[0].open()
    sockets[0].drop(AUTH_FAILED)
    expect(conn.at(-1)).toEqual([false, true])
    expect(timers).toHaveLength(0)
  })

  it('never queues commands while offline', () => {
    const { c, sockets } = rig()
    c.start()
    sockets[0].open()
    sockets[0].recv({ type: 'hello', snapshot: {} })
    sockets[0].drop()
    expect(c.send({ type: 'disarm' })).toBeNull()
  })
})
