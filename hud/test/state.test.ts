import { describe, expect, it } from 'vitest'
import { applyEvent, initialState, inCombat, MAX_LINES, orbMode, reduce } from '../src/state'

const ev = (topic: string, data: unknown) => ({ type: 'event' as const, topic, data })

describe('reducer', () => {
  it('hello replays the snapshot and marks the link online', () => {
    const s = reduce(initialState(), { type: 'hello', snapshot: { 'voice.state': { mode: 'idle', muted: null, interpreter: false, lang: 'ko' }, 'fight.state': { new: 'disarmed' } } }, 0)
    expect(s.connected).toBe(true)
    expect(s.lang).toBe('ko')
    expect(s.fightState).toBe('disarmed')
    expect(orbMode(s, 0)).toBe('idle')
  })

  it('streams a reply into one line and finalizes it', () => {
    let s = reduce(initialState(), { type: 'hello', snapshot: {} }, 0)
    s = reduce(s, ev('voice.transcript', { text: 'open notepad' }), 0)
    s = reduce(s, ev('voice.reply', { turn: 3, delta: 'Opening ' }), 0)
    s = reduce(s, ev('voice.reply', { turn: 3, delta: 'Notepad.' }), 0)
    expect(s.lines.map((l) => [l.who, l.text, l.done])).toEqual([['you', 'open notepad', true], ['jarvis', 'Opening Notepad.', false]])
    s = reduce(s, ev('voice.reply_done', { turn: 3, text: 'Opening Notepad.', heard: null }), 0)
    expect(s.lines[1].done).toBe(true)
  })

  it('an interrupted reply shows only what was heard', () => {
    let s = reduce(initialState(), ev('voice.reply', { turn: 1, delta: 'The first sentence. The second one.' }), 0)
    s = reduce(s, ev('voice.reply_done', { turn: 1, heard: 'The first' }), 0)
    expect(s.lines[0]).toMatchObject({ text: 'The first …', interrupted: true, done: true })
  })

  it('a restarted HUD replays recent history before the snapshot', () => {
    const s = reduce(initialState(), {
      type: 'hello',
      snapshot: { 'voice.state': { mode: 'idle', lang: 'en' } },
      history: [
        { topic: 'voice.transcript', data: { text: 'fight status' } },
        { topic: 'voice.reply_done', data: { turn: 4, text: 'Last match: 3-1 in stocks.', heard: null } },
      ],
    }, 0)
    expect(s.lines.map((l) => [l.who, l.text])).toEqual([['you', 'fight status'], ['jarvis', 'Last match: 3-1 in stocks.']])
  })

  it('caps the transcript', () => {
    let s = initialState()
    for (let i = 0; i < 30; i++) s = reduce(s, ev('voice.transcript', { text: `line ${i}` }), 0)
    expect(s.lines).toHaveLength(MAX_LINES)
    expect(s.lines.at(-1)?.text).toBe('line 29')
  })

  it('voice commands show and hide panels', () => {
    let s = reduce(initialState(), ev('hud.command', { action: 'hide', panel: 'all' }), 0)
    expect(Object.values(s.panels).every((v) => !v)).toBe(true)
    s = reduce(s, ev('hud.command', { action: 'show', panel: 'fight' }), 0)
    expect(s.panels.fight).toBe(true)
    expect(s.panels.stats).toBe(false)
  })

  it('coach comments appear after a match', () => {
    const s = reduce(initialState(), ev('fight.coach', { comment: 'Block earlier against heavies.' }), 0)
    expect(s.lines.at(-1)?.text).toBe('Coach: Block earlier against heavies.')
  })

  it('agent and supervisor notices appear in the transcript', () => {
    const s = reduce(initialState(), ev('hud.notify', { text: 'Download finished' }), 0)
    expect(s.lines.at(-1)).toMatchObject({ who: 'system', text: 'Download finished' })
    const t = reduce(s, ev('sys.notice', { text: 'toolhost keeps crashing' }), 0)
    expect(t.lines.at(-1)).toMatchObject({ who: 'system', text: 'toolhost keeps crashing' })
  })

  it('shows background learning: running job, promotions and drift warnings', () => {
    let s = reduce(initialState(), ev('learn.status', { running: 'nightly' }), 0)
    expect(s.learning).toBe('nightly')
    s = reduce(s, ev('learn.status', { running: null }), 0)
    expect(s.learning).toBeNull()
    s = reduce(s, ev('learn.result', { job: 'nightly', ok: true, promoted: false, ap_300: 0.6 }), 0)
    expect(s.lines).toHaveLength(0)
    s = reduce(s, ev('learn.result', { job: 'nightly', ok: true, promoted: true, ap_300: 0.812 }), 0)
    expect(s.lines.at(-1)?.text).toBe('New threat model promoted (AP@300 0.81)')
    s = reduce(s, ev('learn.drift', { reason: 'HUD confidence fell' }), 0)
    expect(s.lines.at(-1)).toMatchObject({ who: 'system', text: 'HUD needs recalibration — HUD confidence fell' })
  })

  it('ignores unknown topics and malformed data', () => {
    const s0 = initialState()
    expect(applyEvent(s0, 'nope', { x: 1 }, 0)).toBe(s0)
    expect(applyEvent(s0, 'voice.transcript', null, 0)).toBe(s0)
  })
})

describe('orb mode', () => {
  const online = () => reduce(initialState(), { type: 'hello', snapshot: {} }, 0)

  it('offline until the hub says hello; auth failure is an error', () => {
    expect(orbMode(initialState(), 0)).toBe('offline')
    expect(orbMode(reduce(initialState(), { type: 'connection', connected: false, authFailed: true }, 0), 0)).toBe('error')
  })

  it('wake word flashes listening for a moment', () => {
    const s = reduce(online(), ev('voice.wake', { score: 0.9 }), 1000)
    expect(orbMode(s, 1500)).toBe('listening')
    expect(orbMode(s, 4000)).toBe('idle')
  })

  it('fighting unless JARVIS is talking; hard mute wins', () => {
    let s = reduce(online(), ev('fight.state', { new: 'fighting' }), 0)
    expect(inCombat(s)).toBe(true)
    expect(orbMode(s, 0)).toBe('fighting')
    s = reduce(s, ev('voice.state', { mode: 'speaking', muted: null }), 0)
    expect(orbMode(s, 0)).toBe('speaking')
    s = reduce(s, ev('voice.state', { mode: 'idle', muted: 'hard' }), 0)
    expect(orbMode(s, 0)).toBe('muted')
  })

  it('telemetry carries the fight state and frames become data URLs', () => {
    let s = reduce(online(), ev('fight.telemetry', { state: 'queued', hud: {}, threat: {}, match: { won: 0, lost: 0 } }), 0)
    expect(s.fightState).toBe('queued')
    s = reduce(s, ev('fight.frame', { jpg: 'AAAA', w: 320, h: 180 }), 0)
    expect(s.frame?.src).toBe('data:image/jpeg;base64,AAAA')
  })
})
