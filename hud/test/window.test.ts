import { describe, expect, it } from 'vitest'
import { DEFAULT_CONFIG, nextInteractive, parseConfig, pickDisplay, toAccelerator } from '../electron/window'

describe('main-process helpers', () => {
  it('converts config hotkeys to Electron accelerators', () => {
    expect(toAccelerator('ctrl+alt+j')).toBe('Control+Alt+J')
    expect(toAccelerator('ctrl+shift+f12')).toBe('Control+Shift+F12')
    expect(toAccelerator('win+space')).toBe('Super+Space')
  })

  it('prefers a secondary monitor when there is one', () => {
    const a = { id: 1, workArea: { x: 0, y: 0, width: 1920, height: 1040 } }
    const b = { id: 2, workArea: { x: 1920, y: 0, width: 1920, height: 1040 } }
    expect(pickDisplay([a, b], 1, true)).toBe(b)
    expect(pickDisplay([a, b], 1, false)).toBe(a)
    expect(pickDisplay([a], 1, true)).toBe(a)
  })

  it('never allows interactive mode during a fight', () => {
    expect(nextInteractive(false, false)).toBe(true)
    expect(nextInteractive(true, false)).toBe(false)
    expect(nextInteractive(false, true)).toBe(false)
    expect(nextInteractive(true, true, true)).toBe(false)
  })

  it('validates the config file and refuses a remote hub', () => {
    expect(parseConfig(null)).toEqual(DEFAULT_CONFIG)
    const c = parseConfig({ host: 'evil.example', port: 99999, hotkey: 'ctrl+alt+h', combat_mode: 'hide', combat_fps_cap: 24, token_file: '/x/ipc.token' })
    expect(c.host).toBe('127.0.0.1')
    expect(c.port).toBe(DEFAULT_CONFIG.port)
    expect(c.hotkey).toBe('ctrl+alt+h')
    expect(c.combat_mode).toBe('hide')
    expect(c.combat_fps_cap).toBe(24)
    expect(c.token_file).toBe('/x/ipc.token')
  })
})
