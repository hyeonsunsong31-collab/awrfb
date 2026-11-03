import { describe, expect, it } from 'vitest'
import { boxStyle, fightStateLabel, pct, radarPoints, statsText, tacticLabel, threatLevel } from '../src/format'
import { audioLevel, displacement, FpsLimiter, fpsFor, lerpStyle, STYLES } from '../src/orbstyle'

describe('panel formatting', () => {
  it('formats percentages and threat', () => {
    expect(pct(0.554)).toBe('55%')
    expect(pct(null)).toBe('—')
    expect(pct(1.7)).toBe('100%')
    expect(threatLevel({ '100': 0.9, '200': 0.3, '300': 0.45 })).toBe(0.45)
    expect(threatLevel(null)).toBe(0)
  })

  it('draws the opponent radar inside its radius', () => {
    const pts = radarPoints({ aggression: 1, defense: 0, punish: 0.5, range_pref: 0, regen: 0 }, 46, 60, 60).split(' ')
    expect(pts).toHaveLength(5)
    expect(pts[0]).toBe('60.0,14.0') // full aggression points straight up
    expect(pts[1]).toBe('60.0,60.0') // zero defense sits at the centre
  })

  it('labels tactics, states and stats in both languages', () => {
    expect(tacticLabel('counter', 'en')).toBe('Counter-punch')
    expect(tacticLabel('counter', 'ko')).toBe('카운터')
    expect(tacticLabel(null, 'en')).toBe('—')
    expect(fightStateLabel('fighting', 'ko')).toBe('전투 중')
    expect(statsText({ cpu: 23.4, ram_used_gb: 9.4, ram_total_gb: 31.8, gpu: { name: 'RTX', util: 41, vram_used_mb: 3900, vram_total_mb: 8192 } }))
      .toEqual({ cpu: '23%', ram: '9.4 / 32 GB', gpu: '41%', vram: '3.8 / 8 GB' })
  })

  it('places tracking boxes from normalized centres', () => {
    expect(boxStyle([0.5, 0.5, 0.2, 0.4])).toEqual({ left: '40.00%', top: '30.00%', width: '20.00%', height: '40.00%' })
  })
})

describe('orb parameters', () => {
  it('reacts to the right audio source per mode', () => {
    expect(audioLevel('speaking', { mic: 1, out: 0.1 })).toBeCloseTo(0.4)
    expect(audioLevel('listening', { mic: 0.1, out: 1 })).toBeCloseTo(0.6)
    expect(audioLevel('idle', { mic: 1, out: 1 })).toBe(0)
    expect(displacement(STYLES.speaking, 1)).toBeGreaterThan(displacement(STYLES.speaking, 0))
  })

  it('caps frame rate: 30 idle, 60 active, combat cap in fights', () => {
    expect(fpsFor('idle', false, 30)).toBe(30)
    expect(fpsFor('speaking', false, 30)).toBe(60)
    expect(fpsFor('speaking', true, 24)).toBe(24)
    const lim = new FpsLimiter(30)
    expect([0, 10, 20, 34, 40, 67].map((t) => lim.ready(t))).toEqual([true, false, false, true, false, true])
  })

  it('blends styles smoothly', () => {
    const mid = lerpStyle(STYLES.idle, STYLES.fighting, 0.5)
    expect(mid.glow).toBeCloseTo((STYLES.idle.glow + STYLES.fighting.glow) / 2)
  })
})
