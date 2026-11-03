// HUD renderer entry: hub messages -> state -> orb + panels.

import { getBridge } from './bridge'
import { Orb } from './orb'
import { Panels } from './panels'
import type { ServerMsg } from './protocol'
import { type ConnectionMsg, type HudState, inCombat, initialState, orbMode, reduce } from './state'
import { HubClient } from './ws'

async function boot(): Promise<void> {
  const bridge = getBridge()
  const init = await bridge.init()
  let state: HudState = initialState()
  let dirty = true
  let combat = false

  const hub = new HubClient(init.wsUrl, init.token)
  const apply = (m: ServerMsg | ConnectionMsg) => {
    state = reduce(state, m, performance.now())
    dirty = true
  }
  hub.onMessage = apply
  hub.onConnection = (connected, authFailed) => apply({ type: 'connection', connected, authFailed })

  const panels = new Panels(document, (c) => hub.send(c), bridge)
  bridge.onInteractive((on) => panels.setInteractive(on))

  let orb: Orb | null = null
  try {
    orb = new Orb(document.getElementById('orb') as HTMLCanvasElement, { combatFpsCap: init.config.combatFpsCap })
    orb.source = () => ({ mode: orbMode(state, performance.now()), level: state.level })
    orb.start()
  } catch (e) {
    console.error('orb disabled (WebGL unavailable)', e) // panels still work without the orb
  }

  // Panels update at ~15 fps; the orb animates on its own frame budget.
  setInterval(() => {
    const c = inCombat(state)
    if (c !== combat) {
      combat = c
      orb?.setCombat(c, init.config.combatMode === 'hide')
      bridge.setFightActive(c)
      document.body.classList.toggle('combat', c)
      document.body.classList.toggle('combat-hide', c && init.config.combatMode === 'hide')
      dirty = true
    }
    if (dirty || state.wakeUntil > performance.now() - 100) {
      dirty = false
      panels.render(state)
    }
  }, 66)

  hub.start()
}

void boot()
