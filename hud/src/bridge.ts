// Access to the main process (window.jarvis from the preload). In a plain browser (development and the
// end-to-end test) the hub address and token come from the URL instead: ?ws=ws://127.0.0.1:47839&token=...

export interface HudUiConfig {
  combatMode: 'dim' | 'hide'
  combatFpsCap: number
  streamMode: boolean
}

export interface HudInit {
  wsUrl: string
  token: string
  config: HudUiConfig
}

export interface Bridge {
  init(): Promise<HudInit>
  setFightActive(active: boolean): void
  setStreamMode(on: boolean): void
  exitInteractive(): void
  onInteractive(cb: (on: boolean) => void): void
}

declare global {
  interface Window {
    jarvis?: Bridge
  }
}

export function urlBridge(search: string): Bridge {
  const q = new URLSearchParams(search)
  return {
    init: async () => ({
      wsUrl: q.get('ws') ?? 'ws://127.0.0.1:47830',
      token: q.get('token') ?? '',
      config: { combatMode: q.get('combat') === 'hide' ? 'hide' : 'dim', combatFpsCap: Number(q.get('fps') ?? 30), streamMode: false },
    }),
    setFightActive: () => {},
    setStreamMode: () => {},
    exitInteractive: () => {},
    onInteractive: (cb) => cb(q.get('interactive') === '1'),
  }
}

export function getBridge(): Bridge {
  return window.jarvis ?? urlBridge(window.location.search)
}
