// The only bridge between the sandboxed renderer and the main process.

import { contextBridge, ipcRenderer } from 'electron'

contextBridge.exposeInMainWorld('jarvis', {
  init: () => ipcRenderer.invoke('hud:init'),
  setFightActive: (active: boolean) => ipcRenderer.send('hud:fight-active', active),
  setStreamMode: (on: boolean) => ipcRenderer.send('hud:stream-mode', on),
  exitInteractive: () => ipcRenderer.send('hud:exit-interactive'),
  onInteractive: (cb: (on: boolean) => void) => {
    ipcRenderer.on('hud:interactive', (_e, on: boolean) => cb(on))
  },
})
