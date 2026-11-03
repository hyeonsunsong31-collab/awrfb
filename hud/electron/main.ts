// Electron main process: one transparent, click-through, always-on-top window that never takes focus.
// It can only become clickable through the global hotkey, and never while a fight is active, so it
// cannot steal the foreground from Roblox (fightd also refuses to send input when Roblox loses focus).

import { app, BrowserWindow, globalShortcut, ipcMain, screen } from 'electron'
import { existsSync, readFileSync } from 'node:fs'
import { join } from 'node:path'
import { DEFAULT_CONFIG, nextInteractive, parseConfig, pickDisplay, toAccelerator, type HudFileConfig } from './window'

let win: BrowserWindow | null = null
let interactive = false
let fightActive = false
let streamMode = false
let cfg: HudFileConfig = DEFAULT_CONFIG

function loadConfig(): HudFileConfig {
  const path = process.env.JARVIS_HUD_CONFIG
  if (!path || !existsSync(path)) return DEFAULT_CONFIG
  try {
    return parseConfig(JSON.parse(readFileSync(path, 'utf-8')))
  } catch {
    return DEFAULT_CONFIG
  }
}

function readToken(): string {
  try {
    return cfg.token_file ? readFileSync(cfg.token_file, 'utf-8').trim() : ''
  } catch {
    return ''
  }
}

function applyInteractive(on: boolean): void {
  interactive = nextInteractive(interactive, fightActive, on)
  if (!win) return
  win.setIgnoreMouseEvents(!interactive, { forward: true })
  win.setFocusable(interactive)
  if (interactive) win.focus()
  else win.blur()
  win.webContents.send('hud:interactive', interactive)
}

function applyProtection(): void {
  win?.setContentProtection(cfg.content_protection && !streamMode)
}

function createWindow(): void {
  const d = pickDisplay(screen.getAllDisplays(), screen.getPrimaryDisplay().id, cfg.prefer_secondary_monitor)
  win = new BrowserWindow({
    ...d.workArea,
    transparent: true,
    frame: false,
    resizable: false,
    movable: false,
    focusable: false,
    skipTaskbar: true,
    alwaysOnTop: true,
    hasShadow: false,
    show: false,
    backgroundColor: '#00000000',
    webPreferences: {
      preload: join(__dirname, '../preload/preload.js'),
      contextIsolation: true,
      sandbox: true,
      nodeIntegration: false,
      backgroundThrottling: false, // the overlay is never focused but must keep animating
    },
  })
  win.setAlwaysOnTop(true, 'screen-saver')
  win.setVisibleOnAllWorkspaces(true)
  win.setIgnoreMouseEvents(true, { forward: true })
  applyProtection()
  win.once('ready-to-show', () => win?.showInactive()) // never activate on show
  win.webContents.setWindowOpenHandler(() => ({ action: 'deny' }))
  win.webContents.on('will-navigate', (e) => e.preventDefault())
  const dev = process.env.ELECTRON_RENDERER_URL
  if (dev) void win.loadURL(dev)
  else void win.loadFile(join(__dirname, '../renderer/index.html'))
  win.on('closed', () => (win = null))
}

ipcMain.handle('hud:init', () => ({
  wsUrl: `ws://${cfg.host}:${cfg.port}`,
  token: readToken(),
  config: { combatMode: cfg.combat_mode, combatFpsCap: cfg.combat_fps_cap, streamMode: cfg.stream_mode },
}))
ipcMain.on('hud:fight-active', (_e, active: boolean) => {
  fightActive = !!active
  if (fightActive && interactive) applyInteractive(false)
})
ipcMain.on('hud:stream-mode', (_e, on: boolean) => {
  streamMode = !!on
  applyProtection()
})
ipcMain.on('hud:exit-interactive', () => applyInteractive(false))

if (!app.requestSingleInstanceLock()) {
  app.quit()
} else {
  app.whenReady().then(() => {
    cfg = loadConfig()
    streamMode = cfg.stream_mode
    createWindow()
    const accel = toAccelerator(cfg.hotkey)
    if (!globalShortcut.register(accel, () => applyInteractive(!interactive))) {
      console.warn(`HUD hotkey ${accel} is taken by another app`)
    }
  })
  app.on('will-quit', () => globalShortcut.unregisterAll())
  app.on('window-all-closed', () => app.quit())
}
