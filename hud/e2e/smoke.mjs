// End-to-end smoke test (local; not part of CI):
//   1. the real Python demo hub (jarvis.hud_bridge.demo) on a free port with a random token;
//   2. the built renderer in headless Chromium: connects, authenticates, shows the scripted state,
//      and a typed chat message reaches the hub;
//   3. if an Electron binary and a display (e.g. xvfb-run) are available, the real overlay window:
//      transparent, always on top, not focusable, connected.
// Run: npm run build && JARVIS_PYTHON=../.venv/bin/python xvfb-run -a npm run e2e

import { execSync, spawn } from 'node:child_process'
import { randomBytes } from 'node:crypto'
import { existsSync, mkdirSync, readFileSync, writeFileSync } from 'node:fs'
import { createServer } from 'node:http'
import { createRequire } from 'node:module'
import { dirname, extname, join, resolve } from 'node:path'
import { fileURLToPath } from 'node:url'

const here = dirname(fileURLToPath(import.meta.url))
const root = resolve(here, '..')
const renderer = join(root, 'out', 'renderer')
const artifacts = join(here, 'artifacts')
mkdirSync(artifacts, { recursive: true })

async function loadPlaywright() {
  try {
    return await import('playwright')
  } catch {
    const g = execSync('npm root -g', { encoding: 'utf8' }).trim()
    return createRequire(import.meta.url)(join(g, 'playwright'))
  }
}

function startHub(port, token) {
  const py = process.env.JARVIS_PYTHON || 'python'
  const p = spawn(py, ['-m', 'jarvis.hud_bridge.demo', '--port', String(port), '--token', token, '--forever'], {
    cwd: resolve(root, '..'),
    stdio: ['ignore', 'pipe', 'inherit'],
  })
  const lines = []
  let buf = ''
  p.stdout.on('data', (d) => {
    buf += d
    let i
    while ((i = buf.indexOf('\n')) >= 0) {
      lines.push(buf.slice(0, i))
      buf = buf.slice(i + 1)
    }
  })
  return { p, lines }
}

function serve(dir) {
  const types = { '.html': 'text/html', '.js': 'text/javascript', '.css': 'text/css' }
  const srv = createServer((req, res) => {
    const path = join(dir, decodeURIComponent(new URL(req.url, 'http://x').pathname))
    if (!path.startsWith(dir) || !existsSync(path)) return res.writeHead(404).end()
    res.writeHead(200, { 'content-type': types[extname(path)] ?? 'application/octet-stream' }).end(readFileSync(path))
  })
  return new Promise((ok) => srv.listen(0, '127.0.0.1', () => ok(srv)))
}

const waitFor = async (fn, what, ms = 15000) => {
  const t0 = Date.now()
  while (Date.now() - t0 < ms) {
    if (await fn()) return
    await new Promise((r) => setTimeout(r, 100))
  }
  throw new Error(`timed out waiting for ${what}`)
}

const results = []
const check = (name, ok, detail = '') => {
  results.push({ name, ok, detail })
  console.log(`${ok ? 'PASS' : 'FAIL'}  ${name}${detail ? `  (${detail})` : ''}`)
}

const port = 40000 + Math.floor(Math.random() * 20000)
const token = randomBytes(18).toString('base64url')
const hub = startHub(port, token)
await waitFor(() => hub.lines.some((l) => l.includes('"ready"')), 'demo hub')
const pw = await loadPlaywright()
const srv = await serve(renderer)
const httpPort = srv.address().port

try {
  // ---------------------------------------------------------------- renderer in Chromium
  const browser = await pw.chromium.launch({ args: ['--use-angle=swiftshader', '--enable-unsafe-swiftshader'] })
  const page = await browser.newPage({ viewport: { width: 1600, height: 900 } })
  const errors = []
  page.on('pageerror', (e) => errors.push(String(e)))
  await page.goto(`http://127.0.0.1:${httpPort}/index.html?ws=ws://127.0.0.1:${port}&token=${token}&interactive=1`)
  // Screenshots only: a dark, game-like backdrop behind the transparent overlay.
  // (set through the CSSOM: the page's CSP rightly refuses injected <style> tags)
  await page.evaluate(() => {
    document.documentElement.style.background = 'radial-gradient(ellipse at 40% 60%,#2b3a46 0%,#141b22 55%,#07090c 100%)'
  })
  const text = (sel) => page.locator(sel).first().textContent()
  await waitFor(async () => /연결됨|ONLINE/.test((await text('[data-k="link"]')) ?? ''), 'link online')
  check('authenticates with the hub', true)
  await waitFor(async () => ((await text('[data-k="lines"]')) ?? '').includes('자비스, 3판 싸워'), 'transcript')
  check('shows the owner transcript', true)
  await waitFor(async () => ((await text('[data-k="lines"]')) ?? '').includes('경기 시작을 기다립니다'), 'reply')
  check('streams the spoken reply', true)
  await page.screenshot({ path: join(artifacts, 'hud-chromium-talk.png') })
  await waitFor(async () => await page.locator('#p-fight').isVisible() && (await text('[data-k="state"]')) === '전투 중', 'fight panel')
  await waitFor(async () => (await text('[data-k="tactic"]')) === '카운터', 'telemetry')
  const hp = await page.locator('[data-bar="own_hp"]').evaluate((el) => el.style.width)
  check('renders live fight telemetry', parseFloat(hp) > 0, `own HP bar ${hp}`)
  check('combat mode engaged', await page.evaluate(() => document.body.classList.contains('combat')))
  await page.fill('[data-k="input"]', 'status report')
  await page.press('[data-k="input"]', 'Enter')
  await waitFor(() => hub.lines.some((l) => l.includes('"status report"')), 'chat command at hub')
  check('typed chat reaches the hub', true)
  await page.waitForTimeout(800)
  const shot = await page.screenshot({ path: join(artifacts, 'hud-chromium.png') })
  check('orb canvas present', (await page.locator('#orb').count()) === 1, `screenshot ${shot.length} bytes`)
  check('no page errors', errors.length === 0, errors.join(' | ').slice(0, 300))
  await browser.close()

  // ---------------------------------------------------------------- real Electron window
  let electronBin = null
  try {
    electronBin = createRequire(import.meta.url)(join(root, 'node_modules', 'electron'))
  } catch {
    electronBin = null
  }
  if (!electronBin || !existsSync(electronBin) || !process.env.DISPLAY) {
    console.log('SKIP  Electron window (no Electron binary or no DISPLAY; use xvfb-run)')
  } else {
    const tokFile = join(artifacts, 'e2e.token')
    writeFileSync(tokFile, token)
    const conf = join(artifacts, 'hud.json')
    writeFileSync(conf, JSON.stringify({ host: '127.0.0.1', port, token_file: tokFile, hotkey: 'ctrl+alt+j', content_protection: true, combat_mode: 'dim', combat_fps_cap: 30, prefer_secondary_monitor: false }))
    const app = await pw._electron.launch({
      executablePath: electronBin,
      // Test-only switches: running as root in a container, and software WebGL without a GPU.
      args: ['--no-sandbox', '--use-angle=swiftshader', '--enable-unsafe-swiftshader', root],
      env: { ...process.env, JARVIS_HUD_CONFIG: conf },
    })
    const win = await app.firstWindow()
    await win.waitForLoadState('domcontentloaded')
    const props = await app.evaluate(({ BrowserWindow }) => {
      const w = BrowserWindow.getAllWindows()[0]
      return { onTop: w.isAlwaysOnTop(), focusable: w.isFocusable(), visible: w.isVisible(), protected: typeof w.isContentProtected === 'function' ? w.isContentProtected() : null }
    })
    check('electron: always on top', props.onTop)
    check('electron: never takes focus', props.focusable === false)
    if (process.platform === 'linux') console.log('SKIP  electron: content protection (Windows/macOS only)')
    else check('electron: content protection on', props.protected === true, `isContentProtected=${props.protected}`)
    await waitFor(async () => /연결됨|ONLINE/.test((await win.locator('[data-k="link"]').first().textContent()) ?? ''), 'electron link online')
    check('electron: renderer connected through the preload bridge', true)
    await waitFor(async () => ((await win.locator('[data-k="lines"]').first().textContent()) ?? '').includes('자비스, 3판 싸워'), 'electron history')
    check('electron: late-joining HUD shows recent conversation', true)
    await win.waitForTimeout(1500)
    await win.screenshot({ path: join(artifacts, 'hud-electron.png') })
    await app.close()
  }
} catch (e) {
  check('e2e', false, String(e))
} finally {
  srv.close()
  hub.p.kill()
}

const failed = results.filter((r) => !r.ok)
console.log(`\n${results.length - failed.length}/${results.length} checks passed`)
process.exit(failed.length ? 1 : 0)
