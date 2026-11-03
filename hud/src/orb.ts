// The "arc reactor" orb: a noise-displaced shader sphere with a fresnel rim and scanlines, an inner core,
// rotating rings, and bloom. Combat mode drops bloom and caps the frame rate (or hides the orb) so the
// overlay costs Roblox as little GPU time as possible.

import { BloomEffect, EffectComposer, EffectPass, RenderPass, VignetteEffect } from 'postprocessing'
import {
  AdditiveBlending,
  Color,
  Group,
  HalfFloatType,
  IcosahedronGeometry,
  Mesh,
  MeshBasicMaterial,
  PerspectiveCamera,
  Scene,
  ShaderMaterial,
  SphereGeometry,
  TorusGeometry,
  WebGLRenderer,
} from 'three'
import { audioLevel, displacement, FpsLimiter, fpsFor, lerpStyle, STYLES, type OrbStyle } from './orbstyle'
import type { OrbMode } from './state'

const NOISE = /* glsl */ `
vec3 mod289(vec3 x){return x-floor(x*(1.0/289.0))*289.0;}
vec4 mod289(vec4 x){return x-floor(x*(1.0/289.0))*289.0;}
vec4 permute(vec4 x){return mod289(((x*34.0)+1.0)*x);}
vec4 taylorInvSqrt(vec4 r){return 1.79284291400159-0.85373472095314*r;}
float snoise(vec3 v){
  const vec2 C=vec2(1.0/6.0,1.0/3.0); const vec4 D=vec4(0.0,0.5,1.0,2.0);
  vec3 i=floor(v+dot(v,C.yyy)); vec3 x0=v-i+dot(i,C.xxx);
  vec3 g=step(x0.yzx,x0.xyz); vec3 l=1.0-g; vec3 i1=min(g.xyz,l.zxy); vec3 i2=max(g.xyz,l.zxy);
  vec3 x1=x0-i1+C.xxx; vec3 x2=x0-i2+C.yyy; vec3 x3=x0-D.yyy; i=mod289(i);
  vec4 p=permute(permute(permute(i.z+vec4(0.0,i1.z,i2.z,1.0))+i.y+vec4(0.0,i1.y,i2.y,1.0))+i.x+vec4(0.0,i1.x,i2.x,1.0));
  float n_=0.142857142857; vec3 ns=n_*D.wyz-D.xzx; vec4 j=p-49.0*floor(p*ns.z*ns.z);
  vec4 x_=floor(j*ns.z); vec4 y_=floor(j-7.0*x_); vec4 x=x_*ns.x+ns.yyyy; vec4 y=y_*ns.x+ns.yyyy;
  vec4 h=1.0-abs(x)-abs(y); vec4 b0=vec4(x.xy,y.xy); vec4 b1=vec4(x.zw,y.zw);
  vec4 s0=floor(b0)*2.0+1.0; vec4 s1=floor(b1)*2.0+1.0; vec4 sh=-step(h,vec4(0.0));
  vec4 a0=b0.xzyw+s0.xzyw*sh.xxyy; vec4 a1=b1.xzyw+s1.xzyw*sh.zzww;
  vec3 p0=vec3(a0.xy,h.x); vec3 p1=vec3(a0.zw,h.y); vec3 p2=vec3(a1.xy,h.z); vec3 p3=vec3(a1.zw,h.w);
  vec4 norm=taylorInvSqrt(vec4(dot(p0,p0),dot(p1,p1),dot(p2,p2),dot(p3,p3)));
  p0*=norm.x; p1*=norm.y; p2*=norm.z; p3*=norm.w;
  vec4 m=max(0.6-vec4(dot(x0,x0),dot(x1,x1),dot(x2,x2),dot(x3,x3)),0.0); m=m*m;
  return 42.0*dot(m*m,vec4(dot(p0,x0),dot(p1,x1),dot(p2,x2),dot(p3,x3)));
}`

const VERTEX = /* glsl */ `
uniform float uTime; uniform float uDisp; uniform float uSpeed;
varying vec3 vNormal; varying vec3 vView; varying float vNoise; varying vec2 vUv;
${NOISE}
void main(){
  vUv = uv;
  float n = snoise(normal * 1.6 + vec3(uTime * uSpeed));
  vNoise = n;
  vec4 mv = modelViewMatrix * vec4(position + normal * n * uDisp, 1.0);
  vNormal = normalize(normalMatrix * normal);
  vView = normalize(-mv.xyz);
  gl_Position = projectionMatrix * mv;
}`

const FRAGMENT = /* glsl */ `
uniform vec3 uColor; uniform float uGlow; uniform float uTime; uniform float uLevel;
varying vec3 vNormal; varying vec3 vView; varying float vNoise; varying vec2 vUv;
void main(){
  float fres = pow(1.0 - max(dot(vNormal, vView), 0.0), 2.4);
  float scan = 0.75 + 0.25 * sin(vUv.y * 420.0 - uTime * 6.0);
  float body = 0.18 + 0.25 * (vNoise * 0.5 + 0.5);
  float a = clamp((fres * 1.6 + body * 0.35) * uGlow * scan, 0.0, 1.0);
  gl_FragColor = vec4(uColor * (0.6 + fres * 1.4 + uLevel * 0.8), a);
}`

export interface OrbSource {
  mode: OrbMode
  level: { mic: number; out: number }
}

export class Orb {
  source: () => OrbSource = () => ({ mode: 'offline', level: { mic: 0, out: 0 } })

  private readonly renderer: WebGLRenderer
  private readonly composer: EffectComposer
  private readonly bloomPass: EffectPass
  private readonly scene = new Scene()
  private readonly camera = new PerspectiveCamera(35, 1, 0.1, 100)
  private readonly shell: ShaderMaterial
  private readonly core: MeshBasicMaterial
  private readonly rings: Mesh[] = []
  private readonly group = new Group()
  private readonly limiter = new FpsLimiter(30)
  private style: OrbStyle = STYLES.offline
  private level = 0
  private combat = false
  private hidden = false
  private raf = 0
  private t0 = performance.now()

  constructor(private readonly canvas: HTMLCanvasElement, private readonly opts: { combatFpsCap: number }) {
    this.renderer = new WebGLRenderer({ canvas, alpha: true, antialias: false, powerPreference: 'low-power' })
    this.renderer.setClearColor(0x000000, 0)
    this.renderer.setPixelRatio(Math.min(window.devicePixelRatio, 1.5))
    this.camera.position.set(0, 0, 7.6)

    this.shell = new ShaderMaterial({
      vertexShader: VERTEX,
      fragmentShader: FRAGMENT,
      transparent: true,
      depthWrite: false,
      uniforms: {
        uTime: { value: 0 },
        uDisp: { value: 0.04 },
        uSpeed: { value: 0.15 },
        uColor: { value: new Color(...STYLES.offline.color) },
        uGlow: { value: 0.5 },
        uLevel: { value: 0 },
      },
    })
    this.group.add(new Mesh(new IcosahedronGeometry(1, 24), this.shell))
    this.core = new MeshBasicMaterial({ color: new Color(...STYLES.offline.color), transparent: true, opacity: 0.85, blending: AdditiveBlending })
    this.group.add(new Mesh(new SphereGeometry(0.32, 32, 16), this.core))
    for (const [radius, tube, tilt] of [[1.38, 0.008, 0.25], [1.6, 0.005, -0.5], [1.85, 0.012, 1.2]] as const) {
      const m = new Mesh(new TorusGeometry(radius, tube, 8, 160), new MeshBasicMaterial({ color: 0x9fefff, transparent: true, opacity: 0.55, blending: AdditiveBlending }))
      m.rotation.x = Math.PI / 2 + tilt
      this.rings.push(m)
      this.group.add(m)
    }
    this.scene.add(this.group)

    this.composer = new EffectComposer(this.renderer, { frameBufferType: HalfFloatType })
    this.composer.addPass(new RenderPass(this.scene, this.camera))
    // A tight bloom, then a vignette that forces the canvas edges to transparent black: on the overlay
    // the glow must fade out inside the canvas, never end in a visible square over the game.
    const bloom = new BloomEffect({ intensity: 1.25, luminanceThreshold: 0.28, luminanceSmoothing: 0.2, mipmapBlur: true, radius: 0.55, levels: 6 })
    this.bloomPass = new EffectPass(this.camera, bloom, new VignetteEffect({ offset: 0.32, darkness: 1.0 }))
    this.composer.addPass(this.bloomPass)
    this.resize()
    window.addEventListener('resize', () => this.resize())
  }

  setCombat(on: boolean, hide: boolean): void {
    this.combat = on
    this.hidden = on && hide
    this.bloomPass.enabled = !on
    this.canvas.style.visibility = this.hidden ? 'hidden' : 'visible'
    this.renderer.setPixelRatio(on ? 1 : Math.min(window.devicePixelRatio, 1.5))
    this.resize()
  }

  start(): void {
    const loop = (now: number) => {
      this.raf = requestAnimationFrame(loop)
      this.frame(now)
    }
    this.raf = requestAnimationFrame(loop)
  }

  stop(): void {
    cancelAnimationFrame(this.raf)
  }

  private resize(): void {
    const w = this.canvas.clientWidth || 1
    const h = this.canvas.clientHeight || 1
    this.camera.aspect = w / h
    this.camera.updateProjectionMatrix()
    this.renderer.setSize(w, h, false)
    this.composer.setSize(w, h)
  }

  private frame(now: number): void {
    if (this.hidden) return
    const { mode, level } = this.source()
    this.limiter.cap = fpsFor(mode, this.combat, this.opts.combatFpsCap)
    if (!this.limiter.ready(now)) return
    const t = (now - this.t0) / 1000
    this.style = lerpStyle(this.style, STYLES[mode], 0.08)
    this.level += (audioLevel(mode, level) - this.level) * 0.35
    const u = this.shell.uniforms
    u.uTime.value = t
    u.uDisp.value = displacement(this.style, this.level)
    u.uSpeed.value = this.style.speed
    u.uGlow.value = this.style.glow
    u.uLevel.value = this.level
    ;(u.uColor.value as Color).setRGB(...this.style.color)
    this.core.color.setRGB(...this.style.color)
    this.core.opacity = 0.6 + this.level * 0.4
    this.rings.forEach((r, i) => {
      r.rotation.z += 0.004 * this.style.ring * (i % 2 ? -1 : 1) * (1 + i * 0.6)
    })
    this.group.scale.setScalar(1 + this.level * 0.08)
    this.composer.render()
  }
}
