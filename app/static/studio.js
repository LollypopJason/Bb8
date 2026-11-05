// BB-8 Studio front-end: builds the Three.js scene from the server's scene descriptor,
// applies streamed body poses, draws force arrows and live charts, and wires the control panel.
import * as THREE from "three";
import { OrbitControls } from "/static/vendor/three/OrbitControls.js";

THREE.Object3D.DEFAULT_UP.set(0, 0, 1);            // MuJoCo is z-up

const $ = (id) => document.getElementById(id);
const state = { scene: null, status: null, version: -1, bodies: [], geomMeshes: [], keys: new Set(), ws: null,
                trainEpisodes: [] };

// ------------------------------------------------------------------ helpers
async function api(path, body, method) {
  const opts = body === undefined ? { method: method || "GET" } :
    { method: method || "POST", headers: { "Content-Type": "application/json" }, body: JSON.stringify(body) };
  const res = await fetch(path, opts);
  const data = await res.json().catch(() => ({}));
  if (!res.ok) { toast(data.detail || res.statusText); throw new Error(data.detail || res.statusText); }
  return data;
}
let toastTimer = null;
function toast(msg) {
  const el = $("toast");
  el.textContent = msg; el.classList.remove("hidden");
  clearTimeout(toastTimer); toastTimer = setTimeout(() => el.classList.add("hidden"), 6000);
}
function cssVar(name) { return getComputedStyle(document.documentElement).getPropertyValue(name).trim(); }

// ------------------------------------------------------------------ three.js setup
const canvas = $("three");
const renderer = new THREE.WebGLRenderer({ canvas, antialias: true });
renderer.setPixelRatio(Math.min(window.devicePixelRatio, 2));
renderer.shadowMap.enabled = true;
const scene3 = new THREE.Scene();
const camera = new THREE.PerspectiveCamera(45, 1, 0.01, 100);
camera.position.set(1.6, -1.6, 1.2);
const controls = new OrbitControls(camera, canvas);
controls.target.set(0, 0, 0.3);
controls.enableDamping = true;
scene3.add(new THREE.HemisphereLight(0xdfe8ff, 0x30302a, 0.9));
const sun = new THREE.DirectionalLight(0xffffff, 1.6);
sun.position.set(3, -2, 6); sun.castShadow = true;
sun.shadow.mapSize.set(2048, 2048);
Object.assign(sun.shadow.camera, { left: -3, right: 3, top: 3, bottom: -3, near: 0.1, far: 20 });
scene3.add(sun); scene3.add(sun.target);
const root = new THREE.Group();
scene3.add(root);
const arrowGroup = new THREE.Group();
scene3.add(arrowGroup);
const targetMarker = new THREE.Mesh(new THREE.RingGeometry(0.06, 0.08, 32),
  new THREE.MeshBasicMaterial({ color: 0x18c964, transparent: true, opacity: 0.8, side: THREE.DoubleSide }));
targetMarker.position.z = 0.003; targetMarker.visible = false;
scene3.add(targetMarker);

function applyTheme() {
  const dark = window.matchMedia("(prefers-color-scheme: dark)").matches;
  scene3.background = new THREE.Color(dark ? 0x1a1e24 : 0xdfe6ee);
}
applyTheme();
window.matchMedia("(prefers-color-scheme: dark)").addEventListener("change", applyTheme);

function checkerTexture(c1, c2, n, accent) {
  const cv = document.createElement("canvas"); cv.width = cv.height = 256;
  const g = cv.getContext("2d"); const s = 256 / n;
  for (let i = 0; i < n; i++) for (let j = 0; j < n; j++) {
    g.fillStyle = (i + j) % 2 ? c1 : c2; g.fillRect(i * s, j * s, s, s);
  }
  if (accent) { g.strokeStyle = accent; g.lineWidth = 3; g.strokeRect(1, 1, 254, 254); }
  const tex = new THREE.CanvasTexture(cv);
  tex.wrapS = tex.wrapT = THREE.RepeatWrapping; tex.colorSpace = THREE.SRGBColorSpace;
  return tex;
}

function materialFor(g) {
  const [r, gg, b, a] = g.rgba;
  const opts = { color: new THREE.Color(r, gg, b), roughness: 0.55, metalness: 0.05 };
  if (a < 0.999) { opts.transparent = true; opts.opacity = a; opts.depthWrite = false; }
  if (g.material === "mat_grid") {
    const t = checkerTexture("#333a42", "#272c33", 8, "#4a525c");
    t.repeat.set(g.size[0] || 10, g.size[1] || 10);
    opts.map = t; opts.color = new THREE.Color(1, 1, 1);
  } else if (g.material === "mat_ball") {
    const t = checkerTexture("#efeadc", "#dcd3be", 8, "#e8701c"); t.repeat.set(4, 2);
    opts.map = t; opts.color = new THREE.Color(1, 1, 1);
  } else if (g.material === "mat_metal") { opts.metalness = 0.7; opts.roughness = 0.3; }
  return new THREE.MeshStandardMaterial(opts);
}

function geometryFor(g, desc) {
  const s = g.size;
  switch (g.type) {
    case "sphere": return new THREE.SphereGeometry(s[0], 40, 20);
    case "ellipsoid": { const geo = new THREE.SphereGeometry(1, 40, 20); geo.scale(s[0], s[1], s[2]); return geo; }
    case "box": return new THREE.BoxGeometry(2 * s[0], 2 * s[1], 2 * s[2]);
    case "cylinder": { const geo = new THREE.CylinderGeometry(s[0], s[0], 2 * s[1], 40); geo.rotateX(Math.PI / 2); return geo; }
    case "capsule": { const geo = new THREE.CapsuleGeometry(s[0], 2 * s[1], 8, 20); geo.rotateX(Math.PI / 2); return geo; }
    case "plane": return new THREE.PlaneGeometry(2 * (s[0] || 10), 2 * (s[1] || 10));
    case "mesh": {
      const m = desc.meshes[String(g.mesh)];
      const geo = new THREE.BufferGeometry();
      geo.setAttribute("position", new THREE.Float32BufferAttribute(m.vertices, 3));
      geo.setIndex(m.faces); geo.computeVertexNormals();
      return geo;
    }
    case "hfield": {
      const h = desc.hfields[String(g.hfield)];
      const [sx, sy, sz] = h.size;
      const geo = new THREE.PlaneGeometry(2 * sx, 2 * sy, h.ncol - 1, h.nrow - 1);
      const pos = geo.attributes.position;
      for (let i = 0; i < pos.count; i++) {
        const rowTop = Math.floor(i / h.ncol), col = i % h.ncol;
        const row = h.nrow - 1 - rowTop;                       // MuJoCo row 0 is at -y
        pos.setZ(i, h.data[row * h.ncol + col] * sz);
      }
      geo.computeVertexNormals();
      return geo;
    }
  }
  return null;
}

function clearScene() {
  for (const obj of [...root.children]) root.remove(obj);
  root.traverse((o) => { if (o.geometry) o.geometry.dispose(); });
  state.bodies = []; state.geomMeshes = [];
}

function buildScene(desc) {
  clearScene();
  state.scene = desc;
  for (const b of desc.bodies) {
    const grp = new THREE.Group(); grp.name = b.name; root.add(grp); state.bodies.push(grp);
  }
  for (const g of desc.geoms) {
    const geo = geometryFor(g, desc);
    if (!geo) continue;
    const mesh = new THREE.Mesh(geo, materialFor(g));
    mesh.position.set(...g.pos);
    mesh.quaternion.set(g.quat[1], g.quat[2], g.quat[3], g.quat[0]);
    mesh.castShadow = g.type !== "plane" && g.type !== "hfield" && g.rgba[3] > 0.5;
    mesh.receiveShadow = true;
    mesh.userData.group = g.group;
    state.bodies[g.body].add(mesh);
    state.geomMeshes.push(mesh);
  }
  applyVisibility();
  $("scenedesc").textContent = desc.description || "";
  const pend = desc.example === "inverted_pendulum";
  $("t_vel").textContent = pend ? "Joint positions [m, rad]" : desc.example === "kiwi_drive" || desc.example === "single_omniwheel"
    ? "Chassis velocity [m/s, rad/s]" : "Ball velocity [m/s]";
  $("t_trq").textContent = pend ? "Cart force [N]" : "Wheel torque [N·m]";
  fillBodySelects(desc);
  resetCharts();
}

function applyVisibility() {
  const layers = $("showlayers").checked, collision = $("showcollision").checked;
  for (const m of state.geomMeshes) {
    const g = m.userData.group;
    m.visible = g <= 1 || (g === 2 && layers) || (g === 3 && collision);
  }
}
["showlayers", "showcollision"].forEach((id) => $(id).addEventListener("change", applyVisibility));

function fillBodySelects(desc) {
  const names = desc.bodies.filter((b) => b.id > 0 && !b.mocap && b.name && !/roller|hub/.test(b.name)).map((b) => b.name);
  const push = $("pushbody"); push.innerHTML = "";
  for (const n of names) push.add(new Option(n, n, n === "head", n === "head"));
  const mp = $("meshparent"); mp.innerHTML = "";
  mp.add(new Option("world (free body)", "world"));
  for (const n of names) mp.add(new Option(n, n));
}

// arrows
const arrowPool = [];
function setArrows(list) {
  const show = $("showforces").checked;
  let i = 0;
  if (show) for (const a of list) {
    const f = new THREE.Vector3(...a.f); const len = f.length();
    if (len < 1e-6) continue;
    const scale = a.kind === "push" ? 0.02 : 0.006;
    let arrow = arrowPool[i];
    if (!arrow) { arrow = new THREE.ArrowHelper(new THREE.Vector3(1, 0, 0), new THREE.Vector3(), 1, 0xffffff); arrowPool.push(arrow); arrowGroup.add(arrow); }
    arrow.visible = true;
    const color = a.kind === "push" ? 0xe53935 : 0x26a69a;
    arrow.setColor(color);
    const L = Math.min(len * scale, 1.2);
    const origin = new THREE.Vector3(...a.p);
    if (a.kind === "push") origin.sub(f.clone().normalize().multiplyScalar(L));   // push arrows point at the body
    arrow.position.copy(origin);
    arrow.setDirection(f.normalize());
    arrow.setLength(L, Math.min(0.06, 0.3 * L + 0.01), Math.min(0.03, 0.15 * L + 0.005));
    i++;
  }
  for (; i < arrowPool.length; i++) arrowPool[i].visible = false;
}

// ------------------------------------------------------------------ charts
class Chart {
  constructor(canvas, window_s = 10) { this.cv = canvas; this.win = window_s; this.series = new Map(); this.xmax = 0; }
  push(x, name, y) {
    if (!Number.isFinite(y)) return;
    if (!this.series.has(name)) this.series.set(name, []);
    const s = this.series.get(name); s.push([x, y]);
    if (s.length > 4000) s.splice(0, s.length - 4000);
    this.xmax = Math.max(this.xmax, x);
  }
  clear() { this.series.clear(); this.xmax = 0; }
  draw() {
    const cv = this.cv, dpr = window.devicePixelRatio || 1;
    const w = cv.clientWidth, h = cv.clientHeight;
    if (cv.width !== Math.round(w * dpr) || cv.height !== Math.round(h * dpr)) { cv.width = w * dpr; cv.height = h * dpr; }
    const g = cv.getContext("2d"); g.setTransform(dpr, 0, 0, dpr, 0, 0); g.clearRect(0, 0, w, h);
    const x0 = this.win ? this.xmax - this.win : Math.min(...[...this.series.values()].map((s) => s.length ? s[0][0] : 0), 0);
    let lo = Infinity, hi = -Infinity;
    for (const s of this.series.values()) for (const [x, y] of s) if (x >= x0) { lo = Math.min(lo, y); hi = Math.max(hi, y); }
    if (!Number.isFinite(lo)) return;
    if (hi - lo < 1e-6) { hi += 0.5; lo -= 0.5; }
    const pad = 0.08 * (hi - lo); lo -= pad; hi += pad;
    const L = 34, R = 4, T = 4, B = 14;
    const X = (x) => L + (x - x0) / Math.max(this.xmax - x0, 1e-6) * (w - L - R);
    const Y = (y) => T + (hi - y) / (hi - lo) * (h - T - B);
    g.strokeStyle = cssVar("--grid"); g.fillStyle = cssVar("--muted"); g.font = "10px system-ui"; g.lineWidth = 1;
    for (let k = 0; k <= 3; k++) {
      const v = lo + (hi - lo) * k / 3, yy = Y(v);
      g.beginPath(); g.moveTo(L, yy); g.lineTo(w - R, yy); g.stroke();
      g.fillText(Math.abs(v) >= 100 ? v.toFixed(0) : v.toFixed(2), 2, yy + 3);
    }
    if (lo < 0 && hi > 0) { g.strokeStyle = cssVar("--muted"); g.beginPath(); g.moveTo(L, Y(0)); g.lineTo(w - R, Y(0)); g.stroke(); }
    const palette = ["#2a78d6", "#e8701c", "#2e9e5b", "#9c4fd6", "#d6444f", "#1aa3a3", "#b38600", "#6b7480", "#e05fa0"];
    let ci = 0, lx = L + 4;
    for (const [name, s] of this.series) {
      const col = palette[ci++ % palette.length];
      g.strokeStyle = col; g.lineWidth = 1.5; g.beginPath();
      let started = false;
      for (const [x, y] of s) { if (x < x0) continue; if (!started) { g.moveTo(X(x), Y(y)); started = true; } else g.lineTo(X(x), Y(y)); }
      g.stroke();
      g.fillStyle = col; g.fillText(name, lx, h - 3); lx += g.measureText(name).width + 10;
    }
  }
}
const charts = { tilt: new Chart($("c_tilt")), vel: new Chart($("c_vel")), trq: new Chart($("c_trq")),
                 rew: new Chart($("c_rew")), train: new Chart($("c_train"), 0) };
function resetCharts() { for (const k of ["tilt", "vel", "trq", "rew"]) charts[k].clear(); }

function addTelemetry(rows) {
  for (const r of rows) {
    const t = r.t;
    if ("roll_deg" in r) { charts.tilt.push(t, "roll", r.roll_deg); charts.tilt.push(t, "pitch", r.pitch_deg); }
    if ("ball_vx" in r) {
      charts.vel.push(t, "vx", r.ball_vx); charts.vel.push(t, "vy", r.ball_vy);
      charts.vel.push(t, "vx*", r.target_vx); charts.vel.push(t, "vy*", r.target_vy);
    } else if ("body_vx" in r) {
      charts.vel.push(t, "vx", r.body_vx); charts.vel.push(t, "vy", r.body_vy); charts.vel.push(t, "wz", r.body_wz);
    } else if (r.qpos) { r.qpos.forEach((q, i) => charts.vel.push(t, `q${i}`, q)); }
    const trq = r.wheel_torque || r.actuator_force;
    if (trq) trq.forEach((v, i) => charts.trq.push(t, `u${i}`, v));
    if (r.reward_terms) for (const [k, v] of Object.entries(r.reward_terms)) if (k !== "fall") charts.rew.push(t, k, v);
  }
}

// ------------------------------------------------------------------ websocket
function connect() {
  const ws = new WebSocket(`${location.protocol === "https:" ? "wss" : "ws"}://${location.host}/ws`);
  state.ws = ws;
  ws.onmessage = (ev) => {
    const msg = JSON.parse(ev.data);
    if (msg.type === "scene") { state.version = msg.scene.version; buildScene(msg.scene); applyStatus(msg.status); }
    else if (msg.type === "frame") onFrame(msg);
    else if (msg.type === "events") for (const e of msg.events) onTrainEvent(e.event);
  };
  ws.onclose = () => setTimeout(connect, 1000);
}

const tmpV = new THREE.Vector3();
function onFrame(f) {
  if (f.version !== state.version) return;
  const n = state.bodies.length;
  for (let i = 0; i < n; i++) {
    const b = state.bodies[i];
    b.position.set(f.xpos[3 * i], f.xpos[3 * i + 1], f.xpos[3 * i + 2]);
    b.quaternion.set(f.xquat[4 * i + 1], f.xquat[4 * i + 2], f.xquat[4 * i + 3], f.xquat[4 * i]);
  }
  setArrows(f.arrows);
  if (f.target) { targetMarker.visible = true; targetMarker.position.set(f.target[0], f.target[1], 0.003); }
  else targetMarker.visible = false;
  $("simtime").textContent = `t = ${f.t.toFixed(2)} s`;
  $("modelabel").textContent = f.mode + (f.paused ? " (paused)" : "");
  $("fallen").classList.toggle("hidden", !f.fallen);
  if (f.message) toast(f.message);
  addTelemetry(f.telemetry);
  if ($("follow").checked && state.scene && state.scene.focus_body) {
    const fb = state.scene.bodies.find((b) => b.name === state.scene.focus_body);
    if (fb) {
      tmpV.set(f.xpos[3 * fb.id], f.xpos[3 * fb.id + 1], f.xpos[3 * fb.id + 2]);
      const d = tmpV.clone().sub(controls.target).multiplyScalar(0.1);
      controls.target.add(d); camera.position.add(d);
      sun.position.set(tmpV.x + 3, tmpV.y - 2, 6); sun.target.position.copy(tmpV);
    }
  }
}

function onTrainEvent(ev) {
  if (ev.type === "episode") {
    state.trainEpisodes.push(ev);
    charts.train.push(ev.timesteps, "return", ev.reward);
    const last = state.trainEpisodes.slice(-20);
    charts.train.push(ev.timesteps, "mean(20)", last.reduce((a, e) => a + e.reward, 0) / last.length);
  } else if (ev.type === "progress") {
    $("trainprog").value = ev.fraction;
    $("trainstate").textContent = `${ev.timesteps.toLocaleString()} steps · ${ev.fps.toFixed(0)} steps/s`;
  } else if (ev.type === "checkpoint") {
    $("trainstate").textContent = `checkpoint ${ev.tag} saved`; refreshPolicies();
  } else if (ev.type === "finished") {
    $("trainstate").textContent = `training ${ev.state} at ${ev.timesteps.toLocaleString()} steps`; refreshPolicies();
  } else if (ev.type === "error") { toast(`training error: ${ev.error}`); }
}

// ------------------------------------------------------------------ panel
function applyStatus(st) {
  state.status = st;
  $("example").value = st.example;
  const p = st.params;
  const wheel = p.head ? p.head.wheel : p.wheel;
  const motor = p.head ? p.head.motor : p.motor;
  $("omni").disabled = !wheel; if (wheel) $("omni").value = wheel.mode;
  $("drive").disabled = !motor || st.example === "inverted_pendulum"; if (motor) $("drive").value = motor.control_mode;
  const variants = { inverted_pendulum: ["cartpole", "spherical"], planar_balancer: ["1d", "2d"] }[st.example];
  $("variantwrap").classList.toggle("hidden", !variants);
  if (variants) { $("variant").innerHTML = ""; for (const v of variants) $("variant").add(new Option(v, v, false, v === p.variant)); }
  $("pause").textContent = st.paused ? "Resume" : "Pause";
  $("ballpanel").classList.toggle("hidden", !p.ball);
  if (p.ball) {
    $("b_r0").value = (p.ball.inner_radius * 1000).toFixed(1);
    $("b_tf").value = (p.ball.foam_thickness * 1000).toFixed(1);
    $("b_rf").value = p.ball.foam_density;
    $("b_te").value = (p.ball.epoxy_thickness * 1000).toFixed(2);
    $("b_re").value = p.ball.epoxy_density;
    $("b_mu").value = p.ball.surface.sliding_friction;
    previewBall();
  }
  $("headpanel").classList.toggle("hidden", !p.head);
  if (p.head) {
    $("h_n").value = p.head.num_wheels; $("h_alpha").value = p.head.zenith_angle_deg;
    $("h_mass").value = p.head.chassis_mass; $("h_com").value = p.head.com_height;
    $("h_mag").checked = p.head.magnet.enabled; $("h_magf").value = p.head.magnet.force;
    $("h_cw").checked = p.head.counterweight.enabled; $("h_cwm").value = p.head.counterweight.mass;
  }
  $("t_slope").value = st.sim.terrain.floor_slope_deg;
  $("t_bumps").checked = st.sim.terrain.bumps.length > 0;
  $("paramjson").value = JSON.stringify({ params: st.params, sim: st.sim, hardware: st.hardware }, null, 1);
  // controller
  const modes = $("modes"); modes.innerHTML = "";
  const labels = { manual: "Manual", pid: "Cascaded PID", lqr: "LQR", rl: "RL policy", off: "Off" };
  for (const m of st.modes) {
    const b = document.createElement("button"); b.textContent = labels[m]; b.classList.toggle("active", m === st.mode);
    b.onclick = () => setMode(m); modes.appendChild(b);
  }
  $("pidbox").classList.toggle("hidden", !(st.pid && st.mode === "pid"));
  if (st.pid) for (const k of ["kp", "ki", "kd", "outer_pos", "outer_vel"]) {
    const el = $(`pid_${k}`); el.value = st.pid[k]; el.nextElementSibling.textContent = (+st.pid[k]).toFixed(2);
  }
  $("lqrbox").classList.toggle("hidden", st.mode !== "lqr" || !st.pid);
  $("lqrinfo").textContent = st.lqr ? `closed-loop ρ = ${st.lqr.spectral_radius.toFixed(4)}` : "";
  $("rlbox").classList.toggle("hidden", !st.modes.includes("rl"));
  // hardware
  $("noise").checked = st.hardware.noise_enabled; $("curve").checked = st.hardware.torque_speed_curve;
  $("latencyon").checked = st.hardware.latency_enabled;
  $("latency").value = Math.round(st.hardware.latency.delay * 1000);
  $("latency").nextElementSibling.textContent = `${$("latency").value} ms`;
  $("record").checked = st.recording;
  if (st.training) $("trainstate").textContent = `run ${st.training.run}: ${st.training.state}`;
}

async function refreshStatus() { applyStatus(await api("/api/status")); }

async function setMode(mode, extra = {}) {
  try { applyStatus(await api("/api/controller", { mode, ...extra })); } catch (e) { /* toast shown */ }
}

$("load").onclick = async () => {
  const body = { example: $("example").value };
  if (!$("omni").disabled) body.omni_mode = $("omni").value;
  if (!$("drive").disabled) body.control_mode = $("drive").value;
  if (!$("variantwrap").classList.contains("hidden")) body.variant = $("variant").value;
  applyStatus(await api("/api/scene", body));
};
$("example").onchange = async () => { applyStatus(await api("/api/scene", { example: $("example").value })); };
$("reset").onclick = async () => applyStatus(await api("/api/sim", { reset: true }));
$("pause").onclick = async () => applyStatus(await api("/api/sim", { paused: !state.status.paused }));
$("speed").oninput = (e) => { $("speedout").textContent = `${(+e.target.value).toFixed(1)}×`; };
$("speed").onchange = (e) => api("/api/sim", { speed: +e.target.value });
$("record").onchange = (e) => api("/api/sim", { recording: e.target.checked });

function ballBody() {
  return { inner_radius: +$("b_r0").value / 1000, foam_thickness: +$("b_tf").value / 1000,
           foam_density: +$("b_rf").value, epoxy_thickness: +$("b_te").value / 1000,
           epoxy_density: +$("b_re").value,
           surface: { ...state.status.params.ball.surface, sliding_friction: +$("b_mu").value } };
}
let previewTimer = null;
async function previewBall() {
  clearTimeout(previewTimer);
  previewTimer = setTimeout(async () => {
    try {
      const r = await api("/api/ball/preview", ballBody());
      const rows = r.layers.map((l) => `<tr><td>${l.name.replace("_", " ")}</td><td>${(l.thickness * 1e3).toFixed(1)}</td>` +
        `<td>${l.mass.toFixed(3)}</td><td>${l.inertia.toFixed(5)}</td></tr>`).join("");
      $("balltable").innerHTML = `<tr><th>layer</th><th>t [mm]</th><th>m [kg]</th><th>I [kg·m²]</th></tr>${rows}` +
        `<tr class="total"><td>total</td><td>${((r.radii_m.R2 - r.radii_m.R0) * 1e3).toFixed(1)}</td>` +
        `<td>${r.total_mass_kg.toFixed(3)}</td><td>${r.inertia_kgm2.toFixed(5)}</td></tr>` +
        `<tr><td colspan="4">R₂ = ${(r.radii_m.R2 * 1e3).toFixed(1)} mm · I/(mR²) = ${r.inertia_ratio_I_over_mR2.toFixed(3)}</td></tr>`;
      $("ballerr").textContent = "";
    } catch (e) { $("ballerr").textContent = String(e.message); }
  }, 150);
}
for (const id of ["b_r0", "b_tf", "b_rf", "b_te", "b_re", "b_mu"]) $(id).addEventListener("input", previewBall);
$("ballapply").onclick = async () => { await api("/api/ball/apply", ballBody()); refreshStatus(); };

$("headapply").onclick = async () => {
  const st = state.status;
  const params = structuredClone(st.params), sim = structuredClone(st.sim);
  if (params.head) {
    Object.assign(params.head, { num_wheels: +$("h_n").value, zenith_angle_deg: +$("h_alpha").value,
      chassis_mass: +$("h_mass").value, com_height: +$("h_com").value });
    params.head.magnet.enabled = $("h_mag").checked; params.head.magnet.force = +$("h_magf").value;
    params.head.counterweight.enabled = $("h_cw").checked; params.head.counterweight.mass = +$("h_cwm").value;
  }
  sim.terrain.floor_slope_deg = +$("t_slope").value;
  sim.terrain.bumps = $("t_bumps").checked ?
    [{ type: "capsule", pos: [1.0, 0, 0], size: [0.012, 0.8], euler_deg: [90, 0, 0], rgba: [0.45, 0.35, 0.25, 1] },
     { type: "capsule", pos: [-1.0, 0, 0], size: [0.012, 0.8], euler_deg: [90, 0, 0], rgba: [0.45, 0.35, 0.25, 1] },
     { type: "capsule", pos: [0, 1.0, 0], size: [0.012, 0.8], euler_deg: [0, 90, 0], rgba: [0.45, 0.35, 0.25, 1] }] : [];
  applyStatus(await api("/api/params", { params, sim }));
};
$("paramapply").onclick = async () => {
  let body; try { body = JSON.parse($("paramjson").value); } catch (e) { toast(`invalid JSON: ${e.message}`); return; }
  applyStatus(await api("/api/params", body));
};

for (const k of ["kp", "ki", "kd", "outer_pos", "outer_vel"]) {
  const el = $(`pid_${k}`);
  el.oninput = () => { el.nextElementSibling.textContent = (+el.value).toFixed(2); };
  el.onchange = () => setMode("pid", { pid: { [k]: +el.value } });
}
$("lqrdesign").onclick = () => setMode("lqr", { lqr: {
  roll: +$("q_tilt").value, pitch: +$("q_tilt").value, roll_rate: +$("q_rate").value, pitch_rate: +$("q_rate").value,
  ex: +$("q_pos").value, ey: +$("q_pos").value, vx: +$("q_vel").value, vy: +$("q_vel").value } });

async function refreshPolicies() {
  const list = await api("/api/policies");
  const sel = $("policies"); const prev = sel.value; sel.innerHTML = "";
  for (const p of list) sel.add(new Option(p.name, p.path, false, p.path === prev));
}
$("policyrefresh").onclick = refreshPolicies;
$("policyload").onclick = async () => {
  if (!$("policies").value) { toast("no policy selected"); return; }
  await api("/api/policy/load", { path: $("policies").value }); refreshStatus();
};

$("teleop").oninput = (e) => { e.target.nextElementSibling.textContent = `${(+e.target.value).toFixed(2)} m/s`; };
$("teleop").onchange = (e) => api("/api/teleop", { speed: +e.target.value });

document.querySelectorAll("[data-push]").forEach((btn) => btn.addEventListener("click", () => {
  let dir = btn.dataset.push;
  let [x, y] = dir === "random" ? (() => { const a = Math.random() * 2 * Math.PI; return [Math.cos(a), Math.sin(a)]; })()
    : dir.split(",").map(Number);
  const F = +$("pushf").value;
  api("/api/push", { body: $("pushbody").value, force: [F * x, F * y, 0], duration: +$("pushd").value / 1000 });
}));

$("latency").oninput = (e) => { e.target.nextElementSibling.textContent = `${e.target.value} ms`; };
$("hwapply").onclick = async () => applyStatus(await api("/api/hardware", {
  noise: $("noise").checked, torque_curve: $("curve").checked, latency_enabled: $("latencyon").checked,
  latency_ms: +$("latency").value }));

$("meshupload").onclick = async () => {
  const f = $("meshfile").files[0];
  if (!f) { toast("choose a mesh file first"); return; }
  const fd = new FormData();
  fd.append("file", f); fd.append("scale", $("meshscale").value); fd.append("parent", $("meshparent").value);
  fd.append("dynamic", "true"); fd.append("density", $("meshdensity").value); fd.append("z", $("meshz").value);
  const res = await fetch("/api/mesh", { method: "POST", body: fd });
  const data = await res.json();
  if (!res.ok) { toast(data.detail || "upload failed"); return; }
  $("meshinfo").textContent = `${data.name}: m = ${data.mass.toFixed(4)} kg, V = ${(data.volume * 1e6).toFixed(1)} cm³\n` +
    `CoM = ${data.com.map((v) => v.toFixed(4)).join(", ")}\n${data.notes.join("\n")}`;
  refreshStatus();
};
$("primadd").onclick = async () => {
  const t = $("primtype").value, s = +$("primsize").value;
  const size = { box: [s, s, s], sphere: [s], cylinder: [s, s], capsule: [s / 2, s] }[t];
  const hex = $("primcolor").value;
  const rgba = [1, 3, 5].map((i) => parseInt(hex.slice(i, i + 2), 16) / 255).concat([1]);
  applyStatus(await api("/api/primitive", { name: `prop_${Date.now() % 100000}`, type: t, size,
    pos: [+$("primx").value, +$("primy").value, s + 0.02], dynamic: $("primdyn").checked, density: 300, rgba }));
};
$("extrasclear").onclick = async () => applyStatus(await api("/api/extras/clear", {}));

$("trainstart").onclick = async () => {
  state.trainEpisodes = []; charts.train.clear();
  const st = await api("/api/train/start", { algo: $("algo").value, steps: +$("steps").value,
    n_envs: +$("nenvs").value, task: $("task").value });
  $("trainstate").textContent = `run ${st.run}: ${st.state}`;
};
$("trainstop").onclick = () => api("/api/train/stop", {});
$("trainload").onclick = async () => { await api("/api/policy/load", { path: "latest" }); refreshStatus(); };

// keyboard teleoperation
const keyEls = {}; document.querySelectorAll("#keyviz span").forEach((s) => { keyEls[s.textContent.toLowerCase()] = s; });
function sendKeys() {
  for (const [k, el] of Object.entries(keyEls)) el.classList.toggle("on", state.keys.has(k));
  if (state.ws && state.ws.readyState === 1) state.ws.send(JSON.stringify({ type: "keys", keys: [...state.keys] }));
}
window.addEventListener("keydown", (e) => {
  if (["INPUT", "TEXTAREA", "SELECT"].includes(document.activeElement.tagName)) return;
  const k = e.key.toLowerCase(); if (!"wasdqe".includes(k) || k.length !== 1 || state.keys.has(k)) return;
  state.keys.add(k); sendKeys();
});
window.addEventListener("keyup", (e) => { const k = e.key.toLowerCase(); if (state.keys.delete(k)) sendKeys(); });
window.addEventListener("blur", () => { state.keys.clear(); sendKeys(); });

// ------------------------------------------------------------------ main loop
function resize() {
  const vp = $("viewport"); const w = vp.clientWidth, h = vp.clientHeight;
  renderer.setSize(w, h, false); camera.aspect = w / Math.max(h, 1); camera.updateProjectionMatrix();
}
window.addEventListener("resize", resize);
let lastChart = 0;
function animate(t) {
  requestAnimationFrame(animate);
  controls.update();
  renderer.render(scene3, camera);
  if (t - lastChart > 100) { lastChart = t; for (const c of Object.values(charts)) c.draw(); }
}

async function init() {
  resize();
  const examples = await api("/api/examples");
  for (const e of examples) $("example").add(new Option(`${e.number}. ${e.title}`, e.key));
  await refreshStatus();
  refreshPolicies();
  const tr = await api("/api/train/status");
  if (tr.metrics) for (const ev of tr.metrics.episodes) onTrainEvent(ev);
  connect();
  requestAnimationFrame(animate);
}
init();
