// Static WebGL2 ray-marcher for two-channel volumes (no server, no dependencies).
// Data: data/<specimen>/meta.json + <channel>.u8.gz (8-bit, sqrt-encoded, z-y-x order, x fastest).

const $ = (id) => document.getElementById(id);

// ------------------------------------------------------------------ shaders
const VERT = `#version 300 es
void main() {
  vec2 p = vec2(float((gl_VertexID << 1) & 2), float(gl_VertexID & 2));
  gl_Position = vec4(p * 2.0 - 1.0, 0.0, 1.0);
}`;

const FRAG = `#version 300 es
precision highp float;
precision highp sampler3D;
uniform sampler3D uVol0;
uniform sampler3D uVol1;
uniform sampler2D uLut;
uniform vec3 uSize;        // physical extent (um), x y z
uniform vec3 uVox;         // voxel size (um), x y z
uniform vec3 uTexel;       // 1 / dims, x y z
uniform mat3 uRotT;        // view -> data rotation
uniform vec2 uViewport;    // pixels
uniform float uScale;      // pixels per um
uniform vec2 uPan;         // um, view space
uniform float uCamDist;    // 0 = orthographic
uniform int uMode;         // 0 = 3D, 1 = 2D
uniform float uSliceZ;     // um, centred data coordinate (2D)
uniform int uProj;         // 2D: 0 = single z slice, 1 = max projection along z
uniform float uNz;         // number of z slices
uniform float uStep;       // um
uniform float uRelStep;    // step in voxels (attenuated mip)
uniform vec2 uVis;
uniform vec4 uClim;        // lo0 hi0 lo1 hi1 (counts)
uniform vec2 uGamma;
uniform vec2 uOpacity;
uniform vec2 uMaxC;        // 8-bit full scale (counts)
uniform ivec2 uRender;     // 0 mip, 1 attenuated mip, 2 iso, 3 average
uniform vec2 uAtten;
uniform vec2 uIso;         // counts
out vec4 outColor;

float sampleV(int c, vec3 tc) { return c == 0 ? texture(uVol0, tc).r : texture(uVol1, tc).r; }
float counts(int c, float v) { return uMaxC[c] * v * v; }
float normC(int c, float cnt) {
  float lo = c == 0 ? uClim.x : uClim.z;
  float hi = c == 0 ? uClim.y : uClim.w;
  float n = clamp((cnt - lo) / max(hi - lo, 1e-6), 0.0, 1.0);
  return pow(n, uGamma[c]);
}
vec3 lut(int c, float n) { return texture(uLut, vec2((n * 255.0 + 0.5) / 256.0, (float(c) + 0.5) / 2.0)).rgb; }

vec3 isoShade(int c, vec3 p, vec3 dData) {
  vec3 tc = p / uSize + 0.5;
  vec3 g = vec3(
    sampleV(c, tc + vec3(uTexel.x, 0, 0)) - sampleV(c, tc - vec3(uTexel.x, 0, 0)),
    sampleV(c, tc + vec3(0, uTexel.y, 0)) - sampleV(c, tc - vec3(0, uTexel.y, 0)),
    sampleV(c, tc + vec3(0, 0, uTexel.z)) - sampleV(c, tc - vec3(0, 0, uTexel.z))) / uVox;
  float gl = length(g);
  vec3 N = gl > 1e-6 ? -g / gl : -dData;
  vec3 L = -dData;                                   // headlight
  float diff = abs(dot(N, L));
  vec3 H = normalize(L - dData);
  float spec = pow(max(abs(dot(N, H)), 0.0), 32.0);
  return lut(c, 1.0) * (0.22 + 0.78 * diff) + vec3(0.25 * spec);
}

void main() {
  vec2 px = vec2(gl_FragCoord.x, uViewport.y - gl_FragCoord.y);
  vec2 v = (px - 0.5 * uViewport) / uScale + uPan;
  vec3 h = 0.5 * uSize;
  vec3 col = vec3(0.0);

  if (uMode == 1) {
    vec3 p = vec3(v, uSliceZ);
    if (any(greaterThan(abs(p.xy), h.xy))) { outColor = vec4(0, 0, 0, 1); return; }
    vec3 tc = p / uSize + 0.5;
    for (int c = 0; c < 2; c++) {
      if (uVis[c] < 0.5) continue;
      float s = 0.0;
      if (uProj == 1) {
        for (int k = 0; k < 2048; k++) {
          if (float(k) >= uNz) break;
          s = max(s, sampleV(c, vec3(tc.xy, (float(k) + 0.5) / uNz)));
        }
      } else {
        s = sampleV(c, tc);
      }
      col += lut(c, normC(c, counts(c, s))) * uOpacity[c];
    }
    outColor = vec4(min(col, 1.0), 1.0);
    return;
  }

  vec3 ov, dv;
  if (uCamDist > 0.0) { ov = vec3(uPan, -uCamDist); dv = normalize(vec3(v, 0.0) - ov); }
  else { ov = vec3(v, -2.0 * length(uSize)); dv = vec3(0.0, 0.0, 1.0); }
  vec3 o = uRotT * ov;
  vec3 d = uRotT * dv;
  d = mix(d, vec3(1e-7), vec3(lessThan(abs(d), vec3(1e-7))));
  vec3 inv = 1.0 / d;
  vec3 t0 = (-h - o) * inv, t1 = (h - o) * inv;
  vec3 tmn = min(t0, t1), tmx = max(t0, t1);
  float tn = max(max(tmn.x, tmn.y), tmn.z);
  float tf = min(min(tmx.x, tmx.y), tmx.z);
  if (tf <= max(tn, 0.0)) { outColor = vec4(0, 0, 0, 1); return; }
  tn = max(tn, 0.0);

  float jitter = fract(sin(dot(gl_FragCoord.xy, vec2(12.9898, 78.233))) * 43758.5453);
  float t = tn + jitter * uStep;
  vec2 mx = vec2(0.0), sumv = vec2(0.0), am = vec2(0.0), avg = vec2(0.0);
  bvec2 hit = bvec2(false);
  vec3 hp0 = vec3(0.0), hp1 = vec3(0.0);
  float nsteps = 0.0;
  vec2 hiC = vec2(uClim.y, uClim.w), loC = vec2(uClim.x, uClim.z);

  for (int i = 0; i < 8192; i++) {
    if (t > tf) break;
    vec3 p = o + d * t;
    vec3 tc = p / uSize + 0.5;
    for (int c = 0; c < 2; c++) {
      if (uVis[c] < 0.5) continue;
      int r = uRender[c];
      if (r == 2 && hit[c]) continue;
      float s = sampleV(c, tc);
      if (r == 0) { mx[c] = max(mx[c], s); }
      else if (r == 1) {
        float cnt = counts(c, s);
        sumv[c] += clamp((cnt - loC[c]) / max(hiC[c] - loC[c], 1e-6), 0.0, 1.0);
        am[c] = max(am[c], cnt * exp(-uAtten[c] * (sumv[c] - 1.0) * uRelStep));
      }
      else if (r == 3) { avg[c] += counts(c, s); }
      else if (counts(c, s) > uIso[c]) {
        // refine the crossing between the previous and current sample
        float a = t - uStep, b = t;
        for (int k = 0; k < 5; k++) {
          float m = 0.5 * (a + b);
          if (counts(c, sampleV(c, (o + d * m) / uSize + 0.5)) > uIso[c]) b = m; else a = m;
        }
        hit[c] = true;
        if (c == 0) hp0 = o + d * b; else hp1 = o + d * b;
      }
    }
    if ((uRender.x != 2 || hit.x || uVis.x < 0.5) && (uRender.y != 2 || hit.y || uVis.y < 0.5)
        && (uRender.x == 2 || uVis.x < 0.5) && (uRender.y == 2 || uVis.y < 0.5)) break;
    nsteps += 1.0;
    t += uStep;
  }

  for (int c = 0; c < 2; c++) {
    if (uVis[c] < 0.5) continue;
    int r = uRender[c];
    if (r == 0) col += lut(c, normC(c, counts(c, mx[c]))) * uOpacity[c];
    else if (r == 1) col += lut(c, normC(c, am[c])) * uOpacity[c];
    else if (r == 3) col += lut(c, normC(c, avg[c] / max(nsteps, 1.0))) * uOpacity[c];
    else if (hit[c]) col += isoShade(c, c == 0 ? hp0 : hp1, d) * uOpacity[c];
  }
  outColor = vec4(min(col, 1.0), 1.0);
}`;

// ------------------------------------------------------------------ small math
const quat = {
  mul(a, b) {
    return [a[3] * b[0] + a[0] * b[3] + a[1] * b[2] - a[2] * b[1],
            a[3] * b[1] - a[0] * b[2] + a[1] * b[3] + a[2] * b[0],
            a[3] * b[2] + a[0] * b[1] - a[1] * b[0] + a[2] * b[3],
            a[3] * b[3] - a[0] * b[0] - a[1] * b[1] - a[2] * b[2]];
  },
  axisAngle(ax, ang) { const s = Math.sin(ang / 2); return [ax[0] * s, ax[1] * s, ax[2] * s, Math.cos(ang / 2)]; },
  norm(q) { const l = Math.hypot(...q) || 1; return q.map((x) => x / l); },
  // rotation matrix, row-major: R[r][c]
  mat(q) {
    const [x, y, z, w] = q;
    return [[1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
            [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
            [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)]];
  },
};

// ------------------------------------------------------------------ state
const S = {
  specimens: [], colormaps: {}, meta: null, sid: null,
  layers: [], selected: 0,
  q: [0, 0, 0, 1], scale: 1, pan: [0, 0], fov: 0,
  ndisplay: 3, slice: 0, proj: "slice", interp: "linear", quality: 1,
  interacting: false, idleTimer: null, dirty: true, loadToken: 0, key: null,
};

const canvas = $("gl");
const overlay = $("overlay");
const ctx2d = overlay.getContext("2d");
const gl = canvas.getContext("webgl2", { antialias: false, preserveDrawingBuffer: true, alpha: false });
let prog, U = {}, lutTex, vao;

function fail(msg) {
  $("loading").classList.remove("hidden");
  $("loading-text").innerHTML = `<span id="error">${msg}</span>`;
  $("loading-bar").style.width = "0";
  throw new Error(msg);
}

function initGL() {
  if (!gl) fail("This browser does not support WebGL2, which the viewer needs.");
  const sh = (type, src) => {
    const s = gl.createShader(type);
    gl.shaderSource(s, src);
    gl.compileShader(s);
    if (!gl.getShaderParameter(s, gl.COMPILE_STATUS)) fail("Shader error: " + gl.getShaderInfoLog(s));
    return s;
  };
  prog = gl.createProgram();
  gl.attachShader(prog, sh(gl.VERTEX_SHADER, VERT));
  gl.attachShader(prog, sh(gl.FRAGMENT_SHADER, FRAG));
  gl.linkProgram(prog);
  if (!gl.getProgramParameter(prog, gl.LINK_STATUS)) fail("Program error: " + gl.getProgramInfoLog(prog));
  gl.useProgram(prog);
  const n = gl.getProgramParameter(prog, gl.ACTIVE_UNIFORMS);
  for (let i = 0; i < n; i++) {
    const name = gl.getActiveUniform(prog, i).name;
    U[name] = gl.getUniformLocation(prog, name);
  }
  vao = gl.createVertexArray();
  lutTex = gl.createTexture();
  gl.activeTexture(gl.TEXTURE2);
  gl.bindTexture(gl.TEXTURE_2D, lutTex);
  for (const p of [gl.TEXTURE_MIN_FILTER, gl.TEXTURE_MAG_FILTER]) gl.texParameteri(gl.TEXTURE_2D, p, gl.LINEAR);
  gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_WRAP_S, gl.CLAMP_TO_EDGE);
  gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_WRAP_T, gl.CLAMP_TO_EDGE);
  gl.uniform1i(U.uVol0, 0);
  gl.uniform1i(U.uVol1, 1);
  gl.uniform1i(U.uLut, 2);
}

function uploadLut() {
  const data = new Uint8Array(256 * 2 * 4);
  S.layers.forEach((L, c) => {
    const rgb = S.colormaps[L.colormap];
    for (let i = 0; i < 256; i++) {
      data.set([rgb[3 * i], rgb[3 * i + 1], rgb[3 * i + 2], 255], (c * 256 + i) * 4);
    }
  });
  gl.activeTexture(gl.TEXTURE2);
  gl.bindTexture(gl.TEXTURE_2D, lutTex);
  gl.pixelStorei(gl.UNPACK_ALIGNMENT, 1);
  gl.texImage2D(gl.TEXTURE_2D, 0, gl.RGBA8, 256, 2, 0, gl.RGBA, gl.UNSIGNED_BYTE, data);
}

function setFilter() {
  const f = S.interp === "nearest" ? gl.NEAREST : gl.LINEAR;
  S.layers.forEach((L, c) => {
    gl.activeTexture(gl.TEXTURE0 + c);
    gl.bindTexture(gl.TEXTURE_3D, L.tex);
    gl.texParameteri(gl.TEXTURE_3D, gl.TEXTURE_MIN_FILTER, f);
    gl.texParameteri(gl.TEXTURE_3D, gl.TEXTURE_MAG_FILTER, f);
  });
}

// ------------------------------------------------------------------ data
// ------------------------------------------------------------------ encryption (optional)
// If crypto.json exists, every data file is AES-256-GCM encrypted (12-byte nonce || ciphertext || tag) with a key
// derived from the password by PBKDF2-SHA256. Decryption happens here, in the browser; nothing is sent anywhere.
const b64 = (s) => Uint8Array.from(atob(s), (c) => c.charCodeAt(0));
const enc = (path) => (S.key ? `${path}.enc` : path);

async function openSealed(key, buf) {
  return new Uint8Array(await crypto.subtle.decrypt({ name: "AES-GCM", iv: buf.subarray(0, 12) }, key, buf.subarray(12)));
}

async function deriveKey(password, info) {
  const base = await crypto.subtle.importKey("raw", new TextEncoder().encode(password), "PBKDF2", false, ["deriveKey"]);
  return crypto.subtle.deriveKey({ name: "PBKDF2", hash: "SHA-256", salt: b64(info.salt), iterations: info.iterations },
    base, { name: "AES-GCM", length: 256 }, true, ["decrypt"]);
}

async function keyWorks(key, info) {
  try { await openSealed(key, b64(info.check)); return true; } catch { return false; }
}

async function unlock(info) {
  if (!window.crypto || !crypto.subtle) fail("Decryption needs a secure page (https:// or localhost).");
  const slot = `copepod-key-${info.salt}`;
  try {
    const saved = sessionStorage.getItem(slot);
    if (saved) {
      const key = await crypto.subtle.importKey("raw", b64(saved), "AES-GCM", true, ["decrypt"]);
      if (await keyWorks(key, info)) return key;
    }
  } catch { /* storage unavailable: ask again */ }
  $("loading").classList.add("hidden");
  $("lock").classList.remove("hidden");
  $("lock-pw").focus();
  return new Promise((resolve) => {
    $("lock-form").addEventListener("submit", async function onSubmit(e) {
      e.preventDefault();
      $("lock-go").disabled = true;
      $("lock-msg").textContent = "Checking…";
      const key = await deriveKey($("lock-pw").value, info);
      if (!(await keyWorks(key, info))) {
        $("lock-msg").textContent = "Wrong password.";
        $("lock-go").disabled = false;
        $("lock-pw").select();
        return;
      }
      if ($("lock-remember").checked) {
        try {
          const raw = new Uint8Array(await crypto.subtle.exportKey("raw", key));
          sessionStorage.setItem(slot, btoa(String.fromCharCode(...raw)));
        } catch { /* fine without */ }
      }
      $("lock-form").removeEventListener("submit", onSubmit);
      $("lock").classList.add("hidden");
      $("loading").classList.remove("hidden");
      resolve(key);
    });
  });
}

async function fetchJSON(path) {
  const bytes = await fetchBytes(enc(path), () => {});
  return JSON.parse(new TextDecoder().decode(bytes));
}

async function fetchBytes(url, onProgress) {
  const res = await fetch(url);
  if (!res.ok) throw new Error(`${url}: HTTP ${res.status}`);
  const total = Number(res.headers.get("content-length")) || 0;
  const reader = res.body.getReader();
  const parts = [];
  let got = 0;
  for (;;) {
    const { done, value } = await reader.read();
    if (done) break;
    parts.push(value);
    got += value.length;
    if (total) onProgress(Math.min(got / total, 1));
  }
  let buf = new Uint8Array(got);
  let off = 0;
  for (const p of parts) { buf.set(p, off); off += p.length; }
  if (S.key) {
    try { buf = await openSealed(S.key, buf); } catch { throw new Error(`${url}: could not decrypt (wrong password or damaged file)`); }
  }
  if (buf[0] === 0x1f && buf[1] === 0x8b) {           // gzip (unless the server already decompressed it)
    const ds = new Blob([buf]).stream().pipeThrough(new DecompressionStream("gzip"));
    buf = new Uint8Array(await new Response(ds).arrayBuffer());
  }
  return buf;
}

function histogram(u8) {
  const h = new Float64Array(256);
  for (let i = 0; i < u8.length; i += 7) h[u8[i]]++;
  return h;
}

function percentileCounts(L, p) {
  // percentile of the non-zero voxels, from the 8-bit histogram, returned in counts
  const h = L.hist;
  let tot = 0;
  for (let i = 1; i < 256; i++) tot += h[i];
  let acc = 0;
  for (let i = 1; i < 256; i++) {
    acc += h[i];
    if (acc >= p / 100 * tot) return L.maxCounts * (i / 255) ** 2;
  }
  return L.maxCounts;
}

async function loadSpecimen(sid) {
  const token = ++S.loadToken;
  $("loading").classList.remove("hidden");
  $("loading-text").textContent = `Loading ${sid}…`;
  $("loading-bar").style.width = "0";
  const spec = S.specimens.find((x) => x.id === sid);
  const base = `data/${spec.folder || sid}`;
  const meta = await fetchJSON(`${base}/meta.json`);
  const n = meta.channels.length;
  const progress = new Array(n).fill(0);
  const bytes = await Promise.all(meta.channels.map((ch, i) =>
    fetchBytes(enc(`${base}/${ch.file}`), (f) => {
      progress[i] = f;
      $("loading-bar").style.width = `${100 * progress.reduce((a, b) => a + b, 0) / n}%`;
    })));
  if (token !== S.loadToken) return;                  // a newer request won
  const [nz, ny, nx] = meta.shape_zyx;
  const keep = $("keep").checked && S.layers.length === n;
  const old = S.layers;
  S.layers.forEach((L) => gl.deleteTexture(L.tex));
  S.layers = meta.channels.map((ch, c) => {
    if (bytes[c].length !== nx * ny * nz) fail(`${ch.file}: expected ${nx * ny * nz} bytes, got ${bytes[c].length}`);
    const tex = gl.createTexture();
    gl.activeTexture(gl.TEXTURE0 + c);
    gl.bindTexture(gl.TEXTURE_3D, tex);
    gl.pixelStorei(gl.UNPACK_ALIGNMENT, 1);
    gl.texImage3D(gl.TEXTURE_3D, 0, gl.R8, nx, ny, nz, 0, gl.RED, gl.UNSIGNED_BYTE, bytes[c]);
    for (const p of [gl.TEXTURE_WRAP_S, gl.TEXTURE_WRAP_T, gl.TEXTURE_WRAP_R]) gl.texParameteri(gl.TEXTURE_3D, p, gl.CLAMP_TO_EDGE);
    const d = ch.default;
    const L = {
      name: ch.name, data: bytes[c], tex, maxCounts: ch.max_counts, defaults: d,
      visible: true, opacity: d.opacity, clim: [...d.clim], gamma: d.gamma, colormap: d.colormap,
      rendering: d.rendering, attenuation: d.attenuation, iso: d.iso_threshold,
    };
    if (keep) {
      const o = old[c];
      Object.assign(L, { visible: o.visible, opacity: o.opacity, clim: [...o.clim], gamma: o.gamma,
                         colormap: o.colormap, rendering: o.rendering, attenuation: o.attenuation, iso: o.iso });
    }
    L.hist = histogram(L.data);
    return L;
  });
  S.meta = meta;
  S.sid = sid;
  S.probe = null;
  S.slice = Math.floor(nz / 2);
  history.replaceState(null, "", `#${sid}`);
  setFilter();
  uploadLut();
  buildLayerList();
  selectLayer(Math.min(S.selected, S.layers.length - 1));
  updateViewControls();
  if (!keep || !S.everShown) resetView();
  S.everShown = true;
  $("loading").classList.add("hidden");
  requestRender();
}

// ------------------------------------------------------------------ geometry
function extent() {
  const [nz, ny, nx] = S.meta.shape_zyx;
  const [dz, dy, dx] = S.meta.voxel_um_zyx;
  return { size: [nx * dx, ny * dy, nz * dz], vox: [dx, dy, dz], dims: [nx, ny, nz] };
}

function resetView() {
  const { size } = extent();
  const w = canvas.clientWidth, h = canvas.clientHeight;
  S.q = S.ndisplay === 3 ? quat.norm(quat.mul(quat.axisAngle([1, 0, 0], -0.35), quat.axisAngle([0, 1, 0], 0.25))) : [0, 0, 0, 1];
  S.scale = 0.88 * Math.min(w / size[0], h / size[1]);
  S.pan = [0, 0];
  requestRender();
}

// ------------------------------------------------------------------ rendering
function requestRender() {
  if (S.dirty === "queued") return;
  S.dirty = "queued";
  requestAnimationFrame(render);
}

function render() {
  S.dirty = false;
  if (!S.meta) return;
  const dpr = Math.min(window.devicePixelRatio || 1, IS_TOUCH ? 1.25 : 2);
  const res = S.interacting && S.ndisplay === 3 ? (IS_TOUCH ? 0.4 : 0.5) : 1;
  const W = Math.max(1, Math.round(canvas.clientWidth * dpr * res));
  const H = Math.max(1, Math.round(canvas.clientHeight * dpr * res));
  if (canvas.width !== W || canvas.height !== H) { canvas.width = W; canvas.height = H; }
  gl.viewport(0, 0, W, H);
  const { size, vox, dims } = extent();
  const R = quat.mat(S.ndisplay === 3 ? S.q : [0, 0, 0, 1]);
  // uniformMatrix3fv expects column-major; feeding R row by row gives the shader R^T (view -> data)
  const RT = new Float32Array([R[0][0], R[0][1], R[0][2], R[1][0], R[1][1], R[1][2], R[2][0], R[2][1], R[2][2]]);
  const pxScale = S.scale * dpr * res;
  const minVox = Math.min(...vox);
  const step = minVox * S.quality * (S.interacting ? 1.5 : 1);
  const camDist = S.fov > 0 ? (0.5 * canvas.clientHeight / S.scale) / Math.tan((S.fov * Math.PI / 180) / 2) : 0;
  const L0 = S.layers[0], L1 = S.layers[1] || S.layers[0];
  const rmap = { mip: 0, attenuated_mip: 1, iso: 2, average: 3 };
  gl.useProgram(prog);
  gl.bindVertexArray(vao);
  gl.uniform3fv(U.uSize, size);
  gl.uniform3fv(U.uVox, vox);
  gl.uniform3fv(U.uTexel, dims.map((d) => 1 / d));
  gl.uniformMatrix3fv(U.uRotT, false, RT);
  gl.uniform2f(U.uViewport, W, H);
  gl.uniform1f(U.uScale, pxScale);
  gl.uniform2f(U.uPan, S.pan[0], S.pan[1]);
  gl.uniform1f(U.uCamDist, camDist);
  gl.uniform1i(U.uMode, S.ndisplay === 2 ? 1 : 0);
  gl.uniform1f(U.uSliceZ, (S.slice + 0.5) * vox[2] - size[2] / 2);
  gl.uniform1i(U.uProj, S.proj === "mip" ? 1 : 0);
  gl.uniform1f(U.uNz, dims[2]);
  gl.uniform1f(U.uStep, step);
  gl.uniform1f(U.uRelStep, step / minVox);
  gl.uniform2f(U.uVis, L0.visible ? 1 : 0, S.layers[1] && L1.visible ? 1 : 0);
  gl.uniform4f(U.uClim, L0.clim[0], L0.clim[1], L1.clim[0], L1.clim[1]);
  gl.uniform2f(U.uGamma, L0.gamma, L1.gamma);
  gl.uniform2f(U.uOpacity, L0.opacity, L1.opacity);
  gl.uniform2f(U.uMaxC, L0.maxCounts, L1.maxCounts);
  gl.uniform2i(U.uRender, rmap[L0.rendering], rmap[L1.rendering]);
  gl.uniform2f(U.uAtten, L0.attenuation, L1.attenuation);
  gl.uniform2f(U.uIso, L0.iso, L1.iso);
  S.layers.forEach((L, c) => { gl.activeTexture(gl.TEXTURE0 + c); gl.bindTexture(gl.TEXTURE_3D, L.tex); });
  gl.activeTexture(gl.TEXTURE2);
  gl.bindTexture(gl.TEXTURE_2D, lutTex);
  gl.drawArrays(gl.TRIANGLES, 0, 3);
  drawOverlay();
}

function niceLength(target) {
  const p = 10 ** Math.floor(Math.log10(target));
  for (const m of [1, 2, 5, 10]) if (m * p >= target) return m * p;
  return 10 * p;
}

function drawOverlay() {
  const dpr = Math.min(window.devicePixelRatio || 1, 2);
  const w = overlay.clientWidth, h = overlay.clientHeight;
  if (overlay.width !== Math.round(w * dpr) || overlay.height !== Math.round(h * dpr)) {
    overlay.width = Math.round(w * dpr);
    overlay.height = Math.round(h * dpr);
  }
  const g = ctx2d;
  g.setTransform(dpr, 0, 0, dpr, 0, 0);
  g.clearRect(0, 0, w, h);
  g.font = "12px system-ui, sans-serif";
  // scale bar (exact in the plane through the volume centre)
  const um = niceLength(110 / S.scale);
  const len = um * S.scale;
  const x1 = w - 18, x0 = x1 - len, y = h - 40;
  g.strokeStyle = "#fff";
  g.fillStyle = "#fff";
  g.lineWidth = 2;
  g.beginPath();
  g.moveTo(x0, y); g.lineTo(x1, y);
  g.moveTo(x0, y - 5); g.lineTo(x0, y + 5);
  g.moveTo(x1, y - 5); g.lineTo(x1, y + 5);
  g.stroke();
  g.textAlign = "center";
  g.fillText(`${um >= 1 ? um : um.toFixed(1)} µm`, (x0 + x1) / 2, y - 8);
  // tapped point (2D, touch)
  if (S.ndisplay === 2 && S.probe) {
    const px = (S.probe[0] - S.pan[0]) * S.scale + w / 2, py = (S.probe[1] - S.pan[1]) * S.scale + h / 2;
    g.strokeStyle = "#fff";
    g.lineWidth = 1.5;
    g.beginPath(); g.arc(px, py, 7, 0, 2 * Math.PI);
    g.moveTo(px - 12, py); g.lineTo(px - 4, py); g.moveTo(px + 4, py); g.lineTo(px + 12, py);
    g.moveTo(px, py - 12); g.lineTo(px, py - 4); g.moveTo(px, py + 4); g.lineTo(px, py + 12);
    g.stroke();
  }
  // axes (napari colours: x cyan, y magenta, z yellow)
  const R = quat.mat(S.ndisplay === 3 ? S.q : [0, 0, 0, 1]);
  const ox = 52, oy = h - 62, a = 34;
  const axes = [["x", [1, 0, 0], "#00e5ff"], ["y", [0, 1, 0], "#ff3df5"], ["z", [0, 0, 1], "#ffe100"]];
  const shown = S.ndisplay === 3 ? axes : axes.slice(0, 2);
  shown.map(([n, e, c]) => [n, [R[0][0] * e[0] + R[0][1] * e[1] + R[0][2] * e[2],
                                R[1][0] * e[0] + R[1][1] * e[1] + R[1][2] * e[2],
                                R[2][0] * e[0] + R[2][1] * e[1] + R[2][2] * e[2]], c])
    .sort((p, q) => q[1][2] - p[1][2])               // far axes first
    .forEach(([n, v, c]) => {
      const ex = ox + v[0] * a, ey = oy + v[1] * a;
      g.strokeStyle = c;
      g.fillStyle = c;
      g.lineWidth = 2;
      g.beginPath(); g.moveTo(ox, oy); g.lineTo(ex, ey); g.stroke();
      g.beginPath(); g.arc(ex, ey, 2.5, 0, 2 * Math.PI); g.fill();
      g.textAlign = "center";
      g.fillText(n, ox + v[0] * (a + 11), oy + v[1] * (a + 11) + 4);
    });
}

// ------------------------------------------------------------------ interaction
// Mouse: drag rotates (3D) or pans (2D); Shift/right/middle-drag pans; wheel zooms; double-click resets.
// Touch: one finger rotates (3D) or pans (2D); two fingers pinch-zoom and pan; double-tap resets;
//        a tap in 2D shows the coordinates and values under the finger.
const IS_TOUCH = window.matchMedia("(pointer: coarse)").matches;
const pointers = new Map();          // pointerId -> {x, y}
let gesture = null;                  // {mode, ...}
let lastTap = { t: 0, x: 0, y: 0 };
const TAP_MS = 280, TAP_PX = 10, DOUBLE_MS = 320, DOUBLE_PX = 30;

function canvasXY(x, y) {
  const r = canvas.getBoundingClientRect();
  return [x - r.left - r.width / 2, y - r.top - r.height / 2];
}

function zoomAbout(cx, cy, factor) {
  const [mx, my] = canvasXY(cx, cy);
  const before = [mx / S.scale + S.pan[0], my / S.scale + S.pan[1]];
  S.scale = Math.min(Math.max(S.scale * factor, 0.02), 200);
  S.pan = [before[0] - mx / S.scale, before[1] - my / S.scale];
}

function rotateBy(dx, dy) {
  const k = 0.008;
  const qx = quat.axisAngle([0, 1, 0], -dx * k);       // view axes: x right, y down, z into the screen
  const qy = quat.axisAngle([1, 0, 0], dy * k);
  S.q = quat.norm(quat.mul(qy, quat.mul(qx, S.q)));
}

function twoFingerState() {
  const [a, b] = [...pointers.values()];
  return { cx: (a.x + b.x) / 2, cy: (a.y + b.y) / 2, dist: Math.max(Math.hypot(a.x - b.x, a.y - b.y), 1) };
}

canvas.addEventListener("pointerdown", (e) => {
  try { canvas.setPointerCapture(e.pointerId); } catch { /* pointer already gone */ }
  pointers.set(e.pointerId, { x: e.clientX, y: e.clientY });
  if (pointers.size === 1) {
    const pan = e.shiftKey || e.button === 1 || e.button === 2 || S.ndisplay === 2;
    gesture = { mode: pan ? "pan" : "rotate", x: e.clientX, y: e.clientY, x0: e.clientX, y0: e.clientY,
                t0: performance.now(), moved: false, touch: e.pointerType !== "mouse" };
  } else if (pointers.size === 2) {
    gesture = { mode: "pinch", ...twoFingerState(), moved: true };
  }
  canvas.classList.add("dragging");
});

canvas.addEventListener("pointermove", (e) => {
  if (!pointers.has(e.pointerId)) {
    if (S.ndisplay === 2 && e.pointerType === "mouse") updateStatus(e.clientX, e.clientY);
    return;
  }
  pointers.set(e.pointerId, { x: e.clientX, y: e.clientY });
  if (!gesture) return;
  if (gesture.mode === "pinch" && pointers.size >= 2) {
    const st = twoFingerState();
    zoomAbout(st.cx, st.cy, st.dist / gesture.dist);
    S.pan[0] -= (st.cx - gesture.cx) / S.scale;
    S.pan[1] -= (st.cy - gesture.cy) / S.scale;
    Object.assign(gesture, st);
    interact();
    return;
  }
  if (gesture.mode === "pinch") return;               // one finger left after a pinch: wait for lift
  const dx = e.clientX - gesture.x, dy = e.clientY - gesture.y;
  gesture.x = e.clientX;
  gesture.y = e.clientY;
  if (Math.hypot(e.clientX - gesture.x0, e.clientY - gesture.y0) > TAP_PX) gesture.moved = true;
  if (!gesture.moved) return;
  if (gesture.mode === "pan") {
    S.pan[0] -= dx / S.scale;
    S.pan[1] -= dy / S.scale;
  } else {
    rotateBy(dx, dy);
  }
  interact();
});

function endPointer(e) {
  if (!pointers.has(e.pointerId)) return;
  pointers.delete(e.pointerId);
  const g = gesture;
  if (pointers.size === 0) {
    gesture = null;
    canvas.classList.remove("dragging");
    // a tap: quick, no movement
    if (g && g.mode !== "pinch" && !g.moved && e.type === "pointerup" && performance.now() - g.t0 < TAP_MS) {
      const now = performance.now();
      if (g.touch && now - lastTap.t < DOUBLE_MS && Math.hypot(e.clientX - lastTap.x, e.clientY - lastTap.y) < DOUBLE_PX) {
        lastTap.t = 0;
        S.probe = null;
        resetView();
      } else {
        lastTap = { t: now, x: e.clientX, y: e.clientY };
        if (S.ndisplay === 2 && g.touch) { probeAt(e.clientX, e.clientY); }
      }
    }
  } else if (pointers.size === 1 && g && g.mode === "pinch") {
    gesture = { mode: "pinch-end" };                  // ignore the remaining finger until it lifts
  }
}
canvas.addEventListener("pointerup", endPointer);
canvas.addEventListener("pointercancel", endPointer);
canvas.addEventListener("contextmenu", (e) => e.preventDefault());
canvas.addEventListener("dblclick", (e) => { if (!IS_TOUCH) resetView(); e.preventDefault(); });
canvas.addEventListener("wheel", (e) => {
  e.preventDefault();
  zoomAbout(e.clientX, e.clientY, Math.exp(-e.deltaY * (e.deltaMode === 1 ? 0.05 : 0.0015)));
  interact();
}, { passive: false });
canvas.addEventListener("pointerleave", (e) => { if (S.ndisplay === 2 && e.pointerType === "mouse") setStatusHelp(); });
// Safari: keep its own page pinch-zoom out of the canvas
for (const ev of ["gesturestart", "gesturechange", "gestureend"]) canvas.addEventListener(ev, (e) => e.preventDefault());

function probeAt(x, y) {
  S.probe = dataXY(x, y);
  updateStatus(x, y);
  requestRender();
}

function dataXY(x, y) {
  const [mx, my] = canvasXY(x, y);
  return [mx / S.scale + S.pan[0], my / S.scale + S.pan[1]];
}

function interact() {
  S.interacting = true;
  clearTimeout(S.idleTimer);
  S.idleTimer = setTimeout(() => { S.interacting = false; requestRender(); }, 180);
  requestRender();
}

window.addEventListener("keydown", (e) => {
  if (e.target.matches("input[type=number], select")) return;
  if ((e.ctrlKey || e.metaKey) && e.key.toLowerCase() === "y") { e.preventDefault(); toggleNdisplay(); }
  else if (e.key === "Home") resetView();
  else if (S.ndisplay === 2 && S.proj === "slice" && (e.key === "ArrowLeft" || e.key === "ArrowRight") && !e.target.matches("input[type=range]")) {
    setSlice(S.slice + (e.key === "ArrowRight" ? 1 : -1));
  }
});
window.addEventListener("resize", () => requestRender());

function updateStatus(x, y) {
  const { size, vox, dims } = extent();
  const [vx, vy] = dataXY(x, y);
  const i = Math.floor((vx + size[0] / 2) / vox[0]), j = Math.floor((vy + size[1] / 2) / vox[1]);
  if (i < 0 || j < 0 || i >= dims[0] || j >= dims[1]) { setStatusHelp(); return; }
  const plane = dims[0] * dims[1];
  const valueAt = (L) => {
    if (S.proj !== "mip") return L.data[S.slice * plane + j * dims[0] + i];
    let m = 0;
    for (let k = 0; k < dims[2]; k++) m = Math.max(m, L.data[k * plane + j * dims[0] + i]);
    return m;
  };
  const vals = S.layers.map((L) => `${L.name} ${Math.round(L.maxCounts * (valueAt(L) / 255) ** 2)}`).join(" · ");
  const zText = S.proj === "mip" ? "max over z" : `z ${(S.slice * vox[2]).toFixed(1)}`;
  $("status").textContent = `[${zText}, y ${(j * vox[1]).toFixed(1)}, x ${(i * vox[0]).toFixed(1)}] µm   ${vals} counts (8-bit display copy)`;
}

function setStatusHelp() {
  if (IS_TOUCH) {
    $("status").textContent = S.ndisplay === 3
      ? "Drag: rotate · Two fingers: pinch to zoom, drag to pan · Double-tap: reset"
      : "Drag: pan · Two fingers: pinch to zoom · Tap: coordinates and values · Double-tap: reset";
    return;
  }
  $("status").textContent = S.ndisplay === 3
    ? "Drag: rotate · Shift-drag or right-drag: pan · Wheel: zoom · Double-click or Home: reset · Ctrl+Y: 2D/3D"
    : S.proj === "mip"
      ? "Max projection along z · Drag: pan · Wheel: zoom · Hover: coordinates and maximum values · Ctrl+Y: 3D"
      : "Drag: pan · Wheel: zoom · ←/→: change slice · Hover: coordinates and values · Ctrl+Y: 3D";
}

// ------------------------------------------------------------------ UI
const EYE = '<svg viewBox="0 0 24 24"><path d="M2 12s3.6-7 10-7 10 7 10 7-3.6 7-10 7S2 12 2 12z"/><circle cx="12" cy="12" r="3"/></svg>';
const EYE_OFF = '<svg viewBox="0 0 24 24"><path d="M2 12s3.6-7 10-7c2 0 3.7.7 5.1 1.6M22 12s-3.6 7-10 7c-2 0-3.7-.7-5.1-1.6"/><path d="M3 3l18 18"/></svg>';

function gradientCss(name) {
  const rgb = S.colormaps[name];
  const stops = [0, 64, 128, 192, 255].map((i) => `rgb(${rgb[3 * i]},${rgb[3 * i + 1]},${rgb[3 * i + 2]}) ${(i / 255 * 100).toFixed(0)}%`);
  return `linear-gradient(90deg, ${stops.join(",")})`;
}

function buildLayerList() {
  const ul = $("layer-list");
  ul.innerHTML = "";
  // napari lists the top layer first
  [...S.layers.keys()].reverse().forEach((c) => {
    const L = S.layers[c];
    const li = document.createElement("li");
    li.className = (c === S.selected ? "selected " : "") + (L.visible ? "" : "hidden");
    li.innerHTML = `<button class="eye" aria-label="Toggle visibility of ${L.name}" aria-pressed="${L.visible}">${L.visible ? EYE : EYE_OFF}</button>
      <span class="name">${L.name}</span><span class="swatch" style="background:${gradientCss(L.colormap)}"></span>`;
    li.querySelector(".eye").addEventListener("click", (e) => {
      e.stopPropagation();
      L.visible = !L.visible;
      buildLayerList();
      requestRender();
    });
    li.addEventListener("click", () => selectLayer(c));
    ul.appendChild(li);
  });
}

const toSlider = (L, cnt) => Math.round(1000 * Math.sqrt(Math.max(cnt, 0) / L.maxCounts));
const fromSlider = (L, s) => L.maxCounts * (s / 1000) ** 2;
const fmt = (x) => (x >= 100 ? Math.round(x).toString() : x.toFixed(1));

function selectLayer(c) {
  S.selected = c;
  const L = S.layers[c];
  $("lc-title").textContent = `Layer controls: ${L.name}`;
  $("lc-opacity").value = L.opacity;
  $("lc-opacity-v").textContent = L.opacity.toFixed(2);
  $("lc-clim-lo").value = toSlider(L, L.clim[0]);
  $("lc-clim-hi").value = toSlider(L, L.clim[1]);
  $("lc-clim-lo-n").value = fmt(L.clim[0]);
  $("lc-clim-hi-n").value = fmt(L.clim[1]);
  $("lc-gamma").value = L.gamma;
  $("lc-gamma-v").textContent = L.gamma.toFixed(2);
  $("lc-cmap").value = L.colormap;
  $("lc-swatch").style.background = gradientCss(L.colormap);
  $("lc-render").value = L.rendering;
  $("lc-att").value = L.attenuation;
  $("lc-att-v").textContent = L.attenuation.toFixed(2);
  $("lc-iso").value = toSlider(L, L.iso);
  $("lc-iso-v").textContent = fmt(L.iso);
  $("lc-att-row").style.display = L.rendering === "attenuated_mip" ? "" : "none";
  $("lc-iso-row").style.display = L.rendering === "iso" ? "" : "none";
  buildLayerList();
}

function bindLayerControls() {
  const cur = () => S.layers[S.selected];
  $("lc-opacity").addEventListener("input", (e) => { cur().opacity = +e.target.value; selectLayer(S.selected); requestRender(); });
  const setClim = (lo, hi) => {
    const L = cur();
    L.clim = [Math.max(0, Math.min(lo, hi - 1e-3)), Math.max(hi, lo + 1e-3)];
    selectLayer(S.selected);
    requestRender();
  };
  $("lc-clim-lo").addEventListener("input", (e) => {
    const L = cur();
    const lo = fromSlider(L, +e.target.value);
    setClim(Math.min(lo, L.clim[1] * 0.999), L.clim[1]);
  });
  $("lc-clim-hi").addEventListener("input", (e) => {
    const L = cur();
    const hi = fromSlider(L, +e.target.value);
    setClim(L.clim[0], Math.max(hi, L.clim[0] * 1.001 + 1e-3));
  });
  $("lc-clim-lo-n").addEventListener("change", (e) => setClim(+e.target.value, cur().clim[1]));
  $("lc-clim-hi-n").addEventListener("change", (e) => setClim(cur().clim[0], +e.target.value));
  $("lc-auto").addEventListener("click", () => { const L = cur(); setClim(percentileCounts(L, 0.5), percentileCounts(L, 99.95)); });
  $("lc-reset").addEventListener("click", () => { const L = cur(); setClim(...L.defaults.clim); });
  $("lc-gamma").addEventListener("input", (e) => { cur().gamma = +e.target.value; selectLayer(S.selected); requestRender(); });
  $("lc-cmap").addEventListener("change", (e) => { cur().colormap = e.target.value; uploadLut(); selectLayer(S.selected); requestRender(); });
  $("lc-render").addEventListener("change", (e) => { cur().rendering = e.target.value; selectLayer(S.selected); requestRender(); });
  $("lc-att").addEventListener("input", (e) => { cur().attenuation = +e.target.value; selectLayer(S.selected); requestRender(); });
  $("lc-iso").addEventListener("input", (e) => { const L = cur(); L.iso = fromSlider(L, +e.target.value); selectLayer(S.selected); requestRender(); });
}

function toggleNdisplay() {
  S.ndisplay = S.ndisplay === 3 ? 2 : 3;
  S.probe = null;
  updateViewControls();
  const keepScale = S.scale;
  resetView();
  S.scale = keepScale;
  setStatusHelp();
  requestRender();
}

function setSlice(k) {
  S.slice = Math.max(0, Math.min(S.meta.shape_zyx[0] - 1, k));
  $("slice").value = S.slice;
  $("slice-v").textContent = `${(S.slice * S.meta.voxel_um_zyx[0]).toFixed(0)} µm`;
  requestRender();
}

function updateViewControls() {
  $("btn-ndisplay").textContent = S.ndisplay === 3 ? "2D" : "3D";
  $("btn-ndisplay").title = S.ndisplay === 3 ? "Switch to 2D slicing (Ctrl+Y)" : "Switch to 3D rendering (Ctrl+Y)";
  $("proj-row").style.display = S.ndisplay === 2 ? "" : "none";
  $("slice-row").style.display = S.ndisplay === 2 && S.proj === "slice" ? "" : "none";
  $("proj").value = S.proj;
  $("slice").max = S.meta.shape_zyx[0] - 1;
  setSlice(S.slice);
  $("persp-v").textContent = `${S.fov}°`;
}

function bindViewControls() {
  $("btn-ndisplay").addEventListener("click", toggleNdisplay);
  $("btn-reset").addEventListener("click", resetView);
  $("btn-shot").addEventListener("click", screenshot);
  $("slice").addEventListener("input", (e) => setSlice(+e.target.value));
  $("proj").addEventListener("change", (e) => { S.proj = e.target.value; updateViewControls(); setStatusHelp(); requestRender(); });
  $("persp").addEventListener("input", (e) => { S.fov = +e.target.value; $("persp-v").textContent = `${S.fov}°`; requestRender(); });
  $("interp").addEventListener("change", (e) => { S.interp = e.target.value; setFilter(); requestRender(); });
  $("quality").addEventListener("change", (e) => { S.quality = +e.target.value; requestRender(); });
  $("specimen").addEventListener("change", (e) => loadSpecimen(e.target.value).catch((err) => fail(err.message)));
}

function screenshot() {
  render();                                           // full resolution, then copy both canvases
  const c = document.createElement("canvas");
  c.width = canvas.width;
  c.height = canvas.height;
  const g = c.getContext("2d");
  g.drawImage(canvas, 0, 0);
  g.drawImage(overlay, 0, 0, c.width, c.height);
  c.toBlob((b) => {
    const a = document.createElement("a");
    a.href = URL.createObjectURL(b);
    a.download = `${S.sid}_${S.ndisplay === 3 ? "3d" : S.proj === "mip" ? "2d_mip" : `2d_z${S.slice}`}.png`;
    a.click();
    setTimeout(() => URL.revokeObjectURL(a.href), 2000);
  });
}

// ------------------------------------------------------------------ start
async function main() {
  initGL();
  bindLayerControls();
  bindViewControls();
  setStatusHelp();
  const cinfo = await fetch("crypto.json").then((r) => (r.ok ? r.json() : null)).catch(() => null);
  if (cinfo) S.key = await unlock(cinfo);
  const [spec, cmaps] = await Promise.all([
    fetchJSON("specimens.json"),
    fetch("colormaps.json").then((r) => r.json()),
  ]);
  S.specimens = spec.specimens;
  S.colormaps = cmaps;
  $("specimen").innerHTML = S.specimens.map((s) => `<option value="${s.id}">${s.label}</option>`).join("");
  $("lc-cmap").innerHTML = Object.keys(cmaps).map((n) => `<option value="${n}">${n}</option>`).join("");
  const want = decodeURIComponent(location.hash.slice(1));
  const sid = S.specimens.some((s) => s.id === want) ? want : S.specimens[0].id;
  $("specimen").value = sid;
  await loadSpecimen(sid);
  window.addEventListener("hashchange", () => {
    const h = decodeURIComponent(location.hash.slice(1));
    if (h !== S.sid && S.specimens.some((s) => s.id === h)) {
      $("specimen").value = h;
      loadSpecimen(h).catch((err) => fail(err.message));
    }
  });
}

main().catch((err) => { console.error(err); if (!$("error")) fail(err.message); });
