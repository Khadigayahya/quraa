// Quraa in the browser: everything runs on the visitor's device (no server).
// Mirrors quraa/{audio,identify,gallery,quran_text,content,hints,report_html}.py — keep them in sync.
const SR = 16000, WIN = 6 * SR;
const CFG = window.QURAA || {};
const MODEL_BASE = (CFG.MODEL_BASE || "models").replace(/\/$/, "");
const ORT_URL = "https://cdn.jsdelivr.net/npm/onnxruntime-web@1.22.0/dist/ort.min.js";
const TRANSFORMERS_URL = "https://cdn.jsdelivr.net/npm/@huggingface/transformers@3.7.1";
const CACHE = "quraa-models-v2";

// ---------------------------------------------------------------- utils
const cp = c => String.fromCharCode(c);
const range = (a, b) => cp(a) + "-" + cp(b);
const esc = s => String(s ?? "").replace(/[&<>"']/g, c => ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;" }[c]));
const mmss = s => `${Math.floor(s / 60)}:${String(Math.floor(s % 60)).padStart(2, "0")}`;

function loadScript(src) {
  return new Promise((ok, fail) => {
    if ([...document.scripts].some(s => s.src === src)) return ok();
    const s = document.createElement("script");
    s.src = src; s.onload = ok; s.onerror = () => fail(new Error("تعذر تحميل " + src));
    document.head.appendChild(s);
  });
}

// fetch with progress + Cache API (so the 84 MB model downloads once per browser).
// Weak connections drop mid-download: retry up to 4 times, resuming with an HTTP Range request.
async function fetchCached(url, onProgress) {
  let cache = null;
  try { cache = await caches.open(CACHE); const hit = await cache.match(url); if (hit) return await hit.arrayBuffer(); } catch (e) {}
  const chunks = []; let got = 0, total = 0, lastErr = null;
  for (let attempt = 0; attempt < 5; attempt++) {
    try {
      const res = await fetch(url, got ? { headers: { Range: `bytes=${got}-` } } : {});
      if (got && res.status !== 206) { chunks.length = 0; got = 0; }          // server ignored Range: start over
      if (!res.ok) throw new Error(`تعذر تحميل ${url.split("/").pop()} (${res.status})`);
      if (!total) total = +(res.headers.get("content-range") || "").split("/")[1] || +res.headers.get("content-length") || 0;
      const reader = res.body.getReader();
      for (;;) {
        const { done, value } = await reader.read(); if (done) break;
        chunks.push(value); got += value.length; onProgress && onProgress(got, total);
      }
      if (!total || got >= total) { lastErr = null; break; }
    } catch (e) { lastErr = e; await new Promise(r => setTimeout(r, 1500 * (attempt + 1))); }
  }
  if (lastErr) throw lastErr;
  const buf = new Uint8Array(got); let o = 0; for (const c of chunks) { buf.set(c, o); o += c.length; }
  try { cache && await cache.put(url, new Response(buf, { headers: { "content-type": "application/octet-stream" } })); } catch (e) {}
  return buf.buffer;
}

// ---------------------------------------------------------------- audio
export async function decodeFile(file, maxMinutes = 10) {
  if (file.size > 400 * 1048576) throw new Error("الملف كبير جدًا (أكتر من 400MB) — قصّ جزء منه وجرّب تاني.");
  const data = await file.arrayBuffer();
  const Ctx = window.AudioContext || window.webkitAudioContext;
  const ctx = new Ctx();
  let ab;
  try { ab = await ctx.decodeAudioData(data); }
  catch (e) { throw new Error("المتصفح مقدرش يقرا الصوت من الملف ده — جرّب MP3 أو MP4 أو M4A."); }
  finally { ctx.close && ctx.close(); }
  const dur = Math.min(ab.duration, maxMinutes * 60);
  const off = new OfflineAudioContext(1, Math.max(1, Math.ceil(dur * SR)), SR);   // downmix + resample to 16 kHz
  const src = off.createBufferSource(); src.buffer = ab; src.connect(off.destination); src.start(0);
  const out = await off.startRendering();
  return { wav: out.getChannelData(0), duration: ab.duration, analysed: dur };
}

function rms(w, s, n) { let e = 0; for (let i = s; i < s + n; i++) e += w[i] * w[i]; return Math.sqrt(e / n); }

export function windows(w, winS = 6, hopS = 3, maxWindows = 40, rel = 0.3, abs = 1e-3) {
  const win = winS * SR, hop = hopS * SR;
  if (w.length < win) return rms(w, 0, w.length) > abs ? [0] : [];
  let starts = []; for (let s = 0; s + win <= w.length; s += hop) starts.push(s);
  const e = starts.map(s => rms(w, s, win));
  const med = [...e].sort((a, b) => a - b)[Math.floor(e.length / 2)];
  starts = starts.filter((s, i) => e[i] > rel * med && e[i] > abs);
  if (maxWindows && starts.length > maxWindows) {
    const k = starts.length - 1;
    starts = Array.from({ length: maxWindows }, (_, i) => starts[Math.round(i * k / (maxWindows - 1))]);
  }
  return starts;
}

function window6(w, s) {                       // exactly 6 s; short clips are repeated (like np.pad wrap)
  const x = new Float32Array(WIN);
  if (w.length >= s + WIN) { x.set(w.subarray(s, s + WIN)); return x; }
  for (let i = 0; i < WIN; i++) x[i] = w[(s + i) % w.length] || 0;
  return x;
}

// ---------------------------------------------------------------- voiceprint model + gallery
let _ecapa = null, _gallery = null;

export async function loadEcapa(onProgress) {
  if (_ecapa) return _ecapa;
  await loadScript(ORT_URL);
  const ort = window.ort;
  ort.env.wasm.numThreads = self.crossOriginIsolated ? Math.min(4, navigator.hardwareConcurrency || 2) : 1;
  const buf = await fetchCached(`${MODEL_BASE}/ecapa.onnx`, onProgress);
  _ecapa = await ort.InferenceSession.create(buf, { executionProviders: ["wasm"], graphOptimizationLevel: "all" });
  return _ecapa;
}

export async function loadGallery() {
  if (_gallery) return _gallery;
  const [info, bin] = await Promise.all([
    fetch(`${MODEL_BASE}/gallery.json`).then(r => r.json()),
    fetchCached(`${MODEL_BASE}/gallery.bin`)]);
  const f = new Float32Array(bin), D = info.dim, K = info.proj_k || D;
  let o = 0;
  const mean = info.proj_k ? f.subarray(o, o += D) : null;
  const W = info.proj_k ? f.subarray(o, o += D * K) : null;
  const C = f.subarray(o, o + info.n * K);
  _gallery = { ...info, mean, W, C, K };
  return _gallery;
}

async function embed(wavs, onStep) {           // -> array of L2-normalised Float32Array(192)
  const sess = _ecapa, ort = window.ort, out = [], B = 4;
  for (let i = 0; i < wavs.length; i += B) {
    const chunk = wavs.slice(i, i + B), x = new Float32Array(chunk.length * WIN);
    chunk.forEach((w, j) => x.set(w, j * WIN));
    const res = await sess.run({ wav: new ort.Tensor("float32", x, [chunk.length, WIN]) });
    const e = res.emb.data;
    for (let j = 0; j < chunk.length; j++) {
      const v = e.slice(j * 192, (j + 1) * 192); let n = 0; for (const a of v) n += a * a; n = Math.sqrt(n) + 1e-9;
      out.push(v.map(a => a / n));
    }
    onStep && onStep(Math.min(wavs.length, i + B), wavs.length);
  }
  return out;
}

function transform(g, e) {                      // == Gallery.transform (LDA projection + L2 norm)
  let z = e;
  if (g.W) {
    z = new Float32Array(g.K); const D = g.dim;
    for (let d = 0; d < D; d++) { const c = e[d] - g.mean[d]; if (!c) continue; for (let k = 0; k < g.K; k++) z[k] += c * g.W[d * g.K + k]; }
  }
  let n = 0; for (const a of z) n += a * a; n = Math.sqrt(n) + 1e-9;
  return z.map(a => a / n);
}

function scorePeople(g, z) {                    // best centroid per person
  const P = new Float32Array(g.people.length).fill(-1), lab = new Float32Array(g.n);
  for (let j = 0; j < g.n; j++) {
    let s = 0; const off = j * g.K; for (let k = 0; k < g.K; k++) s += z[k] * g.C[off + k];
    lab[j] = s; const p = g.person_index[j]; if (s > P[p]) P[p] = s;
  }
  return { P, lab };
}

export async function identify(wav, spans, onStep) {
  const g = await loadGallery(); await loadEcapa();
  let w = wav;
  if (spans && spans.length) {
    const parts = spans.map(([a, b]) => wav.subarray(Math.floor(a * SR), Math.floor(b * SR)));
    w = new Float32Array(parts.reduce((n, p) => n + p.length, 0)); let o = 0; for (const p of parts) { w.set(p, o); o += p.length; }
  }
  const starts = windows(w);
  if (!starts.length) return { ok: false, reason: "مفيش صوت واضح (صمت أو المقطع قصير)" };
  const E = await embed(starts.map(s => window6(w, s)), onStep);
  const nP = g.people.length, mean = new Float32Array(nP), votes = new Int32Array(nP), labMean = new Float32Array(g.n);
  for (const e of E) {
    const { P, lab } = scorePeople(g, transform(g, e));
    let b = 0; for (let p = 0; p < nP; p++) { mean[p] += P[p] / E.length; if (P[p] > P[b]) b = p; }
    votes[b]++; for (let j = 0; j < g.n; j++) labMean[j] += lab[j] / E.length;
  }
  const order = [...mean.keys()].sort((a, b) => mean[b] - mean[a]), best = order[0];
  const margin = order.length > 1 ? mean[best] - mean[order[1]] : 1;
  let bestLab = -1; for (let j = 0; j < g.n; j++) if (g.person_index[j] === best && (bestLab < 0 || labMean[j] > labMean[bestLab])) bestLab = j;
  const thr = g.threshold;
  return {
    ok: true, person: g.people[best], name: g.names[best], name_en: g.names_en[best], style: g.styles[bestLab] || "",
    similarity: mean[best], margin, votes: votes[best], n_windows: starts.length, threshold: thr,
    unknown: thr != null && mean[best] < thr && margin < g.margin_threshold,
    top: order.slice(0, 5).map(i => ({ person: g.people[i], name: g.names[i], similarity: mean[i], votes: votes[i] })),
  };
}

// ---------------------------------------------------------------- Quran text
export const SURA_NAMES = ("الفاتحة البقرة آل_عمران النساء المائدة الأنعام الأعراف الأنفال التوبة يونس هود يوسف الرعد إبراهيم " +
  "الحجر النحل الإسراء الكهف مريم طه الأنبياء الحج المؤمنون النور الفرقان الشعراء النمل القصص العنكبوت " +
  "الروم لقمان السجدة الأحزاب سبأ فاطر يس الصافات ص الزمر غافر فصلت الشورى الزخرف الدخان الجاثية الأحقاف " +
  "محمد الفتح الحجرات ق الذاريات الطور النجم القمر الرحمن الواقعة الحديد المجادلة الحشر الممتحنة الصف " +
  "الجمعة المنافقون التغابن الطلاق التحريم الملك القلم الحاقة المعارج نوح الجن المزمل المدثر القيامة " +
  "الإنسان المرسلات النبأ النازعات عبس التكوير الانفطار المطففين الانشقاق البروج الطارق الأعلى الغاشية " +
  "الفجر البلد الشمس الليل الضحى الشرح التين العلق القدر البينة الزلزلة العاديات القارعة التكاثر العصر " +
  "الهمزة الفيل قريش الماعون الكوثر الكافرون النصر المسد الإخلاص الفلق الناس").split(" ").map(s => s.replace("_", " "));

const DIAC = new RegExp("[" + range(0x610, 0x61A) + range(0x64B, 0x65F) + cp(0x670) + range(0x6D6, 0x6ED) + cp(0x640) + "]", "g");
const NON_AR = new RegExp("[^" + range(0x621, 0x64A) + " ]+", "g");
const MAP = { "أ": "ا", "إ": "ا", "آ": "ا", "ٱ": "ا", "ى": "ي", "ة": "ه", "ؤ": "و", "ئ": "ي", "ء": "" };
export function normalize(t) {
  return String(t || "").replace(DIAC, "").replace(/[أإآٱىةؤئء]/g, c => MAP[c]).replace(NON_AR, " ").split(/\s+/).filter(Boolean);
}
const ngrams = (w, n) => { const s = new Set(); for (let i = 0; i + n <= w.length; i++) s.add(w.slice(i, i + n).join(" ")); return s; };

let _quran = null;
async function quranIndex() {
  if (_quran) return _quran;
  const txt = await fetch("assets/quran-simple-clean.txt").then(r => r.text());
  const ayat = txt.split("\n").filter(Boolean).map(l => { const [s, a, t] = l.split("|"); return [+s, +a, normalize(t)]; });
  const bi = new Set(), tri = new Set();
  ayat.forEach(([s, , w], i) => {
    const nxt = i + 1 < ayat.length && ayat[i + 1][0] === s ? ayat[i + 1][2].slice(0, 2) : [];
    const ww = w.concat(nxt);
    for (let k = 0; k + 2 <= ww.length; k++) bi.add(ww[k] + " " + ww[k + 1]);
    for (let k = 0; k + 3 <= ww.length; k++) tri.add(ww[k] + " " + ww[k + 1] + " " + ww[k + 2]);
  });
  return (_quran = { bi, tri, ayat });
}
export async function quranScore(text) {
  const { bi, tri } = await quranIndex(), w = normalize(text), b = ngrams(w, 2), t = ngrams(w, 3);
  let hb = 0, ht = 0; b.forEach(x => bi.has(x) && hb++); t.forEach(x => tri.has(x) && ht++);
  return { words: w.length, bigram: b.size ? hb / b.size : 0, trigram: t.size ? ht / t.size : 0 };
}
async function locate(text) {
  const { ayat } = await quranIndex(), t = ngrams(normalize(text), 3); if (!t.size) return null;
  let best = null, bk = 0;
  for (const [s, a, w] of ayat) { let k = 0; for (let i = 0; i + 3 <= w.length; i++) if (t.has(w[i] + " " + w[i + 1] + " " + w[i + 2])) k++; if (k > bk) { bk = k; best = [s, a]; } }
  return best;
}

// ---------------------------------------------------------------- Quran vs lecture (Whisper in the browser)
let _asr = null;
export async function loadWhisper(onProgress) {
  if (_asr) return _asr;
  const { pipeline, env } = await import(TRANSFORMERS_URL);
  env.allowLocalModels = false;
  const model = CFG.WHISPER || "onnx-community/whisper-base";
  const progress_callback = p => onProgress && p.status === "progress" && onProgress(p.loaded, p.total, p.file);
  // navigator.gpu can exist without a usable adapter (old GPUs, some browsers) -> check, and fall back to wasm
  let adapter = null;
  try { adapter = navigator.gpu && await navigator.gpu.requestAdapter(); } catch (e) {}
  if (adapter) {
    try {
      _asr = await pipeline("automatic-speech-recognition", model,
        { device: "webgpu", dtype: { encoder_model: "fp32", decoder_model_merged: "q4" }, progress_callback });
      return _asr;
    } catch (e) { console.warn("WebGPU Whisper failed, using wasm", e); }
  }
  _asr = await pipeline("automatic-speech-recognition", model, { device: "wasm", dtype: "q8", progress_callback });
  return _asr;
}

function looping(text) {                         // == content._looping
  const w = normalize(text), c = new Map(); let mx = 0;
  for (let i = 0; i + 3 <= w.length; i++) { const k = w.slice(i, i + 3).join(" "); const v = (c.get(k) || 0) + 1; c.set(k, v); if (v > mx) mx = v; }
  return mx >= 4;
}

async function classifyChunk(w) {
  const r = await _asr(w, { language: "arabic", task: "transcribe" });
  const text = (r.text || "").trim(), sc = await quranScore(text);
  sc.wps = sc.words / Math.max(w.length / SR, 1);
  let label = "speech";
  if (sc.words < 5 || looping(text)) label = "unclear";      // too short, or Whisper stuck repeating itself
  else if (sc.trigram >= 0.3) label = "quran";
  else if (sc.bigram >= 0.1 && sc.wps < 1.0) label = "quran";
  else if (sc.bigram >= 0.05 && sc.wps < 0.8) label = "quran";      // tarteel pace; small Whisper garbles words
  return { label, text, ...sc };
}

export async function analyzeContent(wav, onStep, chunkS = 30, maxChunks = 4) {
  await loadWhisper();
  const n = chunkS * SR; let starts = [];
  for (let s = 0; s < Math.max(1, wav.length - Math.floor(n / 3)); s += n) starts.push(s);
  if (starts.length > maxChunks) { const k = starts.length - 1; starts = Array.from({ length: maxChunks }, (_, i) => starts[Math.round(i * k / (maxChunks - 1))]); }
  const chunks = [];
  for (const [i, s] of starts.entries()) {
    const r = await classifyChunk(wav.subarray(s, s + n));
    r.start = s / SR; r.end = Math.min(wav.length, s + n) / SR;
    if (r.label === "quran") r.aya = await locate(r.text);
    chunks.push(r); onStep && onStep(i + 1, starts.length);
  }
  const judged = chunks.filter(c => c.label !== "unclear"), q = judged.filter(c => c.label === "quran").length;
  const frac = judged.length ? q / judged.length : 0;
  const verdict = !judged.length ? "unclear" : frac >= 0.6 ? "quran" : frac <= 0.2 ? "lecture" : "mixed";
  const ayat = []; chunks.forEach(c => c.aya && !ayat.some(a => a[0] === c.aya[0] && a[1] === c.aya[1]) && ayat.push(c.aya));
  return { verdict, quran_fraction: frac, chunks, ayat };
}

// ---------------------------------------------------------------- file-name hint (== quraa/hints.py)
const STOP = new Set(["محمد", "احمد", "عبدالله", "عبدالرحمن", "علي", "بن", "ابن", "الشيخ", "القارئ", "abdul", "al", "muhammad", "mohamed", "mohammed", "ahmed", "sheikh", "shaikh", "the", "bin", "ibn"]);
const tokens = s => String(s || "").replace(DIAC, "").replace(/[أإآىةؤئ]/g, c => MAP[c] || c).toLowerCase()
  .replace(/عبد\s+ال/g, "عبدال").split(/[^\p{L}\p{N}]+/u).filter(t => t.length > 1);
let _weak = null;
function weakWords() {
  if (_weak) return _weak;
  _weak = new Set([...STOP, "surah", "sura", "surat", "quran", "rahman", "yasin", "yaseen", "kahf", "mulk", "baqarah", "full"]);
  for (const n of SURA_NAMES) for (const t of tokens(n)) { _weak.add(t); _weak.add(t.startsWith("ال") ? t.slice(2) : "ال" + t); }
  return _weak;
}
export function titleMatches(text, g, top = 3) {
  const words = new Set(tokens(text)); if (!words.size) return [];
  const weak = weakWords(), cands = g.people.map((p, i) => [g.names[i], g.names_en[i]].filter(Boolean).map(n => { const full = tokens(n); return [full, full.filter(t => !weak.has(t))]; }));
  const df = new Map(); cands.forEach(names => names.forEach(([, toks]) => new Set(toks).forEach(t => df.set(t, (df.get(t) || 0) + 1))));
  const out = [];
  cands.forEach((names, i) => {
    let score = 0;
    for (const [full, toks] of names) {
      const hit = toks.filter(t => words.has(t));
      if (full.length >= 2 && toks.length && full.every(t => words.has(t))) score = Math.max(score, 1);
      else if (hit.length && hit.length === toks.length && toks.length >= 2) score = Math.max(score, 0.9);
      else if (hit.length && hit.every(t => df.get(t) === 1) && Math.max(...hit.map(t => t.length)) >= 4) score = Math.max(score, 0.6 * hit.length / Math.max(1, toks.length) + 0.3);
    }
    if (score) out.push({ person: g.people[i], name: g.names[i], score });
  });
  return out.sort((a, b) => b.score - a.score).slice(0, top);
}

// ---------------------------------------------------------------- result card (== quraa/report_html.py)
const VERDICT = { quran: ["تلاوة قرآن", "q-ok"], lecture: ["محاضرة / كلام عادي", "q-warn"], mixed: ["تلاوة وكلام مع بعض", "q-mid"], unclear: ["مش واضح", "q-mute"] };
const CHUNK = { quran: "q-seg-quran", speech: "q-seg-speech", unclear: "q-seg-unclear" };
const pct = (s, lo = 0.25, hi = 0.75) => Math.round(100 * Math.min(1, Math.max(0, (s - lo) / (hi - lo))));

function timeline(c, dur) {
  if (!c || !c.chunks.length || !dur) return "";
  const segs = c.chunks.map(ch => `<span class="q-seg ${CHUNK[ch.label]}" style="right:${(100 * ch.start / dur).toFixed(2)}%;width:${Math.max(1.5, 100 * (ch.end - ch.start) / dur).toFixed(2)}%" title="${mmss(ch.start)}–${mmss(ch.end)}: ${esc(ch.text.slice(0, 80))}"></span>`).join("");
  return `<div class="q-timeline">${segs}</div><div class="q-legend"><span><i class="q-seg-quran"></i>قرآن</span><span><i class="q-seg-speech"></i>كلام</span><span><i class="q-seg-unclear"></i>مش واضح</span><span class="q-dim">اتحلل ${mmss(dur)}</span></div>`;
}
function contentBlock(c, dur) {
  if (!c) return "";
  const [title, cls] = VERDICT[c.verdict];
  const ayat = c.ayat.length ? `<div class="q-row">${c.ayat.slice(0, 5).map(([s, a]) => `<span class="q-chip">سورة ${SURA_NAMES[s - 1]} · ${a}</span>`).join("")}</div>` : "";
  return `<div class="q-panel"><div class="q-panel-head"><span class="q-kicker">نوع المحتوى</span><span class="q-badge ${cls}">${title}</span></div>${timeline(c, dur)}${ayat}</div>`;
}
function bars(rec) {
  return `<div class="q-panel"><div class="q-kicker">أقرب الأصوات</div>${rec.top.map((t, i) => `
    <div class="q-bar ${i ? "" : "q-bar-top"}"><span class="q-bar-name">${esc(t.name)}</span>
    <span class="q-bar-track"><span class="q-bar-fill" style="width:${pct(t.similarity)}%"></span></span>
    <span class="q-bar-val">${t.similarity.toFixed(2)}</span></div>`).join("")}</div>`;
}
function hintBlock(r, rec) {
  const hints = r.title_hints || [], voice = rec && rec.ok && !rec.unknown ? rec.person : null;
  if (hints.length && voice && hints[0].person === voice) return `<div class="q-row q-center"><span class="q-chip q-chip-ok">✓ اسم الملف كمان بيقول كده</span></div>`;
  if (hints.length) return `<div class="q-panel q-hint-box"><div class="q-kicker">📝 اسم الملف بيذكر</div><div class="q-sub">${hints.slice(0, 2).map(h => `<b>${esc(h.name)}</b>`).join("، ")}</div><div class="q-dim">ده من الاسم المكتوب مش من الصوت.</div></div>`;
  return "";
}
export function toHtml(r) {
  const c = r.content, rec = r.reciter;
  const head = `<div class="q-file"><span>🎧 ${esc(r.source)}</span><span class="q-dim">${mmss(r.duration_s)} · تحليل في ${Math.round(r.elapsed_s)} ث</span></div>`;
  if (!rec || !rec.ok) {
    const msg = !rec ? "ده مش تلاوة، فمفيش قارئ نتعرف عليه." : `مقدرناش نسمع صوت واضح (${esc(rec.reason)}).`;
    return `<div class="q-card">${head}<div class="q-hero q-hero-empty"><div class="q-ornament">۞</div><div class="q-sub">${msg}</div></div>${contentBlock(c, r.analysed_s)}</div>`;
  }
  let kicker = "القارئ", note = "", tone = "q-hero-ok";
  if (!rec.unknown) { if (rec.margin < 0.03) note = "الفرق بينه وبين التاني صغير — النتيجة مش أكيدة."; }
  else if (rec.margin >= 0.06) { kicker = "غالبًا القارئ"; tone = "q-hero-mid"; note = "درجة التطابق أقل من العتبة، بس الصوت متميّز بوضوح — غالبًا فيه صدى أو ضجيج."; }
  else { kicker = "أقرب صوت ليه"; tone = "q-hero-unk"; note = "غالبًا القارئ ده مش موجود في قاعدة البيانات لسه."; }
  const hero = `<div class="q-hero ${tone}"><div class="q-ornament">۞</div><div class="q-kicker">${kicker}</div>
    <div class="q-name">${esc(rec.name)}</div><div class="q-name-en">${esc(rec.name_en)}</div>
    <div class="q-row q-center">${rec.style ? `<span class="q-chip q-chip-gold">${esc(rec.style)}</span>` : ""}<span class="q-chip">اتفاق ${rec.votes}/${rec.n_windows} مقطع</span></div>
    <div class="q-meter"><span style="width:${pct(rec.similarity)}%"></span></div>
    <div class="q-dim">درجة التطابق ${rec.similarity.toFixed(2)}${rec.threshold ? ` · العتبة ${rec.threshold.toFixed(2)}` : ""}</div>
    ${note ? `<div class="q-note">${note}</div>` : ""}</div>`;
  return `<div class="q-card">${head}${hero}${hintBlock(r, rec)}${contentBlock(c, r.analysed_s)}${bars(rec)}</div>`;
}

// ---------------------------------------------------------------- full pipeline
export async function analyze(file, { checkContent = true, maxMinutes = 10, onStatus = () => {} } = {}) {
  const t0 = performance.now();
  onStatus("بنقرا الصوت من الملف…");
  const { wav, duration, analysed } = await decodeFile(file, maxMinutes);
  const g = await loadGallery();
  const out = { source: file.name, duration_s: duration, analysed_s: analysed, title_hints: titleMatches(file.name.replace(/\.[^.]+$/, "").replace(/_/g, " "), g) };
  let spans = null;
  if (checkContent) {
    onStatus("بنجهّز Whisper (أول مرة بس بياخد وقت)…");
    await loadWhisper((l, t) => t && onStatus(`بنحمّل موديل الكلام… ${Math.round(100 * l / t)}%`));
    const c = await analyzeContent(wav, (i, n) => onStatus(`بنفرّق القرآن عن الكلام… ${i}/${n}`));
    out.content = c;
    if (c.verdict === "lecture") { out.reciter = null; out.elapsed_s = (performance.now() - t0) / 1000; return out; }
    if (c.verdict === "mixed") spans = c.chunks.filter(ch => ch.label === "quran").map(ch => [ch.start, ch.end]);
  }
  onStatus("بنجهّز موديل البصمة…");
  await loadEcapa((l, t) => onStatus(`بنحمّل موديل البصمة (مرة واحدة بس)… ${t ? Math.round(100 * l / t) + "%" : (l / 1048576).toFixed(0) + " MB"}`));
  out.reciter = await identify(wav, spans, (i, n) => onStatus(`بنقارن الصوت ببصمات ${g.people.length} قارئ… ${i}/${n}`));
  out.elapsed_s = (performance.now() - t0) / 1000;
  return out;
}
