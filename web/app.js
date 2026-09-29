import * as engine from "./engine.js";

const $ = (s, r = document) => r.querySelector(s);
const $$ = (s, r = document) => [...r.querySelectorAll(s)];
const out = $("#q-out");
const state = { tab: "video", files: { video: null, audio: null }, recording: null };

// ---------- theme ----------
$("#q-theme").addEventListener("click", () => {
  const root = document.documentElement;
  const dark = root.dataset.theme ? root.dataset.theme === "dark" : matchMedia("(prefers-color-scheme: dark)").matches;
  root.dataset.theme = dark ? "light" : "dark";
  try { localStorage.setItem("q-theme", root.dataset.theme); } catch (e) {}
});

// ---------- tabs ----------
$$(".q-tab").forEach(btn => btn.addEventListener("click", () => {
  state.tab = btn.dataset.tab;
  $$(".q-tab").forEach(b => b.setAttribute("aria-selected", String(b === btn)));
  $$(".q-pane").forEach(p => { p.hidden = p.dataset.pane !== state.tab; });
  $("#q-go").textContent = state.tab === "video" ? "تعرّف على القارئ" : "حلّل المقطع";
}));

// ---------- drop zones ----------
$$(".q-drop").forEach(zone => {
  const input = $("input", zone), kind = zone.dataset.for, label = $(".q-drop-file", zone);
  const set = f => {
    state.files[kind] = f || null;
    label.textContent = f ? `✓ ${f.name} (${(f.size / 1048576).toFixed(1)} MB)` : "";
    zone.classList.toggle("q-has-file", !!f);
    if (f && kind === "audio") clearRecording();
  };
  input.addEventListener("change", () => set(input.files[0]));
  ["dragenter", "dragover"].forEach(e => zone.addEventListener(e, ev => { ev.preventDefault(); zone.classList.add("q-drag"); }));
  ["dragleave", "drop"].forEach(e => zone.addEventListener(e, ev => { ev.preventDefault(); zone.classList.remove("q-drag"); }));
  zone.addEventListener("drop", ev => set(ev.dataTransfer.files[0]));
});

// ---------- microphone ----------
const recBtn = $("#q-rec"), preview = $("#q-rec-preview");
let rec = null, chunks = [], timer = null, t0 = 0;
function clearRecording() { state.recording = null; preview.hidden = true; preview.removeAttribute("src"); }
recBtn.addEventListener("click", async () => {
  if (rec && rec.state === "recording") { rec.stop(); return; }
  try {
    const stream = await navigator.mediaDevices.getUserMedia({ audio: true });
    rec = new MediaRecorder(stream); chunks = [];
    rec.ondataavailable = e => e.data.size && chunks.push(e.data);
    rec.onstop = () => {
      stream.getTracks().forEach(t => t.stop());
      clearInterval(timer);
      recBtn.classList.remove("q-recording");
      $(".q-rec-label", recBtn).textContent = "سجّل تاني";
      const blob = new Blob(chunks, { type: rec.mimeType || "audio/webm" });
      const ext = blob.type.includes("ogg") ? "ogg" : blob.type.includes("mp4") ? "m4a" : "webm";
      state.recording = new File([blob], `تسجيل.${ext}`, { type: blob.type });
      preview.src = URL.createObjectURL(blob); preview.hidden = false;
      state.files.audio = null; $(".q-drop[data-for=audio] .q-drop-file").textContent = "";
    };
    rec.start(); t0 = Date.now();
    recBtn.classList.add("q-recording");
    $(".q-rec-label", recBtn).textContent = "وقّف التسجيل";
    timer = setInterval(() => { $(".q-rec-time", recBtn).textContent = mmss((Date.now() - t0) / 1000); }, 250);
  } catch (e) {
    showError("مقدرناش نوصل للمايك — اسمح للمتصفح يستخدمه وجرّب تاني.");
  }
});

// ---------- submit ----------
$("#q-input").addEventListener("submit", async ev => {
  ev.preventDefault();
  const f = state.tab === "video" ? state.files.video : (state.files.audio || state.recording);
  if (!f) return showError(state.tab === "video" ? "ارفع فيديو الأول." : "ارفع مقطع صوتي أو سجّل من المايك الأول.");
  const checkContent = state.tab === "audio" || $("#q-check").checked;
  const go = $("#q-go"); go.disabled = true;
  const status = showLoading();
  try {
    const r = await engine.analyze(f, { checkContent, onStatus: status.set });
    out.innerHTML = engine.toHtml(r);
  } catch (e) {
    console.error(e);
    const m = e && e.message ? e.message : String(e);
    showError(/network|fetch|Failed to fetch|Load failed/i.test(m)
      ? "النت فصل وإحنا بنحمّل الموديلات 😕 دوس تاني — اللي اتحمّل خلاص مش هيتحمّل من الأول."
      : m);
  } finally { status.stop(); go.disabled = false; }
});

function showLoading() {
  const s0 = Date.now();
  out.innerHTML = `<div class="q-card q-loading"><div class="q-spin">۞</div>
    <div class="q-sub" id="q-load-msg">بنبدأ…</div><div class="q-dim" id="q-load-time">0:00</div>
    <div class="q-progress"><span id="q-load-bar"></span></div>
    <div class="q-dim q-small">كل التحليل بيحصل على جهازك. أول مرة بس بنحمّل الموديلات وبعدها بتتحفظ.</div></div>`;
  const iv = setInterval(() => { const t = $("#q-load-time"); if (t) t.textContent = mmss((Date.now() - s0) / 1000); }, 500);
  return {
    set: msg => {
      const m = $("#q-load-msg"), bar = $("#q-load-bar"); if (!m) return;
      m.textContent = msg;
      const p = /(\d+)%/.exec(msg) || /(\d+)\/(\d+)/.exec(msg);
      if (bar) bar.style.width = p ? (p[2] ? Math.round(100 * p[1] / p[2]) : +p[1]) + "%" : "0";
    },
    stop: () => clearInterval(iv),
  };
}

function showError(msg) {
  const d = document.createElement("div");
  d.className = "q-card"; d.innerHTML = `<div class="q-hero q-hero-unk"><div class="q-ornament">۞</div><div class="q-sub"></div></div>`;
  $(".q-sub", d).textContent = msg;
  out.replaceChildren(d);
}

function mmss(s) { s = Math.floor(s); return `${Math.floor(s / 60)}:${String(s % 60).padStart(2, "0")}`; }

// show the real number of reciters, and warm the (small) gallery up
engine.loadGallery().then(g => { $("#q-people").textContent = g.people.length; }).catch(() => {});
