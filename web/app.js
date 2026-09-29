(() => {
  const API = (window.QURAA_API || "").replace(/\/$/, "");
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

  $$(".q-ex").forEach(b => b.addEventListener("click", () => { $("#q-url").value = b.dataset.url; }));

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
        state.recording = new File([blob], "recording." + (blob.type.includes("ogg") ? "ogg" : blob.type.includes("mp4") ? "m4a" : "webm"), { type: blob.type });
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
  const LOADING = ["بنجهّز الصوت…", "بنسمع التلاوة…", "بنفرّق القرآن عن الكلام…", "بنقارن الصوت ببصمات القراء…", "لحظات وتظهر النتيجة…"];
  $("#q-input").addEventListener("submit", async ev => {
    ev.preventDefault();
    const fd = new FormData();
    if (state.tab === "video") {
      const url = $("#q-url").value.trim(), f = state.files.video;
      if (!url && !f) return showError("حط رابط أو ارفع فيديو الأول.");
      if (url) fd.append("url", url); else fd.append("file", f);
      fd.append("check_content", $("#q-check").checked ? "true" : "false");
    } else {
      const f = state.files.audio || state.recording;
      if (!f) return showError("ارفع مقطع صوتي أو سجّل من المايك الأول.");
      fd.append("file", f);
      fd.append("check_content", "true");
    }
    const go = $("#q-go"); go.disabled = true;
    const stop = showLoading();
    try {
      const res = await fetch(`${API}/analyze`, { method: "POST", body: fd });
      const data = await res.json().catch(() => ({}));
      if (!res.ok) throw new Error(friendly(data.detail || `HTTP ${res.status}`));
      out.innerHTML = data.html;
    } catch (e) {
      showError(e.message === "Failed to fetch" ? "مقدرناش نوصل للسيرفر — اتأكد من النت وجرّب تاني." : e.message);
    } finally { stop(); go.disabled = false; }
  });

  function friendly(msg) {
    if (/confirm you.?re not a bot|Sign in/i.test(msg)) return "يوتيوب رفض التحميل من السيرفر دلوقتي 😕 نزّل الفيديو وارفعه من جهازك بدل الرابط.";
    if (/Unsupported URL|is not a valid URL/i.test(msg)) return "الرابط ده مش مدعوم — جرّب رابط يوتيوب أو ارفع الملف.";
    if (/No audio track/i.test(msg)) return "الملف ده مفيهوش صوت.";
    return msg;
  }

  function showLoading() {
    let i = 0; const s0 = Date.now();
    out.innerHTML = `<div class="q-card q-loading"><div class="q-spin">۞</div>
      <div class="q-sub" id="q-load-msg">${LOADING[0]}</div><div class="q-dim" id="q-load-time">0:00</div>
      <div class="q-dim q-small">أول طلب بعد فترة ممكن ياخد دقيقة زيادة علشان السيرفر بيصحى</div></div>`;
    const iv = setInterval(() => {
      const m = $("#q-load-msg"), t = $("#q-load-time");
      if (!m) return clearInterval(iv);
      const s = (Date.now() - s0) / 1000;
      t.textContent = mmss(s);
      if (Math.floor(s / 6) !== i && i < LOADING.length - 1) { i = Math.min(LOADING.length - 1, Math.floor(s / 6)); m.textContent = LOADING[i]; }
    }, 500);
    return () => clearInterval(iv);
  }

  function showError(msg) {
    const d = document.createElement("div");
    d.className = "q-card"; d.innerHTML = `<div class="q-hero q-hero-unk"><div class="q-ornament">۞</div><div class="q-sub"></div></div>`;
    $(".q-sub", d).textContent = msg;
    out.replaceChildren(d);
  }

  function mmss(s) { s = Math.floor(s); return `${Math.floor(s / 60)}:${String(s % 60).padStart(2, "0")}`; }

  // people count from the backend (also wakes the server up early)
  fetch(`${API}/health`).then(r => r.json()).then(d => { if (d.people) $("#q-people").textContent = d.people; }).catch(() => {});
})();
