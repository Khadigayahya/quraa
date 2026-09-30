// User feedback on a result: 👍 / 👎 + the right reciter's name.
// Sent to a Supabase table (see build/feedback_schema.sql) when configured in config.js; until then (or when
// offline) it waits in this browser and is retried on the next visit. Only voiceprint numbers are sent,
// never the audio. Corrections also teach this browser immediately (engine.addUserRef).
import * as engine from "./engine.js";

const CFG = window.QURAA || {};
const QUEUE_KEY = "quraa-feedback-queue-v1";
const APP_VERSION = "web-2";
const esc = s => String(s ?? "").replace(/[&<>"']/g, c => ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;" }[c]));
const round = a => Array.from(a, x => +x.toFixed(5));

function queue() { try { const q = JSON.parse(localStorage.getItem(QUEUE_KEY) || "[]"); return Array.isArray(q) ? q : []; } catch (e) { return []; } }
function saveQueue(q) { try { localStorage.setItem(QUEUE_KEY, JSON.stringify(q.slice(-200))); } catch (e) {} }

async function post(row) {
  if (!CFG.SUPABASE_URL || !CFG.SUPABASE_KEY) return false;
  // new-style keys (sb_publishable_…) go in `apikey` only; legacy anon JWTs also as a Bearer token
  const headers = { apikey: CFG.SUPABASE_KEY, "Content-Type": "application/json", Prefer: "return=minimal" };
  if (!CFG.SUPABASE_KEY.startsWith("sb_")) headers.Authorization = `Bearer ${CFG.SUPABASE_KEY}`;
  const res = await fetch(`${CFG.SUPABASE_URL.replace(/\/$/, "")}/rest/v1/feedback`, {
    method: "POST", headers, body: JSON.stringify(row),
  });
  return res.ok;
}

async function send(row) {
  try { if (await post(row)) return "sent"; } catch (e) {}
  const q = queue(); q.push(row); saveQueue(q); return "queued";
}

export async function flushQueue() {             // retry what could not be sent before
  const q = queue(); if (!q.length || !CFG.SUPABASE_URL) return;
  const left = [];
  for (const row of q) { try { if (!(await post(row))) left.push(row); } catch (e) { left.push(row); } }
  saveQueue(left);
}

function baseRow(r) {
  const rec = r.reciter && r.reciter.ok ? r.reciter : null;
  return {
    app_version: APP_VERSION, gallery_version: r.gallery_version || "",
    file_hash: r.file_hash || "", file_name: String(r.source || "").slice(0, 200), duration_s: +(r.duration_s || 0).toFixed(1),
    content_verdict: r.content ? r.content.verdict : null,
    predicted_person: rec ? rec.person : null, predicted_name: rec ? rec.name : null,
    similarity: rec ? +rec.similarity.toFixed(4) : null, margin: rec ? +rec.margin.toFixed(4) : null,
    was_unknown: rec ? !!rec.unknown : null, n_windows: rec ? rec.n_windows : null,
    embedding: rec ? round(rec.embedding) : null,
    window_embeddings: rec ? round(rec.window_embeddings.flatMap(e => Array.from(e))) : null,
  };
}

// Resolve a typed name to a gallery person when it matches one of the 199 names (Arabic or English).
function resolveName(text, g) {
  const t = engine.normalize(text).join(" ");
  if (!t) return null;
  let i = g.names.findIndex(n => engine.normalize(n).join(" ") === t);
  if (i < 0) i = g.names_en.findIndex(n => n && n.toLowerCase().trim() === text.toLowerCase().trim());
  return i >= 0 ? { person: g.people[i], name: g.names[i] } : null;
}

// ---------------------------------------------------------------- UI
export async function render(container, r, { onRecheckAsQuran } = {}) {
  const g = await engine.loadGallery();
  const rec = r.reciter && r.reciter.ok ? r.reciter : null;
  const box = document.createElement("div");
  box.className = "q-fb";

  if (!rec) {
    if (!(r.content && r.content.verdict === "lecture")) return;       // nothing to rate
    box.innerHTML = `<div class="q-fb-q">طلع إن ده محاضرة. صح؟</div>
      <div class="q-fb-row"><button type="button" class="q-fb-btn" data-a="ok">👍 أيوه محاضرة</button>
      <button type="button" class="q-fb-btn" data-a="quran">👎 لأ ده تلاوة، اتعرّف على القارئ</button></div>`;
    box.addEventListener("click", async ev => {
      const a = ev.target.closest("button")?.dataset.a; if (!a) return;
      box.querySelectorAll("button").forEach(b => { b.disabled = true; });
      await send({ ...baseRow(r), kind: "content", verdict: a === "ok" ? "correct" : "wrong" });
      if (a === "ok") box.innerHTML = `<div class="q-fb-done">شكرًا ✓</div>`;
      else onRecheckAsQuran && onRecheckAsQuran();
    });
    container.appendChild(box); return;
  }

  const listId = "q-fb-names";
  box.innerHTML = `
    <div class="q-fb-q">النتيجة دي صح؟</div>
    <div class="q-fb-row">
      <button type="button" class="q-fb-btn q-fb-yes" data-a="yes">👍 أيوه، ده ${esc(rec.name)}</button>
      <button type="button" class="q-fb-btn q-fb-no" data-a="no">👎 لأ، غلط</button>
    </div>
    <form class="q-fb-fix" hidden>
      <label for="q-fb-name">مين القارئ الصح؟</label>
      <input id="q-fb-name" class="q-field" list="${listId}" autocomplete="off" placeholder="اكتب أو اختار من القايمة">
      <datalist id="${listId}">${g.names.map(n => `<option value="${esc(n)}"></option>`).join("")}</datalist>
      <div class="q-fb-row">
        <button type="submit" class="q-fb-btn q-fb-yes">ابعت التصحيح</button>
        <button type="button" class="q-fb-btn" data-a="unsure">مش عارف مين</button>
      </div>
    </form>
    <div class="q-fb-note">بنحفظ بصمة رقمية للصوت (أرقام بس، مش الملف) علشان نحسّن التطبيق.</div>`;
  const form = box.querySelector(".q-fb-fix"), input = box.querySelector("#q-fb-name");
  const done = html => { box.innerHTML = `<div class="q-fb-done">${html}</div>`; };
  const where = st => st === "sent" ? "" : `<span class="q-fb-sub">اتحفظ على جهازك وهيتبعت أول ما يبقى ممكن.</span>`;

  box.addEventListener("click", async ev => {
    const a = ev.target.closest("button")?.dataset.a; if (!a) return;
    if (a === "yes") {
      box.querySelectorAll("button").forEach(b => { b.disabled = true; });
      const st = await send({ ...baseRow(r), kind: "reciter", verdict: "correct", true_person: rec.person, true_name: rec.name });
      engine.addUserRef(rec.person, rec.name, rec.embedding);          // a confirmed real-world sample helps too
      done(`شكرًا! 👍 كده التطبيق اتأكد إن ده صوت ${esc(rec.name)}. ${where(st)}`);
    } else if (a === "no") {
      box.querySelector(".q-fb-row").hidden = true; form.hidden = false; input.focus();
    } else if (a === "unsure") {
      const st = await send({ ...baseRow(r), kind: "reciter", verdict: "wrong" });
      done(`شكرًا، اتسجّل إن النتيجة غلط. ${where(st)}`);
    }
  });
  form.addEventListener("submit", async ev => {
    ev.preventDefault();
    const typed = input.value.trim();
    if (!typed) { input.focus(); return; }
    const hit = resolveName(typed, g);
    const person = hit ? hit.person : engine.userPersonId(typed), name = hit ? hit.name : typed.slice(0, 120);
    if (person === rec.person) {                                        // they picked the same name → it was right
      const st = await send({ ...baseRow(r), kind: "reciter", verdict: "correct", true_person: person, true_name: name });
      engine.addUserRef(person, name, rec.embedding);
      return done(`تمام، يبقى النتيجة كانت صح 👍 ${where(st)}`);
    }
    const st = await send({ ...baseRow(r), kind: "reciter", verdict: "wrong", true_person: hit ? person : null, true_name: name });
    engine.addUserRef(person, name, rec.embedding);
    done(`شكرًا على التصحيح! 🧠 من دلوقتي على جهازك هنتعرف على الصوت ده كـ <b>${esc(name)}</b>.
      ${hit ? "" : "<span class=\"q-fb-sub\">الاسم ده مش في قاعدتنا؛ لو اتكرر من ناس تانية هنضيفه بعد مراجعة.</span>"} ${where(st)}`);
  });
  container.appendChild(box);
}
