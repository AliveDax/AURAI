"use strict";

// Photos are sent as they are when the server can read them, so the EXIF focal length that
// "See it on my wall" needs for the scale survives. Others (HEIC, very large) are re-encoded here.
const SEND_AS_IS = /^image\/(jpeg|png|webp)$/;
const MAX_AS_IS = 12 * 1024 * 1024;
const MAX_SIDE = 1600;
const MAX_PHOTOS = 4;

const $ = (id) => document.getElementById(id);
const steps = ["step-photo", "step-prefs", "step-results"];
const state = { photos: [], mood: new Set(), data: null };

const el = (tag, props = {}, kids = []) => {
  const n = Object.assign(document.createElement(tag), props);
  kids.forEach((k) => n.append(k));
  return n;
};

// ---------------------------------------------------------------- navigation
function show(step) {
  steps.forEach((s) => ($(s).hidden = s !== step));
  $("back").hidden = step === "step-photo";
  window.scrollTo({ top: 0 });
}
$("back").addEventListener("click", () => show($("step-results").hidden ? "step-photo" : "step-prefs"));
$("restart").addEventListener("click", () => show("step-photo"));

function showError(msg) {
  const e = $("error");
  e.textContent = msg;
  e.hidden = false;
  clearTimeout(showError.t);
  showError.t = setTimeout(() => (e.hidden = true), 6000);
}

// ---------------------------------------------------------------- photos
async function shrink(file) {
  // Upright (EXIF applied), at most MAX_SIDE px, as JPEG. If the browser can't decode it, send it as-is.
  try {
    const bmp = await createImageBitmap(file, { imageOrientation: "from-image" });
    const scale = Math.min(1, MAX_SIDE / Math.max(bmp.width, bmp.height));
    const canvas = el("canvas", { width: Math.round(bmp.width * scale), height: Math.round(bmp.height * scale) });
    canvas.getContext("2d").drawImage(bmp, 0, 0, canvas.width, canvas.height);
    bmp.close?.();
    return await new Promise((res) => canvas.toBlob((b) => res(b || file), "image/jpeg", 0.9));
  } catch {
    return file;
  }
}
const prepare = (f) => (SEND_AS_IS.test(f.type) && f.size <= MAX_AS_IS ? f : shrink(f));

async function addPhotos(files, fresh) {
  if (fresh) {
    state.photos.forEach((p) => URL.revokeObjectURL(p.url));
    state.photos = [];
  }
  for (const f of files) {
    if (state.photos.length >= MAX_PHOTOS) { showError(`Up to ${MAX_PHOTOS} photos of the same room.`); break; }
    const blob = await prepare(f);
    state.photos.push({ blob, url: URL.createObjectURL(blob) });
  }
  renderPhotos();
}
function onPick(fresh) {
  return async (e) => {
    const files = [...(e.target.files || [])];
    e.target.value = "";
    if (!files.length) return;
    await addPhotos(files, fresh);
    if (state.photos.length) show("step-prefs");
  };
}
$("camera").addEventListener("change", onPick(true));
$("library").addEventListener("change", onPick(true));
$("add-photo").addEventListener("change", onPick(false));

function renderPhotos() {
  const n = state.photos.length;
  $("photo").src = n ? state.photos[0].url : "";
  $("photos-name").textContent = n > 1 ? `room_photos (${n})` : "room.jpg";
  const row = $("thumbs");
  row.replaceChildren();
  state.photos.forEach((p, i) => {
    const rm = el("button", { type: "button", className: "thumb-x", textContent: "✕" });
    rm.setAttribute("aria-label", `Remove photo ${i + 1}`);
    rm.addEventListener("click", () => {
      URL.revokeObjectURL(p.url);
      state.photos.splice(i, 1);
      renderPhotos();
      if (!state.photos.length) show("step-photo");
    });
    row.append(el("div", { className: "thumb" }, [el("img", { src: p.url, alt: `Photo ${i + 1}` }), rm]));
  });
  if (n < MAX_PHOTOS) {
    row.append(el("label", { className: "thumb add", htmlFor: "add-photo", title: "Add another photo of the same room" },
      [el("span", { textContent: "+" }), el("small", { textContent: "photo" })]));
  }
}

// ---------------------------------------------------------------- preferences
document.querySelectorAll("#mood-chips .chip").forEach((chip) => {
  chip.setAttribute("aria-pressed", "false");
  chip.addEventListener("click", () => {
    const m = chip.dataset.mood;
    const on = !state.mood.has(m);
    on ? state.mood.add(m) : state.mood.delete(m);
    chip.setAttribute("aria-pressed", String(on));
  });
});
$("use-colour").addEventListener("change", (e) => ($("colour").disabled = !e.target.checked));

function segmented(id, onChange) {
  const group = $(id);
  group.addEventListener("click", (e) => {
    const btn = e.target.closest("button[role=radio]");
    if (!btn) return;
    group.querySelectorAll("button").forEach((b) => b.setAttribute("aria-checked", String(b === btn)));
    onChange?.(btn.dataset.value);
  });
}

// The recommender reads one text: it picks out the emotion words, colour words and the subject itself
function queryText() {
  return [...state.mood, $("mood-text").value.trim(), $("description").value.trim()].filter(Boolean).join(", ");
}

$("go").addEventListener("click", async () => {
  if (!state.photos.length) return show("step-photo");
  const fd = new FormData();
  state.photos.forEach((p, i) => fd.append("photos", p.blob, `room_${i + 1}.jpg`));
  fd.append("mood", queryText());
  if ($("use-colour").checked) fd.append("colour", $("colour").value);
  await runSearch(fd);
});

async function runSearch(fd) {
  $("go").disabled = true;
  $("loading").hidden = false;
  setStatus("busy");
  $("loading-text").textContent = "Looking at your room…";
  const slow = setTimeout(() => ($("loading-text").textContent = "Still working. The first search after a restart loads the models, which takes a little longer."), 6000);
  try {
    const r = await fetch("/api/recommend", { method: "POST", body: fd });
    const data = await r.json().catch(() => ({}));
    if (!r.ok) throw new Error(typeof data.detail === "string" ? data.detail : "Something went wrong. Please try again.");
    render(data);
    show("step-results");
  } catch (err) {
    showError(err.message === "Failed to fetch" ? "Can't reach the server. Check your connection." : err.message);
  } finally {
    clearTimeout(slow);
    setStatus();
    $("loading").hidden = true;
    $("go").disabled = false;
  }
}

function setStatus(mode) {
  $("status").textContent = mode === "busy" ? "WORKING" : "READY";
  $("status").classList.toggle("busy", mode === "busy");
}

// ---------------------------------------------------------------- results
const PARTS = {
  description: { label: "Subject", cls: "p-subject" },
  color: { label: "Colour harmony", cls: "p-colour" },
  style: { label: "Room style", cls: "p-style" },
  emotion: { label: "Mood", cls: "p-mood" },
  text: { label: "Mood + subject", cls: "p-subject" },
};
const part = (k) => PARTS[k] || { label: k, cls: "p-style" };
const pct = (v) => `${Math.round(v * 100)}%`;
const slug = (t) => (t || "untitled").toLowerCase().replace(/[^a-z0-9]+/g, "_").replace(/^_|_$/g, "").slice(0, 28) + ".jpg";
const swatches = (hexes) => el("span", { className: "swatches" }, hexes.map((h) => el("i", { style: `background:${h}`, title: h })));
const LEDS = 10;
function leds(v) {
  const n = Math.max(0, Math.min(LEDS, Math.round(v * LEDS)));
  const wrap = el("span", { className: "leds pos", title: v.toFixed(2) });
  for (let i = 0; i < LEDS; i++) wrap.append(el("i", { className: i < n ? "on" : "" }));
  return wrap;
}

function render(d) {
  state.data = d;
  resetWall();

  // What the text was understood as, and what was read from the room
  const sum = $("summary");
  sum.replaceChildren();
  const row = (label, ...content) => sum.append(el("dt", { textContent: label }), el("dd", {}, content));
  const p = d.parsed;
  if (p.emotion) row("Mood", el("span", { textContent: p.emotion }));
  if (p.description) row("Subject", el("span", { textContent: p.description }));
  if (p.colours.length) row("Colour", swatches(p.colours), el("span", { className: "muted", textContent: p.colour_words.length ? ` ${p.colour_words.join(", ")}` : "" }));
  row("Room", swatches(d.room_colours));
  if (d.photos > 1) row("Photos", el("span", { textContent: `${d.photos} of the same room` }));

  const notes = [];
  if (d.subject) {
    notes.push(d.subject.closest_only
      ? `Few paintings clearly show “${d.subject.text}” (${d.subject.clear_matches} found), so the closest matches come first.`
      : `Paintings that clearly show “${d.subject.text}” come first (${d.subject.clear_matches} in the collection).`);
  }
  if (d.colour_picks) {
    const shade = d.recommendations.slice(0, d.colour_picks).some((r) => !r.exact_colour);
    notes.push(`The first ${d.colour_picks} picks contain your colour${shade ? " or the closest shade" : ""}.`);
  }
  if (!p.emotion && !p.description && !p.colours.length) notes.push("No mood or subject given, so the picks are ranked by your room's style and colours alone.");
  $("notice").hidden = !notes.length;
  $("notice").textContent = notes.join(" ");
  $("save-sheet").href = `/api/sheet/${d.session}.jpg`;

  const list = $("results");
  list.replaceChildren();
  $("results-title").hidden = !d.recommendations.length;
  $("count").textContent = `${d.recommendations.length} of ${d.n_catalog.toLocaleString()}`;
  const tpl = $("card-tpl");
  d.recommendations.forEach((r, i) => list.append(card(tpl, r, i, d)));

  // Strip of picks under the room photo, for the wall preview
  const strip = $("strip");
  strip.replaceChildren();
  d.recommendations.forEach((r, i) => {
    const b = el("button", { type: "button", role: "option", title: `${r.title} — ${r.artist}` },
      [el("img", { src: r.image_url, alt: r.title, loading: "lazy" }), el("span", { textContent: String(r.rank).padStart(2, "0") })]);
    b.addEventListener("click", () => openWall(i));
    strip.append(b);
  });
}

function card(tpl, r, i, d) {
  const c = tpl.content.firstElementChild.cloneNode(true);
  const img = c.querySelector(".art");
  img.src = r.image_url;
  img.alt = `${r.title} by ${r.artist}`;
  c.querySelector(".filename").textContent = `${String(r.rank).padStart(2, "0")}_${slug(r.title)}`;
  c.querySelector(".title").textContent = r.title;
  c.querySelector(".artist").textContent = r.artist;

  const tags = c.querySelector(".tags");
  const tag = (text, cls = "") => tags.append(el("span", { className: `tag ${cls}`, textContent: text }));
  if (r.style) tag(r.style);
  if (r.picked_for_color) tag(r.exact_colour ? "colour pick" : "closest shade", "ok");
  if (r.color_match !== null) tag(`your colour ${pct(r.color_match)}`);
  // With few clear matches the subject pool is the closest ones (recommend.subject_pool), so say so
  if (r.subject_match) tag(d.subject.closest_only ? `close to ${d.subject.text}` : `shows ${d.subject.text}`, "ok");
  if (r.size_cm) tag(`${Math.round(r.size_cm[0])} × ${Math.round(r.size_cm[1])} cm`);

  // What the score is made of (each component's share of the final score)
  const parts = Object.entries(r.components).sort((a, b) => b[1].share - a[1].share);
  const mix = c.querySelector(".mix");
  parts.forEach(([k, v]) => mix.append(el("i", { className: part(k).cls, style: `width:${(v.share * 100).toFixed(1)}%`, title: `${part(k).label} ${pct(v.share)}` })));
  const legend = c.querySelector(".mix-legend");
  parts.forEach(([k, v]) => legend.append(el("span", {}, [el("i", { className: part(k).cls }), `${part(k).label} ${pct(v.share)}`])));

  c.querySelector(".palette").append(...r.palette.map((h) => el("i", { style: `background:${h}` })));
  c.querySelector(".view-on-wall").addEventListener("click", () => openWall(i));

  const reasons = c.querySelector(".reasons");
  for (const [k, v] of parts) {
    reasons.append(el("li", {}, [
      el("span", {}, [el("b", { textContent: `${part(k).label} ` }), el("small", { textContent: `score ${v.norm.toFixed(2)} × weight ${v.weight.toFixed(2)} = ${pct(v.share)} of total` })]),
      leds(v.norm),
    ]));
  }
  c.querySelector(".facts").textContent = `Final score ${r.final_score.toFixed(3)} · EmoArt label: ${r.emotion_label || "none"}`;
  return c;
}

// ---------------------------------------------------------------- see it on my wall
// The server hangs the artwork at its real size on the room photo (EXIF focal length + the
// distance the photo was taken from) with a shadow from the wall's light, where the user taps.
const pv = { open: false, index: 0, photo: 0, points: {}, sizes: {}, ctrl: null, url: null, timer: null };
const round1 = (v) => Math.round(v * 10) / 10;

function resetWall() {
  pv.open = false;
  pv.photo = 0;
  pv.points = {};
  pv.sizes = {};
  pv.ctrl?.abort();
  $("wall-panel").hidden = true;
  $("stage").classList.remove("placing", "busy");
  $("stage-hint").hidden = true;
  $("tap-dot").hidden = true;
  $("result-photo").src = state.photos[0]?.url || "";
  $("stage-name").textContent = "room.jpg";
  $("wall-dist").value = state.data?.preview_defaults.distance_ft ?? 6;
  const sel = $("wall-photo");
  sel.replaceChildren();
  sel.hidden = state.photos.length < 2;
  state.photos.forEach((_, i) => {
    const b = el("button", { type: "button", textContent: `Photo ${i + 1}` });
    b.setAttribute("role", "radio");
    b.setAttribute("aria-checked", String(i === 0));
    b.dataset.value = String(i);
    sel.append(b);
  });
  $("strip").querySelectorAll("button").forEach((b) => b.setAttribute("aria-selected", "false"));
}
segmented("wall-photo", (v) => {
  pv.photo = Number(v);
  showPlainPhoto();
  if (pv.points[pv.photo]) requestPreview();
});

function current() { return state.data.recommendations[pv.index]; }

function openWall(i) {
  pv.open = true;
  pv.index = i;
  const r = current();
  const def = state.data.preview_defaults;
  if (!pv.sizes[r.artwork_id]) {
    const w = r.size_cm ? r.size_cm[0] : def.artwork_width_cm;
    pv.sizes[r.artwork_id] = { w: round1(w), h: round1(r.size_cm ? r.size_cm[1] : w / r.aspect) };
  }
  const sz = pv.sizes[r.artwork_id];
  $("wall-w").value = sz.w;
  $("wall-h").value = sz.h;
  ["wall-w", "wall-h", "wall-lock"].forEach((id) => ($(id).disabled = !!r.size_cm));
  $("wall-w").max = $("wall-h").max = def.max_artwork_cm;
  $("wall-title").textContent = `${String(r.rank).padStart(2, "0")} · ${r.title}`;
  $("wall-panel").hidden = false;
  $("wall-save").hidden = true;
  $("stage").classList.add("placing");
  $("stage-name").textContent = `wall_${String(r.rank).padStart(2, "0")}_${slug(r.title)}`;
  $("strip").querySelectorAll("button").forEach((b, j) => b.setAttribute("aria-selected", String(j === i)));
  $("stage").scrollIntoView({ behavior: "smooth", block: "start" });
  if (pv.points[pv.photo]) requestPreview();
  else {
    showPlainPhoto();
    note(r.size_cm ? `Real size: ${r.size_cm[0]} × ${r.size_cm[1]} cm.` : "This artwork's real size isn't known, so enter the size you'd hang it at.");
  }
}

function closeWall() {
  resetWallView();
  pv.open = false;
  $("wall-panel").hidden = true;
  $("stage").classList.remove("placing");
  $("stage-name").textContent = "room.jpg";
  $("strip").querySelectorAll("button").forEach((b) => b.setAttribute("aria-selected", "false"));
}
$("wall-close").addEventListener("click", closeWall);

function resetWallView() {
  pv.ctrl?.abort();
  showPlainPhoto();
  $("stage-hint").hidden = true;
}
function showPlainPhoto() {
  $("result-photo").src = state.photos[pv.photo]?.url || "";
  $("tap-dot").hidden = true;
  $("wall-save").hidden = true;
  hint(pv.open ? "Tap the wall where its centre should go" : "");
}
function hint(text) {
  $("stage-hint").textContent = text;
  $("stage-hint").hidden = !text;
}
function note(text) { $("wall-note").textContent = text; }

function imgRect() {
  // The <img> is object-fit: contain; work out where the picture actually is
  const img = $("result-photo");
  const r = img.getBoundingClientRect();
  const ar = img.naturalWidth / img.naturalHeight || 1;
  let w = r.width, h = r.width / ar;
  if (h > r.height) { h = r.height; w = h * ar; }
  return { left: r.left + (r.width - w) / 2, top: r.top + (r.height - h) / 2, width: w, height: h };
}

$("stage").addEventListener("click", (e) => {
  if (!pv.open) return;
  const r = imgRect();
  const x = (e.clientX - r.left) / r.width, y = (e.clientY - r.top) / r.height;
  if (x < 0 || x > 1 || y < 0 || y > 1) return;
  pv.points[pv.photo] = { x, y };
  const s = $("stage").getBoundingClientRect();
  Object.assign($("tap-dot").style, { left: `${e.clientX - s.left}px`, top: `${e.clientY - s.top}px` });
  $("tap-dot").hidden = false;
  requestPreview();
});

function sizeChanged(which) {
  const r = current();
  const sz = pv.sizes[r.artwork_id];
  const w = parseFloat($("wall-w").value), h = parseFloat($("wall-h").value);
  if (which === "w" && w > 0) {
    sz.w = w;
    if ($("wall-lock").checked) $("wall-h").value = sz.h = round1(w / r.aspect);
  } else if (which === "h" && h > 0) {
    sz.h = h;
    if ($("wall-lock").checked) $("wall-w").value = sz.w = round1(h * r.aspect);
  }
  later();
}
$("wall-w").addEventListener("input", () => sizeChanged("w"));
$("wall-h").addEventListener("input", () => sizeChanged("h"));
$("wall-lock").addEventListener("change", () => $("wall-lock").checked && sizeChanged("w"));
$("wall-dist").addEventListener("input", later);
function later() {
  clearTimeout(pv.timer);
  pv.timer = setTimeout(() => pv.open && pv.points[pv.photo] && requestPreview(), 400);
}

async function requestPreview() {
  const d = state.data, r = current(), pt = pv.points[pv.photo], sz = pv.sizes[r.artwork_id];
  if (!pt) return;
  pv.ctrl?.abort();
  const ctrl = (pv.ctrl = new AbortController());
  const fd = new FormData();
  Object.entries({ session: d.session, artwork_id: r.artwork_id, photo: pv.photo, x: pt.x.toFixed(4), y: pt.y.toFixed(4),
    width_cm: sz.w, height_cm: sz.h, distance_ft: $("wall-dist").value || d.preview_defaults.distance_ft })
    .forEach(([k, v]) => fd.append(k, v));
  $("stage").classList.add("busy");
  hint("Hanging it…");
  try {
    const res = await fetch("/api/preview", { method: "POST", body: fd, signal: ctrl.signal });
    if (!res.ok) {
      const j = await res.json().catch(() => ({}));
      hint("");
      note(typeof j.detail === "string" ? j.detail : "Couldn't place it there. Try again.");
      return;
    }
    const blob = await res.blob();
    if (pv.url) URL.revokeObjectURL(pv.url);
    pv.url = URL.createObjectURL(blob);
    $("result-photo").src = pv.url;
    $("tap-dot").hidden = true;
    hint("Tap anywhere on the wall to move it");
    const h = (k) => decodeURIComponent(res.headers.get(k) || "");
    const [w, hh] = h("X-Size-Cm").split("x");
    note(`${w} × ${hh} cm${h("X-Known-Size") === "1" ? " (real size)" : ""}. Shadow: ${h("X-Light")}. ` +
      `Scale: ${h("X-Scale")} px per inch of wall. Lens: ${h("X-Lens")}. ` +
      "This assumes the photo was taken straight at the wall and not cropped.");
    const save = $("wall-save");
    save.href = pv.url;
    save.download = `wall_preview_${r.artwork_id}.jpg`;
    save.hidden = false;
  } catch (err) {
    if (err.name !== "AbortError") { hint(""); note("Can't reach the server. Check your connection."); }
  } finally {
    if (pv.ctrl === ctrl) $("stage").classList.remove("busy");
  }
}

// ---------------------------------------------------------------- install
if ("serviceWorker" in navigator) {
  window.addEventListener("load", () => navigator.serviceWorker.register("sw.js").catch(() => {}));
}
