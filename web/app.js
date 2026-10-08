"use strict";

const MAX_SIDE = 1600; // matches the server's PHOTO_MAX_SIDE; shrinks a 12 MP upload to ~300 KB

const $ = (id) => document.getElementById(id);
const steps = ["step-photo", "step-prefs", "step-results"];
const state = { blob: null, photoUrl: null, mood: new Set(), box: null };

// ---------------------------------------------------------------- navigation
function show(step) {
  steps.forEach((s) => ($(s).hidden = s !== step));
  $("back").hidden = step === "step-photo";
  window.scrollTo({ top: 0 });
}
$("back").addEventListener("click", () => {
  show($("step-results").hidden ? "step-photo" : "step-prefs");
});
$("restart").addEventListener("click", () => show("step-photo"));

function showError(msg) {
  const el = $("error");
  el.textContent = msg;
  el.hidden = false;
  clearTimeout(showError.t);
  showError.t = setTimeout(() => (el.hidden = true), 6000);
}

// ---------------------------------------------------------------- photo input
async function shrink(file) {
  // Draw the photo upright (EXIF applied) at most MAX_SIDE px and re-encode as JPEG.
  // If the browser can't decode it (e.g. HEIC on some Android browsers), send it as-is.
  try {
    const bmp = await createImageBitmap(file, { imageOrientation: "from-image" });
    const scale = Math.min(1, MAX_SIDE / Math.max(bmp.width, bmp.height));
    const canvas = document.createElement("canvas");
    canvas.width = Math.round(bmp.width * scale);
    canvas.height = Math.round(bmp.height * scale);
    canvas.getContext("2d").drawImage(bmp, 0, 0, canvas.width, canvas.height);
    bmp.close?.();
    return await new Promise((res) => canvas.toBlob((b) => res(b || file), "image/jpeg", 0.88));
  } catch {
    return file;
  }
}

async function onPhoto(e) {
  const file = e.target.files?.[0];
  e.target.value = "";
  if (!file) return;
  state.blob = await shrink(file);
  if (state.photoUrl) URL.revokeObjectURL(state.photoUrl);
  state.photoUrl = URL.createObjectURL(state.blob);
  $("photo").src = state.photoUrl;
  $("result-photo").src = state.photoUrl;
  state.box = null;
  $("draw-box").hidden = true;
  show("step-prefs");
}
$("camera").addEventListener("change", onPhoto);
$("library").addEventListener("change", onPhoto);

// ---------------------------------------------------------------- controls
document.querySelectorAll("#mood-chips .chip").forEach((chip) => {
  chip.setAttribute("aria-pressed", "false");
  chip.addEventListener("click", () => {
    const m = chip.dataset.mood;
    const on = !state.mood.has(m);
    on ? state.mood.add(m) : state.mood.delete(m);
    chip.setAttribute("aria-pressed", String(on));
  });
});

function segmented(id, onChange) {
  const group = $(id);
  group.addEventListener("click", (e) => {
    const btn = e.target.closest("button[role=radio]");
    if (!btn) return;
    group.querySelectorAll("button").forEach((b) => b.setAttribute("aria-checked", String(b === btn)));
    onChange?.(btn.dataset.value);
  });
}
const value = (id) => $(id).querySelector("[aria-checked=true]").dataset.value;

segmented("size-mode", (v) => ($("size-inputs").hidden = v !== "manual"));
segmented("shape");
segmented("placement", (v) => {
  const marking = v === "mark";
  $("draw-hint").hidden = !marking || !!state.box;
  $("draw-box").hidden = !marking || !state.box;
  $("photo-wrap").style.touchAction = marking ? "none" : "auto";
});
$("use-colour").addEventListener("change", (e) => ($("colour").disabled = !e.target.checked));

// ---------------------------------------------------------------- marking the spot
// The box is kept as fractions of the displayed image, so it is independent of resolution.
(function setupDrawing() {
  const wrap = $("photo-wrap");
  const img = $("photo");
  const boxEl = $("draw-box");
  let start = null;

  function imgRect() {
    // The <img> is object-fit: contain; work out where the picture actually is.
    const r = img.getBoundingClientRect();
    const ar = img.naturalWidth / img.naturalHeight || 1;
    let w = r.width, h = r.width / ar;
    if (h > r.height) { h = r.height; w = h * ar; }
    return { left: r.left + (r.width - w) / 2, top: r.top + (r.height - h) / 2, width: w, height: h };
  }
  const clamp = (v) => Math.min(1, Math.max(0, v));
  function frac(e) {
    const r = imgRect();
    return { x: clamp((e.clientX - r.left) / r.width), y: clamp((e.clientY - r.top) / r.height) };
  }
  function draw(b) {
    const r = imgRect(), w = wrap.getBoundingClientRect();
    Object.assign(boxEl.style, {
      left: `${r.left - w.left + b.x * r.width}px`, top: `${r.top - w.top + b.y * r.height}px`,
      width: `${b.w * r.width}px`, height: `${b.h * r.height}px`,
    });
    boxEl.hidden = false;
  }

  wrap.addEventListener("pointerdown", (e) => {
    if (value("placement") !== "mark") return;
    start = frac(e);
    wrap.setPointerCapture(e.pointerId);
    $("draw-hint").hidden = true;
  });
  wrap.addEventListener("pointermove", (e) => {
    if (!start) return;
    const p = frac(e);
    state.box = { x: Math.min(start.x, p.x), y: Math.min(start.y, p.y), w: Math.abs(p.x - start.x), h: Math.abs(p.y - start.y) };
    draw(state.box);
  });
  const end = () => {
    if (start && state.box && (state.box.w < 0.03 || state.box.h < 0.03)) {
      state.box = null; // a tap, not a drag
      boxEl.hidden = true;
      $("draw-hint").hidden = false;
    }
    start = null;
  };
  wrap.addEventListener("pointerup", end);
  wrap.addEventListener("pointercancel", end);
  window.addEventListener("resize", () => state.box && value("placement") === "mark" && draw(state.box));
})();

// ---------------------------------------------------------------- request
function moodText() {
  const words = [...state.mood];
  const free = $("mood-text").value.trim();
  if (free) words.push(free);
  return words.join(", ");
}

$("go").addEventListener("click", async () => {
  if (!state.blob) return show("step-photo");
  const sizeMode = value("size-mode");
  const fd = new FormData();
  fd.append("photo", state.blob, "wall.jpg");
  fd.append("mood", moodText());
  fd.append("description", $("description").value.trim());
  if ($("use-colour").checked) fd.append("colour", $("colour").value);
  fd.append("orientation", value("shape"));
  fd.append("size_mode", sizeMode);
  if (sizeMode === "manual") {
    fd.append("width_cm", $("width-cm").value);
    fd.append("height_cm", $("height-cm").value);
  }
  if (value("placement") === "mark") {
    if (!state.box) return showError("Drag on the photo to mark where the art should go.");
    const b = state.box;
    fd.append("box", [b.x, b.y, b.w, b.h].map((v) => v.toFixed(4)).join(","));
  }

  $("go").disabled = true;
  $("loading").hidden = false;
  setStatus("busy");
  $("loading-text").textContent = "Looking at your room…";
  const slow = setTimeout(() => ($("loading-text").textContent = "Still working — the first search after a restart loads the models, which takes a little longer."), 6000);
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
});

function setStatus(mode) {
  $("status").textContent = mode === "busy" ? "WORKING" : "READY";
  $("status").classList.toggle("busy", mode === "busy");
}

// ---------------------------------------------------------------- results
const slug = (t) => (t || "untitled").toLowerCase().replace(/[^a-z0-9]+/g, "_").replace(/^_|_$/g, "").slice(0, 28) + ".jpg";
const LEDS = 10;
function leds(v, maxAbs) {
  const n = Math.max(1, Math.round((Math.abs(v) / maxAbs) * LEDS));
  const wrap = el("span", { className: `leds ${v >= 0 ? "pos" : "neg"}`, title: v.toFixed(2) });
  for (let i = 0; i < LEDS; i++) wrap.append(el("i", { className: i < n ? "on" : "" }));
  return wrap;
}
const el = (tag, props = {}, kids = []) => {
  const n = Object.assign(document.createElement(tag), props);
  kids.forEach((k) => n.append(k));
  return n;
};
const swatches = (hexes) => el("span", { className: "swatches" }, hexes.map((h) => el("i", { style: `background:${h}`, title: h })));
const cm = (w, h) => `${Math.round(w)} × ${Math.round(h)} cm`;

function render(d) {
  // Overlay: chosen space (green) and the A4 sheet (red), in the analysed photo's pixel space
  const svg = $("overlay");
  svg.setAttribute("viewBox", `0 0 ${d.photo.width} ${d.photo.height}`);
  const t = Math.max(3, d.photo.width / 220);
  let shapes = "";
  if (d.space.box) {
    const [x, y, w, h] = d.space.box;
    shapes += `<rect x="${x}" y="${y}" width="${w}" height="${h}" fill="rgba(47,91,255,.14)" stroke="#2f5bff" stroke-width="${t}"/>`;
  }
  if (d.a4_corners) shapes += `<polygon points="${d.a4_corners.map((p) => p.join(",")).join(" ")}" fill="none" stroke="#ff3d7f" stroke-width="${t}"/>`;
  svg.innerHTML = shapes;
  // The overlay must match the contained image, so size the wrapper to the photo's aspect ratio
  $("result-photo").parentElement.style.aspectRatio = `${d.photo.width} / ${d.photo.height}`;

  const sum = $("summary");
  sum.replaceChildren();
  const row = (label, content) => sum.append(el("dt", { textContent: label }), el("dd", {}, [content]));
  if (d.measurement) row("Space", el("span", { textContent: `${cm(d.measurement.width_cm, d.measurement.height_cm)} ${d.measurement.source === "a4" ? "· from A4" : "· entered"}` }));
  if (d.room_style.length) row("Room", el("span", { textContent: d.room_style.slice(0, 1).map((s) => `${s.style} ${Math.round(s.p * 100)}%`).join(" · ") }));
  row("Wall", swatches(d.wall_colours));
  if (d.decor_colours.length) row("Decor", swatches(d.decor_colours));

  const notes = [];
  if (d.a4_requested && !d.measurement) notes.push("No A4 sheet found in the photo, so sizes weren't checked. You can enter the size of the space instead.");
  if (d.space.source === "none") notes.push("Couldn't find a clear area on the wall. Try marking the spot yourself.");
  if (d.measurement && d.n_size_unknown) notes.push(`${d.n_candidates - d.n_size_unknown} artworks are known to fit; ${d.n_size_unknown} more have no recorded size, so only their shape was checked.`);
  if (d.message) notes.push(d.message);
  $("notice").hidden = !notes.length;
  $("notice").textContent = notes.join(" ");

  const list = $("results");
  list.replaceChildren();
  $("results-title").hidden = !d.recommendations.length;
  $("count").textContent = `${d.recommendations.length} of ${d.n_candidates}`;
  const tpl = $("card-tpl");
  const maxAbs = Math.max(0.01, ...d.recommendations.flatMap((r) => r.reasons.map((x) => Math.abs(x.value))));
  for (const r of d.recommendations) {
    const card = tpl.content.firstElementChild.cloneNode(true);
    const img = card.querySelector(".art");
    img.src = r.image_url;
    img.alt = `${r.title} by ${r.artist}`;
    card.querySelector(".filename").textContent = `${String(r.rank).padStart(2, "0")}_${slug(r.title)}`;
    card.querySelector(".title").textContent = r.title;
    card.querySelector(".artist").textContent = r.artist;
    const tags = card.querySelector(".tags");
    const tag = (text, cls = "") => tags.append(el("span", { className: `tag ${cls}`, textContent: text }));
    if (r.style) tag(r.style);
    if (r.emotion) tag(r.emotion);
    if (r.width_cm && r.height_cm) {
      tag(cm(r.width_cm, r.height_cm));
      if (d.measurement) tag("✓ fits", "ok");
    } else {
      tag("size ?");
      if (d.measurement) tag("not size-checked", "warn");
    }
    card.querySelector(".palette").append(...r.palette.map((h) => el("i", { style: `background:${h}` })));
    const reasons = card.querySelector(".reasons");
    for (const x of r.reasons) {
      const label = x.value >= 0 ? x.label : `${x.label} (−)`;
      reasons.append(el("li", {}, [el("span", { textContent: label }), leds(x.value, maxAbs)]));
    }
    list.append(card);
  }
}

// ---------------------------------------------------------------- install
if ("serviceWorker" in navigator) {
  window.addEventListener("load", () => navigator.serviceWorker.register("sw.js").catch(() => {}));
}
