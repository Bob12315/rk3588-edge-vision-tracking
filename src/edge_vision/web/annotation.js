const elements = {
  canvas: document.querySelector("#annotationCanvas"),
  canvasStage: document.querySelector("#canvasStage"),
  loadingFrame: document.querySelector("#loadingFrame"),
  videoName: document.querySelector("#videoName"),
  videoMeta: document.querySelector("#videoMeta"),
  previousFrame: document.querySelector("#previousFrame"),
  nextFrame: document.querySelector("#nextFrame"),
  nextUnreviewed: document.querySelector("#nextUnreviewed"),
  frameSlider: document.querySelector("#frameSlider"),
  frameNumber: document.querySelector("#frameNumber"),
  frameTotal: document.querySelector("#frameTotal"),
  reviewedCount: document.querySelector("#reviewedCount"),
  annotationCount: document.querySelector("#annotationCount"),
  progressBar: document.querySelector("#progressBar"),
  identityInput: document.querySelector("#identityInput"),
  visibilityInput: document.querySelector("#visibilityInput"),
  visibilityValue: document.querySelector("#visibilityValue"),
  applyIdentity: document.querySelector("#applyIdentity"),
  deleteBox: document.querySelector("#deleteBox"),
  acceptSuggestions: document.querySelector("#acceptSuggestions"),
  clearBoxes: document.querySelector("#clearBoxes"),
  saveFrame: document.querySelector("#saveFrame"),
  reviewBadge: document.querySelector("#reviewBadge"),
  boxList: document.querySelector("#boxList"),
  outputPath: document.querySelector("#outputPath"),
  saveState: document.querySelector(".save-state"),
  saveLabel: document.querySelector("#saveLabel"),
  toast: document.querySelector("#toast"),
};

const context = elements.canvas.getContext("2d");
let project = null;
let frameId = 1;
let framePayload = null;
let frameImage = null;
let annotations = [];
let suggestions = [];
let selectedIndex = -1;
let drawing = null;
let dirty = false;
let loading = false;
let toastTimer = null;

async function api(path, options = {}) {
  const response = await fetch(path, options);
  const payload = await response.json().catch(() => ({}));
  if (!response.ok) throw new Error(payload.error || `Request failed (${response.status})`);
  return payload;
}

function jsonOptions(payload) {
  return {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(payload),
  };
}

function showToast(message) {
  elements.toast.textContent = message;
  elements.toast.classList.add("visible");
  clearTimeout(toastTimer);
  toastTimer = setTimeout(() => elements.toast.classList.remove("visible"), 3500);
}

function setSaveState(state, label) {
  elements.saveState.className = `save-state ${state}`;
  elements.saveLabel.textContent = label;
}

function markDirty() {
  dirty = true;
  setSaveState("dirty", "Unsaved frame");
  renderFrameState();
}

function videoFilename(path) {
  return String(path || "Video").split(/[\\/]/).pop();
}

function updateProjectUi() {
  const video = project.video;
  elements.videoName.textContent = videoFilename(video.source);
  elements.videoMeta.textContent = `${video.width}×${video.height} · ${Number(video.fps).toFixed(2)} FPS · ${Number(video.duration_s).toFixed(1)} s`;
  elements.frameSlider.max = String(video.frame_count);
  elements.frameNumber.max = String(video.frame_count);
  elements.frameTotal.textContent = `/ ${video.frame_count}`;
  elements.reviewedCount.textContent = String(project.reviewed_frame_count || 0);
  elements.annotationCount.textContent = `${project.annotation_count || 0} ground-truth boxes`;
  elements.progressBar.style.width = `${100 * (project.reviewed_frame_count || 0) / video.frame_count}%`;
  elements.outputPath.textContent = `MOT: ${project.outputs.mot_ground_truth}`;
}

function canvasPoint(event) {
  const rectangle = elements.canvas.getBoundingClientRect();
  return {
    x: Math.max(0, Math.min(elements.canvas.width, (event.clientX - rectangle.left) * elements.canvas.width / rectangle.width)),
    y: Math.max(0, Math.min(elements.canvas.height, (event.clientY - rectangle.top) * elements.canvas.height / rectangle.height)),
  };
}

function identityColor(identity) {
  const hue = (Number(identity) * 67) % 360;
  return `hsl(${hue} 78% 63%)`;
}

function drawBox(box, color, label, dashed = false, selected = false) {
  context.save();
  context.lineWidth = selected ? 5 : 3;
  context.strokeStyle = color;
  context.fillStyle = color;
  if (dashed) context.setLineDash([12, 8]);
  context.strokeRect(box.x, box.y, box.width, box.height);
  context.setLineDash([]);
  context.font = "bold 18px ui-monospace, monospace";
  const metrics = context.measureText(label);
  const labelY = Math.max(0, box.y - 26);
  context.globalAlpha = 0.88;
  context.fillRect(box.x, labelY, metrics.width + 14, 26);
  context.globalAlpha = 1;
  context.fillStyle = "#051013";
  context.fillText(label, box.x + 7, labelY + 19);
  context.restore();
}

function renderCanvas() {
  if (!frameImage) return;
  context.clearRect(0, 0, elements.canvas.width, elements.canvas.height);
  context.drawImage(frameImage, 0, 0, elements.canvas.width, elements.canvas.height);
  for (const suggestion of suggestions) {
    drawBox(
      suggestion,
      "#f59f45",
      `P${suggestion.suggested_identity_id} ${Number(suggestion.confidence).toFixed(2)}`,
      true,
      false,
    );
  }
  annotations.forEach((box, index) => {
    drawBox(
      box,
      index === selectedIndex ? "#ffd15a" : identityColor(box.identity_id),
      `GT ${box.identity_id}`,
      false,
      index === selectedIndex,
    );
  });
  if (drawing) {
    const box = normalizedDrawingBox();
    drawBox(box, "#ffd15a", `NEW ${elements.identityInput.value}`, true, true);
  }
}

function renderFrameState() {
  const reviewed = framePayload?.reviewed && !dirty;
  elements.reviewBadge.textContent = reviewed ? "REVIEWED" : "UNREVIEWED";
  elements.reviewBadge.classList.toggle("reviewed", reviewed);
  elements.previousFrame.disabled = loading || frameId <= 1;
  elements.nextFrame.disabled = loading || frameId >= project.video.frame_count;
  elements.acceptSuggestions.disabled = loading || suggestions.length === 0;
  elements.saveFrame.disabled = loading;
  renderBoxList();
}

function renderBoxList() {
  elements.boxList.replaceChildren();
  if (!annotations.length) {
    const empty = document.createElement("p");
    empty.className = "muted";
    empty.textContent = suggestions.length
      ? `${suggestions.length} suggestion(s); review before accepting.`
      : "No person boxes on this frame.";
    elements.boxList.append(empty);
    return;
  }
  annotations.forEach((box, index) => {
    const row = document.createElement("div");
    row.className = `box-row${index === selectedIndex ? " selected" : ""}`;
    const identity = document.createElement("span");
    identity.className = "identity-chip";
    identity.textContent = `ID ${box.identity_id}`;
    const coordinates = document.createElement("code");
    coordinates.textContent = `${Math.round(box.x)},${Math.round(box.y)} · ${Math.round(box.width)}×${Math.round(box.height)}`;
    const visibility = document.createElement("small");
    visibility.textContent = `${Number(box.visibility).toFixed(2)}`;
    row.append(identity, coordinates, visibility);
    row.addEventListener("click", () => selectBox(index));
    elements.boxList.append(row);
  });
}

function selectBox(index) {
  selectedIndex = index;
  if (index >= 0 && annotations[index]) {
    elements.identityInput.value = String(annotations[index].identity_id);
    elements.visibilityInput.value = String(annotations[index].visibility);
    elements.visibilityValue.textContent = Number(annotations[index].visibility).toFixed(2);
  }
  renderFrameState();
  renderCanvas();
}

function boxAt(point) {
  for (let index = annotations.length - 1; index >= 0; index -= 1) {
    const box = annotations[index];
    if (point.x >= box.x && point.x <= box.x + box.width && point.y >= box.y && point.y <= box.y + box.height) return index;
  }
  return -1;
}

function normalizedDrawingBox() {
  const x = Math.min(drawing.start.x, drawing.end.x);
  const y = Math.min(drawing.start.y, drawing.end.y);
  return {
    x,
    y,
    width: Math.abs(drawing.end.x - drawing.start.x),
    height: Math.abs(drawing.end.y - drawing.start.y),
  };
}

async function loadFrame(targetFrame) {
  if (loading) return;
  const bounded = Math.max(1, Math.min(project.video.frame_count, Number(targetFrame)));
  if (dirty) await saveCurrentFrame();
  loading = true;
  elements.loadingFrame.classList.remove("hidden");
  try {
    const [payload, image] = await Promise.all([
      api(`/api/frame?frame=${bounded}`),
      new Promise((resolve, reject) => {
        const candidate = new Image();
        candidate.onload = () => resolve(candidate);
        candidate.onerror = () => reject(new Error(`Cannot load frame ${bounded}`));
        candidate.src = `/api/frame-image?frame=${bounded}&v=${Date.now()}`;
      }),
    ]);
    frameId = bounded;
    framePayload = payload;
    frameImage = image;
    annotations = (payload.annotations || []).map((item) => ({ ...item }));
    suggestions = (payload.suggestions || []).map((item) => ({ ...item }));
    selectedIndex = -1;
    dirty = false;
    elements.canvas.width = project.video.width;
    elements.canvas.height = project.video.height;
    elements.frameSlider.value = String(frameId);
    elements.frameNumber.value = String(frameId);
    setSaveState("saved", payload.reviewed ? "Reviewed frame" : "No unsaved changes");
    renderCanvas();
  } catch (error) {
    setSaveState("error", "Load failed");
    showToast(error.message || String(error));
  } finally {
    loading = false;
    elements.loadingFrame.classList.add("hidden");
    renderFrameState();
  }
}

async function saveCurrentFrame() {
  if (loading) return;
  loading = true;
  setSaveState("", "Saving…");
  try {
    const result = await api("/api/frame", jsonOptions({
      frame_id: frameId,
      annotations,
    }));
    framePayload = result.frame;
    project = result.project;
    dirty = false;
    setSaveState("saved", "Saved as reviewed");
    updateProjectUi();
  } catch (error) {
    setSaveState("error", "Save failed");
    showToast(error.message || String(error));
    throw error;
  } finally {
    loading = false;
    renderFrameState();
  }
}

function acceptSuggestions() {
  const usable = suggestions.filter((item) => Number(item.suggested_identity_id) >= 1);
  const identities = new Set();
  annotations = usable.filter((item) => {
    const identity = Number(item.suggested_identity_id);
    if (identities.has(identity)) return false;
    identities.add(identity);
    return true;
  }).map((item) => ({
    identity_id: Number(item.suggested_identity_id),
    x: item.x,
    y: item.y,
    width: item.width,
    height: item.height,
    visibility: 1,
  }));
  selectedIndex = annotations.length ? 0 : -1;
  if (selectedIndex >= 0) selectBox(selectedIndex);
  markDirty();
  renderCanvas();
}

function applySelectedIdentity() {
  if (selectedIndex < 0 || !annotations[selectedIndex]) return showToast("Select a box first");
  const identity = Number(elements.identityInput.value);
  if (!Number.isInteger(identity) || identity < 1) return showToast("Identity ID must be a positive integer");
  if (annotations.some((box, index) => index !== selectedIndex && box.identity_id === identity)) {
    return showToast(`Identity ${identity} already exists on this frame`);
  }
  annotations[selectedIndex].identity_id = identity;
  annotations[selectedIndex].visibility = Number(elements.visibilityInput.value);
  markDirty();
  renderCanvas();
}

function deleteSelected() {
  if (selectedIndex < 0) return;
  annotations.splice(selectedIndex, 1);
  selectedIndex = Math.min(selectedIndex, annotations.length - 1);
  markDirty();
  renderCanvas();
}

function nextUnreviewedFrame() {
  const reviewed = new Set(project.reviewed_frames || []);
  for (let candidate = frameId + 1; candidate <= project.video.frame_count; candidate += 1) {
    if (!reviewed.has(candidate)) return loadFrame(candidate);
  }
  for (let candidate = 1; candidate < frameId; candidate += 1) {
    if (!reviewed.has(candidate)) return loadFrame(candidate);
  }
  showToast("Every frame is reviewed");
}

elements.canvas.addEventListener("pointerdown", (event) => {
  if (loading) return;
  const point = canvasPoint(event);
  const existing = boxAt(point);
  if (existing >= 0) {
    selectBox(existing);
    return;
  }
  selectedIndex = -1;
  drawing = { start: point, end: point };
  elements.canvas.setPointerCapture(event.pointerId);
  renderCanvas();
});

elements.canvas.addEventListener("pointermove", (event) => {
  if (!drawing) return;
  drawing.end = canvasPoint(event);
  renderCanvas();
});

elements.canvas.addEventListener("pointerup", (event) => {
  if (!drawing) return;
  drawing.end = canvasPoint(event);
  const box = normalizedDrawingBox();
  drawing = null;
  const identity = Number(elements.identityInput.value);
  if (box.width < 5 || box.height < 5) return renderCanvas();
  if (!Number.isInteger(identity) || identity < 1) return showToast("Set a positive identity ID first");
  if (annotations.some((item) => item.identity_id === identity)) return showToast(`Identity ${identity} already exists on this frame`);
  annotations.push({ ...box, identity_id: identity, visibility: Number(elements.visibilityInput.value) });
  selectBox(annotations.length - 1);
  markDirty();
  renderCanvas();
});

elements.previousFrame.addEventListener("click", () => loadFrame(frameId - 1));
elements.nextFrame.addEventListener("click", () => loadFrame(frameId + 1));
elements.nextUnreviewed.addEventListener("click", nextUnreviewedFrame);
elements.frameSlider.addEventListener("change", () => loadFrame(elements.frameSlider.value));
elements.frameNumber.addEventListener("change", () => loadFrame(elements.frameNumber.value));
elements.acceptSuggestions.addEventListener("click", acceptSuggestions);
elements.clearBoxes.addEventListener("click", () => {
  annotations = [];
  selectedIndex = -1;
  markDirty();
  renderCanvas();
});
elements.saveFrame.addEventListener("click", () => saveCurrentFrame().catch(() => {}));
elements.applyIdentity.addEventListener("click", applySelectedIdentity);
elements.deleteBox.addEventListener("click", deleteSelected);
elements.visibilityInput.addEventListener("input", () => {
  elements.visibilityValue.textContent = Number(elements.visibilityInput.value).toFixed(2);
});

document.addEventListener("keydown", (event) => {
  if (event.ctrlKey && event.key.toLowerCase() === "s") {
    event.preventDefault();
    saveCurrentFrame().catch(() => {});
    return;
  }
  if (["INPUT", "TEXTAREA"].includes(document.activeElement?.tagName)) return;
  if (event.key === "ArrowLeft") loadFrame(frameId - 1);
  if (event.key === "ArrowRight") loadFrame(frameId + 1);
  if (event.key === "Delete" || event.key === "Backspace") deleteSelected();
});

window.addEventListener("beforeunload", (event) => {
  if (!dirty) return;
  event.preventDefault();
  event.returnValue = "";
});

async function initialize() {
  try {
    project = await api("/api/project");
    updateProjectUi();
    await loadFrame(1);
  } catch (error) {
    setSaveState("error", "Service unavailable");
    showToast(error.message || String(error));
  }
}

initialize();
