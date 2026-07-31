const elements = {
  statusDot: document.querySelector("#statusDot"),
  phaseLabel: document.querySelector("#phaseLabel"),
  sourceMeta: document.querySelector("#sourceMeta"),
  videoEmpty: document.querySelector("#videoEmpty"),
  cameraTab: document.querySelector("#cameraTab"),
  uploadTab: document.querySelector("#uploadTab"),
  cameraPane: document.querySelector("#cameraPane"),
  uploadPane: document.querySelector("#uploadPane"),
  cameraIndex: document.querySelector("#cameraIndex"),
  connectCamera: document.querySelector("#connectCamera"),
  videoFile: document.querySelector("#videoFile"),
  fileName: document.querySelector("#fileName"),
  uploadVideo: document.querySelector("#uploadVideo"),
  analyzeScene: document.querySelector("#analyzeScene"),
  analysisOutput: document.querySelector("#analysisOutput"),
  targetInput: document.querySelector("#targetInput"),
  useVlmGrounding: document.querySelector("#useVlmGrounding"),
  activePrompts: document.querySelector("#activePrompts"),
  startTracking: document.querySelector("#startTracking"),
  stopTracking: document.querySelector("#stopTracking"),
  detectionBadge: document.querySelector("#detectionBadge"),
  detectionsList: document.querySelector("#detectionsList"),
  metricFrame: document.querySelector("#metricFrame"),
  metricDetections: document.querySelector("#metricDetections"),
  metricTracks: document.querySelector("#metricTracks"),
  metricFps: document.querySelector("#metricFps"),
  metricLatency: document.querySelector("#metricLatency"),
  busyOverlay: document.querySelector("#busyOverlay"),
  busyTitle: document.querySelector("#busyTitle"),
  busyDetail: document.querySelector("#busyDetail"),
  toast: document.querySelector("#toast"),
};

const phaseNames = {
  idle: "等待视频源",
  preview: "画面已就绪",
  tracking: "正在检测跟踪",
  finished: "视频处理完成",
  error: "运行异常",
};

let localBusy = false;
let toastTimer = null;
let lastAnalysisSignature = "";

async function api(path, options = {}) {
  const response = await fetch(path, options);
  let payload = {};
  try {
    payload = await response.json();
  } catch (_) {
    throw new Error(`服务返回了无效响应 (${response.status})`);
  }
  if (!response.ok) {
    throw new Error(payload.error || `请求失败 (${response.status})`);
  }
  return payload;
}

function jsonOptions(payload = {}) {
  return {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(payload),
  };
}

function setBusy(visible, title = "正在处理", detail = "本地模型可能需要几秒钟") {
  localBusy = visible;
  elements.busyTitle.textContent = title;
  elements.busyDetail.textContent = detail;
  elements.busyOverlay.classList.toggle("visible", visible);
  elements.busyOverlay.setAttribute("aria-hidden", String(!visible));
}

async function withBusy(title, detail, operation) {
  setBusy(true, title, detail);
  try {
    return await operation();
  } catch (error) {
    showToast(error.message || String(error));
    throw error;
  } finally {
    setBusy(false);
  }
}

function showToast(message) {
  if (!message) return;
  elements.toast.textContent = message;
  elements.toast.classList.add("visible");
  clearTimeout(toastTimer);
  toastTimer = setTimeout(() => elements.toast.classList.remove("visible"), 5000);
}

function switchSourceTab(tab) {
  const camera = tab === "camera";
  elements.cameraTab.classList.toggle("active", camera);
  elements.uploadTab.classList.toggle("active", !camera);
  elements.cameraPane.classList.toggle("active", camera);
  elements.uploadPane.classList.toggle("active", !camera);
}

function makeChip(text) {
  const chip = document.createElement("span");
  chip.className = "chip";
  chip.textContent = text;
  return chip;
}

function makeTargetChoice(prompt) {
  const button = document.createElement("button");
  button.type = "button";
  button.className = "target-choice";
  button.textContent = prompt;
  button.addEventListener("click", () => {
    elements.targetInput.value = prompt;
    elements.useVlmGrounding.checked = false;
    showToast(`已选择 ${prompt}，点击“开始检测跟踪”即可`);
  });
  return button;
}

function renderAnalysis(analysis) {
  if (!analysis) return;
  const signature = JSON.stringify(analysis);
  if (signature === lastAnalysisSignature) return;
  lastAnalysisSignature = signature;
  elements.analysisOutput.replaceChildren();
  elements.analysisOutput.classList.remove("empty");

  const summary = document.createElement("p");
  summary.textContent = analysis.summary || "未返回场景摘要";
  elements.analysisOutput.append(summary);

  const chips = document.createElement("div");
  chips.className = "object-chips";
  for (const object of analysis.objects || []) {
    const attributes = (object.attributes || []).join(" / ");
    const count = object.count == null ? "" : ` ×${object.count}`;
    chips.append(makeChip(`${object.name}${count}${attributes ? ` · ${attributes}` : ""}`));
  }
  if (chips.childElementCount) elements.analysisOutput.append(chips);

  const prompts = analysis.grounding?.yolo_world_prompts || [];
  if (prompts.length) {
    const title = document.createElement("p");
    title.className = "choice-title";
    title.textContent = "可框选目标（点击选择）";
    const choices = document.createElement("div");
    choices.className = "target-choices";
    for (const prompt of prompts) choices.append(makeTargetChoice(prompt));
    elements.analysisOutput.append(title, choices);
  }
}

function renderPrompts(tracking) {
  elements.activePrompts.replaceChildren();
  if (!tracking.prompts || !tracking.prompts.length) {
    const empty = document.createElement("span");
    empty.textContent = "YOLO-World 检测词尚未生成";
    elements.activePrompts.append(empty);
    return;
  }
  for (const prompt of tracking.prompts) elements.activePrompts.append(makeChip(prompt));
  if (tracking.attribute_filter) elements.activePrompts.append(makeChip("+ 白衣 HSV 复核"));
}

function renderDetections(tracking) {
  const detections = tracking.detections || [];
  elements.detectionBadge.textContent = String(detections.length);
  elements.detectionsList.replaceChildren();
  if (!detections.length) {
    elements.detectionsList.className = "detections-list empty-list";
    elements.detectionsList.textContent = tracking.active ? "当前帧没有找到目标" : "暂无检测结果";
    return;
  }
  elements.detectionsList.className = "detections-list";
  for (const detection of detections) {
    const row = document.createElement("div");
    row.className = "detection-row";
    const id = document.createElement("span");
    id.className = "track-tag";
    id.textContent = `ID ${detection.track_id == null ? "?" : detection.track_id}`;
    const name = document.createElement("span");
    name.className = "detection-name";
    name.textContent = detection.label;
    const score = document.createElement("span");
    score.className = "detection-score";
    score.textContent = Number(detection.confidence || 0).toFixed(2);
    row.append(id, name, score);
    elements.detectionsList.append(row);
  }
}

function renderStatus(status) {
  const source = status.source || {};
  const tracking = status.tracking || {};
  elements.phaseLabel.textContent = phaseNames[status.phase] || status.phase;
  elements.statusDot.className = "status-dot";
  if (status.error) elements.statusDot.classList.add("error");
  else if (source.connected) elements.statusDot.classList.add("active");

  elements.videoEmpty.style.display = source.connected ? "none" : "flex";
  elements.sourceMeta.textContent = source.connected
    ? `${source.label} · ${source.width}×${source.height} · ${Number(source.fps).toFixed(1)} FPS`
    : "尚未连接";
  const hasRun = tracking.active || status.phase === "finished";
  const visibleTrackIds = tracking.track_ids?.length
    ? tracking.track_ids
    : tracking.seen_track_ids;
  elements.metricFrame.textContent = source.connected ? String(tracking.frame_id || 0) : "—";
  elements.metricDetections.textContent = hasRun ? String(tracking.detection_count || 0) : "—";
  elements.metricTracks.textContent = hasRun && visibleTrackIds?.length
    ? visibleTrackIds.join(", ")
    : "—";
  elements.metricFps.textContent = hasRun
    ? `${Number(tracking.processing_fps || 0).toFixed(1)} FPS`
    : "—";
  elements.metricLatency.textContent = hasRun
    ? `${Number(tracking.mean_inference_ms || 0).toFixed(1)} ms`
    : "—";

  const disabled = localBusy || status.busy;
  elements.analyzeScene.disabled = disabled || !source.connected;
  elements.startTracking.disabled = disabled || !source.connected;
  elements.stopTracking.disabled = disabled || !tracking.active;
  elements.connectCamera.disabled = disabled;
  elements.uploadVideo.disabled = disabled || !elements.videoFile.files.length;
  renderAnalysis(status.scene_analysis);
  renderPrompts(tracking);
  renderDetections(tracking);
  if (status.error) showToast(status.error);
}

async function pollStatus() {
  try {
    const status = await api("/api/status");
    renderStatus(status);
  } catch (error) {
    elements.phaseLabel.textContent = "服务连接失败";
    elements.statusDot.className = "status-dot error";
  }
}

elements.cameraTab.addEventListener("click", () => switchSourceTab("camera"));
elements.uploadTab.addEventListener("click", () => switchSourceTab("upload"));

elements.videoFile.addEventListener("change", () => {
  const file = elements.videoFile.files[0];
  elements.fileName.textContent = file ? file.name : "选择本地视频";
  elements.uploadVideo.disabled = !file;
});

elements.connectCamera.addEventListener("click", async () => {
  await withBusy("正在连接摄像头", "首次打开设备可能需要几秒钟", async () => {
    const status = await api("/api/source/camera", jsonOptions({
      index: Number(elements.cameraIndex.value || 0),
    }));
    renderStatus(status);
  }).catch(() => {});
});

elements.uploadVideo.addEventListener("click", async () => {
  const file = elements.videoFile.files[0];
  if (!file) return showToast("请先选择视频文件");
  await withBusy("正在上传视频", `${file.name} · ${(file.size / 1048576).toFixed(1)} MB`, async () => {
    const form = new FormData();
    form.append("file", file);
    const status = await api("/api/source/upload", { method: "POST", body: form });
    renderStatus(status);
  }).catch(() => {});
});

elements.analyzeScene.addEventListener("click", async () => {
  await withBusy("VLM 正在理解画面", "Qwen3-VL 正在分析当前帧", async () => {
    const result = await api("/api/analyze", jsonOptions());
    lastAnalysisSignature = "";
    renderAnalysis(result.analysis);
    await pollStatus();
  }).catch(() => {});
});

elements.startTracking.addEventListener("click", async () => {
  const target = elements.targetInput.value.trim();
  if (!target) return showToast("请输入要识别和跟踪的目标");
  const useVlm = elements.useVlmGrounding.checked;
  const detail = useVlm
    ? "VLM 先生成检测词，然后加载 YOLO-World 与 ByteTrack"
    : "正在加载 YOLO-World 与 ByteTrack";
  await withBusy("正在启动检测跟踪", detail, async () => {
    const status = await api("/api/tracking/start", jsonOptions({
      target,
      use_vlm_grounding: useVlm,
    }));
    renderStatus(status);
  }).catch(() => {});
});

elements.stopTracking.addEventListener("click", async () => {
  try {
    const status = await api("/api/tracking/stop", jsonOptions());
    renderStatus(status);
  } catch (error) {
    showToast(error.message || String(error));
  }
});

pollStatus();
setInterval(pollStatus, 700);
