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
  scanPeople: document.querySelector("#scanPeople"),
  analysisOutput: document.querySelector("#analysisOutput"),
  peopleCatalog: document.querySelector("#peopleCatalog"),
  peopleCatalogStatus: document.querySelector("#peopleCatalogStatus"),
  peopleCards: document.querySelector("#peopleCards"),
  targetInput: document.querySelector("#targetInput"),
  useVlmGrounding: document.querySelector("#useVlmGrounding"),
  performanceMode: document.querySelector("#performanceMode"),
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
  people_scanning: "正在扫描稳定人物",
  people_analyzing: "VLM正在分析人物",
  people_ready: "请选择要跟踪的人物",
  tracking: "正在检测跟踪",
  finished: "视频处理完成",
  error: "运行异常",
};

let localBusy = false;
let toastTimer = null;
let lastAnalysisSignature = "";
let lastPeopleSignature = "";

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
  if (tracking.target_color) {
    elements.activePrompts.append(makeChip(`+ ${tracking.target_color} clothing verification`));
  }
}

function visibleAttributeLabels(attributes) {
  if (!attributes) return [];
  const labels = [];
  const garment = (color, type, fallback) => {
    const parts = [];
    if (color && color !== "unknown") parts.push(color);
    if (type && type !== "unknown") parts.push(type);
    else if (parts.length) parts.push(fallback);
    return parts.join(" ");
  };
  const upper = garment(attributes.upper_color, attributes.upper_type, "upper clothing");
  const lower = garment(attributes.lower_color, attributes.lower_type, "lower clothing");
  if (upper) labels.push(upper);
  if (lower) labels.push(lower);
  if (attributes.headwear && !["none", "unknown"].includes(attributes.headwear)) {
    labels.push(`${attributes.headwear_color !== "unknown" ? `${attributes.headwear_color} ` : ""}${attributes.headwear}`);
  }
  if (attributes.carried_object && !["none", "unknown"].includes(attributes.carried_object)) {
    labels.push(`${attributes.carried_object_color !== "unknown" ? `${attributes.carried_object_color} ` : ""}${attributes.carried_object}`);
  }
  if (attributes.safety_vest === "yes") labels.push("safety vest");
  if (attributes.action && attributes.action !== "unknown") {
    labels.push(attributes.action.replaceAll("_", " "));
  }
  return [...new Set(labels)];
}

async function selectPerson(candidate) {
  await withBusy("正在锁定人物", `保持扫描 Track ID ${candidate.scan_track_id}`, async () => {
    const status = await api("/api/people/select", jsonOptions({
      candidate_id: candidate.candidate_id,
    }));
    elements.targetInput.value = candidate.recommended_label || candidate.display_name;
    elements.useVlmGrounding.checked = false;
    renderStatus(status);
    showToast(`已锁定 ${candidate.display_name} · Track ID ${candidate.scan_track_id}`);
  }).catch(() => {});
}

function renderPeopleCatalog(catalog) {
  const value = catalog || { state: "idle", candidates: [] };
  const signature = JSON.stringify(value);
  if (signature === lastPeopleSignature) return;
  lastPeopleSignature = signature;
  const candidates = value.candidates || [];
  elements.peopleCards.replaceChildren();
  elements.peopleCatalog.classList.toggle("empty", value.state === "idle");

  const stateMessages = {
    idle: "人物卡片会显示在这里。",
    scanning: `正在累计稳定轨迹：${value.frames_processed || 0}/${value.frames_target || 0} 帧`,
    analyzing: `已找到 ${candidates.length} 人，Qwen3-VL 正在逐人分析…`,
    ready: `已生成 ${candidates.filter((item) => item.state === "ready").length} 张人物卡片，请直接选择。`,
    selected: "已选择具体人物，系统只输出锁定的 Track ID。",
    error: "人物扫描未完成，请查看错误并重试。",
  };
  elements.peopleCatalogStatus.textContent = stateMessages[value.state] || value.state;

  for (const candidate of candidates) {
    const card = document.createElement("article");
    card.className = "person-card";
    if (candidate.candidate_id === value.selected_candidate_id) card.classList.add("selected");

    const image = document.createElement("img");
    image.src = candidate.crop_url;
    image.alt = `${candidate.display_name} crop`;
    image.loading = "lazy";

    const body = document.createElement("div");
    body.className = "person-card-body";
    const heading = document.createElement("div");
    heading.className = "person-card-heading";
    const name = document.createElement("strong");
    name.textContent = candidate.display_name;
    const track = document.createElement("span");
    track.textContent = `SCAN ID ${candidate.scan_track_id}`;
    heading.append(name, track);

    const recommendation = document.createElement("p");
    recommendation.className = "person-recommendation";
    recommendation.textContent = candidate.state === "analyzing"
      ? "Analyzing visible attributes…"
      : candidate.error || candidate.recommended_label || "person";

    const chips = document.createElement("div");
    chips.className = "person-attributes";
    for (const label of visibleAttributeLabels(candidate.attributes)) chips.append(makeChip(label));

    const meta = document.createElement("small");
    const vlmConfidence = candidate.attributes?.confidence;
    meta.textContent = `track samples ${candidate.sample_count} · detector ${Number(candidate.detector_confidence || 0).toFixed(2)}${vlmConfidence == null ? "" : ` · VLM ${Number(vlmConfidence).toFixed(2)}`}`;

    const button = document.createElement("button");
    button.type = "button";
    button.className = "button person-select";
    const selected = candidate.candidate_id === value.selected_candidate_id;
    button.textContent = selected ? "TRACKING THIS PERSON" : "TRACK THIS PERSON";
    button.disabled = candidate.state !== "ready" || selected || value.state !== "ready";
    button.addEventListener("click", () => selectPerson(candidate));

    body.append(heading, recommendation, chips, meta, button);
    card.append(image, body);
    elements.peopleCards.append(card);
  }
}

function renderDetections(tracking, phase) {
  const detections = tracking.detections || [];
  elements.detectionBadge.textContent = String(detections.length);
  elements.detectionsList.replaceChildren();
  if (!detections.length) {
    elements.detectionsList.className = "detections-list empty-list";
    if (phase === "finished" && tracking.seen_track_ids?.length) {
      const ids = tracking.seen_track_ids.join(", ");
      elements.detectionsList.textContent = `视频已结束：目标累计出现在 ${tracking.matched_frame_count || 0} 帧，Track ID ${ids}；最后一帧没有目标。`;
    } else {
      elements.detectionsList.textContent = tracking.active ? "当前帧没有找到目标" : "暂无检测结果";
    }
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
    const detectorScore = Number(detection.confidence || 0).toFixed(2);
    const colorScore = detection.color_label
      ? ` · ${detection.color_label} ${Number(detection.color_score || 0).toFixed(2)}`
      : "";
    score.textContent = `${detectorScore}${colorScore}`;
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
  elements.scanPeople.disabled = disabled || !source.connected || tracking.active;
  elements.startTracking.disabled = disabled || !source.connected || tracking.active;
  elements.stopTracking.disabled = disabled || !tracking.active;
  elements.connectCamera.disabled = disabled;
  elements.uploadVideo.disabled = disabled || !elements.videoFile.files.length;
  renderAnalysis(status.scene_analysis);
  renderPeopleCatalog(status.people_catalog);
  renderPrompts(tracking);
  renderDetections(tracking, status.phase);
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

elements.scanPeople.addEventListener("click", async () => {
  await withBusy("正在启动人物扫描", "加载 YOLO-World + ByteTrack，随后自动分析人物裁剪", async () => {
    const status = await api("/api/people/scan", jsonOptions({
      performance_mode: elements.performanceMode.value,
    }));
    lastPeopleSignature = "";
    renderStatus(status);
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
      performance_mode: elements.performanceMode.value,
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
