// Elements
const form = document.querySelector("#analysis-form") || document.body;
const fileInput = document.querySelector("#audio-file");
const dropZone = document.querySelector("#drop-zone");
const userInput = document.querySelector("#user-id");
const usersDatalist = document.querySelector("#users-datalist");
const apiInput = document.querySelector("#api-url");
const message = document.querySelector("#message");
const analyzeButton = document.querySelector("#analyze-button");
const baselineButton = document.querySelector("#baseline-button");
const sampleButtons = document.querySelectorAll(".sample-btn");
const health = document.querySelector("#health");
const heading = document.querySelector("#result-heading");
const statusBadge = document.querySelector("#status-badge");
const speakerBadge = document.querySelector("#speaker-badge");
const emptyState = document.querySelector("#empty-state");
const resultsContent = document.querySelector("#results-content");
const jsonResult = document.querySelector("#json-result");
const copyJsonBtn = document.querySelector("#copy-json-btn");
const playerContainer = document.querySelector("#player-container");
const audioPlayer = document.querySelector("#audio-player");
const fileMeta = document.querySelector("#file-meta");

// Gauge elements
const gaugeBar = document.querySelector("#gauge-bar");
const gaugeNumber = document.querySelector("#gauge-number");
const confidenceText = document.querySelector("#confidence-text");
const methodText = document.querySelector("#method-text");

// Emotion elements
const emotionIcon = document.querySelector("#emotion-icon");
const dominantEmotionLabel = document.querySelector("#dominant-emotion-label");
const emotionConfidenceText = document.querySelector("#emotion-confidence-text");
const barNeu = document.querySelector("#bar-neu");
const barAng = document.querySelector("#bar-ang");
const barHap = document.querySelector("#bar-hap");
const barSad = document.querySelector("#bar-sad");
const valNeu = document.querySelector("#val-neu");
const valAng = document.querySelector("#val-ang");
const valHap = document.querySelector("#val-hap");
const valSad = document.querySelector("#val-sad");

// XAI elements
const xaiSection = document.querySelector("#xai-section");
const xaiAlerts = document.querySelector("#xai-alerts");

// Physics elements
const valF0 = document.querySelector("#val-f0");
const baselineF0 = document.querySelector("#baseline-f0");
const zscoreF0 = document.querySelector("#zscore-f0");
const badgeF0 = document.querySelector("#badge-f0");

const valJitter = document.querySelector("#val-jitter");
const baselineJitter = document.querySelector("#baseline-jitter");
const zscoreJitter = document.querySelector("#zscore-jitter");
const badgeJitter = document.querySelector("#badge-jitter");

const valShimmer = document.querySelector("#val-shimmer");
const baselineShimmer = document.querySelector("#baseline-shimmer");
const zscoreShimmer = document.querySelector("#zscore-shimmer");
const badgeShimmer = document.querySelector("#badge-shimmer");

const valHnr = document.querySelector("#val-hnr");
const baselineHnr = document.querySelector("#baseline-hnr");
const zscoreHnr = document.querySelector("#zscore-hnr");
const badgeHnr = document.querySelector("#badge-hnr");

const breakdownTbody = document.querySelector("#breakdown-tbody");

// Initialize API URL
if (!apiInput.value) {
  if (window.location.protocol.startsWith("http")) {
    apiInput.value = window.location.origin;
  } else {
    apiInput.value = "http://127.0.0.1:8000";
  }
}

function setMessage(text, isError = false) {
  message.textContent = text;
  message.classList.toggle("error", isError);
}

function getBaseUrl() {
  return apiInput.value.trim().replace(/\/+$/, "");
}

// Check Health & Load Baselines
async function checkHealth() {
  const base = getBaseUrl();
  try {
    const response = await fetch(`${base}/health`);
    if (!response.ok) throw new Error("API chưa sẵn sàng");
    const result = await response.json();
    health.className = result.pipeline_ready ? "health ok" : "health error";
    health.lastElementChild.textContent = result.pipeline_ready
      ? `API Sẵn sàng ${result.stress_model_ready ? "· Model Đã Nạp" : ""}`
      : "API Đang khởi động...";
  } catch {
    health.className = "health error";
    health.lastElementChild.textContent = "Không kết nối được API";
  }
}

async function loadBaselines() {
  const base = getBaseUrl();
  try {
    const res = await fetch(`${base}/api/v1/baselines`);
    if (res.ok) {
      const data = await res.json();
      usersDatalist.replaceChildren();
      for (const u of data.registered_users || []) {
        const opt = document.createElement("option");
        opt.value = u;
        usersDatalist.appendChild(opt);
      }
    }
  } catch {
    // Ignore baseline fetch errors
  }
}

// File handling & Audio preview
let currentFile = null;

function setAudioSource(fileOrUrl, fileName, fileSize = "") {
  playerContainer.classList.remove("hidden");
  if (typeof fileOrUrl === "string") {
    audioPlayer.src = fileOrUrl;
    fileMeta.textContent = `${fileName} ${fileSize ? `· ${fileSize}` : ""}`;
  } else {
    currentFile = fileOrUrl;
    audioPlayer.src = URL.createObjectURL(fileOrUrl);
    fileMeta.textContent = `${fileOrUrl.name} · ${(fileOrUrl.size / (1024 * 1024)).toFixed(2)} MiB`;
  }
  audioPlayer.load();
}

fileInput.addEventListener("change", () => {
  if (fileInput.files?.[0]) {
    currentFile = fileInput.files[0];
    setAudioSource(currentFile);
  }
});

// Drag & Drop
if (dropZone) {
  ["dragenter", "dragover"].forEach((evt) => {
    dropZone.addEventListener(evt, (e) => {
      e.preventDefault();
      dropZone.classList.add("dragover");
    });
  });
  ["dragleave", "drop"].forEach((evt) => {
    dropZone.addEventListener(evt, (e) => {
      e.preventDefault();
      dropZone.classList.remove("dragover");
    });
  });
  dropZone.addEventListener("drop", (e) => {
    if (e.dataTransfer.files?.length) {
      currentFile = e.dataTransfer.files[0];
      setAudioSource(currentFile);
    }
  });
}

// Sample buttons
sampleButtons.forEach((btn) => {
  btn.addEventListener("click", async () => {
    const sampleId = btn.getAttribute("data-sample");
    const base = getBaseUrl();
    setMessage(`Đang tải mẫu âm thanh ${btn.textContent.trim()}...`);
    try {
      const audioUrl = `${base}/api/v1/sample-audio/${sampleId}`;
      const res = await fetch(audioUrl);
      if (!res.ok) throw new Error("Không thể tải mẫu âm thanh");
      const blob = await res.blob();
      currentFile = new File([blob], `${sampleId}.wav`, { type: "audio/wav" });
      setAudioSource(currentFile);
      if (sampleId.includes("calm_male") || sampleId.includes("angry_male")) {
        userInput.value = "driver_actor_01";
      }
      setMessage(`Đã tải mẫu: ${btn.textContent.trim()}. Bấm "Phân tích Stress" để kiểm tra.`);
    } catch (err) {
      setMessage(`Lỗi tải mẫu: ${err.message}`, true);
    }
  });
});

// Format numbers
function fmt(val, digits = 1) {
  if (val === null || val === undefined || Number.isNaN(val)) return "--";
  return typeof val === "number" ? val.toFixed(digits) : String(val);
}

// Update Gauge (circumference = 2 * PI * 50 = 314.159)
function updateGauge(score, status) {
  const circumference = 314.159;
  const numScore = typeof score === "number" ? Math.max(0, Math.min(100, score)) : 0;
  const offset = circumference - (circumference * numScore) / 100;
  gaugeBar.style.strokeDashoffset = offset;
  gaugeNumber.textContent = fmt(score, 1);

  if (status === "HIGH_STRESS") {
    gaugeBar.style.stroke = "#ef4444";
  } else if (status === "MODERATE_STRESS") {
    gaugeBar.style.stroke = "#f59e0b";
  } else {
    gaugeBar.style.stroke = "#10b981";
  }
}

// Render Analysis Results
function renderResult(data) {
  emptyState.classList.add("hidden");
  resultsContent.classList.remove("hidden");

  const assessment = data.assessment || {};
  const emotion = data.emotion || {};
  const speaker = data.speaker || {};
  const acoustic = data.acoustic_breakdown || {};
  const explanations = Array.isArray(data.explanations) ? data.explanations : [];

  // Status Badge
  statusBadge.classList.remove("hidden");
  const status = assessment.status || "CALM";
  if (status === "HIGH_STRESS") {
    heading.textContent = "Phát hiện Căng thẳng Cao (High Stress)";
    statusBadge.textContent = "HIGH STRESS";
    statusBadge.className = "badge danger";
  } else if (status === "MODERATE_STRESS") {
    heading.textContent = "Căng thẳng Vừa phải (Moderate Stress)";
    statusBadge.textContent = "MODERATE STRESS";
    statusBadge.className = "badge warning";
  } else if (status === "UNCALIBRATED") {
    heading.textContent = "Chưa hiệu chuẩn (Uncalibrated)";
    statusBadge.textContent = "UNCALIBRATED";
    statusBadge.className = "badge neutral";
  } else {
    heading.textContent = "Trạng thái Bình tĩnh (Calm)";
    statusBadge.textContent = "CALM";
    statusBadge.className = "badge calm";
  }

  // Speaker Verification Badge
  if (speaker.user_id && speaker.user_id !== "null") {
    speakerBadge.classList.remove("hidden");
    const sim = speaker.similarity_score !== null ? ` (Cosine: ${speaker.similarity_score.toFixed(3)})` : "";
    if (speaker.verified === true) {
      speakerBadge.textContent = `✓ Đã xác minh: ${speaker.user_id}${sim}`;
      speakerBadge.className = "badge verified";
    } else if (speaker.verified === false) {
      speakerBadge.textContent = `✗ Không khớp: ${speaker.user_id}${sim}`;
      speakerBadge.className = "badge unverified";
    } else {
      speakerBadge.textContent = `Người nói: ${speaker.user_id}`;
      speakerBadge.className = "badge neutral";
    }
  } else {
    speakerBadge.classList.add("hidden");
  }

  // Stress Gauge
  updateGauge(assessment.stress_score, status);
  confidenceText.textContent = `Độ tin cậy: ${assessment.confidence !== null ? (assessment.confidence * 100).toFixed(1) + "%" : "--"}`;

  const methodMap = {
    acoustic_safeguard_hybrid: "Bảo vệ Âm học Sinh lý (Safeguard)",
    trained_hybrid_model: "Mô hình Lai Học Sâu (Wav2Vec2)",
    baseline_zscore_hybrid: "Hiệu chuẩn Z-Score Nền",
  };
  methodText.textContent = methodMap[assessment.method] || assessment.method || "Mô hình lai";

  // Dominant Emotion
  const emoMap = {
    neu: { label: "Bình thường / Trung tính (Neutral)", icon: "😐" },
    ang: { label: "Giận dữ / Căng thẳng (Anger)", icon: "😠" },
    hap: { label: "Vui vẻ / Hào hứng (Happy)", icon: "😊" },
    sad: { label: "Buồn bã / Trầm uất (Sadness)", icon: "😢" },
  };
  const currentEmo = emoMap[emotion.label] || { label: emotion.label || "Chưa có", icon: "🎙️" };
  dominantEmotionLabel.textContent = currentEmo.label;
  emotionIcon.textContent = currentEmo.icon;
  emotionConfidenceText.textContent = `Độ tin cậy cảm xúc: ${fmt(emotion.confidence, 2)}`;

  // Probabilities
  const probs = emotion.probabilities || {};
  const neuP = (probs.neu || 0) * 100;
  const angP = (probs.ang || 0) * 100;
  const hapP = (probs.hap || 0) * 100;
  const sadP = (probs.sad || 0) * 100;

  barNeu.style.width = `${Math.min(100, neuP)}%`;
  valNeu.textContent = `${neuP.toFixed(1)}%`;
  barAng.style.width = `${Math.min(100, angP)}%`;
  valAng.textContent = `${angP.toFixed(1)}%`;
  barHap.style.width = `${Math.min(100, hapP)}%`;
  valHap.textContent = `${hapP.toFixed(1)}%`;
  barSad.style.width = `${Math.min(100, sadP)}%`;
  valSad.textContent = `${sadP.toFixed(1)}%`;

  // XAI Explanations
  if (explanations.length > 0) {
    xaiSection.classList.remove("hidden");
    xaiAlerts.replaceChildren();
    for (const exp of explanations) {
      const alert = document.createElement("div");
      let icon = "🔍";
      let typeClass = "info";
      if (exp.includes("safeguard") || exp.includes("thư giãn") || exp.includes("bảo vệ")) {
        icon = "🛡️";
        typeClass = "safeguard";
      } else if (exp.includes("vượt ngưỡng") || exp.includes("stress") || exp.includes("căng cứng")) {
        icon = "⚠️";
        typeClass = "warning";
      }
      alert.className = `xai-alert ${typeClass}`;
      alert.innerHTML = `<span class="xai-icon">${icon}</span><span>${exp}</span>`;
      xaiAlerts.appendChild(alert);
    }
  } else {
    xaiSection.classList.add("hidden");
  }

  // Acoustic Physics Cards
  function setAcousticCard(elVal, elBase, elZ, elBadge, item, unit) {
    elVal.textContent = fmt(item?.value, 2);
    elBase.textContent = `Baseline: ${fmt(item?.baseline, 2)} ${unit}`;
    const z = item?.z_score;
    elZ.textContent = `Z: ${z !== null && z !== undefined ? (z > 0 ? "+" : "") + z.toFixed(2) : "--"}`;

    if (z !== null && z !== undefined) {
      if (z >= 1.5) {
        elBadge.textContent = "Elevated (+Z)";
        elBadge.className = "mini-badge elevated";
      } else if (z <= -1.5) {
        elBadge.textContent = "Reduced (-Z)";
        elBadge.className = "mini-badge reduced";
      } else {
        elBadge.textContent = "Normal";
        elBadge.className = "mini-badge normal";
      }
    } else {
      elBadge.textContent = "Measured";
      elBadge.className = "mini-badge";
    }
  }

  setAcousticCard(valF0, baselineF0, zscoreF0, badgeF0, acoustic.f0_mean_hz, "Hz");
  setAcousticCard(valJitter, baselineJitter, zscoreJitter, badgeJitter, acoustic.jitter_percent, "%");
  setAcousticCard(valShimmer, baselineShimmer, zscoreShimmer, badgeShimmer, acoustic.shimmer_percent, "%");
  setAcousticCard(valHnr, baselineHnr, zscoreHnr, badgeHnr, acoustic.hnr_db, "dB");

  // Breakdown Table
  const tableDescriptions = {
    f0_mean_hz: "Tần số cơ bản (Pitch) - Phản ánh độ căng dây thanh âm",
    jitter_percent: "Rung tần số chu kỳ vi mô - Dấu hiệu run giọng khi kích động",
    shimmer_percent: "Rung biên độ chu kỳ vi mô - Áp lực khí dưới thanh môn",
    hnr_db: "Tỷ số sóng điều hòa trên nhiễu - Độ trong thanh âm vs hụt hơi",
  };

  breakdownTbody.replaceChildren();
  for (const [key, row] of Object.entries(acoustic)) {
    const tr = document.createElement("tr");
    const z = row.z_score;
    const zText = z !== null && z !== undefined ? (z > 0 ? "+" : "") + z.toFixed(2) : "--";
    tr.innerHTML = `
      <td><strong>${key}</strong></td>
      <td>${fmt(row.value, 2)}</td>
      <td>${fmt(row.baseline, 2)}</td>
      <td>${zText}</td>
      <td>${tableDescriptions[key] || "Chỉ số âm học thanh quản"}</td>
    `;
    breakdownTbody.appendChild(tr);
  }

  // Raw JSON
  jsonResult.textContent = JSON.stringify(data, null, 2);
}

// Send Audio API Request
async function sendAudio(endpoint, successMessage, render = true) {
  if (!currentFile) {
    setMessage("Vui lòng chọn hoặc kéo thả tệp âm thanh trước.", true);
    return;
  }

  const formData = new FormData();
  formData.append("file", currentFile);
  const uid = userInput.value.trim();
  if (uid) {
    formData.append("user_id", uid);
  }

  analyzeButton.disabled = true;
  baselineButton.disabled = true;
  setMessage("Đang truyền và phân tích âm thanh qua SpeechBrain & Acoustic Physics...");

  try {
    const base = getBaseUrl();
    const response = await fetch(`${base}${endpoint}`, {
      method: "POST",
      body: formData,
    });
    const result = await response.json();
    if (!response.ok) {
      throw new Error(result.detail || "Yêu cầu API không thành công.");
    }

    if (render) {
      renderResult(result);
    } else {
      emptyState.classList.add("hidden");
      resultsContent.classList.remove("hidden");
      heading.textContent = `Đã Đăng Ký Baseline Cho ${result.result?.user_id || uid}`;
      statusBadge.textContent = "BASELINE REGISTERED";
      statusBadge.className = "badge verified";
      jsonResult.textContent = JSON.stringify(result, null, 2);
      loadBaselines();
    }
    setMessage(successMessage);
  } catch (err) {
    setMessage(`Lỗi: ${err.message}`, true);
  } finally {
    analyzeButton.disabled = false;
    baselineButton.disabled = false;
  }
}

// Button Events
analyzeButton.addEventListener("click", () => {
  sendAudio("/api/v1/analyze-stress", "Phân tích âm thanh hoàn tất!");
});

baselineButton.addEventListener("click", () => {
  if (!userInput.value.trim()) {
    setMessage("Vui lòng nhập Mã người dùng (User ID) để lưu giọng nền baseline.", true);
    userInput.focus();
    return;
  }
  sendAudio("/api/v1/register-baseline", "Đăng ký baseline cá nhân thành công!", false);
});

// Copy JSON
copyJsonBtn?.addEventListener("click", () => {
  navigator.clipboard.writeText(jsonResult.textContent).then(() => {
    copyJsonBtn.textContent = "✓ Đã sao chép!";
    setTimeout(() => {
      copyJsonBtn.textContent = "📋 Sao chép JSON";
    }, 2000);
  });
});

apiInput.addEventListener("change", () => {
  checkHealth();
  loadBaselines();
});

// Initial boot
checkHealth();
loadBaselines();
