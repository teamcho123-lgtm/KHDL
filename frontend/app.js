// Elements
const fileInput = document.querySelector("#audio-file");
const dropZone = document.querySelector("#drop-zone");
const userInput = document.querySelector("#user-id");
const overwriteCheckbox = document.querySelector("#overwrite-baseline");
const identifyCheckbox = document.querySelector("#identify-speaker");
const apiInput = document.querySelector("#api-url");
const apiKeyInput = document.querySelector("#api-key");
const message = document.querySelector("#message");
const analyzeButton = document.querySelector("#analyze-button");
const baselineButton = document.querySelector("#baseline-button");
const deleteBaselineButton = document.querySelector("#delete-baseline-button");
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
const durationText = document.querySelector("#duration-text");
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

function getAuthHeaders() {
  const headers = {};
  const key = apiKeyInput?.value.trim();
  if (key) {
    headers["X-API-Key"] = key;
  }
  return headers;
}

// Check Health (Readiness)
async function checkHealth() {
  const base = getBaseUrl();
  try {
    const response = await fetch(`${base}/health`, { headers: getAuthHeaders() });
    if (!response.ok) throw new Error("API chưa sẵn sàng");
    const result = await response.json();
    const isReady = Boolean(result.ready);
    health.className = isReady ? "health ok" : "health error";
    if (result.status === "healthy") {
      health.lastElementChild.textContent = "API Sẵn sàng · Đủ Model";
    } else if (result.status === "degraded") {
      health.lastElementChild.textContent = "API Hoạt động (Thiếu Model / Cần Baseline)";
    } else {
      health.lastElementChild.textContent = "API Đang khởi động...";
    }
  } catch {
    health.className = "health error";
    health.lastElementChild.textContent = "Không kết nối được API";
  }
}

// File handling & Audio preview
let currentFile = null;

// Sniff audio format by reading magic bytes to prevent browser demuxer errors
// when files have mismatched extensions (e.g. ADTS AAC or MP3 named .wav)
async function sniffAudioMime(file) {
  try {
    const slice = await file.slice(0, 16).arrayBuffer();
    const bytes = new Uint8Array(slice);
    if (bytes[0] === 0x52 && bytes[1] === 0x49 && bytes[2] === 0x46 && bytes[3] === 0x46) {
      return "audio/wav";
    }
    if (bytes[0] === 0xFF && (bytes[1] & 0xF6) === 0xF0) {
      return "audio/aac";
    }
    if (
      (bytes[0] === 0x49 && bytes[1] === 0x44 && bytes[2] === 0x33) ||
      (bytes[0] === 0xFF && (bytes[1] & 0xFE) === 0xFA)
    ) {
      return "audio/mpeg";
    }
    if (bytes[0] === 0x4F && bytes[1] === 0x67 && bytes[2] === 0x67 && bytes[3] === 0x53) {
      return "audio/ogg";
    }
    if (bytes[0] === 0x66 && bytes[1] === 0x4C && bytes[2] === 0x61 && bytes[3] === 0x43) {
      return "audio/flac";
    }
    if (bytes[4] === 0x66 && bytes[5] === 0x74 && bytes[6] === 0x79 && bytes[7] === 0x70) {
      return "audio/mp4";
    }
  } catch {}
  return file.type || "audio/wav";
}

// Convert AudioBuffer to a clean 16-bit PCM WAV Blob with exact RIFF headers.
// Optimized using typed Int16Array and pre-cached channels to eliminate GC freezing on long audio.
function bufferToWavBlob(buffer) {
  const numChannels = buffer.numberOfChannels;
  const sampleRate = buffer.sampleRate;
  const format = 1; // PCM
  const bitDepth = 16;
  const bytesPerSample = bitDepth / 8;
  const blockAlign = numChannels * bytesPerSample;
  const byteRate = sampleRate * blockAlign;
  const dataSize = buffer.length * blockAlign;
  const totalSize = 44 + dataSize;

  const arrayBuffer = new ArrayBuffer(totalSize);
  const view = new DataView(arrayBuffer);

  function writeString(offset, string) {
    for (let i = 0; i < string.length; i++) {
      view.setUint8(offset + i, string.charCodeAt(i));
    }
  }

  writeString(0, "RIFF");
  view.setUint32(4, 36 + dataSize, true);
  writeString(8, "WAVE");
  writeString(12, "fmt ");
  view.setUint32(16, 16, true);
  view.setUint16(20, format, true);
  view.setUint16(22, numChannels, true);
  view.setUint32(24, sampleRate, true);
  view.setUint32(28, byteRate, true);
  view.setUint16(32, blockAlign, true);
  view.setUint16(34, bitDepth, true);
  writeString(36, "data");
  view.setUint32(40, dataSize, true);

  // Pre-fetch channel Float32Arrays once to avoid millions of allocations in loop
  const channels = [];
  for (let c = 0; c < numChannels; c++) {
    channels.push(buffer.getChannelData(c));
  }

  // Fast write using typed Int16Array directly on buffer offset
  const int16 = new Int16Array(arrayBuffer, 44, buffer.length * numChannels);
  let idx = 0;
  for (let i = 0; i < buffer.length; i++) {
    for (let c = 0; c < numChannels; c++) {
      const s = Math.max(-1, Math.min(1, channels[c][i]));
      int16[idx++] = s < 0 ? s * 0x8000 : s * 0x7FFF;
    }
  }

  return new Blob([arrayBuffer], { type: "audio/wav" });
}

async function setAudioSource(fileOrUrl, fileName, fileSize = "") {
  playerContainer.classList.remove("hidden");
  if (typeof fileOrUrl === "string") {
    audioPlayer.src = fileOrUrl;
    fileMeta.textContent = `${fileName} ${fileSize ? `· ${fileSize}` : ""}`;
    audioPlayer.load();
  } else {
    currentFile = fileOrUrl;
    const sizeStr = `${(fileOrUrl.size / (1024 * 1024)).toFixed(2)} MiB`;
    fileMeta.textContent = `${fileOrUrl.name} · ${sizeStr}`;

    // Sniff actual audio format so browser player uses the right demuxer
    const sniffedMime = await sniffAudioMime(fileOrUrl);
    const initialBlob = sniffedMime !== fileOrUrl.type
      ? new Blob([fileOrUrl], { type: sniffedMime })
      : fileOrUrl;
    audioPlayer.src = URL.createObjectURL(initialBlob);
    audioPlayer.load();

    // Decode audio buffer via Web Audio API to obtain exact physical duration (e.g. 2:55)
    // and re-feed as a clean WAV Blob to fix Chrome native audio player 50:00 bug.
    try {
      const AudioCtx = window.AudioContext || window.webkitAudioContext;
      if (AudioCtx) {
        const buf = await fileOrUrl.arrayBuffer();
        const ctx = new AudioCtx();
        ctx.decodeAudioData(
          buf.slice(0),
          (decoded) => {
            const sec = decoded.duration;
            const m = Math.floor(sec / 60);
            const s = Math.floor(sec % 60);
            fileMeta.textContent = `${fileOrUrl.name} · ${sizeStr} · ⏱️ ${m}:${s.toString().padStart(2, "0")}`;

            // Re-feed as clean WAV so Chrome's native player controls display exact duration
            try {
              const cleanWavBlob = bufferToWavBlob(decoded);
              audioPlayer.src = URL.createObjectURL(cleanWavBlob);
              audioPlayer.load();
              // Replace currentFile with the pristine WAV so upload is standard PCM
              currentFile = new File(
                [cleanWavBlob],
                fileOrUrl.name.replace(/\.[^/.]+$/, "") + ".wav",
                { type: "audio/wav" }
              );
            } catch {}

            ctx.close();
          },
          () => {
            // If browser decode fails (e.g. exotic container), keep original file
            // and let backend PyAV handle universal decoding
            ctx.close();
          }
        );
      }
    } catch {}
  }
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
      const res = await fetch(audioUrl, { headers: getAuthHeaders() });
      if (!res.ok) throw new Error("Không thể tải mẫu âm thanh (kiểm tra API hoặc X-API-Key)");
      const blob = await res.blob();
      currentFile = new File([blob], `${sampleId}.wav`, { type: "audio/wav" });
      setAudioSource(currentFile);
      if (sampleId.includes("calm_male") || sampleId.includes("angry_male")) {
        userInput.value = "driver_actor_01";
      }
      setMessage(`Đã tải mẫu: ${btn.textContent.trim()}. Bấm "Phân tích Audio" để kiểm tra.`);
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
  const hasValidScore = typeof score === "number" && !Number.isNaN(score);
  const numScore = hasValidScore ? Math.max(0, Math.min(100, score)) : 0;
  const offset = circumference - (circumference * numScore) / 100;
  gaugeBar.style.strokeDashoffset = offset;
  gaugeNumber.textContent = hasValidScore ? fmt(score, 1) : "--";

  if (!hasValidScore || status === "UNCALIBRATED" || status === "INSUFFICIENT_AUDIO") {
    gaugeBar.style.stroke = "#64748b";
  } else if (status === "HIGH_STRESS") {
    gaugeBar.style.stroke = "#ef4444";
  } else if (status === "MODERATE_STRESS") {
    gaugeBar.style.stroke = "#f59e0b";
  } else if (status === "CALM") {
    gaugeBar.style.stroke = "#10b981";
  } else {
    gaugeBar.style.stroke = "#64748b";
  }
}

// Render Analysis Results (Strictly using DOM textContent to prevent XSS)
function renderResult(data) {
  emptyState.classList.add("hidden");
  resultsContent.classList.remove("hidden");

  const assessment = data.assessment || {};
  const emotion = data.emotion || {};
  const speaker = data.speaker || {};
  const acoustic = data.acoustic_breakdown || {};
  const explanations = Array.isArray(data.explanations) ? data.explanations : [];

  // Status Badge (Explicit checks; unknown statuses default to neutral/unknown, NOT CALM)
  statusBadge.classList.remove("hidden");
  const status = assessment.status || "UNKNOWN";
  if (status === "HIGH_STRESS") {
    heading.textContent = "Chỉ báo Kích hoạt / Stress Proxy Cao (High)";
    statusBadge.textContent = "HIGH STRESS (PROXY)";
    statusBadge.className = "badge danger";
  } else if (status === "MODERATE_STRESS") {
    heading.textContent = "Chỉ báo Kích hoạt / Stress Proxy Vừa (Moderate)";
    statusBadge.textContent = "MODERATE STRESS (PROXY)";
    statusBadge.className = "badge warning";
  } else if (status === "CALM") {
    heading.textContent = "Chỉ báo Kích hoạt Thấp / Bình tĩnh (Calm Proxy)";
    statusBadge.textContent = "CALM (PROXY)";
    statusBadge.className = "badge calm";
  } else if (status === "INSUFFICIENT_AUDIO") {
    heading.textContent = "Chất lượng âm học chưa đủ để kết luận";
    statusBadge.textContent = "INSUFFICIENT AUDIO";
    statusBadge.className = "badge neutral";
  } else if (status === "UNCALIBRATED") {
    heading.textContent = "Chưa hiệu chuẩn (Uncalibrated)";
    statusBadge.textContent = "UNCALIBRATED";
    statusBadge.className = "badge neutral";
  } else {
    heading.textContent = "Trạng thái Chưa xác định";
    statusBadge.textContent = String(status);
    statusBadge.className = "badge neutral";
  }

  // Speaker Verification Badge
  if (speaker.user_id && speaker.user_id !== "null" && speaker.user_id !== "anonymous") {
    speakerBadge.classList.remove("hidden");
    const sim = typeof speaker.similarity_score === "number" ? ` (Cosine: ${speaker.similarity_score.toFixed(3)})` : "";
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
  confidenceText.textContent = `Độ tin cậy phân loại: ${typeof assessment.confidence === "number" ? (assessment.confidence * 100).toFixed(1) + "%" : "--"}`;

  if (durationText) {
    const durSec = assessment.analyzed_duration_seconds;
    if (typeof durSec === "number" && durSec > 0) {
      const dm = Math.floor(durSec / 60);
      const ds = Math.floor(durSec % 60);
      durationText.textContent = `⏱️ Thời lượng: ${dm}:${ds.toString().padStart(2, "0")} (${durSec.toFixed(1)}s)`;
      durationText.classList.remove("hidden");
      if (fileMeta && currentFile) {
        const sizeStr = `${(currentFile.size / (1024 * 1024)).toFixed(2)} MiB`;
        fileMeta.textContent = `${currentFile.name} · ${sizeStr} · ⏱️ Thời lượng: ${dm}:${ds.toString().padStart(2, "0")} (${durSec.toFixed(1)}s)`;
      }
    } else {
      durationText.textContent = "";
      durationText.classList.add("hidden");
    }
  }

  const methodMap = {
    acoustic_safeguard_hybrid: "Điều chỉnh Âm học + Mô hình Lai (Proxy)",
    trained_hybrid_model: "Mô hình Lai Học Sâu (RAVDESS Proxy)",
    calibrated_hybrid_model: "Hiệu chuẩn Z-Score Nền Cá nhân",
    personal_baseline_rules: "Quy tắc Z-Score Baseline Cá nhân",
    insufficient_acoustic_quality: "Chất lượng Âm học Chưa Đạt",
    uncalibrated: "Chưa hiệu chuẩn",
  };
  methodText.textContent = methodMap[assessment.method] || assessment.method || "Chưa xác định";

  // Dominant Emotion
  const emoMap = {
    neu: { label: "Bình thường / Điềm đạm (Neutral)", icon: "😐" },
    ang: { label: "Giận dữ / Căng thẳng (Anger)", icon: "😠" },
    hap: { label: "Vui vẻ / Hào hứng (Happy)", icon: "😊" },
    sad: { label: "Buồn bã / Trầm uất (Sadness)", icon: "😢" },
  };
  const currentEmo = emoMap[emotion.label] || { label: emotion.label || "Chưa có", icon: "🎙️" };
  dominantEmotionLabel.textContent = currentEmo.label;
  emotionIcon.textContent = currentEmo.icon;
  let confText = `Độ tin cậy cảm xúc: ${fmt(emotion.confidence, 2)}`;
  if (emotion.adjusted_by_safeguard) {
    confText += " (🛡️ Đã chuẩn hóa âm học)";
  }
  emotionConfidenceText.textContent = confText;

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

  // XAI Explanations (Safe DOM creation via textContent, no innerHTML)
  if (explanations.length > 0) {
    xaiSection.classList.remove("hidden");
    xaiAlerts.replaceChildren();
    for (const exp of explanations) {
      const expText = String(exp);
      const alert = document.createElement("div");
      let icon = "🔍";
      let typeClass = "info";
      if (expText.includes("ổn định") || expText.includes("điều chỉnh giảm")) {
        icon = "🛡️";
        typeClass = "safeguard";
      } else if (expText.includes("vượt ngưỡng") || expText.includes("tăng vọt") || expText.includes("không đủ")) {
        icon = "⚠️";
        typeClass = "warning";
      }
      alert.className = `xai-alert ${typeClass}`;
      const iconSpan = document.createElement("span");
      iconSpan.className = "xai-icon";
      iconSpan.textContent = icon;
      const textSpan = document.createElement("span");
      textSpan.textContent = expText;
      alert.append(iconSpan, textSpan);
      xaiAlerts.appendChild(alert);
    }
  } else {
    xaiSection.classList.add("hidden");
  }

  // Acoustic Physics Cards
  function setAcousticCard(elVal, elBase, elZ, elBadge, item, unit) {
    const hasVal = item && item.value !== null && item.value !== undefined;
    elVal.textContent = hasVal ? fmt(item.value, 2) : "--";
    elBase.textContent = `Baseline: ${fmt(item?.baseline, 2)} ${unit}`;
    const z = item?.z_score;
    elZ.textContent = `Z: ${z !== null && z !== undefined ? (z > 0 ? "+" : "") + z.toFixed(2) : "--"}`;

    if (!hasVal) {
      elBadge.textContent = "Invalid/Unvoiced";
      elBadge.className = "mini-badge reduced";
    } else if (z !== null && z !== undefined) {
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

  // Breakdown Table (Safe DOM creation via textContent, no innerHTML)
  const tableDescriptions = {
    f0_mean_hz: "Tần số cơ bản (Pitch) - Phản ánh độ căng dây thanh âm",
    jitter_percent: "Rung tần số chu kỳ vi mô - Dấu hiệu run giọng khi kích động",
    shimmer_percent: "Rung biên độ chu kỳ vi mô - Áp lực khí dưới thanh môn",
    hnr_db: "Tỷ số sóng điều hòa trên nhiễu - Độ trong thanh âm vs hụt hơi",
  };

  breakdownTbody.replaceChildren();
  for (const [key, row] of Object.entries(acoustic)) {
    const tr = document.createElement("tr");
    const z = row?.z_score;
    const zText = z !== null && z !== undefined ? (z > 0 ? "+" : "") + z.toFixed(2) : "--";

    const tdKey = document.createElement("td");
    const strongKey = document.createElement("strong");
    strongKey.textContent = key;
    tdKey.appendChild(strongKey);

    const tdVal = document.createElement("td");
    tdVal.textContent = fmt(row?.value, 2);

    const tdBase = document.createElement("td");
    tdBase.textContent = fmt(row?.baseline, 2);

    const tdZ = document.createElement("td");
    tdZ.textContent = zText;

    const tdDesc = document.createElement("td");
    tdDesc.textContent = tableDescriptions[key] || "Chỉ số âm học thanh quản";

    tr.append(tdKey, tdVal, tdBase, tdZ, tdDesc);
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
  if (endpoint === "/api/v1/register-baseline" && overwriteCheckbox?.checked) {
    formData.append("overwrite", "true");
  }
  if (endpoint === "/api/v1/analyze-stress" && !uid && identifyCheckbox?.checked) {
    formData.append("identify_speaker", "true");
  }

  analyzeButton.disabled = true;
  baselineButton.disabled = true;
  setMessage("Đang truyền và phân tích âm thanh qua SpeechBrain & Acoustic Physics (tệp dài có thể mất 15-35 giây)...");

  try {
    const base = getBaseUrl();
    const response = await fetch(`${base}${endpoint}`, {
      method: "POST",
      headers: getAuthHeaders(),
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
      statusBadge.classList.remove("hidden");
      statusBadge.textContent = "BASELINE REGISTERED";
      statusBadge.className = "badge verified";
      jsonResult.textContent = JSON.stringify(result, null, 2);
    }
    setMessage(successMessage);
  } catch (err) {
    setMessage(`Lỗi: ${err.message}`, true);
  } finally {
    analyzeButton.disabled = false;
    baselineButton.disabled = false;
  }
}

// Delete Baseline
deleteBaselineButton?.addEventListener("click", async () => {
  const uid = userInput.value.trim();
  if (!uid) {
    setMessage("Nhập User ID cần xóa baseline.", true);
    userInput.focus();
    return;
  }
  try {
    const base = getBaseUrl();
    const response = await fetch(`${base}/api/v1/baseline/${encodeURIComponent(uid)}`, {
      method: "DELETE",
      headers: getAuthHeaders(),
    });
    const result = await response.json();
    if (!response.ok) {
      throw new Error(result.detail || "Không thể xóa baseline.");
    }
    setMessage(`Đã xóa baseline của '${uid}' thành công.`);
  } catch (err) {
    setMessage(`Lỗi: ${err.message}`, true);
  }
});

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

apiInput.addEventListener("change", checkHealth);
apiKeyInput?.addEventListener("change", checkHealth);

// Initial boot
checkHealth();
