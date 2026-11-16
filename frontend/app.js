const form = document.querySelector("#analysis-form");
const fileInput = document.querySelector("#audio-file");
const userInput = document.querySelector("#user-id");
const apiInput = document.querySelector("#api-url");
const message = document.querySelector("#message");
const analyzeButton = document.querySelector("#analyze-button");
const baselineButton = document.querySelector("#baseline-button");
const health = document.querySelector("#health");
const heading = document.querySelector("#result-heading");
const summary = document.querySelector("#summary");
const details = document.querySelector("#details");
const badge = document.querySelector("#status-badge");
const jsonResult = document.querySelector("#json-result");

function setMessage(text, isError = false) {
  message.textContent = text;
  message.classList.toggle("error", isError);
}

function metric(label, value) {
  const item = document.createElement("div");
  item.className = "metric";
  const caption = document.createElement("span");
  caption.textContent = label;
  const content = document.createElement("strong");
  content.textContent = value;
  item.append(caption, content);
  return item;
}

function formatValue(value, suffix = "") {
  if (value === null || value === undefined || Number.isNaN(value)) return "Chưa có";
  return typeof value === "number" ? `${value.toFixed(1)}${suffix}` : String(value);
}

function renderResult(data) {
  const assessment = data.assessment || {};
  const emotion = data.emotion;
  const speaker = data.speaker || {};
  heading.textContent = assessment.status || "Kết quả phân tích";
  badge.textContent = assessment.status || "Hoàn tất";
  badge.className = "badge";
  if (assessment.status === "UNCALIBRATED") badge.classList.add("warning");
  if (assessment.status === "MODERATE_STRESS") badge.classList.add("warning");
  if (assessment.status === "HIGH_STRESS") badge.classList.add("danger");

  summary.className = "summary";
  summary.replaceChildren();
  const grid = document.createElement("div");
  grid.className = "metric-grid";
  grid.append(
    metric("Điểm stress", formatValue(assessment.stress_score, "%")),
    metric("Cảm xúc", emotion?.label || "Chưa có"),
    metric("Độ tin cậy", formatValue(assessment.confidence)),
  );
  summary.append(grid);

  details.replaceChildren();
  const meta = document.createElement("div");
  meta.className = "metric-grid";
  meta.append(
    metric("Phương pháp", assessment.method || "Chưa có"),
    metric("Người nói", speaker.verified === true ? "Đã xác minh" : speaker.verified === false ? "Không khớp" : "Chưa kiểm tra"),
    metric("Emotion confidence", formatValue(emotion?.confidence)),
  );
  details.append(meta);

  const explanations = Array.isArray(data.explanations) ? data.explanations : [];
  if (explanations.length) {
    const title = document.createElement("h3");
    title.textContent = "Giải thích";
    const list = document.createElement("ul");
    for (const text of explanations) {
      const item = document.createElement("li");
      item.textContent = text;
      list.append(item);
    }
    details.append(title, list);
  }

  const breakdown = data.acoustic_breakdown || {};
  const entries = Object.entries(breakdown);
  if (entries.length) {
    const title = document.createElement("h3");
    title.textContent = "Đặc trưng âm học";
    const table = document.createElement("table");
    table.className = "breakdown";
    const thead = document.createElement("thead");
    thead.innerHTML = "<tr><th>Đặc trưng</th><th>Giá trị</th><th>Baseline</th><th>Z-score</th></tr>";
    const tbody = document.createElement("tbody");
    for (const [name, row] of entries) {
      const tr = document.createElement("tr");
      for (const value of [name, formatValue(row.value), formatValue(row.baseline), formatValue(row.z_score)]) {
        const td = document.createElement("td");
        td.textContent = value;
        tr.append(td);
      }
      tbody.append(tr);
    }
    table.append(thead, tbody);
    details.append(title, table);
  }

  jsonResult.textContent = JSON.stringify(data, null, 2);
}

async function checkHealth() {
  const base = apiInput.value.trim().replace(/\/+$/, "");
  try {
    const response = await fetch(`${base}/health`);
    if (!response.ok) throw new Error("API chưa sẵn sàng");
    const result = await response.json();
    health.className = result.pipeline_ready ? "health ok" : "health error";
    health.lastElementChild.textContent = result.pipeline_ready
      ? `API sẵn sàng${result.stress_model_ready ? " · stress model" : " · cần baseline"}`
      : "API chưa sẵn sàng";
  } catch {
    health.className = "health error";
    health.lastElementChild.textContent = "Không kết nối được API";
  }
}

async function sendAudio(endpoint, successMessage, render = true) {
  const file = fileInput.files?.[0];
  if (!file) {
    setMessage("Hãy chọn file audio trước.", true);
    return;
  }
  const formData = new FormData();
  formData.append("file", file);
  const userId = userInput.value.trim();
  if (userId) formData.append("user_id", userId);

  analyzeButton.disabled = true;
  baselineButton.disabled = true;
  setMessage("Đang xử lý audio…");
  try {
    const base = apiInput.value.trim().replace(/\/+$/, "");
    const response = await fetch(`${base}${endpoint}`, { method: "POST", body: formData });
    const result = await response.json();
    if (!response.ok) throw new Error(result.detail || "Yêu cầu không thành công");
    if (render) renderResult(result);
    else {
      heading.textContent = "Baseline đã đăng ký";
      summary.className = "summary";
      summary.textContent = `Đã lưu baseline cho ${result.result.user_id}.`;
      jsonResult.textContent = JSON.stringify(result, null, 2);
    }
    setMessage(successMessage);
  } catch (error) {
    setMessage(error.message || "Không thể xử lý audio.", true);
  } finally {
    analyzeButton.disabled = false;
    baselineButton.disabled = false;
  }
}

form.addEventListener("submit", (event) => {
  event.preventDefault();
  sendAudio("/api/v1/analyze-stress", "Phân tích hoàn tất.");
});

baselineButton.addEventListener("click", () => {
  if (!userInput.value.trim()) {
    setMessage("Nhập mã người dùng để đăng ký baseline.", true);
    userInput.focus();
    return;
  }
  sendAudio("/api/v1/register-baseline", "Baseline đã được lưu.", false);
});

fileInput.addEventListener("change", () => {
  const file = fileInput.files?.[0];
  document.querySelector("#file-meta").textContent = file
    ? `${file.name} · ${(file.size / (1024 * 1024)).toFixed(2)} MiB`
    : "Chọn WAV, FLAC hoặc OGG tối đa 20 MiB, dài tối đa 30 giây.";
});

apiInput.addEventListener("change", checkHealth);
checkHealth();
