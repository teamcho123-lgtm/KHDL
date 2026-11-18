# Quy Trình Phát Triển Hệ Thống Phân Tích Stress Giọng Nói Với AI

Tài liệu này dành cho người phát triển khi cộng tác với trợ lý AI (Codex / Antigravity) để nghiên cứu, cải tiến mô hình học sâu, phát triển tính năng API/Frontend và kiểm thử hệ thống phân tích stress giọng nói trên nền tảng Ubuntu Linux.

Hướng dẫn trực tiếp dành cho Codex/Agent được lưu tại [`AGENTS.md`](./AGENTS.md). Quy trình kiểm thử toàn diện 5 tầng xem tại [`QUY-TRINH-DEV-WORKFLOW.md`](./QUY-TRINH-DEV-WORKFLOW.md).

---

## 1. Nguyên Tắc Làm Việc Cốt Lõi

1. **Một mục tiêu rõ ràng**: Mỗi phiên làm việc chỉ giao một nhiệm vụ cụ thể (ví dụ: sửa lỗi đọc audio VBR, tinh chỉnh ngưỡng F0 của Safeguard, huấn luyện lại mô hình 15 epochs).
2. **Khảo sát trước khi sửa**: Luôn yêu cầu AI đọc mã nguồn hiện tại, cấu hình YAML và log huấn luyện trước khi can thiệp code.
3. **Thay đổi từng bước nhỏ**: Dễ dàng kiểm soát độ chênh lệch (diff), tránh làm gãy các bộ kiểm thử tự động sẵn có.
4. **Đề xuất trước, triển khai sau**: Với tính năng mới hoặc can thiệp thuật toán ML, AI cần phân tích hiện trạng và đưa ra phương án để thống nhất trước khi code.
5. **Đánh giá bằng số liệu định lượng**: Không dựa vào cảm tính. Mọi cải tiến mô hình phải được đo bằng: **Accuracy**, **Macro F1**, **Normal Recall**, **Stress Recall** và **Confusion Matrix**.
6. **Bảo vệ dữ liệu & quyền riêng tư**: Không đưa file audio thật chứa thông tin nhạy cảm cá nhân, mã token Hugging Face lên Git.

---

## 2. Chuẩn Bị Trước Khi Giao Việc

Mở terminal trong thư mục dự án trên Ubuntu:

```bash
cd ~/Downloads/KHDL
git status --short
```

Kiểm tra trạng thái Git để biết các tệp đang thay đổi. Nếu có công việc dang dở, thông báo cho AI để tránh sửa đè lên.

### Cấu Trúc Thành Phần Dự Án:

| Thành phần | Đường dẫn | Vai trò |
| :--- | :--- | :--- |
| **Pipeline Trung Tâm** | `src/pipeline.py` | Điều phối trích xuất, chuẩn hóa, phân loại và giải thích sinh lý |
| **Mô hình Lai (DL)** | `src/stress_model.py` | Kết hợp Wav2Vec2 + 5 đặc trưng vật lý âm học |
| **Vật lý Âm học** | `src/acoustic_extractor.py` | Trích xuất $F_0$, Jitter, Shimmer, HNR qua Parselmouth |
| **Người nói & Baseline**| `src/speaker_calibrator.py` | Mô hình ECAPA-TDNN xác minh người nói và lưu profile cá nhân |
| **Khử Nhiễu** | `src/enhancer.py` | Tiền xử lý MetricGAN+ nâng cao chất lượng giọng nói |
| **API Backend** | `api/main.py` | FastAPI Server, xử lý upload, giới hạn 20MB/300s, API endpoints |
| **Web Dashboard** | `frontend/` | Giao diện HTML5/JS/CSS với đồng hồ đo stress, biểu đồ 4 cảm xúc |
| **Cấu hình** | `configs/` | `hparams_train.yaml` và `hparams_inference.yaml` |
| **Bộ Kiểm Thử** | `tests/` | 5 script kiểm thử từ Giai đoạn 1 đến Giai đoạn 5 |
| **Agent Skill** | `.agents/skills/speech-stress-workflow/` | Skill chuẩn hóa quy trình sửa lỗi và huấn luyện lại |

---

## 3. Cách Viết Yêu Cầu Cho AI

Một yêu cầu hiệu quả gồm 5 yếu tố: **Mục tiêu**, **Hiện trạng/Lỗi**, **Kết quả mong muốn**, **Phạm vi tác động**, và **Tiêu chí nghiệm thu**.

### Mẫu Yêu Cầu Mới (Chế độ Khảo sát & Đề xuất):

```text
Tôi muốn: [Mục tiêu, ví dụ: Khắc phục tình trạng mô hình dự đoán nhầm giọng bình thường thành stress].
Hiện trạng: [Mô tả hiện tượng, ví dụ: Normal Recall chỉ đạt 54.2%, 22/48 mẫu normal bị đoán sai].
Mong muốn: [Kết quả cụ thể, ví dụ: Tăng Normal Recall >= 70% mà không làm giảm Stress Recall].
Phạm vi: [configs/hparams_train.yaml, src/brain_trainer.py, src/pipeline.py].
Giới hạn: [Giữ nguyên seed=1986, không làm tăng thời gian suy luận trên CPU].
Chế độ: Đề xuất phương án trước, chưa sửa code.
Tiêu chí hoàn thành:
- Giải thích nguyên nhân gốc rễ (mất cân bằng mẫu, thiếu pooling...).
- Đề xuất thay đổi cấu hình hoặc thuật toán.
- Nêu rõ lệnh kiểm thử sẽ chạy để chứng minh cải thiện.
```

### Mẫu Giao Triển Khai Sau Khi Thống Nhất:

```text
Triển khai phương án vừa đề xuất cho [tính năng / sửa lỗi ...].
Phạm vi đã đồng ý: [Nêu các file cụ thể].
Thực hiện trọn vẹn, chạy kiểm thử tự động (.venv/bin/python tests/test_inference_sanity.py) và báo cáo kết quả thực tế.
```

---

## 4. Kiểm Tra Kết Quả Trên Ubuntu

### 1. Khởi động Web Dashboard & Backend:
```bash
.venv/bin/uvicorn api.main:app --host 127.0.0.1 --port 8000 --reload
```
- Mở trình duyệt vào Dashboard: **`http://127.0.0.1:8000`**
- Mở trang Swagger Docs: **`http://127.0.0.1:8000/docs`**

### 2. Kiểm tra theo từng cấp độ:

| Phạm vi cần kiểm tra | Lệnh thực thi | Mục tiêu xác minh |
| :--- | :--- | :--- |
| **Độ nhạy & Không thiên vị** | `.venv/bin/python tests/test_inference_sanity.py` | Normal Recall $\ge 70\%$, Stress Recall $\ge 75\%$ |
| **E2E Backend & Dashboard** | `.venv/bin/python tests/test_phase5.py` | API, upload an toàn, static frontend, chunking $>180$s |
| **Baseline & Nhận diện người nói** | `.venv/bin/python tests/test_phase4.py` | Cosine Similarity cùng người $\ge 0.70$, khác người bị từ chối |
| **Mô hình & Đánh giá Test Set** | `.venv/bin/python tests/test_phase3.py` | Accuracy $\ge 85\%$, Macro F1 $\ge 85\%$, checkpoint hợp lệ |
| **DataIO & Không rò rỉ dữ liệu** | `.venv/bin/python tests/test_phase2.py` | Zero speaker leakage giữa train/valid/test |
| **Vật lý Âm học & ECAPA-TDNN** | `.venv/bin/python tests/test_phase1.py` | Đo đúng $F_0$, Jitter, Shimmer, HNR và vector 192-dim |

---

## 5. Cách Gửi Lỗi Để AI Xử Lý Nhanh

Khi gặp lỗi (ví dụ audio bị tính sai thời lượng, lỗi shape tensor, lỗi HTTP 400/500), cung cấp cho AI:
1. Lệnh bạn vừa chạy hoặc thao tác trên UI.
2. Thông số file audio thử nghiệm (định dạng, thời lượng thực, nguồn thu âm).
3. Toàn bộ log lỗi / stack trace từ terminal.

**Mẫu prompt báo lỗi:**
```text
Tôi đang chạy server trên Ubuntu và upload file âm thanh 2 phút 55 giây.
Hiện tượng lỗi: [Mô tả chi tiết, ví dụ: Audio player hiển thị thời lượng thành 50:00].
Log terminal API:
[Dán đoạn log traceback vào đây]

Hãy đọc mã nguồn xử lý audio trong src/pipeline.py và frontend/app.js, xác định nguyên nhân và đề xuất bản vá.
```

---

## 6. Xem Thay Đổi và Quản Lý Git

Trước khi commit, luôn xem lại toàn bộ thay đổi:

```bash
git status --short
git diff --stat
git diff
```

Thực hiện commit các file đã kiểm tra đạt chuẩn:

```bash
git add <cac_file_da_kiem_tra>
git commit -m "feat/fix: mô tả súc tích nội dung cải tiến"
```

---

## 7. Checklist Nghiệm Thu Trước Khi Kết Thúc Phiên

- [ ] AI đã đọc code và làm đúng phạm vi yêu cầu.
- [ ] Không có file tạm, file test thừa (`test_*.npz`, `__pycache__`) sót lại trong repo.
- [ ] Các script kiểm thử liên quan đã chạy và đạt $100\%$ PASS.
- [ ] Đã thử nghiệm trực tiếp trên Web Dashboard (`http://127.0.0.1:8000`).
- [ ] `git diff` sạch sẽ, không có secret / token / absolute path ngoài ý muốn.
- [ ] Báo cáo bàn giao rõ ràng kèm hướng dẫn chạy thử.
