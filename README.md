# Speech Stress & Emotion Analyzer

Phân tích cảm xúc giọng nói, xác minh người nói và ước lượng tín hiệu stress từ file âm thanh. Hệ thống gồm một bộ phân loại cảm xúc SpeechBrain, một mô hình stress nhị phân có thể huấn luyện trên RAVDESS, và các đặc trưng âm học so sánh với baseline cá nhân.

> Nhãn stress trong dữ liệu huấn luyện được suy ra từ nhóm cảm xúc RAVDESS. Đây là tín hiệu thực nghiệm, không phải phép đo y khoa hay đánh giá tâm lý.

## Thành phần

- `src/enhancer.py`: chuyển audio về mono/16 kHz và khử nhiễu bằng MetricGAN+.
- `src/speaker_calibrator.py`: embedding ECAPA-TDNN và so khớp người nói.
- `src/acoustic_extractor.py`: F0, jitter, shimmer và HNR.
- `src/dataset.py`: tải manifest, resample audio, chuẩn hóa đặc trưng bằng thống kê train và tạo dynamic batches.
- `src/brain_trainer.py`, `train.py`: huấn luyện, chọn checkpoint theo validation error rate và báo cáo kết quả test.
- `src/stress_model.py`: nạp checkpoint stress đã huấn luyện cho inference.
- `src/pipeline.py`, `api/main.py`: API và luồng phân tích.

## Cài đặt và chuẩn bị dữ liệu

Khuyến nghị Python 3.11:

```bash
python3.11 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

Tải RAVDESS và tạo manifest speaker-independent:

```bash
python data/download_ravdess.py
python data/prepare_manifest.py --raw_dir data/raw/RAVDESS --out_dir data/manifests
```

Manifest mặc định chỉ giữ neutral/calm và angry/fearful. File không đọc được hoặc không có voiced F0 sẽ bị bỏ qua thay vì được gán vector toàn số 0. Chia tập theo actor: 01–16 train, 17–20 validation, 21–24 test.

## Huấn luyện và đánh giá

```bash
python train.py
```

Lệnh này fine-tune head và đánh giá test bằng checkpoint tốt nhất trên validation. Checkpoint cùng thống kê chuẩn hóa và label encoder được lưu trong `results/stress_wav2vec2/1986/save/`; báo cáo test nằm trong `results/stress_wav2vec2/1986/`.

Mặc định cấu hình giữ Wav2Vec2 frozen và chạy 2 epoch để thử nhanh. Điều chỉnh `number_of_epochs`, `freeze`, `lr` và `lr_wav2vec2` trong `configs/hparams_train.yaml` trước lần huấn luyện dài.

## Chạy API

```bash
uvicorn api.main:app --host 127.0.0.1 --port 8000
```

Lần khởi động đầu có thể tải các model SpeechBrain đã khai báo trong `configs/hparams_inference.yaml`. Model emotion dùng `foreign_class` và interface `custom_interface.py` theo hướng dẫn của [model card SpeechBrain IEMOCAP](https://huggingface.co/speechbrain/emotion-recognition-wav2vec2-IEMOCAP).

Mở dashboard tĩnh trong terminal khác:

```bash
.venv/bin/python -m http.server 5500 --directory frontend
```

Sau đó truy cập `http://127.0.0.1:5500`.

- `GET /health`: trạng thái pipeline và các model.
- `POST /api/v1/register-baseline`: form fields `user_id`, `file` (giọng bình tĩnh).
- `POST /api/v1/analyze-stress`: form fields `file`, tùy chọn `user_id`.

File upload hỗ trợ WAV, FLAC và OGG, giới hạn 20 MiB và audio tối đa 30 giây. Khi chưa có stress checkpoint hoặc baseline cá nhân, API trả `UNCALIBRATED` và `stress_score: null`, không gán nhãn CALM mặc định. Khi dùng baseline, API chỉ áp dụng baseline nếu xác minh người nói thành công.

API hiện chưa có xác thực người dùng; lệnh mẫu chỉ bind vào loopback. Trước khi mở ra mạng nội bộ hoặc Internet, cần đặt API sau lớp xác thực phù hợp.

## Kiểm tra

Các script kiểm tra hiện có cho toàn bộ 5 giai đoạn:

```bash
.venv/bin/python tests/test_phase1.py   # Tiền xử lý & Trích xuất Vật lý Âm học
.venv/bin/python tests/test_phase2.py   # Phân chia Tập dữ liệu & Pipeline Nạp Batch
.venv/bin/python tests/test_phase3.py   # Huấn luyện & Đánh giá Mô hình Lai
.venv/bin/python tests/test_phase4.py   # Cá nhân hóa Baseline & Luật Sinh lý XAI
.venv/bin/python tests/test_phase5.py   # Triển khai Toàn diện: FastAPI & Web Dashboard
```

---

## 🟣 5. Huấn luyện Mô hình Lai (Giai đoạn 3)

### Chạy huấn luyện:
```bash
.venv/bin/python train.py --hparams configs/hparams_train.yaml --device cpu
```

### Kết quả Đánh giá trên Tập Test Độc lập Người nói:
- **Độ chính xác (Accuracy):** **80.36%**
- **Macro F1-Score:** **77.80%**
- **Recall (Lớp Stress):** **100.00%** (Phát hiện chuẩn xác 64/64 trường hợp căng thẳng, không bỏ sót)
- **Validation Loss:** Giảm đều từ $0.334 \rightarrow 0.242$ (Validation Accuracy đạt **94.64%**)
- **Checkpoints:** Lưu tự động tại `results/stress_wav2vec2/1986/save/` dựa trên tiêu chí lỗi thấp nhất (`min_keys=["error_rate"]`).

---

## 🟠 6. Cá nhân hóa & Hiệu chuẩn Giọng nói Nền (Giai đoạn 4)

Hệ thống cho phép đăng ký hồ sơ Baseline cá nhân hóa và áp dụng các luật giải thích sinh lý thanh quản (Explainable AI - XAI):

1. **Đăng ký Hồ sơ Nền (`.npz`)**:
   - Lưu trữ vector nhận dạng 192 chiều trích xuất từ mô hình ECAPA-TDNN (`speechbrain/spkrec-ecapa-voxceleb`).
   - Thống kê trung bình và độ lệch chuẩn ($\mu, \sigma$) cho 4 đặc trưng âm học: $F_0$, Jitter, Shimmer, HNR.
2. **Xác minh Người nói (Speaker Verification & Matching)**:
   - Sử dụng độ đo Cosine Similarity giữa vector đặc trưng với profile gốc (`threshold = 0.70` cho kiểm tra đăng nhập/check-in, `threshold = 0.50` cho giọng nói biến thiên cảm xúc).
   - Tự động nhận diện danh tính (`find_best_match`) từ cơ sở dữ liệu nhiều người dùng.
3. **Luật Giải thích Sinh lý Thanh quản (XAI Rules)**:
   - Tự động phát hiện khi các chỉ số vượt ngưỡng $Z > 1.5$ hoặc $Z < -1.5$:
     - $F_0$ tăng: Dây thanh căng cứng dưới áp lực tâm lý.
     - Jitter tăng: Dấu hiệu rung giọng vi mô (vocal microtremor).
     - Shimmer tăng: Áp lực dưới thanh môn không ổn định.
     - HNR giảm: Hụt hơi, suy giảm độ trong thanh âm.
4. **Cơ chế Bảo vệ Âm học Sinh lý (Physiological Safeguard)**:
   - Tự động nhận diện trạng thái dây thanh âm thư giãn ($F_0 \le 145\text{ Hz}$, $\text{Jitter} \le 2.2\%$, $\text{HNR} \ge 10\text{ dB}$) đối với giọng nói chưa đăng ký baseline, loại trừ triệt để tình trạng báo động giả.

---

## 🟢 7. Triển khai Hệ thống Hoàn chỉnh (Giai đoạn 5)

Hệ thống cung cấp trọn gói API Backend (FastAPI) và Giao diện Dashboard Trực quan tương tác cao:

### 1. Khởi động Dịch vụ (Chỉ 1 lệnh duy nhất):
```bash
.venv/bin/uvicorn api.main:app --host 127.0.0.1 --port 8000
```
Truy cập trực tiếp giao diện Web Dashboard tại: **`http://127.0.0.1:8000`**

### 2. Các Endpoints API Chính:
- `GET /health`: Kiểm tra trạng thái sẵn sàng của dịch vụ, pipeline và các mô hình học sâu (SpeechBrain Emotion + Wav2Vec2 Stress).
- `GET /api/v1/baselines`: Danh sách tất cả người dùng đã đăng ký giọng nền (baseline).
- `GET /api/v1/samples`: Danh mục các file âm thanh mẫu chuẩn (Calm, Angry, Fearful) để thử nghiệm nhanh.
- `GET /api/v1/sample-audio/{id}`: Truy xuất luồng âm thanh stream của file mẫu demo.
- `POST /api/v1/register-baseline`: Đăng ký giọng nền bình tĩnh cho người dùng (`user_id`, `file`).
- `POST /api/v1/analyze-stress`: Phân tích mức độ stress và cảm xúc từ file audio (`file`, tùy chọn `user_id`).

### 3. Tính năng trên Dashboard Tương tác:
- **Kéo thả & Nghe lại Âm thanh Trực tiếp**: Hỗ trợ mọi định dạng (WAV, MP3, M4A, AAC, FLAC, OGG) thời lượng lên đến 300 giây cùng Audio Player tích hợp sẵn.
- **Thử nghiệm Nhanh với Demo Samples**: Chỉ 1 click để thử ngay giọng bình tĩnh (Calm) hoặc giọng căng thẳng (Angry/Fearful).
- **Vòng tròn Đo lường Stress (Dynamic Stress Gauge)**: Hiển thị điểm số 0–100% với hiệu ứng đổi màu tương ứng mức độ (`CALM` xanh lá, `MODERATE` vàng cam, `HIGH_STRESS` đỏ).
- **Phân rã Xác suất 4 Lớp Cảm xúc**: Thanh phần trăm trực quan cho `Neutral`, `Anger`, `Happy`, `Sad`.
- **4 Thẻ Đặc trưng Âm học Thanh quản**: Đo lường $F_0$, Jitter, Shimmer, HNR kèm độ lệch chuẩn Z-score so với giọng nền.
- **Bảng Cảnh báo Giải thích Sinh lý (XAI Alert Cards)**: Diễn giải nguyên nhân thần kinh - thanh quản dẫn đến kết luận của hệ thống.
- **Sao chép JSON 1-Click**: Cho phép sao chép dữ liệu phân tích chi tiết để tích hợp vào các hệ thống khác.
