# Speech Emotion & Vocal Arousal/Stress Proxy Analyzer

Hệ thống phân tích cảm xúc giọng nói, xác minh người nói (ECAPA-TDNN) và ước lượng **chỉ báo kích hoạt âm học / stress proxy** từ file âm thanh. Hệ thống kết hợp bộ phân loại cảm xúc SpeechBrain (IEMOCAP), mô hình lai Wav2Vec2 + 5 đặc trưng vật lý thanh quản huấn luyện trên RAVDESS, và hiệu chuẩn Z-score theo baseline cá nhân.

> [!IMPORTANT]
> **Lưu ý về Construct Validity (Giá trị Khái niệm của Nhãn):**
> Nhãn `normal` / `stress` trong quá trình huấn luyện được ánh xạ từ các lớp **cảm xúc diễn xuất (acted emotions)** của tập dữ liệu RAVDESS (`neutral`, `calm` $\rightarrow$ `normal`; `angry`, `fearful` $\rightarrow$ `stress`). Do đó, điểm số đầu ra là **chỉ báo kích hoạt giọng nói thực nghiệm (Vocal Arousal / Stress Proxy)** phản ánh mức độ căng giọng/cảm xúc mạnh giống trạng thái giận dữ hoặc sợ hãi trong dữ liệu diễn xuất — **không phải phép đo stress sinh lý/tâm lý lâm sàng hay kết luận y khoa**.

---

## 1. Thành phần Kiến trúc

- [`src/acoustic_extractor.py`](src/acoustic_extractor.py): Trích xuất 5 đặc trưng vật lý thanh quản bằng Praat-Parselmouth ($F_0$ mean, $F_0$ std, Jitter local, Shimmer local, HNR) kèm cờ kiểm định chất lượng (`voiced_fraction`, `all_valid`). Các phép đo thất bại được giữ nguyên là `None` thay vì gán `0.0`.
- [`src/speaker_calibrator.py`](src/speaker_calibrator.py): Trích xuất vector định danh người nói 192 chiều (ECAPA-TDNN), xác minh người nói (`verify_speaker`) và tự động khớp danh tính (`find_best_match`).
- [`src/enhancer.py`](src/enhancer.py): Mô-đun khử nhiễu tùy chọn MetricGAN+ (`enable_enhancer: false` mặc định trong inference để tối ưu RAM khởi động; có thể bật trong [`configs/hparams_inference.yaml`](configs/hparams_inference.yaml)).
- [`src/dataset.py`](src/dataset.py): Tải manifest độc lập người nói, chuẩn hóa đặc trưng theo thống kê tập `train` và tạo dynamic batches.
- [`src/brain_trainer.py`](src/brain_trainer.py), [`train.py`](train.py): Huấn luyện mô hình lai, chọn checkpoint theo validation error rate và xuất báo cáo đánh giá trên tập test độc lập người nói.
- [`src/stress_model.py`](src/stress_model.py): Nạp checkpoint đã huấn luyện và suy luận trên toàn bộ thời lượng audio (tối đa 300 giây) bằng cơ chế chia cửa sổ 30 giây kết hợp **trung bình có trọng số theo thời lượng (duration-weighted pooling)**.
- [`src/pipeline.py`](src/pipeline.py), [`api/main.py`](api/main.py): Luồng điều phối phân tích, giải mã âm thanh theo luồng có giới hạn mẫu (streaming sample cap), kiểm soát đồng thời (concurrency semaphore), xác thực API và phục vụ Web Dashboard.

---

## 2. Cài đặt và Chuẩn bị Dữ liệu

Khuyến nghị Python 3.11:

```bash
python3.11 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

Tải tập dữ liệu RAVDESS và tạo manifest độc lập người nói (Actor-disjoint split):

```bash
.venv/bin/python data/download_ravdess.py
.venv/bin/python data/prepare_manifest.py --raw_dir data/raw/RAVDESS --out_dir data/manifests
```

- **Quy tắc ánh xạ nhãn proxy:** Mặc định giữ `neutral`/`calm` (`normal`) và `angry`/`fearful` (`stress`). Các file không đọc được hoặc không có đoạn hữu thanh ($F_0$) hợp lệ sẽ bị loại bỏ.
- **Phân chia độc lập người nói:** Actor `01–16` cho `train` (448 mẫu), Actor `17–20` cho `valid` (112 mẫu), Actor `21–24` cho `test` (112 mẫu).

---

## 3. Huấn luyện và Đánh giá Mô hình Proxy (Giai đoạn 3)

```bash
.venv/bin/python train.py --hparams configs/hparams_train.yaml --device cpu
```

- Checkpoint tốt nhất theo `error_rate` trên tập validation cùng thống kê chuẩn hóa (`acoustic_normalization.npz`) và `label_encoder.txt` được lưu tại `results/stress_wav2vec2/1986/save/`.
- **Kết quả thực nghiệm trên tập Test độc lập người nói (RAVDESS Acted Emotion Proxy, 112 mẫu):**
  - **Accuracy:** `87.50%`
  - **Macro F1-Score:** `86.67%`
  - **Normal Recall:** `72.92%` (35/48 mẫu)
  - **Recall (nhóm nhãn `stress` proxy):** `98.44%` (63/64 mẫu `angry`/`fearful`)
  - **Confusion Matrix (`[normal, stress]`):** `[[35, 13], [1, 63]]`
  - *Lưu ý:* Các chỉ số này đánh giá khả năng phân biệt nhóm cảm xúc diễn xuất trong RAVDESS giữa các diễn viên chưa từng xuất hiện trong tập train; không đại diện cho độ chính xác đo stress ngoài đời thực.

---

## 4. Cá nhân hóa Baseline & Kiểm định Chất lượng Âm học (Giai đoạn 4)

1. **Đăng ký Hồ sơ Nền (`.npz`) có Kiểm định Chất lượng & Chống Ghi đè**:
   - Yêu cầu đoạn thu âm bình tĩnh tối thiểu 1 giây có tỷ lệ khung hữu thanh `voiced_fraction >= 0.20` và đo được hợp lệ cả 4 chỉ số ($F_0$, Jitter, Shimmer, HNR).
   - Mặc định chặn ghi đè (`overwrite=False` trả lỗi `FileExistsError` / HTTP `409 Conflict`) trừ khi người dùng chủ động bật cờ `overwrite=True`.
2. **Ngưỡng Xác minh và Nhận diện Người nói (ECAPA-TDNN Cosine Similarity)**:
   - `speaker_similarity_threshold: 0.50`: Dùng khi xác minh một `user_id` đã khai báo trong lúc phân tích cảm xúc/căng thẳng (do giọng nói khi kích động mạnh có biến thiên thanh quản làm giảm Cosine xuống vùng `0.55–0.66`, trong khi người lạ nằm dưới `0.45`).
   - `speaker_identification_threshold: 0.70`: Ngưỡng nghiêm ngặt dùng cho kiểm tra giọng bình tĩnh (check-in) và tính năng tự động tìm người nói (`identify_speaker=True` qua `find_best_match`).
3. **Cổng Chất lượng Âm học (Acoustic Quality Gate)**:
   - Nếu Praat không trích xuất được $F_0$, Jitter, Shimmer hoặc HNR trên đoạn thu, hệ thống giữ giá trị `null` (`None`), trả trạng thái `INSUFFICIENT_AUDIO` và **tuyệt đối không sử dụng giá trị lỗi `0.0` để kết luận `CALM`**.

---

## 5. Chạy API & Web Dashboard (Giai đoạn 5)

### Khởi động dịch vụ (Mặc định chỉ mở trên Loopback):

```bash
.venv/bin/uvicorn api.main:app --host 127.0.0.1 --port 8000
```

Truy cập trực tiếp Web Dashboard tại: **`http://127.0.0.1:8000`**

### Bảo mật & Giới hạn Tài nguyên khi Triển khai:
- **Xác thực (`API_KEY`)**:
  - Khi chạy cục bộ không đặt biến môi trường `API_KEY`, API chỉ cho phép kết nối từ loopback (`127.0.0.1`, `::1`, `localhost`) và từ chối mọi IP mạng ngoài với mã `403 Forbidden`.
  - Khi đặt `export API_KEY="your-secret-key"`, mọi request tới `/api/v1/*` bắt buộc gửi header `X-API-Key: your-secret-key` (hoặc `Authorization: Bearer your-secret-key`), nếu thiếu hoặc sai sẽ trả về `401 Unauthorized`.
- **CORS Allow-list**: Cấu hình qua biến môi trường `ALLOWED_ORIGINS` (mặc định chỉ cho phép `http://127.0.0.1:8000`, `http://localhost:8000`, `http://127.0.0.1:5500`, `http://localhost:5500`).
- **Giới hạn Sớm & Giải mã Theo Luồng**:
  - Middleware ASGI kiểm tra `Content-Length` và đếm byte luồng nhận trước khi multipart parser hoạt động (chặn ngay ở ngưỡng **20 MiB** với HTTP `413`).
  - Hỗ trợ các định dạng **WAV, MP3, M4A, AAC, FLAC, OGG, WebM** với thời lượng từ **0.5 giây đến tối đa 300 giây**. Quá trình giải mã kiểm tra metadata và đếm số mẫu theo từng frame để dừng ngay lập tức nếu vượt quá 300 giây.
  - Giới hạn số tác vụ inference đồng thời (`max_concurrent_inferences: 2`) và timeout (`inference_timeout_seconds: 120.0`).

### Danh sách Endpoints:
- `GET /health/live`: Kiểm tra tiến trình server còn sống (`{"status": "alive"}`).
- `GET /health` (hoặc `GET /health/ready`): Kiểm tra trạng thái sẵn sàng của pipeline và các model (`healthy` khi đủ cả emotion + stress model, `degraded` khi thiếu model).
- `POST /api/v1/register-baseline`: Form fields `user_id`, `file`, tùy chọn `overwrite=false|true`.
- `GET /api/v1/baseline/{user_id}`: Kiểm tra trạng thái đăng ký baseline của một `user_id` cụ thể (không liệt kê danh sách ID toàn hệ thống).
- `DELETE /api/v1/baseline/{user_id}`: Xóa hồ sơ baseline của `user_id`.
- `POST /api/v1/analyze-stress`: Form fields `file`, tùy chọn `user_id`, tùy chọn `identify_speaker=false|true`.
- `GET /api/v1/samples` & `GET /api/v1/sample-audio/{sample_id}`: Lấy danh sách và audio mẫu thử nghiệm.

---

## 6. Kiểm thử Tự động

```bash
.venv/bin/python tests/test_phase1.py   # Tiền xử lý & Trích xuất Vật lý Âm học
.venv/bin/python tests/test_phase2.py   # Phân chia Tập dữ liệu & Pipeline Nạp Batch
.venv/bin/python tests/test_phase3.py   # Huấn luyện & Đánh giá Mô hình Lai
.venv/bin/python tests/test_phase4.py   # Cá nhân hóa Baseline, Overwrite Guard & Luật XAI
.venv/bin/python tests/test_phase5.py   # Bảo mật API, Quality Gate, Chunking >180s & E2E
```
