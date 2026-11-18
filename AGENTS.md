# Hướng Dẫn Codex & AI Agent cho Hệ Thống Phân Tích Cảm Xúc & Stress Giọng Nói (KHDL)

Áp dụng cho toàn bộ dự án. Ưu tiên yêu cầu hiện tại của người dùng; nếu có hướng dẫn riêng trong thư mục con hoặc skill chuyên biệt, đọc hướng dẫn đó khi làm việc tại phần tương ứng.

---

## 1. Giao tiếp và Nguyên tắc Thực Hiện

- **Ngôn ngữ & Môi trường**: Trao đổi bằng tiếng Việt, ngắn gọn, súc tích, logic. Hướng dẫn dòng lệnh theo chuẩn **Ubuntu / Linux Bash**, luôn sử dụng dấu gạch chéo `/` trong đường dẫn, kích hoạt môi trường ảo `.venv/bin/python` và `.venv/bin/uvicorn`.
- **Chế độ đề xuất trước**: Với yêu cầu thêm tính năng mới, thay đổi kiến trúc mô hình học sâu hoặc can thiệp pipeline âm học chưa có chỉ dẫn rõ ràng, mặc định đọc hiện trạng và đưa đề xuất trước: phạm vi, mô hình AI/đặc trưng âm học bị tác động, rủi ro data leakage, tiêu chí nghiệm thu và cách kiểm tra. Không tự ý sửa code chức năng ở bước đề xuất.
- **Thực thi dứt điểm**: Khi người dùng đã yêu cầu triển khai, sửa ngay hoặc đồng ý phương án, thực hiện đến kết quả có thể kiểm tra được trong phạm vi đó; không hỏi lại các bước đã được cho phép. Yêu cầu review chỉ tạo báo cáo phân tích. Yêu cầu hoàn thiện tài liệu cho phép cập nhật tài liệu tương ứng.
- **Tự quyết định chi tiết kỹ thuật**: Tự quyết định các chi tiết nhỏ trong phạm vi được giao. Chỉ hỏi khi thiếu dữ liệu làm thay đổi bản chất thuật toán (ví dụ: thay đổi nhãn dataset, thay đổi kiến trúc trích xuất âm học, can thiệp trọng số mô hình) hoặc cần quyền thực hiện thao tác ngoài phạm vi.
- **Báo cáo trung thực & minh bạch**: Báo ngắn gọn mục tiêu trước khi làm; khi bàn giao, nêu rõ thay đổi, kiểm tra đã chạy, cách thử nghiệm và giới hạn còn lại.
- **Không ảo giác (Zero Hallucination)**: Tuyệt đối không khẳng định mô hình đã hội tụ, pipeline đã chạy đúng hoặc tính năng đã chuẩn khi chưa chạy lệnh kiểm thử thực tế. Phân biệt rõ giữa việc đọc code, biên dịch cú pháp, chạy script kiểm thử tự động và thử nghiệm trực tiếp trên giao diện Dashboard.

---

## 2. Tài Liệu và Agent Skills

- **Agent Skill chính**: [`.agents/skills/speech-stress-workflow/SKILL.md`](file:///.agents/skills/speech-stress-workflow/SKILL.md) — Quy trình toàn diện về sửa lỗi, tối ưu hóa, chuẩn hóa Masked Mean Pooling, mở rộng Acoustic Safeguard, huấn luyện lại (SpeechBrain, Wav2Vec2) và kiểm thử hồi quy.
- **Quy trình làm việc với AI**: [`QUY-TRINH-LAM-VIEC-VOI-AI.md`](file:///QUY-TRINH-LAM-VIEC-VOI-AI.md) — Tài liệu dành cho người dùng hướng dẫn giao việc, chạy thử, xử lý lỗi và bàn giao.
- **Quy trình kiểm thử chất lượng E2E**: [`QUY-TRINH-DEV-WORKFLOW.md`](file:///QUY-TRINH-DEV-WORKFLOW.md) — Quy trình kiểm thử 5 tầng từ cú pháp Python đến API Swagger và Web Dashboard.
- **Báo cáo kỹ thuật & Đánh giá**: [`PROJECT_REVIEW.md`](file:///PROJECT_REVIEW.md) và [`README.md`](file:///README.md) — Đối chiếu hiện trạng hệ thống trước khi sửa đổi.
- **Tự động áp dụng Skill**: Tự chọn skill liên quan cho mỗi yêu cầu, không bắt người dùng nhắc lại tên skill. Không tự ý tạo skill mới cho mỗi prompt trừ khi có yêu cầu đóng gói quy trình tái sử dụng rõ ràng.

---

## 3. Bản Đồ Dự Án (Project Architecture Map)

| Đường dẫn | Vai trò | Công nghệ chính |
| :--- | :--- | :--- |
| `src/pipeline.py` | Pipeline trung tâm điều phối phân tích âm thanh, baseline, safeguard và giải thích XAI | Python 3.11, NumPy, Torch, SoundFile, PyAV |
| `src/stress_model.py` | Mô hình lai (Hybrid Classifier): Wav2Vec2 + 5D Acoustic Features kèm Duration-Weighted Pooling | PyTorch, SpeechBrain, HuggingFace Transformers |
| `src/acoustic_extractor.py` | Trích xuất 5 chỉ số vật lý thanh môn ($F_0$, $F_0\text{ std}$, Jitter, Shimmer, HNR) | Praat Parselmouth, NumPy |
| `src/speaker_calibrator.py` | Nhận dạng người nói, trích xuất 192-dim vector và xác minh Cosine Similarity | SpeechBrain ECAPA-TDNN (`spkrec-ecapa-voxceleb`) |
| `src/enhancer.py` | Tiền xử lý khử nhiễu giọng nói thích ứng | SpeechBrain MetricGAN+ (`metricgan-plus-voicebank`) |
| `src/brain_trainer.py` | Lớp huấn luyện SpeechBrain Brain tùy biến với Weighted Loss & Masked Pooling | SpeechBrain, PyTorch NLLLoss |
| `src/dataset.py` | Nạp dữ liệu DynamicItemDataset và DynamicBatchSampler | SpeechBrain DataIO, TorchAudio |
| `api/main.py` | REST API Backend, xác thực, giới hạn dung lượng, phục vụ Dashboard tĩnh | FastAPI, Uvicorn, Starlette |
| `frontend/` | Web Dashboard trực quan: Vòng tròn Stress Gauge, 4 Cảm xúc, Thẻ Âm học, XAI Cards | HTML5, CSS3, JavaScript ES6+, Web Audio API |
| `configs/` | Cấu hình huấn luyện (`hparams_train.yaml`) và suy luận (`hparams_inference.yaml`) | HyperPyYAML |
| `data/` | Bộ dữ liệu RAVDESS, script tạo manifest, hồ sơ baseline cá nhân | Zenodo RAVDESS, JSON Manifests, NPZ Profiles |
| `tests/` | Bộ kiểm thử tự động 5 giai đoạn (`test_phase1.py` $\rightarrow$ `test_phase5.py`, `test_inference_sanity.py`) | Python unittest/pytest, FastAPI TestClient |
| `train.py` | Entrypoint huấn luyện và đánh giá mô hình học sâu | SpeechBrain CLI |

---

## 4. Quy Trình 5 Bước Thay Đổi & Tối Ưu Code Bắt Buộc

Áp dụng cho mọi yêu cầu tạo mới hoặc chỉnh sửa mã nguồn:

1. **Khảo sát hiện trạng (Read & Scope)**:
   - Kiểm tra `git status --short` để giữ nguyên các công việc chưa commit của người dùng.
   - Đọc kỹ mã nguồn liên quan (Pipeline, Model, API, Frontend, Tests, YAML configs).
   - Nếu là yêu cầu mới chưa rõ ràng, dừng ở bước khảo sát và đề xuất. Khi người dùng đã yêu cầu triển khai hoặc sửa ngay, chuyển sang bước 2.

2. **Viết code & Tối ưu hóa (Code & Optimize)**:
   - Giữ vững tính tương thích của kiến trúc (Python 3.11, SpeechBrain, FastAPI).
   - **Tối ưu âm học & tensor**: Tránh cấp phát tensor thừa trên CPU/GPU, sử dụng `torch.inference_mode()`, đồng bộ Masked Mean Pooling, bảo đảm chia tỷ lệ chuẩn hóa đúng (ví dụ: chuyển đổi số nguyên PCM về $[-1.0, 1.0]$).
   - **Bảo toàn cơ chế an toàn**: Duy trì cơ chế tự động dọn dẹp tệp tạm thời qua khối `try...finally`, ghi tệp nguyên tử bằng `os.replace` để chống hỏng file, và giữ vững cơ chế bảo vệ âm học (*Acoustic Safeguard* $F_0 \in [75, 320]\text{ Hz}$, $F_0\text{ std} \le 35\text{ Hz}$).
   - **Bổ sung trọn vẹn**: Xử lý đầy đủ 4 trạng thái giao diện (`Loading`, `Empty`, `Error`, `Success`), validate chặt chẽ ở API bằng Pydantic/Regex. Tuyệt đối không để lại code dở dang hoặc placeholder rỗng.

3. **Kiểm thử tự động bắt buộc (Test & Verify)**:
   - Chạy lệnh kiểm thử tương ứng với phần bị sửa đổi:
     - Cú pháp mã nguồn: `.venv/bin/python -m py_compile src/*.py api/*.py tests/*.py`
     - Độ nhạy & Ma trận nhầm lẫn: `.venv/bin/python tests/test_inference_sanity.py`
     - E2E Backend & Dashboard: `.venv/bin/python tests/test_phase5.py`
     - Baseline & Xác minh người nói: `.venv/bin/python tests/test_phase4.py`
     - Trích xuất âm học & Huấn luyện: `.venv/bin/python tests/test_phase1.py` đến `test_phase3.py`
   - Tuyệt đối không khẳng định hoàn thành khi chưa chạy kiểm thử thực tế.

4. **Tự sửa lỗi đến khi đạt (Self-Correction Loop)**:
   - Nếu lệnh build/test phát sinh lỗi (AssertionError, CUDA OOM, PyTorch shape mismatch, HTTP 4xx/5xx):
     - Đọc kỹ log lỗi và stack trace, xác định nguyên nhân gốc rễ.
     - Tự động sửa lỗi ngay lập tức, sau đó chạy lại lệnh test.
     - Lặp lại cho đến khi toàn bộ kiểm tra vượt qua (100% PASS) trước khi bàn giao.

5. **Đối chiếu Git & Bàn giao minh bạch (Diff & Handoff)**:
   - Kiểm tra `git status --short` và `git diff` để đảm bảo sạch sẽ, đúng phạm vi, không lọt tệp rác hay dữ liệu nhạy cảm.
   - Bàn giao ngắn gọn theo 4 mục: (1) Files đã thay đổi/tạo mới, (2) Kết quả kiểm thử thực tế kèm lệnh đã chạy, (3) Hướng dẫn chạy thử tính năng, (4) Giới hạn hoặc lưu ý còn lại (nếu có).

---

## 5. Lệnh Chạy và Kiểm Tra Chuẩn

Chạy từ thư mục gốc của dự án bằng môi trường ảo `.venv`:

| Mục đích | Lệnh thực thi |
| :--- | :--- |
| **Khởi động Server Backend + Web Dashboard** | `.venv/bin/uvicorn api.main:app --host 127.0.0.1 --port 8000 --reload` |
| **Kiểm thử Sanity & Độ lệch mô hình** | `.venv/bin/python tests/test_inference_sanity.py` |
| **Kiểm thử E2E Giai đoạn 5 (FastAPI & Dashboard)** | `.venv/bin/python tests/test_phase5.py` |
| **Kiểm thử Giai đoạn 4 (Xác minh người nói & XAI)** | `.venv/bin/python tests/test_phase4.py` |
| **Kiểm thử Giai đoạn 3 (Checkpoint & Metrics Test Set)** | `.venv/bin/python tests/test_phase3.py` |
| **Kiểm thử Giai đoạn 2 (Manifest & Zero Leakage)** | `.venv/bin/python tests/test_phase2.py` |
| **Kiểm thử Giai đoạn 1 (Parselmouth & ECAPA-TDNN)** | `.venv/bin/python tests/test_phase1.py` |
| **Chạy toàn bộ Pytest** | `.venv/bin/python -m pytest tests/ -v` |
| **Tái tạo Manifest dữ liệu** | `.venv/bin/python data/prepare_manifest.py --raw_dir data/raw/RAVDESS --out_dir data/manifests` |
| **Huấn luyện lại mô hình (Retrain)** | `.venv/bin/python train.py --hparams configs/hparams_train.yaml` |

- Dịch vụ FastAPI và Web Dashboard chạy tại cổng `8000`:
  - Web Dashboard: `http://127.0.0.1:8000`
  - Swagger API Docs: `http://127.0.0.1:8000/docs`
- Kiểm tra tiến trình cổng 8000 trước khi chạy lại nếu xảy ra xung đột `Address already in use`.

---

## 6. Dữ Liệu, Bảo Mật và Git

- **Bảo mật sinh trắc học giọng nói**: Không in, log hoặc commit các tệp vector embedding giọng nói (`.npz`) chứa thông tin sinh trắc học người dùng thực. Các tệp profile thử nghiệm phải gắn tiền tố `test_` và được dọn dẹp sạch sau khi test.
- **Không rò rỉ bí mật**: Tuyệt đối không commit tệp `.env`, mã thông báo Hugging Face Token (`HF_TOKEN`) hoặc khóa API vào Git. Luôn kiểm tra `.gitignore`.
- **An toàn tệp âm thanh**: Luôn giới hạn dung lượng tải lên tối đa $20\text{ MB}$, giới hạn thời lượng $300\text{ giây}$, kiểm tra định dạng magic bytes và sử dụng đường dẫn tệp tạm thời ngẫu nhiên an toàn qua `tempfile.NamedTemporaryFile`.
- **Kiểm soát đường dẫn (Path Traversal Guard)**: Mọi định danh `user_id` bắt buộc phải được kiểm tra qua biểu thức chính quy `^[A-Za-z0-9][A-Za-z0-9_-]{0,63}$` trước khi truy cập ổ cứng.
- **An toàn Git**: Chỉ stage file thuộc công việc đã kiểm tra. Không dùng `git reset --hard` hoặc force push làm mất thay đổi của người dùng.

---

## 7. Bàn Giao

Trả lời ngắn gọn, nêu rõ đường dẫn file có thể mở, kết quả kiểm tra định lượng (Accuracy, F1, Precision, Recall) và lệnh chạy thử cần thiết.
Không tự ý tạo mới hoặc sửa đổi lung tung các file cấu hình khi chưa có bằng chứng kiểm thử thực tế.

---

## graphify

This project has a graphify knowledge graph at graphify-out/.

Rules:
- For codebase or architecture questions, when `graphify-out/graph.json` exists, first run `graphify query "<question>"` (CLI) or `query_graph` (MCP). Use `graphify path "<A>" "<B>"` / `shortest_path` for relationships and `graphify explain "<concept>"` / `get_node` for focused concepts. These return a scoped subgraph, usually much smaller than `GRAPH_REPORT.md` or raw grep output.
- If graphify-out/wiki/index.md exists, navigate it instead of reading raw files
- Read graphify-out/GRAPH_REPORT.md only for broad architecture review or when query/path/explain do not surface enough context
- After modifying code files in this session, run `graphify update .` to keep the graph current (AST-only, no API cost)

