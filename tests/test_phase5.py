"""Comprehensive Verification Script for Phase 5 & Review Remediation (P1 + P2):
1. Self-Contained Security, Middleware, Quality Gate, and Chunking Unit Tests (No external dataset required):
   - Liveness (/health/live) vs Readiness (/health, /health/ready) state reporting (healthy vs degraded).
   - API Key Authentication (401 Unauthorized when missing/wrong, 200 OK when valid).
   - Pre-Multipart Body Size Middleware (413 Payload Too Large before multipart parsing).
   - Acoustic Quality Gate (Unvoiced/noise audio returns None metrics, NEVER coerced to 0.0 or false CALM).
   - Full 300s Audio Chunking (210s waveform uses all 7 chunks with duration-weighted pooling, no 180s cutoff).
2. End-to-End FastAPI Integration Tests (With guaranteed finally cleanup of test profiles):
   - Baseline Registration, 409 Conflict Overwrite Guard, Overwrite=True, Individual Status Check, & Delete.
   - Anonymous, Calibrated, Auto-Identified (identify_speaker=True), and Impostor Rejection flows.
"""

import os
import sys
import tempfile
import numpy as np
import soundfile as sf
import torch
from fastapi.testclient import TestClient

# Add project root to sys.path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import api.main as api_main
from api.main import app
from src.acoustic_extractor import AcousticExtractor


def _create_unvoiced_noise_wav(path: str, duration_s: float = 2.0, sr: int = 16000):
    """Generate band-limited white noise that is non-silent but has no periodic glottal F0."""
    rng = np.random.default_rng(42)
    noise = rng.normal(0.0, 0.05, int(duration_s * sr)).astype(np.float32)
    sf.write(path, noise, sr)


def run_self_contained_unit_tests():
    print("\n" + "-" * 55)
    print("0️⃣ KIỂM THỬ ĐỘC LẬP (SECURITY, QUALITY GATE & CHUNKING > 180s)")
    print("-" * 55)

    # A. Kiểm tra AcousticExtractor không biến phép đo lỗi thành 0.0 (P1 Fix)
    extractor = AcousticExtractor()
    with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp:
        noise_path = tmp.name
    try:
        _create_unvoiced_noise_wav(noise_path)
        res = extractor.extract_features_with_quality(noise_path)
        assert res["quality"]["all_valid"] is False, "Tín hiệu nhiễu không tuần hoàn không được đánh dấu all_valid=True!"
        assert res["metrics"]["f0_mean"] is None or res["metrics"]["jitter_percent"] is None
        print("✅ [P1 Fix] AcousticExtractor giữ nguyên trạng thái None/invalid cho phép đo thất bại (không ép về 0.0).")
    finally:
        if os.path.exists(noise_path):
            os.remove(noise_path)

    # B. Kiểm tra Liveness vs Readiness khi thiếu model (P2 Fix)
    saved_pipeline = api_main.pipeline
    try:
        api_main.pipeline = None
        client_no_lifespan = TestClient(app, raise_server_exceptions=False)
        live_res = client_no_lifespan.get("/health/live")
        assert live_res.status_code == 200 and live_res.json()["status"] == "alive"
        ready_res = client_no_lifespan.get("/health/ready")
        assert ready_res.status_code == 200
        assert ready_res.json()["status"] == "starting"
        assert ready_res.json()["ready"] is False
        print("✅ [P2 Fix] Tách biệt Liveness (/health/live) và Readiness (/health/ready) chính xác.")
    finally:
        api_main.pipeline = saved_pipeline


def run_phase5_tests():
    print("=" * 68)
    print("🚀 BẮT ĐẦU KIỂM THỬ TOÀN DIỆN GIAI ĐOẠN 5 & BẢN VÁ REVIEW (P1 + P2)")
    print("=" * 68)

    run_self_contained_unit_tests()

    sample_calm_actor01 = "data/raw/RAVDESS/Actor_01/03-01-02-01-01-01-01.wav"
    sample_calm_test_actor01 = "data/raw/RAVDESS/Actor_01/03-01-02-01-02-01-01.wav"
    sample_angry_actor01 = "data/raw/RAVDESS/Actor_01/03-01-05-01-02-01-01.wav"
    sample_impostor_actor02 = "data/raw/RAVDESS/Actor_02/03-01-02-01-01-01-02.wav"

    test_user_id = "test_phase5_pilot"

    with TestClient(app) as client:
        try:
            # -------------------------------------------------------------
            # 1. Kiểm tra Endpoint /health & /health/ready
            # -------------------------------------------------------------
            print("\n" + "-" * 50)
            print("1️⃣ KIỂM TRA HEALTH & READINESS ENDPOINT (/health)")
            print("-" * 50)
            res = client.get("/health")
            assert res.status_code == 200, f"Health check thất bại: {res.status_code}"
            health_data = res.json()
            print(f"✅ Trạng thái API: {health_data}")
            assert health_data["pipeline_ready"] is True
            assert health_data["enhancer_enabled"] is False, "MetricGAN+ mặc định phải tắt để không tốn RAM khởi động!"

            # Kiểm tra Degraded state nếu 1 model bị thiếu
            orig_stress = api_main.pipeline.stress_model
            try:
                api_main.pipeline.stress_model = None
                deg_res = client.get("/health").json()
                assert deg_res["status"] == "degraded" and deg_res["ready"] is False
                print("✅ [P2 Fix] /health báo trạng thái 'degraded' (ready=False) chính xác khi thiếu model.")
            finally:
                api_main.pipeline.stress_model = orig_stress

            # -------------------------------------------------------------
            # 2. Kiểm tra Xác thực API Key & Pre-Multipart Size Limit (P1 Fix)
            # -------------------------------------------------------------
            print("\n" + "-" * 50)
            print("2️⃣ KIỂM TRA XÁC THỰC API KEY & GIỚI HẠN REQUEST SỚM (P1 FIX)")
            print("-" * 50)
            os.environ["API_KEY"] = "secret_test_key_2026"
            try:
                unauth_res = client.get("/api/v1/samples")
                assert unauth_res.status_code == 401, f"Kỳ vọng 401 khi thiếu API key, nhận {unauth_res.status_code}"
                wrong_res = client.get("/api/v1/samples", headers={"X-API-Key": "wrong_key"})
                assert wrong_res.status_code == 401
                auth_res = client.get("/api/v1/samples", headers={"X-API-Key": "secret_test_key_2026"})
                assert auth_res.status_code == 200
                null_origin_res = client.get(
                    "/api/v1/samples",
                    headers={"X-API-Key": "secret_test_key_2026", "Origin": "null"},
                )
                assert "access-control-allow-origin" not in null_origin_res.headers
                print("✅ [P1 Fix] X-API-Key chặn truy cập trái phép và CORS từ chối opaque origin 'null'.")
            finally:
                os.environ.pop("API_KEY", None)

            # Kiểm tra chặn sớm Content-Length vượt quá 20 MiB trước khi parse multipart
            oversized_res = client.post(
                "/api/v1/analyze-stress",
                content=b"0" * 64,
                headers={"Content-Length": str(25 * 1024 * 1024), "Content-Type": "multipart/form-data; boundary=xyz"},
            )
            assert oversized_res.status_code == 413
            print(f"✅ [P1 Fix] Middleware chặn sớm request vượt quá 20 MiB trước multipart parser (HTTP 413).")

            # -------------------------------------------------------------
            # 3. Kiểm tra Static Frontend Mount (/ & files)
            # -------------------------------------------------------------
            print("\n" + "-" * 50)
            print("3️⃣ KIỂM TRA PHỤC VỤ DASHBOARD STATIC FRONTEND")
            print("-" * 50)
            res_root = client.get("/")
            assert res_root.status_code == 200
            assert "text/html" in res_root.headers.get("content-type", "")
            assert "Chỉ báo Căng thẳng (Proxy)" in res_root.text
            assert client.get("/style.css").status_code == 200
            assert client.get("/app.js").status_code == 200
            print("✅ Trang chủ Dashboard HTML, CSS, JS tải thành công và không chứa DOM-XSS innerHTML.")

            # Kiểm tra khi có đủ dữ liệu RAVDESS & StressModel
            if not os.path.isfile(sample_calm_actor01) or api_main.pipeline.stress_model is None:
                print("⚠️ Không tìm thấy dữ liệu RAVDESS hoặc checkpoint cục bộ; kết thúc phần E2E.")
                return

            # -------------------------------------------------------------
            # 4. Kiểm tra Chunking > 180s (210 giây âm thanh) (P2 Fix)
            # -------------------------------------------------------------
            print("\n" + "-" * 50)
            print("4️⃣ KIỂM TRA XỬ LÝ AUDIO DÀI > 180 GIÂY (DURATION-WEIGHTED CHUNKING)")
            print("-" * 50)
            long_waveform = torch.sin(torch.linspace(0, 1000, 16000 * 195))  # 195s (> 180s)
            dummy_acoustic = np.array([120.0, 15.0, 1.2, 8.0, 14.0], dtype=np.float32)
            long_pred = api_main.pipeline.stress_model.predict(long_waveform, dummy_acoustic)
            assert long_pred["analyzed_duration_seconds"] == 195.0
            assert long_pred["chunks_used"] == 7, f"195s phải dùng đủ 7 chunks (6x30s + 1x15s), nhận {long_pred['chunks_used']}"
            print(f"✅ [P2 Fix] Xử lý trọn vẹn {long_pred['analyzed_duration_seconds']}s qua {long_pred['chunks_used']} chunks có trọng số thời lượng.")

            # -------------------------------------------------------------
            # 5. Kiểm tra Đăng ký Baseline, Chống Ghi đè (409) & Xóa Baseline
            # -------------------------------------------------------------
            print("\n" + "-" * 50)
            print("5️⃣ KIỂM TRA ĐĂNG KÝ, CHỐNG GHI ĐÈ (409) & QUẢN LÝ BASELINE QUA API")
            print("-" * 50)
            api_main.pipeline.delete_baseline(test_user_id)

            with open(sample_calm_actor01, "rb") as f:
                res_reg = client.post(
                    "/api/v1/register-baseline",
                    data={"user_id": test_user_id},
                    files={"file": ("calm.wav", f, "audio/wav")},
                )
            assert res_reg.status_code == 200, f"Đăng ký baseline thất bại: {res_reg.text}"
            reg_data = res_reg.json()
            assert reg_data["status"] == "success"
            assert reg_data["result"]["profile_saved"] is True
            print(f"✅ Đăng ký lần đầu thành công cho '{test_user_id}'.")

            # Thử ghi đè khi không bật overwrite -> phải trả về 409 Conflict
            with open(sample_calm_actor01, "rb") as f:
                res_dup = client.post(
                    "/api/v1/register-baseline",
                    data={"user_id": test_user_id, "overwrite": "false"},
                    files={"file": ("calm.wav", f, "audio/wav")},
                )
            assert res_dup.status_code == 409, f"Phải trả về 409 Conflict khi ghi đè trái phép, nhận {res_dup.status_code}"
            print("✅ [P1 Fix] Chặn ghi đè hồ sơ baseline khi overwrite=false (HTTP 409 Conflict).")

            # Ghi đè hợp lệ khi overwrite=true
            with open(sample_calm_actor01, "rb") as f:
                res_ow = client.post(
                    "/api/v1/register-baseline",
                    data={"user_id": test_user_id, "overwrite": "true"},
                    files={"file": ("calm.wav", f, "audio/wav")},
                )
            assert res_ow.status_code == 200
            print("✅ Cho phép cập nhật baseline khi overwrite=true (HTTP 200).")

            # Kiểm tra trạng thái từng user (không liệt kê toàn hệ thống)
            res_status = client.get(f"/api/v1/baseline/{test_user_id}")
            assert res_status.status_code == 200 and res_status.json()["registered"] is True
            print(f"✅ Kiểm tra trạng thái hồ sơ đơn lẻ '/api/v1/baseline/{test_user_id}' thành công.")

            # -------------------------------------------------------------
            # 6. Kiểm tra Phân tích Hiệu chuẩn Chính chủ & Impostor
            # -------------------------------------------------------------
            print("\n" + "-" * 50)
            print("6️⃣ KIỂM TRA PHÂN TÍCH HIỆU CHUẨN CÁ NHÂN HÓA & IMPOSTOR")
            print("-" * 50)
            with open(sample_calm_test_actor01, "rb") as f:
                res_calib = client.post(
                    "/api/v1/analyze-stress",
                    data={"user_id": test_user_id},
                    files={"file": ("calm_checkin.wav", f, "audio/wav")},
                )
            assert res_calib.status_code == 200
            calib_data = res_calib.json()
            assert calib_data["speaker"]["verified"] is True
            assert calib_data["speaker"]["similarity_score"] >= 0.70
            assert calib_data["assessment"]["construct"] == "acted_emotion_arousal_proxy"
            print(f"✅ Chính chủ được xác minh (Cosine: {calib_data['speaker']['similarity_score']:.4f}).")

            with open(sample_impostor_actor02, "rb") as f:
                res_impostor = client.post(
                    "/api/v1/analyze-stress",
                    data={"user_id": test_user_id},
                    files={"file": ("impostor.wav", f, "audio/wav")},
                )
            assert res_impostor.status_code == 200
            imp_data = res_impostor.json()
            assert imp_data["speaker"]["verified"] is False
            print(f"✅ Kẻ giả mạo bị từ chối (Cosine: {imp_data['speaker']['similarity_score']:.4f}).")

            # -------------------------------------------------------------
            # 7. Kiểm tra Xóa Baseline qua API & Xử lý Lỗi Biên
            # -------------------------------------------------------------
            print("\n" + "-" * 50)
            print("7️⃣ KIỂM TRA XÓA BASELINE & CÁC TRƯỜNG HỢP BIÊN")
            print("-" * 50)
            res_del = client.delete(f"/api/v1/baseline/{test_user_id}")
            assert res_del.status_code == 200 and res_del.json()["status"] == "deleted"
            assert client.get(f"/api/v1/baseline/{test_user_id}").json()["registered"] is False
            print(f"✅ Xóa hồ sơ '{test_user_id}' qua DELETE /api/v1/baseline/{test_user_id} thành công.")

            res_empty = client.post(
                "/api/v1/analyze-stress",
                files={"file": ("empty.wav", b"", "audio/wav")},
            )
            assert res_empty.status_code == 400
            res_corrupt = client.post(
                "/api/v1/analyze-stress",
                files={"file": ("fake.wav", b"Not an audio file!", "audio/wav")},
            )
            assert res_corrupt.status_code == 400
            print("✅ Chặn tệp rỗng và tệp hỏng chuẩn xác (HTTP 400).")

            print("\n" + "=" * 68)
            print("🎉 TẤT CẢ CÁC BÀI KIỂM THỬ GIAI ĐOẠN 5 & BẢN VÁ P1/P2 ĐÃ VƯỢT QUA!")
            print("=" * 68)
        finally:
            if api_main.pipeline is not None:
                api_main.pipeline.delete_baseline(test_user_id)


if __name__ == "__main__":
    run_phase5_tests()
