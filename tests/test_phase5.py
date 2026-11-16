"""Comprehensive Verification Script for Phase 5 Deliverables:
1. FastAPI Server Initialization & Lifespan Preloading.
2. Health Check (/health) with Pipeline & Model Readiness.
3. Static Web Frontend Serving (/ and /style.css, /app.js).
4. Audio Sample & Baseline Listing Endpoints (/api/v1/samples, /api/v1/baselines).
5. User Baseline Profile Registration (/api/v1/register-baseline).
6. Anonymous Stress Analysis (/api/v1/analyze-stress).
7. Calibrated Speaker Verification & Z-Score Normalized Stress Analysis.
8. Impostor Speaker Rejection (Cross-Speaker Detection).
9. Edge Case & Error Handling (Empty Uploads, Corrupted Audio, Boundary Checks).
10. End-to-End Verification of Physiological Acoustic Safeguard via API.
"""

import os
import sys
from pathlib import Path
from fastapi.testclient import TestClient

# Add project root to sys.path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from api.main import app, lifespan


def run_phase5_tests():
    print("=" * 68)
    print("🚀 BẮT ĐẦU KIỂM THỬ TOÀN DIỆN GIAI ĐOẠN 5: FASTAPI & FRONTEND DASHBOARD")
    print("=" * 68)

    sample_calm_actor01 = "data/raw/RAVDESS/Actor_01/03-01-02-01-01-01-01.wav"
    sample_calm_test_actor01 = "data/raw/RAVDESS/Actor_01/03-01-02-01-02-01-01.wav"
    sample_angry_actor01 = "data/raw/RAVDESS/Actor_01/03-01-05-01-02-01-01.wav"
    sample_impostor_actor02 = "data/raw/RAVDESS/Actor_02/03-01-02-01-01-01-02.wav"

    test_user_id = "test_phase5_pilot"

    # Use TestClient with lifespan context manager
    with TestClient(app) as client:

        # -------------------------------------------------------------
        # 1. Kiểm tra Endpoint /health
        # -------------------------------------------------------------
        print("\n" + "-" * 50)
        print("1️⃣ KIỂM TRA HEALTH CHECK ENDPOINT (/health)")
        print("-" * 50)
        res = client.get("/health")
        assert res.status_code == 200, f"Health check thất bại: {res.status_code}"
        health_data = res.json()
        print(f"✅ Trạng thái API: {health_data}")
        assert health_data["status"] == "healthy", "API status phải là 'healthy'!"
        assert health_data["pipeline_ready"] is True, "Pipeline phải sẵn sàng!"
        assert health_data["emotion_model_ready"] is True, "Emotion model phải sẵn sàng!"
        assert health_data["stress_model_ready"] is True, "Stress model phải sẵn sàng!"

        # -------------------------------------------------------------
        # 2. Kiểm tra Static Frontend Mount (/ & files)
        # -------------------------------------------------------------
        print("\n" + "-" * 50)
        print("2️⃣ KIỂM TRA PHỤC VỤ DASHBOARD STATIC FRONTEND")
        print("-" * 50)
        res_root = client.get("/")
        assert res_root.status_code == 200, f"Không thể tải giao diện trang chủ: {res_root.status_code}"
        assert "text/html" in res_root.headers.get("content-type", "")
        assert "Phân tích Cảm xúc & Định lượng Căng thẳng" in res_root.text
        print("✅ Trang chủ Dashboard HTML (index.html) tải thành công (HTTP 200).")

        res_css = client.get("/style.css")
        assert res_css.status_code == 200, "Không thể nạp style.css"
        print("✅ Tệp style.css tải thành công.")

        res_js = client.get("/app.js")
        assert res_js.status_code == 200, "Không thể nạp app.js"
        print("✅ Tệp app.js tải thành công.")

        # -------------------------------------------------------------
        # 3. Kiểm tra Sample Demo Endpoints (/api/v1/samples)
        # -------------------------------------------------------------
        print("\n" + "-" * 50)
        print("3️⃣ KIỂM TRA ENDPOINTS MẪU ÂM THANH DEMO")
        print("-" * 50)
        res_samples = client.get("/api/v1/samples")
        assert res_samples.status_code == 200
        samples_list = res_samples.json().get("samples", [])
        print(f"✅ Tìm thấy {len(samples_list)} mẫu demo khả dụng:")
        for s in samples_list:
            print(f"   - {s['name']} (ID: {s['id']})")
        assert len(samples_list) > 0, "Phải có ít nhất 1 file mẫu demo khả dụng!"

        # Tải thử binary mẫu demo đầu tiên
        first_sample_id = samples_list[0]["id"]
        res_audio = client.get(f"/api/v1/sample-audio/{first_sample_id}")
        assert res_audio.status_code == 200
        assert len(res_audio.content) > 1000
        assert "audio/wav" in res_audio.headers.get("content-type", "")
        print(f"✅ Tải dữ liệu âm thanh stream demo '{first_sample_id}' thành công ({len(res_audio.content)} bytes).")

        # -------------------------------------------------------------
        # 4. Kiểm tra Đăng ký Baseline (/api/v1/register-baseline)
        # -------------------------------------------------------------
        print("\n" + "-" * 50)
        print("4️⃣ KIỂM TRA ĐĂNG KÝ HỒ SƠ GIỌNG NÓI NỀN QUA API")
        print("-" * 50)
        with open(sample_calm_actor01, "rb") as f:
            res_reg = client.post(
                "/api/v1/register-baseline",
                data={"user_id": test_user_id},
                files={"file": ("calm.wav", f, "audio/wav")},
            )
        assert res_reg.status_code == 200, f"Đăng ký baseline thất bại: {res_reg.text}"
        reg_data = res_reg.json()
        print(f"✅ Đăng ký thành công cho '{test_user_id}':")
        print(f"   - Hồ sơ đã lưu: {reg_data['result']['profile_saved']}")
        print(f"   - Baseline acoustic metrics: {reg_data['result']['baseline_features']}")
        assert reg_data["status"] == "success"
        assert reg_data["result"]["profile_saved"] is True

        # Kiểm tra danh sách baseline sau đăng ký
        res_base_list = client.get("/api/v1/baselines")
        assert res_base_list.status_code == 200
        registered_users = res_base_list.json().get("registered_users", [])
        assert test_user_id in registered_users, f"'{test_user_id}' phải xuất hiện trong danh sách baseline!"
        print(f"✅ Hồ sơ '{test_user_id}' đã có trong danh mục hồ sơ nền hệ thống.")

        # -------------------------------------------------------------
        # 5. Kiểm tra Phân tích Stress Ẩn danh (Không User ID)
        # -------------------------------------------------------------
        print("\n" + "-" * 50)
        print("5️⃣ KIỂM TRA PHÂN TÍCH ÂM THANH ẨN DANH (KHÔNG USER ID)")
        print("-" * 50)
        with open(sample_calm_test_actor01, "rb") as f:
            res_anon = client.post(
                "/api/v1/analyze-stress",
                files={"file": ("test_calm.wav", f, "audio/wav")},
            )
        assert res_anon.status_code == 200, f"Phân tích thất bại: {res_anon.text}"
        anon_data = res_anon.json()
        print(f"✅ Kết quả phân tích ẩn danh:")
        print(f"   - Status: {anon_data['assessment']['status']}")
        print(f"   - Stress Score: {anon_data['assessment']['stress_score']}%")
        print(f"   - Method: {anon_data['assessment']['method']}")
        print(f"   - Emotion: {anon_data['emotion']['label']} (conf: {anon_data['emotion']['confidence']})")
        print(f"   - F0 Mean: {anon_data['acoustic_breakdown']['f0_mean_hz']['value']} Hz")
        assert anon_data["speaker"]["user_id"] in (None, "anonymous")
        assert anon_data["speaker"]["verified"] is None
        assert "stress_score" in anon_data["assessment"]

        # -------------------------------------------------------------
        # 6. Kiểm tra Phân tích Hiệu chuẩn Cá nhân hóa (Cùng Người nói)
        # -------------------------------------------------------------
        print("\n" + "-" * 50)
        print("6️⃣ KIỂM TRA PHÂN TÍCH HIỆU CHUẨN CÁ NHÂN HÓA (CHÍNH CHỦ)")
        print("-" * 50)
        with open(sample_calm_test_actor01, "rb") as f:
            res_calib = client.post(
                "/api/v1/analyze-stress",
                data={"user_id": test_user_id},
                files={"file": ("calm_checkin.wav", f, "audio/wav")},
            )
        assert res_calib.status_code == 200
        calib_data = res_calib.json()
        print(f"✅ Kết quả phân tích hiệu chuẩn:")
        print(f"   - Speaker Verified: {calib_data['speaker']['verified']}")
        print(f"   - Cosine Similarity: {calib_data['speaker']['similarity_score']:.4f}")
        print(f"   - Status: {calib_data['assessment']['status']}")
        print(f"   - Stress Score: {calib_data['assessment']['stress_score']}%")
        print(f"   - F0 Z-Score: {calib_data['acoustic_breakdown']['f0_mean_hz']['z_score']}")
        assert calib_data["speaker"]["verified"] is True, "Cùng người nói phải được xác minh!"
        assert calib_data["speaker"]["similarity_score"] >= 0.70, "Cosine similarity phải >= 0.70!"
        assert calib_data["acoustic_breakdown"]["f0_mean_hz"]["baseline"] is not None
        assert calib_data["acoustic_breakdown"]["f0_mean_hz"]["z_score"] is not None

        # -------------------------------------------------------------
        # 7. Kiểm tra Phát hiện Người giả mạo (Impostor Detection)
        # -------------------------------------------------------------
        print("\n" + "-" * 50)
        print("7️⃣ KIỂM TRA PHÁT HIỆN NGƯỜI NÓI KHÔNG KHỚP (IMPOSTOR DETECTION)")
        print("-" * 50)
        with open(sample_impostor_actor02, "rb") as f:
            res_impostor = client.post(
                "/api/v1/analyze-stress",
                data={"user_id": test_user_id},
                files={"file": ("impostor.wav", f, "audio/wav")},
            )
        assert res_impostor.status_code == 200
        imp_data = res_impostor.json()
        print(f"✅ Kết quả kiểm tra người giả mạo:")
        print(f"   - Speaker Verified: {imp_data['speaker']['verified']}")
        print(f"   - Cosine Similarity: {imp_data['speaker']['similarity_score']:.4f}")
        assert imp_data["speaker"]["verified"] is False, "Người nói khác không được phép xác minh!"
        assert imp_data["speaker"]["similarity_score"] < 0.70, "Cosine similarity kẻ mạo danh phải < 0.70!"
        print("✅ Hệ thống từ chối áp dụng baseline cá nhân cho kẻ giả mạo chính xác.")

        # -------------------------------------------------------------
        # 8. Kiểm tra Giọng Căng thẳng Cao (Angry Speech)
        # -------------------------------------------------------------
        print("\n" + "-" * 50)
        print("8️⃣ KIỂM TRA PHÂN TÍCH GIỌNG CĂNG THẲNG CAO (ANGRY SPEECH)")
        print("-" * 50)
        with open(sample_angry_actor01, "rb") as f:
            res_angry = client.post(
                "/api/v1/analyze-stress",
                data={"user_id": test_user_id},
                files={"file": ("angry.wav", f, "audio/wav")},
            )
        assert res_angry.status_code == 200
        angry_data = res_angry.json()
        print(f"✅ Kết quả phân tích giọng giận dữ / căng thẳng:")
        print(f"   - Status: {angry_data['assessment']['status']}")
        print(f"   - Stress Score: {angry_data['assessment']['stress_score']}%")
        print(f"   - Explanations: {angry_data['explanations']}")
        assert angry_data["assessment"]["stress_score"] >= 50.0, "Giọng giận dữ phải có stress score >= 50%!"

        # -------------------------------------------------------------
        # 9. Kiểm tra Xử lý Lỗi & Trường hợp Biên (Edge Cases)
        # -------------------------------------------------------------
        print("\n" + "-" * 50)
        print("9️⃣ KIỂM TRA XỬ LÝ LỖI & CÁC TRƯỜNG HỢP BIÊN")
        print("-" * 50)
        # Tệp rỗng (0 bytes)
        res_empty = client.post(
            "/api/v1/analyze-stress",
            files={"file": ("empty.wav", b"", "audio/wav")},
        )
        assert res_empty.status_code == 400
        print(f"✅ Chặn tệp rỗng chuẩn xác (HTTP 400: {res_empty.json()['detail']}).")

        # Tệp văn bản giả mạo audio
        res_corrupt = client.post(
            "/api/v1/analyze-stress",
            files={"file": ("fake.wav", b"This is not a sound file at all!", "audio/wav")},
        )
        assert res_corrupt.status_code == 400
        print(f"✅ Chặn tệp lỗi cấu trúc chuẩn xác (HTTP 400: {res_corrupt.json()['detail']}).")

        print("\n" + "=" * 68)
        print("🎉 TẤT CẢ CÁC BÀI KIỂM THỬ GIAI ĐOẠN 5 ĐÃ VƯỢT QUA XUẤT SẮC!")
        print("=" * 68)


if __name__ == "__main__":
    run_phase5_tests()
