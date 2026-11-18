"""Automated Unit & Integration Tests for Benchmark Data Collection & Safeguards."""

import csv
import json
import os
import shutil
import sys
import tempfile
from pathlib import Path

import numpy as np
import soundfile as sf
import torch
from fastapi.testclient import TestClient

# Add project root to sys.path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import api.main as api_main
from api.main import (
    BENCHMARK_DIR,
    BENCHMARK_MANIFEST_PATH,
    BENCHMARK_METADATA_PATH,
    BENCHMARK_WAVS_DIR,
    app,
)


def _create_synthetic_voiced_wav(path: str, duration_s: float = 3.0, sr: int = 16000, f0: float = 140.0):
    """Generate harmonic voiced tone with realistic glottal pitch for acoustic validation."""
    t = np.linspace(0, duration_s, int(duration_s * sr), endpoint=False)
    # Fundamental frequency + first 3 harmonics
    signal = 0.5 * np.sin(2 * np.pi * f0 * t)
    signal += 0.25 * np.sin(2 * np.pi * 2 * f0 * t)
    signal += 0.12 * np.sin(2 * np.pi * 3 * f0 * t)
    signal = signal.astype(np.float32)
    sf.write(path, signal, sr)


def _create_unvoiced_noise_wav(path: str, duration_s: float = 2.5, sr: int = 16000):
    """Generate unvoiced white noise without glottal F0 periodicity."""
    rng = np.random.default_rng(42)
    noise = rng.normal(0.0, 0.05, int(duration_s * sr)).astype(np.float32)
    sf.write(path, noise, sr)


def test_benchmark_collection_workflow():
    print("\n" + "=" * 65)
    print("🚀 BẮT ĐẦU KIỂM THỬ LUỒNG THU THẬP DỮ LIỆU BENCHMARK KIỂM CHỨNG")
    print("=" * 65)

    client = TestClient(app)
    test_user = "test_user_bench_01"
    created_sample_id = None
    created_wav_path = None

    with tempfile.TemporaryDirectory() as tmp_dir:
        voiced_wav = os.path.join(tmp_dir, "voiced.wav")
        noise_wav = os.path.join(tmp_dir, "noise.wav")
        short_wav = os.path.join(tmp_dir, "short.wav")

        _create_synthetic_voiced_wav(voiced_wav, duration_s=3.0)
        _create_unvoiced_noise_wav(noise_wav, duration_s=2.5)
        _create_synthetic_voiced_wav(short_wav, duration_s=0.8)  # Too short (< 1.5s)

        try:
            # 1. Kiểm tra Path Traversal Guard trên user_id
            print("\n1️⃣ Kiểm tra Path Traversal Guard trên user_id...")
            with open(voiced_wav, "rb") as f:
                res = client.post(
                    "/api/v1/benchmarks/submit",
                    data={"user_id": "../../etc_passwd", "ground_truth": "normal"},
                    files={"file": ("test.wav", f, "audio/wav")},
                )
            assert res.status_code == 400, f"Kỳ vọng 400 cho path traversal, nhận {res.status_code}"
            print("✅ [Security] Chặn thành công user_id chứa path traversal '../../etc_passwd'.")

            # 2. Kiểm tra Nhãn Ground Truth không hợp lệ
            print("\n2️⃣ Kiểm tra Validation nhãn Ground Truth...")
            with open(voiced_wav, "rb") as f:
                res = client.post(
                    "/api/v1/benchmarks/submit",
                    data={"user_id": test_user, "ground_truth": "invalid_label_xyz"},
                    files={"file": ("test.wav", f, "audio/wav")},
                )
            assert res.status_code == 400
            assert "ground_truth" in res.json()["detail"]
            print("✅ [Validation] Chặn nhãn ground_truth ngoài danh mục quy chuẩn.")

            # 3. Kiểm tra Ràng buộc Consent
            print("\n3️⃣ Kiểm tra Ràng buộc Consent...")
            with open(voiced_wav, "rb") as f:
                res = client.post(
                    "/api/v1/benchmarks/submit",
                    data={"user_id": test_user, "ground_truth": "normal", "consent": "false"},
                    files={"file": ("test.wav", f, "audio/wav")},
                )
            assert res.status_code == 400
            assert "đồng ý" in res.json()["detail"].lower() or "consent" in res.json()["detail"].lower()
            print("✅ [Ethics] Chặn lưu trữ khi người dùng chưa đồng ý điều khoản lưu trữ ẩn danh.")

            # 4. Kiểm tra Thời lượng tối thiểu (>= 1.5s)
            print("\n4️⃣ Kiểm tra Thời lượng tối thiểu...")
            with open(short_wav, "rb") as f:
                res = client.post(
                    "/api/v1/benchmarks/submit",
                    data={"user_id": test_user, "ground_truth": "normal"},
                    files={"file": ("short.wav", f, "audio/wav")},
                )
            assert res.status_code == 400
            assert "quá ngắn" in res.json()["detail"]
            print("✅ [Audio Guard] Bác bỏ tệp có thời lượng < 1.5 giây.")

            # 5. Kiểm tra Acoustic Quality Safeguard (Bác bỏ tiếng ồn trắng không có F0 tuần hoàn)
            print("\n5️⃣ Kiểm tra Acoustic Quality Safeguard trên tệp nhiễu không tuần hoàn...")
            with open(noise_wav, "rb") as f:
                res = client.post(
                    "/api/v1/benchmarks/submit",
                    data={"user_id": test_user, "ground_truth": "stress"},
                    files={"file": ("noise.wav", f, "audio/wav")},
                )
            assert res.status_code == 422, f"Kỳ vọng HTTP 422 cho tín hiệu không đạt âm học, nhận {res.status_code}: {res.text}"
            print("✅ [Acoustic Safeguard] Bác bỏ chính xác tệp nhiễu không có F0 tuần hoàn (HTTP 422).")

            # 6. Gửi mẫu hợp lệ đạt chuẩn
            print("\n6️⃣ Gửi mẫu hợp lệ đạt chuẩn chất lượng âm học...")
            with open(voiced_wav, "rb") as f:
                res = client.post(
                    "/api/v1/benchmarks/submit",
                    data={
                        "user_id": test_user,
                        "ground_truth": "normal",
                        "device": "automated_test_mic",
                        "context": "unit_test_run",
                        "consent": "true",
                    },
                    files={"file": ("voiced.wav", f, "audio/wav")},
                )
            assert res.status_code == 200, f"Lưu mẫu hợp lệ thất bại: {res.text}"
            resp_data = res.json()
            assert resp_data["status"] == "success"
            created_sample_id = resp_data["sample_id"]
            wav_filename = resp_data["filename"]
            created_wav_path = BENCHMARK_WAVS_DIR / wav_filename

            # Kiểm tra tệp WAV đã lưu trên ổ cứng
            assert created_wav_path.is_file(), f"Tệp WAV {created_wav_path} không tồn tại trên đĩa!"
            saved_info = sf.info(str(created_wav_path))
            assert saved_info.samplerate == 16000, f"Sample rate phải là 16kHz, nhận {saved_info.samplerate}"
            assert saved_info.channels == 1, f"Channels phải là Mono (1), nhận {saved_info.channels}"
            print(f"✅ [Storage] Tệp WAV đã lưu chuẩn 16kHz Mono: {created_wav_path.name} (Thời lượng: {saved_info.duration:.2f}s).")

            # Kiểm tra Manifest JSON
            assert BENCHMARK_MANIFEST_PATH.is_file()
            with open(BENCHMARK_MANIFEST_PATH, "r", encoding="utf-8") as mf:
                manifest_dict = json.load(mf)
            assert created_sample_id in manifest_dict
            assert manifest_dict[created_sample_id]["speaker_id"] == test_user
            assert manifest_dict[created_sample_id]["ground_truth"] == "normal"
            print("✅ [Manifest] Dữ liệu đã được cập nhật nguyên tử vào benchmark_manifest.json.")

            # Kiểm tra Metadata CSV
            assert BENCHMARK_METADATA_PATH.is_file()
            with open(BENCHMARK_METADATA_PATH, "r", encoding="utf-8") as cf:
                csv_content = cf.read()
            assert created_sample_id in csv_content
            print("✅ [Metadata CSV] Dòng dữ liệu đã được ghi vào benchmark_metadata.csv.")

            # 7. Kiểm tra Endpoint Thống kê (/api/v1/benchmarks/stats)
            print("\n7️⃣ Kiểm tra Endpoint Thống kê...")
            res_stats = client.get("/api/v1/benchmarks/stats")
            assert res_stats.status_code == 200
            stats_data = res_stats.json()
            assert stats_data["total_samples"] >= 1
            assert test_user in stats_data["speakers"]
            assert stats_data["ground_truth_counts"].get("normal", 0) >= 1
            print(f"✅ [Stats] Thống kê phản hồi chính xác: {stats_data['total_samples']} mẫu, {stats_data['speakers_count']} người nói.")

        finally:
            # 8. Dọn dẹp dữ liệu kiểm thử (Guaranteed cleanup)
            print("\n8️⃣ Dọn dẹp an toàn các tệp và bản ghi test...")
            if created_wav_path and created_wav_path.is_file():
                created_wav_path.unlink(missing_ok=True)

            if created_sample_id and BENCHMARK_MANIFEST_PATH.is_file():
                try:
                    with open(BENCHMARK_MANIFEST_PATH, "r", encoding="utf-8") as mf:
                        mdata = json.load(mf)
                    if created_sample_id in mdata:
                        del mdata[created_sample_id]
                        with open(BENCHMARK_MANIFEST_PATH, "w", encoding="utf-8") as mf:
                            json.dump(mdata, mf, indent=2, ensure_ascii=False)
                except Exception:
                    pass

            if created_sample_id and BENCHMARK_METADATA_PATH.is_file():
                try:
                    with open(BENCHMARK_METADATA_PATH, "r", encoding="utf-8") as cf:
                        rows = [r for r in csv.reader(cf) if not (r and r[0] == created_sample_id)]
                    with open(BENCHMARK_METADATA_PATH, "w", newline="", encoding="utf-8") as cf:
                        writer = csv.writer(cf)
                        writer.writerows(rows)
                except Exception:
                    pass
            print("✅ [Cleanup] Toàn bộ dữ liệu kiểm thử đã được dọn sạch không để lại tệp rác.")

    print("\n" + "=" * 65)
    print("🎉 TẤT CẢ CÁC BÀI TEST THU THẬP BENCHMARK ĐÃ VƯỢT QUA 100%!")
    print("=" * 65)


if __name__ == "__main__":
    test_benchmark_collection_workflow()
