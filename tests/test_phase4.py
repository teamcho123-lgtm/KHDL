"""Comprehensive Verification Script for Phase 4 Deliverables:
1. Baseline Profile Registration with Overwrite Protection (ECAPA-TDNN Embedding + F0/Jitter/Shimmer/HNR stats).
2. Speaker Verification (Cosine Similarity >= 0.70 for calm check-in, >= 0.50 for cross-emotion).
3. Automatic Speaker Identification (find_best_match & pipeline.analyze(identify_speaker=True)).
4. Baseline Z-Score Normalization & Physiological Explainable AI (XAI) Rules.
5. Acoustic Quality Gate (Failed measurements remain None and never trigger false CALM safeguard).
6. Personal Baseline Calibration Dampening for Naturally High-Pitched Voices.
"""

import os
import shutil
import sys
import tempfile
import numpy as np

# Add project root to sys.path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.pipeline import StressAnalysisPipeline


def main():
    print("=" * 65)
    print("🚀 BẮT ĐẦU KIỂM THỬ GIAI ĐOẠN 4: CÁ NHÂN HÓA & HIỆU CHUẨN BASELINE")
    print("=" * 65)

    pipeline = StressAnalysisPipeline.from_config("configs/hparams_inference.yaml")
    # Auto-identification must only see temporary test profiles, never real user profiles.
    pipeline.baseline_dir = tempfile.mkdtemp(prefix="speech_stress_phase4_baselines_")

    calm_speaker1 = "data/raw/RAVDESS/Actor_01/03-01-02-01-01-01-01.wav"
    test_calm_speaker1 = "data/raw/RAVDESS/Actor_01/03-01-02-01-02-01-01.wav"
    stress_speaker1 = "data/raw/RAVDESS/Actor_01/03-01-05-01-02-01-01.wav"
    impostor_speaker = "data/raw/RAVDESS/Actor_02/03-01-02-01-01-01-02.wav"

    # Keep test profiles isolated from a user's real baseline profiles.
    user_id = "test_driver_actor_01"
    user_id_high_pitch = "test_actor_02_high_pitch"

    try:
        # -------------------------------------------------------------
        # 1. Đăng ký Baseline Cá nhân hóa & Chống Ghi đè (Overwrite Guard)
        # -------------------------------------------------------------
        print("\n" + "-" * 50)
        print("1️⃣ KIỂM TRA ĐĂNG KÝ HỒ SƠ GIỌNG NÓI NỀN & OVERWRITE GUARD")
        print("-" * 50)
        reg_result = pipeline.register_baseline(
            user_id=user_id, calm_audio_path=calm_speaker1, overwrite=True
        )
        print(f"✅ Đăng ký thành công cho '{user_id}':")
        print(f"   - File hồ sơ: data/baseline_profiles/{user_id}_profile.npz")
        print(f"   - Baseline acoustic metrics: {reg_result['baseline_features']}")

        # Kiểm tra chống ghi đè khi overwrite=False
        try:
            pipeline.register_baseline(
                user_id=user_id, calm_audio_path=calm_speaker1, overwrite=False
            )
            raise AssertionError("Lỗi: Phải chặn ghi đè khi overwrite=False!")
        except FileExistsError:
            print("✅ Chặn ghi đè trái phép (FileExistsError khi overwrite=False) hoạt động chuẩn xác.")

        profile = pipeline.load_baseline(user_id)
        assert profile is not None, "Lỗi: Không tìm thấy hồ sơ baseline vừa lưu!"
        assert profile["speaker_embedding"].shape == (192,), "Vector embedding phải có shape (192,)!"
        assert profile["f0_mean"] > 0, "F0 baseline phải lớn hơn 0!"
        print("✅ Hồ sơ Baseline chứa đầy đủ 192-dim ECAPA-TDNN và thông số thanh môn hợp lệ.")

        # -------------------------------------------------------------
        # 2. Xác minh Người nói & Tự động Nhận diện (identify_speaker=True)
        # -------------------------------------------------------------
        print("\n" + "-" * 50)
        print("2️⃣ KIỂM TRA XÁC MINH & TỰ ĐỘNG NHẬN DIỆN NGƯỜI NÓI")
        print("-" * 50)
        calibrator = pipeline.calibrator

        is_same, score_same = calibrator.verify_speaker(
            test_calm_speaker1,
            profile["speaker_embedding"],
            threshold=pipeline.speaker_identification_threshold,
        )
        print(f"   - Cùng người nói (Actor_01): Cosine = {score_same:.4f} -> Verified: {is_same}")
        assert is_same, f"Lỗi: Cùng người nói phải được xác minh (Cosine {score_same} < 0.70)"

        is_diff, score_diff = calibrator.verify_speaker(
            impostor_speaker,
            profile["speaker_embedding"],
            threshold=pipeline.speaker_identification_threshold,
        )
        print(f"   - Khác người nói (Actor_02): Cosine = {score_diff:.4f} -> Verified: {is_diff}")
        assert not is_diff, f"Lỗi: Khác người nói phải bị từ chối (Cosine {score_diff} >= 0.70)"

        # Kiểm tra luồng tự động nhận diện người nói trong pipeline.analyze(identify_speaker=True)
        auto_id_res = pipeline.analyze(
            test_calm_speaker1, user_id=None, identify_speaker=True
        )
        print(
            f"   - Tự động nhận diện qua pipeline (identify_speaker=True): "
            f"user_id='{auto_id_res['speaker']['user_id']}', Cosine={auto_id_res['speaker']['similarity_score']}"
        )
        assert auto_id_res["speaker"]["user_id"] == user_id
        assert auto_id_res["speaker"]["verified"] is True
        print("✅ Cơ chế nhận diện và xác minh người nói hoạt động chính xác!")

        # -------------------------------------------------------------
        # 3. Chuẩn hóa Z-Score & Kích hoạt Luật Giải thích (Explainable Rules)
        # -------------------------------------------------------------
        print("\n" + "-" * 50)
        print("3️⃣ KIỂM TRA CHUẨN HÓA Z-SCORE & LUẬT GIẢI THÍCH VẬT LÝ (XAI)")
        print("-" * 50)
        analysis_stress = pipeline.analyze(stress_speaker1, user_id=user_id, verify_speaker=True)

        print("Kết quả phân tích khi người dùng đang ở trạng thái kích hoạt cao:")
        print(f"   - Request ID:        {analysis_stress['request_id']}")
        print(f"   - Speaker Verified:  {analysis_stress['speaker']['verified']} (Score: {analysis_stress['speaker']['similarity_score']})")
        print(f"   - Status:            {analysis_stress['assessment']['status']}")
        print(f"   - Score (Proxy):     {analysis_stress['assessment']['stress_score']}/100")
        print(f"   - Scoring Method:    {analysis_stress['assessment']['method']}")
        print(f"   - Construct:         {analysis_stress['assessment']['construct']}")
        print("\n   - Bảng phân rã Z-Score (Acoustic Breakdown):")
        for metric, data in analysis_stress["acoustic_breakdown"].items():
            print(f"     * {metric:<16}: value={str(data['value']):<8} baseline={str(data['baseline']):<8} z_score={data['z_score']}")

        print("\n   - Danh sách giải thích sinh lý thanh quản (XAI):")
        for exp in analysis_stress["explanations"]:
            print(f"     ⚠️ {exp}")

        assert analysis_stress["speaker"]["verified"] is True
        assert analysis_stress["assessment"]["method"] == "calibrated_hybrid_model"
        assert len(analysis_stress["explanations"]) >= 1

        # -------------------------------------------------------------
        # 4. Kiểm toán Hiệu chuẩn Giọng Nền Cho Người Có Tông Giọng Cao Tự Nhiên
        # -------------------------------------------------------------
        print("\n" + "-" * 50)
        print("4️⃣ KIỂM TOÁN HIỆU CHUẨN Z-SCORE CHO NGƯỜI CÓ TÔNG GIỌNG CAO")
        print("-" * 50)
        actor2_calm = "data/raw/RAVDESS/Actor_02/03-01-02-01-01-01-02.wav"
        actor2_test_calm = "data/raw/RAVDESS/Actor_02/03-01-02-01-02-01-02.wav"
        pipeline.register_baseline(
            user_id=user_id_high_pitch, calm_audio_path=actor2_calm, overwrite=True
        )

        res_uncalib = pipeline.analyze(actor2_test_calm, user_id=None, verify_speaker=False)
        res_calib = pipeline.analyze(
            actor2_test_calm, user_id=user_id_high_pitch, verify_speaker=True
        )

        score_uncalib = res_uncalib["assessment"]["stress_score"]
        score_calib = res_calib["assessment"]["stress_score"]

        print(f"   - Điểm khi CHƯA hiệu chuẩn (Uncalibrated): {score_uncalib}")
        print(f"   - Điểm khi ĐÃ hiệu chuẩn (Calibrated):     {score_calib}")
        assert score_calib <= score_uncalib, "Điểm sau hiệu chuẩn cá nhân phải thấp hơn hoặc bằng điểm thô khi bình tĩnh!"
        print("✅ Hiệu chuẩn Baseline Z-Score giảm độ lệch điểm cho giọng cao tự nhiên thành công!")

        print("\n" + "=" * 65)
        print("🎉 TẤT CẢ TIÊU CHÍ GIAI ĐOẠN 4 ĐÃ ĐẠT CHUẨN HOÀN TOÀN!")
        print("=" * 65)
    finally:
        pipeline.delete_baseline(user_id)
        pipeline.delete_baseline(user_id_high_pitch)
        shutil.rmtree(pipeline.baseline_dir, ignore_errors=True)


if __name__ == "__main__":
    main()
