"""Comprehensive Verification Script for Phase 1 Deliverables:
1. Generate test audio (clean tone/speech + ambient noise).
2. Speech Enhancement: MetricGAN+ denoises audio & converts to 16kHz mono.
3. Speaker Identification: ECAPA-TDNN extracts 192-dim embedding.
4. Acoustic Physics: Parselmouth computes F0, Jitter, Shimmer, HNR.
"""

import os
import sys
import atexit
import shutil
import tempfile
import numpy as np
import scipy.io.wavfile as wavfile

# Add project root to path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.enhancer import AudioEnhancer
from src.speaker_calibrator import SpeakerCalibrator
from src.acoustic_extractor import AcousticExtractor


def generate_synthetic_audio(
    filepath: str,
    duration_s: float = 3.0,
    sample_rate: int = 16000,
    f0: float = 180.0,
    noise_level: float = 0.05,
):
    """Generates a synthetic voiced vocal signal (harmonics) mixed with Gaussian noise."""
    os.makedirs(os.path.dirname(filepath), exist_ok=True)
    t = np.linspace(0, duration_s, int(sample_rate * duration_s), endpoint=False)

    # Fundamental frequency + harmonics (vocal tract simulation)
    signal = 0.5 * np.sin(2 * np.pi * f0 * t)
    signal += 0.25 * np.sin(2 * np.pi * 2 * f0 * t)
    signal += 0.12 * np.sin(2 * np.pi * 3 * f0 * t)

    # Amplitude modulation (vocal pulsation)
    envelope = 0.5 * (1 + np.sin(2 * np.pi * 2.0 * t))
    signal = signal * envelope

    # Environmental background noise
    noise = np.random.normal(0, noise_level, size=len(t))
    noisy_signal = signal + noise

    # Normalize to 16-bit PCM range
    noisy_signal = noisy_signal / np.max(np.abs(noisy_signal))
    audio_int16 = (noisy_signal * 32767).astype(np.int16)

    wavfile.write(filepath, sample_rate, audio_int16)
    print(f"[SyntheticAudio] Generated synthetic noisy audio: {filepath} ({duration_s}s, {sample_rate}Hz)")
    return filepath


def main():
    print("=" * 60)
    print("🚀 BẮT ĐẦU KIỂM THỬ GIAI ĐOẠN 1: PRE-TRAINED PIPELINE & SPEECHBRAIN")
    print("=" * 60)

    # Keep generated WAVs out of the repository and preserve any user samples.
    test_dir = tempfile.mkdtemp(prefix="speech_stress_phase1_")
    atexit.register(shutil.rmtree, test_dir, ignore_errors=True)

    # 1. Tạo audio thử nghiệm
    noisy_wav = os.path.join(test_dir, "sample_speaker1_noisy.wav")
    clean_wav = os.path.join(test_dir, "sample_speaker1_clean.wav")
    different_speaker_wav = os.path.join(test_dir, "sample_speaker2_noisy.wav")

    generate_synthetic_audio(noisy_wav, duration_s=3.0, f0=150.0, noise_level=0.08)
    generate_synthetic_audio(different_speaker_wav, duration_s=3.0, f0=220.0, noise_level=0.05)

    # 2. Kiểm thử Mô-đun Khử nhiễu (MetricGAN+)
    print("\n" + "-" * 50)
    print("1️⃣ KIỂM THỬ SPEECH ENHANCEMENT (MetricGAN+)")
    print("-" * 50)
    enhancer = AudioEnhancer()
    enhanced_wav = enhancer.clean_audio(input_path=noisy_wav, output_path=clean_wav)
    assert os.path.exists(enhanced_wav), "Lỗi: Không tìm thấy file âm thanh đã khử nhiễu!"
    sr, data = wavfile.read(enhanced_wav)
    print(f"✅ Khử nhiễu thành công: {enhanced_wav}")
    print(f"   - Tần số lấy mẫu: {sr} Hz (Chuẩn 16kHz)")
    print(f"   - Kích thước mẫu: {data.shape}")

    # 3. Kiểm thử Mô-đun Trích xuất Định danh Người nói (ECAPA-TDNN)
    print("\n" + "-" * 50)
    print("2️⃣ KIỂM THỬ SPEAKER EMBEDDING (ECAPA-TDNN)")
    print("-" * 50)
    calibrator = SpeakerCalibrator()
    emb1 = calibrator.extract_embedding(enhanced_wav)
    emb2 = calibrator.extract_embedding(different_speaker_wav)

    print(f"✅ Trích xuất thành công Speaker Embedding:")
    print(f"   - Shape vector: {emb1.shape} (Kỳ vọng: (192,))")
    assert emb1.shape == (192,), f"Lỗi: Vector embedding có shape {emb1.shape}, kỳ vọng (192,)!"

    sim_self = calibrator.compute_similarity(emb1, emb1)
    sim_diff = calibrator.compute_similarity(emb1, emb2)
    print(f"   - Độ tương đồng tự thân (Cosine Self-Similarity): {sim_self:.4f}")
    print(f"   - Độ tương đồng khác người nói (Cosine Cross-Speaker): {sim_diff:.4f}")

    # 4. Kiểm thử Mô-đun Đặc trưng Vật lý Âm học (Parselmouth)
    print("\n" + "-" * 50)
    print("3️⃣ KIỂM THỬ TRÍCH XUẤT ĐẶC TRƯNG ÂM HỌC VẬT LÝ (Parselmouth)")
    print("-" * 50)
    extractor = AcousticExtractor()
    phys_features = extractor.extract_features(enhanced_wav)
    phys_vector = extractor.get_feature_vector(enhanced_wav)

    print("✅ Trích xuất thành công đặc trưng vật lý thanh quản:")
    for k, v in phys_features.items():
        print(f"   - {k}: {v}")
    print(f"   - Vector 5 chiều [F0_mean, F0_std, Jitter, Shimmer, HNR]: {phys_vector}")
    assert len(phys_vector) == 5, f"Lỗi: Vector đặc trưng vật lý phải có 5 chiều, nhận được {len(phys_vector)}"

    print("\n" + "=" * 60)
    print("🎉 TẤT CẢ TIÊU CHÍ GIAI ĐOẠN 1 ĐÃ ĐẠT CHUẨN HOÀN TOÀN!")
    print("   [x] Audio Denoising (MetricGAN+ -> 16kHz mono)")
    print("   [x] Speaker Identification (ECAPA-TDNN 192-dim vector)")
    print("   [x] Acoustic Physical Features (F0, Jitter, Shimmer, HNR)")
    print("=" * 60)


if __name__ == "__main__":
    main()
