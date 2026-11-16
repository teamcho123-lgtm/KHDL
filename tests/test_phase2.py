"""Comprehensive Verification Script for Phase 2 Deliverables:
1. Manifests Verification (train.json, valid.json, test.json).
2. Speaker-Independent Split Audit (Zero overlap between Train, Valid, and Test speakers).
3. SpeechBrain DynamicItemDataset Integration.
4. DynamicBatchSampler Test (Batching by length, verifying tensor shapes).
"""

import os
import sys
import json
import atexit
import shutil
import tempfile
import torch
import speechbrain as sb

# Add project root to sys.path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.dataset import dataio_prep, create_dynamic_dataloader


def verify_manifests(manifest_dir: str = "data/manifests"):
    print("\n" + "-" * 50)
    print("1️⃣ KIỂM TRA TẬP TIN MANIFEST & SPEAKER-INDEPENDENT SPLIT")
    print("-" * 50)

    train_path = os.path.join(manifest_dir, "train.json")
    valid_path = os.path.join(manifest_dir, "valid.json")
    test_path = os.path.join(manifest_dir, "test.json")

    for p in [train_path, valid_path, test_path]:
        assert os.path.exists(p), f"Lỗi: Không tìm thấy file manifest tại {p}"

    with open(train_path, "r", encoding="utf-8") as f:
        train_manifest = json.load(f)
    with open(valid_path, "r", encoding="utf-8") as f:
        valid_manifest = json.load(f)
    with open(test_path, "r", encoding="utf-8") as f:
        test_manifest = json.load(f)

    print(f"✅ Đã tải các tập manifest:")
    print(f"   - Tập Train: {len(train_manifest)} mẫu")
    print(f"   - Tập Valid: {len(valid_manifest)} mẫu")
    print(f"   - Tập Test:  {len(test_manifest)} mẫu")

    assert len(train_manifest) > 0, "Lỗi: Tập train rỗng!"
    assert len(valid_manifest) > 0, "Lỗi: Tập valid rỗng!"
    assert len(test_manifest) > 0, "Lỗi: Tập test rỗng!"

    # Kiểm tra tính độc lập tuyệt đối của người nói (Speaker-Independent)
    train_speakers = {v["speaker_id"] for v in train_manifest.values()}
    valid_speakers = {v["speaker_id"] for v in valid_manifest.values()}
    test_speakers = {v["speaker_id"] for v in test_manifest.values()}

    print(f"\nPhân bổ người nói:")
    print(f"   - Train speakers ({len(train_speakers)}): {sorted(list(train_speakers))}")
    print(f"   - Valid speakers ({len(valid_speakers)}): {sorted(list(valid_speakers))}")
    print(f"   - Test speakers  ({len(test_speakers)}): {sorted(list(test_speakers))}")

    overlap_train_val = train_speakers.intersection(valid_speakers)
    overlap_train_test = train_speakers.intersection(test_speakers)
    overlap_val_test = valid_speakers.intersection(test_speakers)

    assert len(overlap_train_val) == 0, f"Lỗi rò rỉ dữ liệu giữa Train và Valid: {overlap_train_val}"
    assert len(overlap_train_test) == 0, f"Lỗi rò rỉ dữ liệu giữa Train và Test: {overlap_train_test}"
    assert len(overlap_val_test) == 0, f"Lỗi rò rỉ dữ liệu giữa Valid và Test: {overlap_val_test}"
    print("✅ Kiểm toán Speaker-Independent: KHÔNG CÓ BẤT KỲ RÒ RỈ NGƯỜI NÓI NÀO (Zero Speaker Leakage)!")


def verify_dynamic_dataset():
    print("\n" + "-" * 50)
    print("2️⃣ KIỂM TRA SPEECHBRAIN DYNAMIC ITEM DATASET & PIPELINE")
    print("-" * 50)

    save_folder = tempfile.mkdtemp(prefix="speech_stress_phase2_")
    atexit.register(shutil.rmtree, save_folder, ignore_errors=True)
    hparams = {
        "train_json": "data/manifests/train.json",
        "valid_json": "data/manifests/valid.json",
        "test_json": "data/manifests/test.json",
        "save_folder": save_folder,
    }

    train_data, valid_data, test_data, label_encoder = dataio_prep(hparams)

    print(f"✅ DynamicItemDataset khởi tạo thành công:")
    print(f"   - Train dataset: {len(train_data)} items")
    print(f"   - Label encoder mapping: {label_encoder.lab2ind}")

    # Lấy thử 1 phần tử
    first_item = train_data[0]
    print(f"\nKiểm tra 1 item đơn:")
    print(f"   - ID: {first_item['id']}")
    print(f"   - Sóng âm shape: {first_item['sig'].shape} (dtype: {first_item['sig'].dtype})")
    print(f"   - Acoustic Physics Tensor: {first_item['acoustic_phys_tensor'].shape} (Values: {first_item['acoustic_phys_tensor'].tolist()})")
    print(f"   - Stress Label Encoded: {first_item['stress_label_encoded'].item()}")

    assert first_item["sig"].ndim == 1, "Sóng âm phải là 1D tensor [samples]"
    assert first_item["acoustic_phys_tensor"].shape == (5,), "Vector vật lý phải có đúng 5 chiều!"

    return train_data


def verify_dynamic_batching(train_data):
    print("\n" + "-" * 50)
    print("3️⃣ KIỂM TRA SPEECHBRAIN DYNAMIC BATCH SAMPLER")
    print("-" * 50)

    # Gom các câu âm thanh theo độ dài tương đương lên tới 25 giây/batch
    batch_loader = create_dynamic_dataloader(
        train_data,
        max_batch_length=25.0,
        num_buckets=10,
        shuffle=True,
    )

    print("✅ Khởi tạo DynamicBatchSampler thành công. Đang nạp thử 1 batch...")
    sample_batch = next(iter(batch_loader))

    wavs, lens = sample_batch.sig
    acoustic_feats = sample_batch.acoustic_phys_tensor.data
    labels = sample_batch.stress_label_encoded.data

    print(f"\nThông tin Batch mẫu thu được:")
    print(f"   - Số lượng mẫu trong batch (Batch size động): {wavs.shape[0]}")
    print(f"   - Shape sóng âm batch (wavs): {wavs.shape} [Batch_Size, Max_Time_Samples]")
    print(f"   - Relative lengths (lens): {lens.shape} -> min: {lens.min().item():.3f}, max: {lens.max().item():.3f}")
    print(f"   - Shape đặc trưng vật lý: {acoustic_feats.shape} [Batch_Size, 5]")
    print(f"   - Shape nhãn lớp: {labels.shape} [Batch_Size, 1]")

    assert wavs.ndim == 2, "Wavs batch phải có shape [Batch, Time]"
    assert acoustic_feats.shape == (wavs.shape[0], 5), "Vector acoustic_phys_tensor phải khớp kích thước batch và 5 chiều!"
    assert labels.shape == (wavs.shape[0], 1), "Nhãn lớp phải khớp kích thước batch!"

    print("\n" + "=" * 60)
    print("🎉 TẤT CẢ TIÊU CHÍ GIAI ĐOẠN 2 ĐÃ ĐẠT CHUẨN HOÀN TOÀN!")
    print("   [x] Speaker-Independent Split Audit Passed")
    print("   [x] Pre-computed Acoustic Physics (5-dim) in Manifests")
    print("   [x] SpeechBrain DynamicItemDataset & DynamicBatchSampler")
    print("=" * 60)


if __name__ == "__main__":
    print("=" * 60)
    print("🚀 BẮT ĐẦU KIỂM THỬ GIAI ĐOẠN 2: DỮ LIỆU & DYNAMIC BATCHING")
    print("=" * 60)
    verify_manifests()
    train_dataset = verify_dynamic_dataset()
    verify_dynamic_batching(train_dataset)
