---
name: speech-stress-workflow
description: >-
  Workflow toàn diện sửa lỗi và tối ưu hóa hệ thống nhận diện âm thanh, cảm xúc và phân loại stress giọng nói (SpeechBrain, Wav2Vec2, MetricGAN+, FastAPI, Parselmouth).
  Sử dụng skill này khi cần kiểm tra, huấn luyện lại, hiệu chuẩn hoặc khắc phục lỗi trong pipeline phân loại stress và nhận diện cảm xúc.
---

# Quy Trình Sửa Lỗi & Tối Ưu Hệ Thống Nhận Diện Âm Thanh & Cảm Xúc (Speech Stress Workflow)

Hệ thống kết hợp mô hình học sâu Wav2Vec2 và các đặc trưng vật lý âm học ($F_0$, Jitter, Shimmer, HNR) cùng mô hình nhận diện cảm xúc (SpeechBrain Wav2Vec2 IEMOCAP).

---

## 1. Danh Mục Các Lỗi Đã Được Khắc Phục

| # | Vấn đề | Nguyên nhân | Giải pháp |
|---|---|---|---|
| **Lỗi #1** | Emotion IEMOCAP Domain Mismatch | IEMOCAP 4 lớp (`ang`, `hap`, `sad`, `neu`) lệch miền so với RAVDESS | Chuẩn hóa ánh xạ nhãn cảm xúc và đảm bảo tương thích đa miền |
| **Lỗi #2** | Stress Model Overfit & False Positive 45.8% | Mất cân bằng mẫu ($\text{normal}=192$, $\text{stress}=256$), quá ít epochs (2 epochs) | Áp dụng `class_weights: [1.33, 1.0]`, tăng lên 15 epochs, tối ưu ngưỡng |
| **Lỗi #3** | Mâu thuẫn `output_norm` | `output_norm: False` trong train config nhưng `True` trong checkpoint | Đặt đồng nhất `output_norm: True` trong cả train config và inference |
| **Lỗi #4** | Xung đột `save_path` | Thư mục checkpoint bị phân mảnh (`wav2vec2` vs `wav2vec2_checkpoint`) | Đồng bộ `save_path: !ref <save_folder>/wav2vec2` |
| **Lỗi #5 & #7** | Bất đồng bộ Pooling khi Inference | Train dùng Masked Mean Pooling, Inference dùng `mean(dim=1)` làm loãng padding | Bổ sung `_masked_mean_pool` chuẩn xác trong `src/stress_model.py` |
| **Lỗi #6** | Safeguard loại bỏ giọng nữ | $F_0 \le 220\text{ Hz}$ loại hầu hết giọng nữ ($200\text{--}300\text{ Hz}$) | Mở rộng $F_0 \in [75, 320]\text{ Hz}$ và kiểm soát $F_0\text{ std} \le 35\text{ Hz}$ |
| **P1--P4** | Vấn đề phụ | Thiếu weights, epoch ít, đọc lại file disk | Sửa trong `brain_trainer.py`, `hparams_train.yaml`, `pipeline.py` |

---

## 2. Quy Trình Sửa Đổi & Huấn Luyện (Workflow Steps)

### Bước 1 & 2: Đồng bộ `output_norm`, `save_path` và `class_weights`
File: `configs/hparams_train.yaml`
```yaml
number_of_epochs: 15
lr_patience: 3

wav2vec2: !new:speechbrain.integrations.huggingface.wav2vec2.Wav2Vec2
    source: !ref <wav2vec2_hub>
    save_path: !ref <save_folder>/wav2vec2
    output_norm: True
    freeze: True
    freeze_feature_extractor: True

class_weights: [1.33, 1.0]
```

### Bước 3: Đồng bộ Masked Mean Pooling khi Suy luận
File: `src/stress_model.py`
```python
def _masked_mean_pool(self, features: torch.Tensor, lengths: torch.Tensor) -> torch.Tensor:
    frame_lens = torch.round(lengths * features.shape[1]).long()
    frame_lens = frame_lens.clamp(min=1, max=features.shape[1])
    mask = (
        torch.arange(features.shape[1], device=features.device)[None, :]
        < frame_lens[:, None]
    )
    pooled = (features * mask.unsqueeze(-1)).sum(dim=1)
    return pooled / frame_lens.unsqueeze(-1)
```

### Bước 4: Mở rộng Acoustic Safeguard ($F_0: 75\text{--}320\text{ Hz}$)
File: `src/pipeline.py`
```python
f0_val = float(raw_metrics["f0_mean"])
f0_std_val = float(raw_metrics.get("f0_std", 0.0))
jitter_val = float(raw_metrics["jitter_percent"])
shimmer_val = float(raw_metrics["shimmer_percent"])
hnr_val = float(raw_metrics["hnr_db"])

is_pitch_calm = (75.0 <= f0_val <= 320.0) and (f0_std_val <= 35.0)
is_jitter_calm = 0.05 <= jitter_val <= 2.2
is_shimmer_calm = 0.10 <= shimmer_val <= 12.0
is_hnr_clean = hnr_val >= 10.0
```

### Bước 5: Ánh xạ và chuẩn hóa Emotion Recognition
File: `src/pipeline.py` (`_classify_emotion`)
Ánh xạ `iemocap_mapping: {"neu": "neutral", "ang": "angry", "hap": "happy", "sad": "sad"}`.

### Bước 6: Cân bằng hàm mất mát trong `StressBrain`
File: `src/brain_trainer.py`
```python
if hasattr(self.hparams, "class_weights") and self.hparams.class_weights:
    weights = torch.tensor(
        self.hparams.class_weights,
        dtype=predictions.dtype,
        device=self.device,
    )
    loss = torch.nn.functional.nll_loss(predictions, labels, weight=weights)
else:
    loss = self.hparams.compute_cost(predictions, labels)
```

### Bước 7 & 8: Tái tạo Manifest dữ liệu
```bash
.venv/bin/python data/prepare_manifest.py --raw_dir data/raw/RAVDESS --out_dir data/manifests
```

### Bước 9: Xóa Checkpoint cũ và Huấn luyện lại
```bash
rm -rf results/stress_wav2vec2/1986/save/CKPT+*
rm -rf results/stress_wav2vec2/1986/save/wav2vec2_checkpoint
.venv/bin/python train.py --hparams configs/hparams_train.yaml
```

### Bước 10: Kiểm thử tự động (Sanity & Regression Test)
File: `tests/test_inference_sanity.py`
```bash
.venv/bin/python -m pytest tests/test_inference_sanity.py -v
.venv/bin/python tests/test_phase5.py
```
Mục tiêu chất lượng:
- **Normal Recall** $\ge 75\%$
- **Stress Recall** $\ge 80\%$
- **Macro F1** $\ge 80\%$
