"""Kiểm thử hồi quy: Đảm bảo mô hình không bị thiên vị dự đoán 100% thành stress."""

import json
from pathlib import Path


def test_confusion_matrix_not_biased():
    metrics_path = Path("results/stress_wav2vec2/1986/test_metrics.json")
    assert metrics_path.exists(), "Không tìm thấy file test_metrics.json!"

    with open(metrics_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    # Kiểm tra tỷ lệ dự đoán lớp normal và recall từng lớp
    cm = data.get("confusion_matrix", [[0, 0], [0, 0]])
    tn, fp = cm[0][0], cm[0][1]
    fn, tp = cm[1][0], cm[1][1]
    total = tn + fp + fn + tp

    predicted_normal_ratio = (tn + fn) / max(total, 1)
    normal_recall = tn / max(tn + fp, 1)
    stress_recall = tp / max(fn + tp, 1)

    assert predicted_normal_ratio >= 0.30, (
        f"Mô hình vẫn bị lệch! Tỷ lệ dự đoán 'normal' chỉ đạt {predicted_normal_ratio:.1%} (< 30%)."
    )
    assert normal_recall >= 0.70, f"Normal Recall quá thấp: {normal_recall:.1%} (< 70%)."
    assert stress_recall >= 0.75, f"Stress Recall dưới ngưỡng: {stress_recall:.1%} (< 75%)."


if __name__ == "__main__":
    test_confusion_matrix_not_biased()
    print("✅ All sanity checks passed!")
