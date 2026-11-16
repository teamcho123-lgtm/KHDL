"""Train and evaluate the hybrid Wav2Vec2 + acoustic stress classifier."""

import argparse
import json
import os

import speechbrain as sb
import torch
from hyperpyyaml import load_hyperpyyaml

from src.brain_trainer import StressBrain
from src.dataset import create_dynamic_dataloader, dataio_prep


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--hparams", default="configs/hparams_train.yaml")
    parser.add_argument("--device", default=None, help="cpu, cuda, or cuda:0")
    args = parser.parse_args()

    with open(args.hparams, "r", encoding="utf-8") as stream:
        hparams = load_hyperpyyaml(stream)

    sb.utils.seed_everything(int(hparams["seed"]))
    device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    os.makedirs(hparams["output_folder"], exist_ok=True)
    os.makedirs(hparams["save_folder"], exist_ok=True)

    train_data, valid_data, test_data, _ = dataio_prep(hparams)
    train_loader = create_dynamic_dataloader(
        train_data,
        max_batch_length=float(hparams["max_batch_length"]),
        num_buckets=int(hparams["num_buckets"]),
        shuffle=True,
        num_workers=int(hparams["num_workers"]),
    )
    valid_loader = create_dynamic_dataloader(
        valid_data,
        max_batch_length=float(hparams["max_batch_length"]),
        num_buckets=int(hparams["num_buckets"]),
        shuffle=False,
        num_workers=int(hparams["num_workers"]),
    )
    test_loader = create_dynamic_dataloader(
        test_data,
        max_batch_length=float(hparams["max_batch_length"]),
        num_buckets=int(hparams["num_buckets"]),
        shuffle=False,
        num_workers=int(hparams["num_workers"]),
    )

    brain = StressBrain(
        modules=hparams["modules"],
        hparams=hparams,
        run_opts={"device": device},
        checkpointer=hparams["checkpointer"],
    )
    brain.fit(
        epoch_counter=hparams["epoch_counter"],
        train_set=train_loader,
        valid_set=valid_loader,
    )

    from sklearn.metrics import classification_report, confusion_matrix, f1_score, accuracy_score, recall_score
    test_loss = brain.evaluate(test_loader, min_key="error_rate")
    preds = getattr(brain, "all_preds", [])
    targets = getattr(brain, "all_targets", [])

    acc = accuracy_score(targets, preds) if len(preds) > 0 else (1.0 - brain.last_metrics.get("error_rate", 0.0))
    macro_f1 = f1_score(targets, preds, average="macro", zero_division=0) if len(preds) > 0 else 0.0
    stress_recall = recall_score(targets, preds, pos_label=1, zero_division=0) if len(preds) > 0 else 0.0
    cm = confusion_matrix(targets, preds).tolist() if len(preds) > 0 else []
    target_names = ["normal", "stress"]

    metrics = {
        "test_loss": float(test_loss),
        "accuracy": float(acc),
        "macro_f1": float(macro_f1),
        "stress_recall": float(stress_recall),
        "confusion_matrix": cm,
        "target_names": target_names,
        **getattr(brain, "last_metrics", {}),
    }

    metrics_path = os.path.join(hparams["output_folder"], "test_metrics.json")
    with open(metrics_path, "w", encoding="utf-8") as stream:
        json.dump(metrics, stream, indent=2)

    report_path = os.path.join(hparams["output_folder"], "test_evaluation_report.json")
    with open(report_path, "w", encoding="utf-8") as stream:
        json.dump(metrics, stream, indent=2)

    print("\n" + "=" * 65)
    print("📊 BẢNG BÁO CÁO ĐÁNH GIÁ CHẤT LƯỢNG MÔ HÌNH (TEST EVALUATION REPORT)")
    print("=" * 65)
    print(f"   - Độ chính xác (Accuracy): {acc * 100:.2f}%")
    print(f"   - Macro F1-Score:           {macro_f1 * 100:.2f}%")
    print(f"   - Stress Class Recall:      {stress_recall * 100:.2f}%")
    print(f"   - Ma trận nhầm lẫn:         {cm}")
    if len(preds) > 0:
        print(classification_report(targets, preds, target_names=target_names, zero_division=0))
    print(f"Test metrics saved to {metrics_path} and {report_path}")


if __name__ == "__main__":
    main()
