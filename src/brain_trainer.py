"""SpeechBrain trainer for the hybrid speech stress classifier."""

import torch
import speechbrain as sb


class StressBrain(sb.Brain):
    """Fine-tune Wav2Vec2 and fuse its pooled output with acoustic features."""

    def compute_forward(self, batch, stage):
        batch = batch.to(self.device)
        wavs, wav_lens = batch.sig
        acoustic = batch.acoustic_phys_tensor.data.float().to(self.device)

        features = self.modules.wav2vec2(wavs, wav_lens)
        frame_lens = torch.round(wav_lens * features.shape[1]).long()
        frame_lens = frame_lens.clamp(min=1, max=features.shape[1])
        mask = (
            torch.arange(features.shape[1], device=features.device)[None, :]
            < frame_lens[:, None]
        )
        pooled = (features * mask.unsqueeze(-1)).sum(dim=1)
        pooled = pooled / frame_lens.unsqueeze(-1)

        fused = torch.cat((pooled, acoustic), dim=-1)
        logits = self.modules.output_mlp(fused)
        return torch.log_softmax(logits, dim=-1)

    def compute_objectives(self, predictions, batch, stage):
        labels = batch.stress_label_encoded.data.long().view(-1)
        loss = self.hparams.compute_cost(predictions, labels)

        if stage != sb.Stage.TRAIN:
            predicted = predictions.argmax(dim=-1)
            self.correct += int((predicted == labels).sum().item())
            self.num_examples += int(labels.numel())
            if stage == sb.Stage.TEST:
                self.all_preds.extend(predicted.cpu().tolist())
                self.all_targets.extend(labels.cpu().tolist())
        return loss

    def on_stage_start(self, stage, epoch=None):
        if stage != sb.Stage.TRAIN:
            self.correct = 0
            self.num_examples = 0
            if stage == sb.Stage.TEST:
                self.all_preds = []
                self.all_targets = []

    def on_stage_end(self, stage, stage_loss, epoch=None):
        if stage == sb.Stage.TRAIN:
            self.train_loss = stage_loss
            return

        accuracy = self.correct / max(self.num_examples, 1)
        stats = {
            "loss": float(stage_loss),
            "accuracy": accuracy,
            "error_rate": 1.0 - accuracy,
        }
        self.last_metrics = stats

        if stage == sb.Stage.VALID:
            if stage_loss < self.best_valid_loss - self.hparams.lr_min_delta:
                self.best_valid_loss = stage_loss
                self.bad_valid_epochs = 0
            else:
                self.bad_valid_epochs += 1
                if self.bad_valid_epochs >= self.hparams.lr_patience:
                    for optimizer in self.optimizers_dict.values():
                        for group in optimizer.param_groups:
                            group["lr"] = max(
                                group["lr"] * self.hparams.lr_factor,
                                self.hparams.min_learning_rate,
                            )
                    self.bad_valid_epochs = 0

            learning_rates = {
                name: optimizer.param_groups[0]["lr"]
                for name, optimizer in self.optimizers_dict.items()
            }
            self.hparams.train_logger.log_stats(
                stats_meta={"epoch": epoch, **learning_rates},
                train_stats={"loss": self.train_loss},
                valid_stats=stats,
            )
            self.checkpointer.save_and_keep_only(
                meta={"error_rate": stats["error_rate"]},
                min_keys=["error_rate"],
            )
        elif stage == sb.Stage.TEST:
            self.hparams.train_logger.log_stats(
                stats_meta={},
                test_stats=stats,
            )

    def init_optimizers(self):
        """Keep separate learning rates for the pretrained encoder and head."""
        self.optimizers_dict = {
            "output_mlp": torch.optim.AdamW(
                self.modules.output_mlp.parameters(),
                lr=self.hparams.lr,
                weight_decay=self.hparams.weight_decay,
            ),
        }
        if not self.modules.wav2vec2.freeze:
            self.optimizers_dict["wav2vec2"] = torch.optim.AdamW(
                self.modules.wav2vec2.parameters(),
                lr=self.hparams.lr_wav2vec2,
                weight_decay=self.hparams.weight_decay,
            )
        if self.checkpointer is not None:
            for name, optimizer in self.optimizers_dict.items():
                self.checkpointer.add_recoverable(f"{name}_optimizer", optimizer)

        self.best_valid_loss = float("inf")
        self.bad_valid_epochs = 0
