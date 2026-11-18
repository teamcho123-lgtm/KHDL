"""Load a trained hybrid classifier for inference when a checkpoint exists."""

import glob
import os
from pathlib import Path

import numpy as np
import torch
from hyperpyyaml import load_hyperpyyaml
from speechbrain.dataio.encoder import CategoricalEncoder


class StressModel:
    def __init__(self, hparams_path: str, checkpoint_dir: str, device: str):
        checkpoint_files = glob.glob(os.path.join(checkpoint_dir, "CKPT+*"))
        if not checkpoint_files:
            raise FileNotFoundError(f"No SpeechBrain checkpoint found in {checkpoint_dir}")

        with open(hparams_path, "r", encoding="utf-8") as stream:
            checkpoint_dir = str(Path(checkpoint_dir).resolve())
            hparams = load_hyperpyyaml(
                stream,
                overrides={
                    "output_folder": str(Path(checkpoint_dir).parent),
                    "save_folder": checkpoint_dir,
                },
            )
        self.device = torch.device(device)
        self.modules = torch.nn.ModuleDict(hparams["modules"])
        self.checkpointer = hparams["checkpointer"]
        self.checkpointer.recover_if_possible(min_key="error_rate")
        self.modules.to(self.device).eval()

        normalization_path = os.path.join(checkpoint_dir, "acoustic_normalization.npz")
        with np.load(normalization_path, allow_pickle=False) as data:
            self.acoustic_mean = data["mean"].astype(np.float32)
            self.acoustic_std = data["std"].astype(np.float32)

        encoder_path = os.path.join(checkpoint_dir, "label_encoder.txt")
        encoder = CategoricalEncoder.from_saved(encoder_path)
        if set(encoder.lab2ind) != {"normal", "stress"}:
            raise ValueError("Checkpoint label encoder must define normal and stress")
        self.stress_index = int(encoder.lab2ind["stress"])

    def _masked_mean_pool(self, features: torch.Tensor, lengths: torch.Tensor) -> torch.Tensor:
        """Calculate masked mean pooling matching brain_trainer.py exactly."""
        frame_lens = torch.round(lengths * features.shape[1]).long()
        frame_lens = frame_lens.clamp(min=1, max=features.shape[1])
        mask = (
            torch.arange(features.shape[1], device=features.device)[None, :]
            < frame_lens[:, None]
        )
        pooled = (features * mask.unsqueeze(-1)).sum(dim=1)
        return pooled / frame_lens.unsqueeze(-1)

    @torch.inference_mode()
    def predict(
        self,
        waveform: torch.Tensor,
        acoustic_features: np.ndarray,
        sample_rate: int = 16000,
    ):
        """Return arousal/stress proxy probability and predicted class from 16 kHz mono audio.

        Processes the entire waveform in 30-second windows with duration-weighted pooling
        so no trailing audio within the configured duration limit is truncated or over-weighted.
        """
        waveform = waveform.flatten().float().to(self.device)
        total_samples = int(waveform.numel())
        if total_samples == 0:
            raise ValueError("Cannot run StressModel on an empty waveform")

        max_chunk = int(sample_rate * 30)  # 30-second window
        min_tail = int(sample_rate * 1)    # 1-second minimum standalone tail

        if total_samples > max_chunk:
            raw_chunks = [
                waveform[i : i + max_chunk]
                for i in range(0, total_samples, max_chunk)
            ]
            # Merge a very short trailing segment (< 1s) into the preceding chunk
            if len(raw_chunks) > 1 and raw_chunks[-1].numel() < min_tail:
                raw_chunks[-2] = torch.cat((raw_chunks[-2], raw_chunks[-1]), dim=0)
                raw_chunks.pop()

            pooled_list = []
            weights = []
            for chunk in raw_chunks:
                w_in = chunk.unsqueeze(0)
                lens = torch.ones(1, dtype=torch.float32, device=self.device)
                feat = self.modules.wav2vec2(w_in, lens)
                pooled_list.append(self._masked_mean_pool(feat, lens))
                weights.append(float(chunk.numel()))

            weight_tensor = torch.tensor(weights, dtype=torch.float32, device=self.device)
            weight_tensor = (weight_tensor / weight_tensor.sum()).view(-1, 1, 1)
            stacked = torch.stack(pooled_list, dim=0)
            pooled = (stacked * weight_tensor).sum(dim=0)
            chunks_used = len(raw_chunks)
        else:
            w_in = waveform.unsqueeze(0)
            lengths = torch.ones(1, dtype=torch.float32, device=self.device)
            features = self.modules.wav2vec2(w_in, lengths)
            pooled = self._masked_mean_pool(features, lengths)
            chunks_used = 1

        normalized_acoustic = (
            np.asarray(acoustic_features, dtype=np.float32).flatten() - self.acoustic_mean
        ) / self.acoustic_std
        acoustic = torch.from_numpy(normalized_acoustic).to(self.device).unsqueeze(0)

        logits = self.modules.output_mlp(torch.cat((pooled, acoustic), dim=-1))
        probabilities = torch.softmax(logits, dim=-1)[0]
        prediction = int(probabilities.argmax().item())
        return {
            "stress_probability": float(probabilities[self.stress_index].item()),
            "confidence": float(probabilities[prediction].item()),
            "predicted_label": "stress" if prediction == self.stress_index else "normal",
            "analyzed_duration_seconds": round(total_samples / float(sample_rate), 2),
            "chunks_used": chunks_used,
        }
