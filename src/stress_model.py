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

    @torch.inference_mode()
    def predict(self, waveform: torch.Tensor, acoustic_features: np.ndarray):
        """Return stress probability and predicted class from 16 kHz mono audio."""
        waveform = waveform.flatten().float().to(self.device)
        max_chunk = 16000 * 30  # 30 seconds
        if waveform.numel() > max_chunk:
            chunks = [
                waveform[i : i + max_chunk]
                for i in range(0, min(waveform.numel(), 16000 * 180), max_chunk)
                if waveform[i : i + max_chunk].numel() >= 16000 * 3
            ]
            pooled_list = []
            for chunk in chunks:
                w_in = chunk.unsqueeze(0)
                lens = torch.ones(1, dtype=torch.float32, device=self.device)
                feat = self.modules.wav2vec2(w_in, lens)
                pooled_list.append(feat.mean(dim=1))
            pooled = torch.stack(pooled_list, dim=0).mean(dim=0)
        else:
            w_in = waveform.unsqueeze(0)
            lengths = torch.ones(1, dtype=torch.float32, device=self.device)
            features = self.modules.wav2vec2(w_in, lengths)
            pooled = features.mean(dim=1)

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
        }
