"""Speaker Identification & Baseline Calibrator Module.
Extracts 192-dimensional speaker embeddings using SpeechBrain ECAPA-TDNN
and computes cosine similarity for speaker verification and profiling.
"""

import os
import torch
import torchaudio
import numpy as np
from typing import Optional, Tuple

try:
    from speechbrain.inference.speaker import EncoderClassifier
except ImportError:
    from speechbrain.pretrained import EncoderClassifier


class SpeakerCalibrator:
    """Speaker recognition and baseline calibration using ECAPA-TDNN."""

    def __init__(
        self,
        model_source: str = "speechbrain/spkrec-ecapa-voxceleb",
        savedir: str = "pretrained_models/spkrec-ecapa-voxceleb",
        device: Optional[str] = None,
    ):
        if device is None:
            self.device = "cuda" if torch.cuda.is_available() else "cpu"
        else:
            self.device = device

        print(f"[SpeakerCalibrator] Initializing ECAPA-TDNN on device: {self.device}...")
        self.classifier = EncoderClassifier.from_hparams(
            source=model_source,
            savedir=savedir,
            run_opts={"device": self.device},
        )
        print("[SpeakerCalibrator] ECAPA-TDNN loaded successfully.")

    def extract_embedding(self, audio_path: str) -> np.ndarray:
        """Extracts 192-dimensional speaker embedding vector from a wav file.
        Returns:
            np.ndarray: 1D array of shape (192,)
        """
        if not os.path.exists(audio_path):
            raise FileNotFoundError(f"Audio file not found: {audio_path}")

        try:
            import soundfile as sf
            data, sr = sf.read(audio_path, dtype="float32")
            if data.ndim == 1:
                sig = torch.from_numpy(data).unsqueeze(0)
            else:
                sig = torch.from_numpy(data.T)
        except Exception:
            sig, sr = torchaudio.load(audio_path)
        # Resample to 16kHz if needed
        if sr != 16000:
            resampler = torchaudio.transforms.Resample(orig_freq=sr, new_freq=16000)
            sig = resampler(sig)
        if sig.shape[0] > 1:
            sig = torch.mean(sig, dim=0, keepdim=True)

        sig = sig.to(self.device)

        with torch.no_grad():
            embeddings = self.classifier.encode_batch(sig)
            # embeddings shape: [batch, 1, 192] -> squeeze to [192]
            emb_vector = embeddings.squeeze().cpu().numpy()

        return emb_vector

    def compute_similarity(self, emb1: np.ndarray, emb2: np.ndarray) -> float:
        """Calculates Cosine Similarity between two 192-dimensional embeddings."""
        norm1 = np.linalg.norm(emb1)
        norm2 = np.linalg.norm(emb2)
        if norm1 == 0 or norm2 == 0:
            return 0.0
        return float(np.dot(emb1, emb2) / (norm1 * norm2))

    def verify_speaker(
        self,
        audio_path: str,
        baseline_embedding: np.ndarray,
        threshold: float = 0.75,
    ) -> Tuple[bool, float]:
        """Checks if speaker in audio_path matches the baseline embedding."""
        current_emb = self.extract_embedding(audio_path)
        similarity = self.compute_similarity(current_emb, baseline_embedding)
        is_verified = bool(similarity >= threshold)
        return is_verified, similarity

    def find_best_match(
        self,
        audio_path: str,
        candidates: dict,
        threshold: float = 0.75,
    ) -> Tuple[Optional[str], float]:
        """Identifies the best matching registered speaker from {user_id: embedding} candidates."""
        if not candidates:
            return None, 0.0

        current_emb = self.extract_embedding(audio_path)
        best_user = None
        best_score = -1.0

        for user_id, embedding in candidates.items():
            emb = np.asarray(embedding)
            sim = self.compute_similarity(current_emb, emb)
            if sim > best_score:
                best_score = sim
                if sim >= threshold:
                    best_user = user_id

        return best_user, float(best_score)
