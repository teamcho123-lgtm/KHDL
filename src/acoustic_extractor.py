"""Acoustic Physical Feature Extraction Module.
Uses Praat-Parselmouth to compute physiological vocal biomarkers:
- F0 Mean & Std (Hz) - Fundamental frequency / pitch
- Jitter Local (%) - Cycle-to-cycle frequency variation (Micro-tremors)
- Shimmer Local (%) - Cycle-to-cycle amplitude variation
- HNR (dB) - Harmonics-to-Noise Ratio (Breathiness / vocal clarity)
"""

import os
import parselmouth
from parselmouth.praat import call
import numpy as np
from typing import Dict, Any


class AcousticExtractor:
    """Extracts domain-specific acoustic physics features with voiced-frame safety checks."""

    def __init__(
        self,
        pitch_floor: float = 75.0,
        pitch_ceiling: float = 600.0,
        time_step: float = 0.01,
    ):
        self.pitch_floor = pitch_floor
        self.pitch_ceiling = pitch_ceiling
        self.time_step = time_step

    def extract_features(self, audio_path: str) -> Dict[str, float]:
        """Extracts the 5 core physical features from a wav file.
        Returns:
            Dict containing:
                - f0_mean: float (Hz)
                - f0_std: float (Hz)
                - jitter_percent: float (%)
                - shimmer_percent: float (%)
                - hnr_db: float (dB)
        """
        if not os.path.exists(audio_path):
            raise FileNotFoundError(f"Audio file not found: {audio_path}")

        sound = parselmouth.Sound(audio_path)
        samples = np.asarray(sound.values, dtype=np.float64)
        if samples.ndim == 2 and samples.shape[0] > 1:
            mono = samples.mean(axis=0)
            sound = parselmouth.Sound(mono, sampling_frequency=sound.sampling_frequency)
            samples = np.asarray(sound.values, dtype=np.float64)
        if sound.duration < 0.1 or samples.size == 0:
            raise ValueError("Audio is too short for acoustic feature extraction")
        if not np.isfinite(samples).all() or float(np.sqrt(np.mean(samples ** 2))) < 1e-8:
            raise ValueError("Audio is silent or contains invalid samples")

        # 1. Extract Pitch (F0)
        pitch = sound.to_pitch(
            time_step=self.time_step,
            pitch_floor=self.pitch_floor,
            pitch_ceiling=self.pitch_ceiling,
        )
        f0_values = pitch.selected_array["frequency"]
        voiced_f0 = f0_values[f0_values > 0]

        if len(voiced_f0) > 0:
            f0_mean = float(np.mean(voiced_f0))
            f0_std = float(np.std(voiced_f0))
        else:
            f0_mean = 0.0
            f0_std = 0.0

        # 2. Extract PointProcess for Jitter & Shimmer
        point_process = call(
            sound,
            "To PointProcess (periodic, cc)",
            self.pitch_floor,
            self.pitch_ceiling,
        )

        # Local Jitter (relative variation in frequency)
        try:
            jitter_local = call(
                point_process,
                "Get jitter (local)",
                0.0,
                0.0,
                0.0001,
                0.02,
                1.3,
            )
            # Convert to percentage
            jitter_percent = float(jitter_local * 100.0) if np.isfinite(jitter_local) else 0.0
        except Exception:
            jitter_percent = 0.0

        # Local Shimmer (relative variation in amplitude)
        try:
            shimmer_local = call(
                [sound, point_process],
                "Get shimmer (local)",
                0.0,
                0.0,
                0.0001,
                0.02,
                1.3,
                1.6,
            )
            shimmer_percent = float(shimmer_local * 100.0) if np.isfinite(shimmer_local) else 0.0
        except Exception:
            shimmer_percent = 0.0

        # 3. Harmonics-to-Noise Ratio (HNR)
        try:
            harmonicity = sound.to_harmonicity(
                time_step=self.time_step,
                minimum_pitch=self.pitch_floor,
            )
            hnr_mean = call(harmonicity, "Get mean", 0.0, 0.0)
            hnr_db = float(hnr_mean) if np.isfinite(hnr_mean) else 0.0
        except Exception:
            hnr_db = 0.0

        features = {
            "f0_mean": round(f0_mean, 2),
            "f0_std": round(f0_std, 2),
            "jitter_percent": round(jitter_percent, 4),
            "shimmer_percent": round(shimmer_percent, 4),
            "hnr_db": round(hnr_db, 2),
        }
        if not all(np.isfinite(value) for value in features.values()):
            raise ValueError(f"Non-finite acoustic feature produced for {audio_path}")
        return features

    def get_feature_vector(self, audio_path: str) -> np.ndarray:
        """Returns 5-element array: [f0_mean, f0_std, jitter_percent, shimmer_percent, hnr_db]"""
        feats = self.extract_features(audio_path)
        return np.array([
            feats["f0_mean"],
            feats["f0_std"],
            feats["jitter_percent"],
            feats["shimmer_percent"],
            feats["hnr_db"],
        ], dtype=np.float32)
