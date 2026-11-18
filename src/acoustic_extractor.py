"""Acoustic Physical Feature Extraction Module.
Uses Praat-Parselmouth to compute physiological vocal biomarkers:
- F0 Mean & Std (Hz) - Fundamental frequency / pitch
- Jitter Local (%) - Cycle-to-cycle frequency variation (Micro-tremors)
- Shimmer Local (%) - Cycle-to-cycle amplitude variation
- HNR (dB) - Harmonics-to-Noise Ratio (Breathiness / vocal clarity)
"""

import os
from typing import Any, Dict, Optional

import numpy as np
import parselmouth
from parselmouth.praat import call


class AcousticExtractor:
    """Extracts domain-specific acoustic physics features with voiced-frame and quality checks."""

    def __init__(
        self,
        pitch_floor: float = 75.0,
        pitch_ceiling: float = 600.0,
        time_step: float = 0.01,
        min_voiced_frames: int = 5,
        min_voiced_fraction: float = 0.15,
    ):
        self.pitch_floor = pitch_floor
        self.pitch_ceiling = pitch_ceiling
        self.time_step = time_step
        self.min_voiced_frames = min_voiced_frames
        self.min_voiced_fraction = min_voiced_fraction

    def extract_features_with_quality(self, audio_path: str) -> Dict[str, Any]:
        """Extract acoustic features while preserving None for failed/unvoiced measurements.

        Never substitutes 0.0 for a failed Praat measurement in the raw `metrics` dictionary,
        preventing downstream logic from mistaking extraction failures for calm vocal folds.
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
        total_frames = int(len(f0_values))
        voiced_f0 = f0_values[f0_values > 0]
        voiced_frames = int(len(voiced_f0))
        voiced_fraction = float(voiced_frames / max(total_frames, 1))

        f0_mean: Optional[float] = None
        f0_std: Optional[float] = None
        if voiced_frames >= self.min_voiced_frames and voiced_fraction >= self.min_voiced_fraction:
            mean_val = float(np.mean(voiced_f0))
            std_val = float(np.std(voiced_f0))
            if np.isfinite(mean_val) and mean_val > 0 and np.isfinite(std_val):
                f0_mean = round(mean_val, 2)
                f0_std = round(std_val, 2)

        # 2. Extract PointProcess for Jitter & Shimmer
        jitter_percent: Optional[float] = None
        shimmer_percent: Optional[float] = None
        if f0_mean is not None:
            try:
                point_process = call(
                    sound,
                    "To PointProcess (periodic, cc)",
                    self.pitch_floor,
                    self.pitch_ceiling,
                )
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
                    if np.isfinite(jitter_local) and jitter_local > 0:
                        jitter_percent = round(float(jitter_local * 100.0), 4)
                except Exception:
                    jitter_percent = None

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
                    if np.isfinite(shimmer_local) and shimmer_local > 0:
                        shimmer_percent = round(float(shimmer_local * 100.0), 4)
                except Exception:
                    shimmer_percent = None
            except Exception:
                jitter_percent = None
                shimmer_percent = None

        # 3. Harmonics-to-Noise Ratio (HNR)
        hnr_db: Optional[float] = None
        if f0_mean is not None:
            try:
                harmonicity = sound.to_harmonicity(
                    time_step=self.time_step,
                    minimum_pitch=self.pitch_floor,
                )
                hnr_mean = call(harmonicity, "Get mean", 0.0, 0.0)
                if np.isfinite(hnr_mean) and hnr_mean > -100.0:
                    hnr_db = round(float(hnr_mean), 2)
            except Exception:
                hnr_db = None

        valid_flags = {
            "f0": f0_mean is not None and f0_std is not None,
            "jitter": jitter_percent is not None,
            "shimmer": shimmer_percent is not None,
            "hnr": hnr_db is not None,
        }
        all_valid = all(valid_flags.values())

        return {
            "metrics": {
                "f0_mean": f0_mean,
                "f0_std": f0_std,
                "jitter_percent": jitter_percent,
                "shimmer_percent": shimmer_percent,
                "hnr_db": hnr_db,
            },
            "quality": {
                "voiced_frames": voiced_frames,
                "total_frames": total_frames,
                "voiced_fraction": round(voiced_fraction, 4),
                "valid_flags": valid_flags,
                "all_valid": all_valid,
            },
        }

    def extract_features(self, audio_path: str, strict: bool = False) -> Dict[str, float]:
        """Extracts the 5 core physical features from a wav file.
        If `strict=True`, raises ValueError if any acoustic measurement failed.
        """
        result = self.extract_features_with_quality(audio_path)
        metrics = result["metrics"]
        quality = result["quality"]

        if strict and not quality["all_valid"]:
            missing = [k for k, v in quality["valid_flags"].items() if not v]
            raise ValueError(
                f"Insufficient acoustic quality; failed to measure: {', '.join(missing)} "
                f"(voiced_fraction={quality['voiced_fraction']:.2f})"
            )

        features = {
            "f0_mean": float(metrics["f0_mean"]) if metrics["f0_mean"] is not None else 0.0,
            "f0_std": float(metrics["f0_std"]) if metrics["f0_std"] is not None else 0.0,
            "jitter_percent": float(metrics["jitter_percent"]) if metrics["jitter_percent"] is not None else 0.0,
            "shimmer_percent": float(metrics["shimmer_percent"]) if metrics["shimmer_percent"] is not None else 0.0,
            "hnr_db": float(metrics["hnr_db"]) if metrics["hnr_db"] is not None else 0.0,
        }
        if not all(np.isfinite(value) for value in features.values()):
            raise ValueError(f"Non-finite acoustic feature produced for {audio_path}")
        return features

    def get_feature_vector(
        self, audio_path: str, features: Optional[Dict[str, float]] = None
    ) -> np.ndarray:
        """Returns 5-element array: [f0_mean, f0_std, jitter_percent, shimmer_percent, hnr_db]"""
        feats = features if features is not None else self.extract_features(audio_path, strict=True)
        return np.array(
            [
                float(feats["f0_mean"]),
                float(feats["f0_std"]),
                float(feats["jitter_percent"]),
                float(feats["shimmer_percent"]),
                float(feats["hnr_db"]),
            ],
            dtype=np.float32,
        )
