"""Audio analysis, baseline management, and optional model inference."""

import logging
import os
import re
import tempfile
import time
import uuid
from pathlib import Path
from typing import Any, Dict, Optional

import numpy as np
import soundfile as sf
import torch
import torchaudio
import yaml

from src.acoustic_extractor import AcousticExtractor
from src.enhancer import AudioEnhancer
from src.speaker_calibrator import SpeakerCalibrator
from src.stress_model import StressModel

logger = logging.getLogger(__name__)


class StressAnalysisPipeline:
    """Produce emotion predictions and calibrated vocal stress indicators."""

    def __init__(
        self,
        baseline_dir: str = "data/baseline_profiles",
        device: Optional[str] = None,
        sample_rate: int = 16000,
        metricgan_hub: str = "speechbrain/metricgan-plus-voicebank",
        metricgan_savedir: str = "pretrained_models/metricgan-plus-voicebank",
        ecapa_hub: str = "speechbrain/spkrec-ecapa-voxceleb",
        ecapa_savedir: str = "pretrained_models/spkrec-ecapa-voxceleb",
        emotion_hub: str = "speechbrain/emotion-recognition-wav2vec2-IEMOCAP",
        emotion_savedir: str = "pretrained_models/emotion-recognition-wav2vec2-IEMOCAP",
        enable_emotion: bool = True,
        emotion_pymodule_file: str = "custom_interface.py",
        emotion_classname: str = "CustomEncoderWav2vec2Classifier",
        stress_hparams: Optional[str] = None,
        stress_checkpoint_dir: Optional[str] = None,
        speaker_similarity_threshold: float = 0.75,
        stress_threshold: float = 50.0,
        high_stress_threshold: float = 70.0,
        zscore_alert_threshold: float = 1.5,
        max_audio_seconds: float = 30.0,
    ):
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.sample_rate = int(sample_rate)
        self.baseline_dir = os.path.abspath(baseline_dir)
        os.makedirs(self.baseline_dir, exist_ok=True)
        self.speaker_similarity_threshold = float(speaker_similarity_threshold)
        self.stress_threshold = float(stress_threshold)
        self.high_stress_threshold = float(high_stress_threshold)
        self.zscore_alert_threshold = float(zscore_alert_threshold)
        self.max_audio_seconds = float(max_audio_seconds)

        self.enhancer = AudioEnhancer(
            model_source=metricgan_hub,
            savedir=metricgan_savedir,
            device=self.device,
        )
        self.calibrator = SpeakerCalibrator(
            model_source=ecapa_hub,
            savedir=ecapa_savedir,
            device=self.device,
        )
        self.acoustic_extractor = AcousticExtractor()

        self.emotion_model = None
        self.emotion_load_error = None
        if enable_emotion:
            try:
                from speechbrain.inference.interfaces import foreign_class

                self.emotion_model = foreign_class(
                    source=emotion_hub,
                    savedir=emotion_savedir,
                    pymodule_file=emotion_pymodule_file,
                    classname=emotion_classname,
                    run_opts={"device": self.device},
                )
            except Exception as exc:
                self.emotion_load_error = str(exc)
                logger.exception("Could not load the configured emotion model")

        self.stress_model = None
        if stress_hparams and stress_checkpoint_dir:
            try:
                self.stress_model = StressModel(
                    hparams_path=stress_hparams,
                    checkpoint_dir=stress_checkpoint_dir,
                    device=self.device,
                )
            except FileNotFoundError:
                logger.info("No trained stress checkpoint is available yet")

    @classmethod
    def from_config(cls, config_path: str):
        """Build the pipeline from the inference YAML with repo-relative paths."""
        config_path = Path(config_path).resolve()
        root = config_path.parent.parent
        with config_path.open("r", encoding="utf-8") as stream:
            config = yaml.safe_load(stream) or {}

        def path_value(key):
            value = config.get(key)
            if value is None:
                return None
            path = Path(value)
            return str(path if path.is_absolute() else (root / path).resolve())

        return cls(
            baseline_dir=path_value("baseline_dir"),
            device=config.get("device"),
            sample_rate=config.get("sample_rate", 16000),
            metricgan_hub=config.get("metricgan_hub"),
            metricgan_savedir=path_value("metricgan_savedir"),
            ecapa_hub=config.get("ecapa_hub"),
            ecapa_savedir=path_value("ecapa_savedir"),
            emotion_hub=config.get("emotion_hub"),
            emotion_savedir=path_value("emotion_savedir"),
            enable_emotion=config.get("enable_emotion", True),
            emotion_pymodule_file=config.get(
                "emotion_pymodule_file", "custom_interface.py"
            ),
            emotion_classname=config.get(
                "emotion_classname", "CustomEncoderWav2vec2Classifier"
            ),
            stress_hparams=path_value("stress_hparams"),
            stress_checkpoint_dir=path_value("stress_checkpoint_dir"),
            speaker_similarity_threshold=config.get(
                "speaker_similarity_threshold", 0.75
            ),
            stress_threshold=config.get("stress_threshold", 50.0),
            high_stress_threshold=config.get("high_stress_threshold", 70.0),
            zscore_alert_threshold=config.get("zscore_alert_threshold", 1.5),
            max_audio_seconds=config.get("max_audio_seconds", 30.0),
        )

    @property
    def ready(self):
        return True

    def _read_audio_universal(self, audio_path: str):
        """Read any audio file (WAV, FLAC, OGG, MP3, AAC, M4A, WebM) into float32 mono."""
        if not os.path.isfile(audio_path):
            raise FileNotFoundError(f"Audio file not found: {audio_path}")
        try:
            audio, sample_rate = sf.read(
                audio_path, dtype="float32", always_2d=True
            )
            mono = audio.mean(axis=1)
            is_native = True
        except Exception:
            try:
                import av
                container = av.open(audio_path)
                stream = container.streams.audio[0]
                frames = [f.to_ndarray() for f in container.decode(stream)]
                if not frames:
                    raise ValueError("Audio file contains no decodable audio frames")
                audio = np.concatenate(frames, axis=1)
                mono = audio.mean(axis=0).astype(np.float32)
                sample_rate = stream.rate
                is_native = False
            except Exception as av_exc:
                raise ValueError(f"Unsupported or invalid audio file: {av_exc}") from av_exc
        return mono, sample_rate, is_native

    def _validate_and_prepare_audio(self, audio_path: str, min_seconds: float = 0.5):
        mono, sample_rate, is_native = self._read_audio_universal(audio_path)
        duration = len(mono) / float(sample_rate)
        if duration < min_seconds:
            raise ValueError(f"Audio must be at least {min_seconds:g} seconds long")
        if duration > self.max_audio_seconds:
            raise ValueError(
                f"Audio exceeds the {self.max_audio_seconds:g} second limit ({duration:.1f}s > {self.max_audio_seconds:g}s)"
            )
        if mono.size == 0 or not np.isfinite(mono).all():
            raise ValueError("Audio is empty or contains invalid samples")
        if float(np.sqrt(np.mean(np.square(mono)))) < 1e-6:
            raise ValueError("Audio is silent")

        waveform = torch.from_numpy(mono.copy())
        if sample_rate != self.sample_rate:
            waveform = torchaudio.functional.resample(
                waveform, orig_freq=sample_rate, new_freq=self.sample_rate
            )

        temp_wav_path = None
        if not is_native or sample_rate != self.sample_rate:
            descriptor, temp_wav_path = tempfile.mkstemp(
                prefix="standard_16k_", suffix=".wav"
            )
            os.close(descriptor)
            sf.write(temp_wav_path, waveform.numpy(), self.sample_rate)
            effective_wav_path = temp_wav_path
        else:
            effective_wav_path = audio_path

        return waveform, effective_wav_path, temp_wav_path

    def _validate_audio(self, audio_path: str, min_seconds: float = 0.5):
        waveform, _, temp_wav = self._validate_and_prepare_audio(audio_path, min_seconds)
        if temp_wav and os.path.exists(temp_wav):
            os.remove(temp_wav)
        return waveform

    @staticmethod
    def _validate_user_id(user_id: str):
        if not isinstance(user_id, str) or not re.fullmatch(
            r"[A-Za-z0-9][A-Za-z0-9_-]{0,63}", user_id
        ):
            raise ValueError(
                "user_id must be 1-64 characters using letters, numbers, '_' or '-'"
            )

    def register_baseline(self, user_id: str, calm_audio_path: str) -> Dict[str, Any]:
        """Register a vocal baseline for a stable, non-silent recording."""
        self._validate_user_id(user_id)
        waveform, effective_wav, temp_wav = self._validate_and_prepare_audio(calm_audio_path, min_seconds=1.0)
        try:
            speaker_emb = self.calibrator.extract_embedding(effective_wav)
            acoustic_feats = self.acoustic_extractor.extract_features(effective_wav)
            if acoustic_feats["f0_mean"] <= 0:
                raise ValueError("Baseline recording does not contain enough voiced speech")

            baseline_profile = {
                "speaker_embedding": speaker_emb.astype(np.float32),
                "f0_mean": np.float32(acoustic_feats["f0_mean"]),
                "f0_std": np.float32(max(acoustic_feats["f0_std"], 15.0)),
                "jitter_mean": np.float32(acoustic_feats["jitter_percent"]),
                "jitter_std": np.float32(max(acoustic_feats["jitter_percent"] * 0.35, 0.2)),
                "shimmer_mean": np.float32(acoustic_feats["shimmer_percent"]),
                "shimmer_std": np.float32(max(acoustic_feats["shimmer_percent"] * 0.35, 0.5)),
                "hnr_mean": np.float32(acoustic_feats["hnr_db"]),
                "hnr_std": np.float32(max(abs(acoustic_feats["hnr_db"]) * 0.2, 2.0)),
            }

            profile_path = os.path.join(self.baseline_dir, f"{user_id}_profile.npz")
            descriptor, temporary_path = tempfile.mkstemp(
                prefix=f".{user_id}_", suffix=".npz", dir=self.baseline_dir
            )
            os.close(descriptor)
            try:
                np.savez(temporary_path, **baseline_profile)
                os.replace(temporary_path, profile_path)
            finally:
                if os.path.exists(temporary_path):
                    os.remove(temporary_path)

            return {
                "user_id": user_id,
                "profile_saved": True,
                "baseline_features": acoustic_feats,
            }
        finally:
            if temp_wav and os.path.exists(temp_wav):
                os.remove(temp_wav)

    def load_baseline(self, user_id: str) -> Optional[Dict[str, Any]]:
        self._validate_user_id(user_id)
        profile_path = os.path.join(self.baseline_dir, f"{user_id}_profile.npz")
        if not os.path.isfile(profile_path):
            return None
        with np.load(profile_path, allow_pickle=False) as data:
            profile = {key: data[key] for key in data.files}
        required = {
            "speaker_embedding",
            "f0_mean",
            "f0_std",
            "jitter_mean",
            "jitter_std",
            "shimmer_mean",
            "shimmer_std",
            "hnr_mean",
            "hnr_std",
        }
        if not required.issubset(profile):
            raise ValueError(f"Baseline profile for {user_id} is incomplete")
        embedding = np.asarray(profile["speaker_embedding"])
        if embedding.shape != (192,) or not np.isfinite(embedding).all():
            raise ValueError(f"Baseline profile for {user_id} has an invalid speaker embedding")
        for key in required - {"speaker_embedding"}:
            value = np.asarray(profile[key])
            if value.size != 1 or not np.isfinite(value).all():
                raise ValueError(f"Baseline profile for {user_id} has invalid {key}")
            profile[key] = value.reshape(()).item()
        if any(float(profile[key]) <= 0 for key in ("f0_std", "jitter_std", "shimmer_std", "hnr_std")):
            raise ValueError(f"Baseline profile for {user_id} has invalid deviation values")
        return profile

    def _classify_emotion(self, audio_path: str):
        if self.emotion_model is None:
            return None
        probabilities, _, _, labels = self.emotion_model.classify_file(audio_path)
        probabilities = probabilities.reshape(-1).cpu()
        label = labels[0] if labels else None
        label_encoder = getattr(self.emotion_model.hparams, "label_encoder", None)
        index_to_label = getattr(label_encoder, "ind2lab", {})
        probability_labels = [
            str(index_to_label.get(index, index)) for index in range(len(probabilities))
        ]
        return {
            "label": str(label) if label is not None else None,
            "confidence": round(float(probabilities.max().item()), 4),
            "probabilities": {
                name: round(float(probability.item()), 4)
                for name, probability in zip(probability_labels, probabilities)
            },
        }

    def analyze(
        self,
        audio_path: str,
        user_id: Optional[str] = None,
        verify_speaker: bool = True,
    ) -> Dict[str, Any]:
        """Analyze a speech recording without overstating uncalibrated results."""
        start_time = time.time()
        request_id = f"req_{uuid.uuid4().hex[:8]}"
        waveform, effective_wav, temp_wav = self._validate_and_prepare_audio(audio_path)
        try:
            current_feats = self.acoustic_extractor.extract_features(effective_wav)
            if current_feats["f0_mean"] <= 0:
                raise ValueError("Audio does not contain enough voiced speech")

            speaker_info = {
                "user_id": user_id or "anonymous",
                "verified": None,
                "similarity_score": None,
            }
            baseline = None
            if user_id:
                self._validate_user_id(user_id)
                baseline = self.load_baseline(user_id)

            if baseline is not None and verify_speaker:
                verified, similarity = self.calibrator.verify_speaker(
                    effective_wav,
                    baseline["speaker_embedding"],
                    threshold=self.speaker_similarity_threshold,
                )
                speaker_info["verified"] = verified
                speaker_info["similarity_score"] = round(similarity, 3)

            emotion = None
            if self.emotion_model is not None:
                try:
                    emotion = self._classify_emotion(effective_wav)
                except Exception:
                    logger.exception("Emotion inference failed for request %s", request_id)

            if baseline is not None and verify_speaker and speaker_info["verified"] is False:
                baseline = None

            explanations = []
            breakdown = {}
            z_scores = []
            if baseline is not None:
                metric_specs = [
                    ("f0_mean_hz", "f0_mean", "F0", 1),
                    ("jitter_percent", "jitter_mean", "Jitter", 1),
                    ("shimmer_percent", "shimmer_mean", "Shimmer", 1),
                    ("hnr_db", "hnr_mean", "HNR", -1),
                ]
                std_keys = {
                    "f0_mean_hz": "f0_std",
                    "jitter_percent": "jitter_std",
                    "shimmer_percent": "shimmer_std",
                    "hnr_db": "hnr_std",
                }
                for output_key, baseline_key, label, direction in metric_specs:
                    value_key = "f0_mean" if output_key == "f0_mean_hz" else output_key
                    value = float(current_feats[value_key])
                    center = float(baseline[baseline_key])
                    deviation = max(float(baseline[std_keys[output_key]]), 1e-6)
                    z_value = (value - center) / deviation
                    breakdown[output_key] = {
                        "value": value,
                        "baseline": round(center, 3),
                        "z_score": round(z_value, 3),
                    }
                    if direction * z_value >= self.zscore_alert_threshold:
                        if output_key == "f0_mean_hz":
                            explanations.append(
                                f"Cao độ giọng nói (F0) tăng vọt +{z_value:.2f} SD so với trạng thái bình tĩnh (Dây thanh căng cứng)."
                            )
                        elif output_key == "jitter_percent":
                            explanations.append(
                                f"Độ rung tần số chu kỳ (Jitter) vượt ngưỡng ổn định (+{z_value:.2f} SD, dấu hiệu rung giọng vi mô/Microtremor)."
                            )
                        elif output_key == "shimmer_percent":
                            explanations.append(
                                f"Biến thiên biên độ vi mô (Shimmer) bất thường (+{z_value:.2f} SD, áp lực dưới thanh môn không ổn định)."
                            )
                        elif output_key == "hnr_db":
                            explanations.append(
                                f"Tỷ lệ hài trên nhiễu (HNR) sụt giảm mạnh ({z_value:.2f} SD, dấu hiệu hụt hơi hoặc suy giảm độ trong thanh âm)."
                            )
                    z_scores.append(direction * z_value)
            else:
                breakdown = {
                    "f0_mean_hz": {"value": current_feats["f0_mean"], "baseline": None, "z_score": None},
                    "jitter_percent": {"value": current_feats["jitter_percent"], "baseline": None, "z_score": None},
                    "shimmer_percent": {"value": current_feats["shimmer_percent"], "baseline": None, "z_score": None},
                    "hnr_db": {"value": current_feats["hnr_db"], "baseline": None, "z_score": None},
                }

            if self.stress_model is not None:
                prediction = self.stress_model.predict(
                    waveform, self.acoustic_extractor.get_feature_vector(effective_wav)
                )
                model_score = prediction["stress_probability"] * 100.0
                confidence = round(prediction["confidence"], 3)

                if baseline is not None and speaker_info.get("verified") is not False:
                    # Calibrated Fusion: Adjust model score with personal baseline Z-scores
                    # Prevents false positives for users who naturally speak at high pitch or loud intensity
                    avg_z = float(np.mean(z_scores)) if z_scores else 0.0
                    if avg_z <= 1.0:
                        # Acoustic metrics within normal calm range: apply baseline dampening to reduce false positives
                        dampening = 0.85 + 0.15 * max(-1.0, min(1.0, avg_z))
                        calibrated_score = model_score * dampening
                    else:
                        # Significant acoustic divergence: elevate stress score proportionally
                        z_boost = 100.0 / (1.0 + np.exp(-1.5 * (avg_z - 1.5)))
                        calibrated_score = 0.50 * model_score + 0.50 * z_boost
                    stress_score = round(float(np.clip(calibrated_score, 0.0, 100.0)), 1)
                    scoring_method = "calibrated_hybrid_model"
                    match_info = (
                        f" (Độ khớp giọng {speaker_info['similarity_score']})"
                        if speaker_info.get("similarity_score") is not None
                        else ""
                    )
                    explanations.append(
                        f"Đã hiệu chuẩn Z-Score với Baseline cá nhân của '{user_id}'{match_info}."
                    )
                else:
                    # General Population Physiological Gating (Acoustic Safeguard)
                    # When no personal baseline is registered, protect against neural false alarms
                    # by checking universal vocal fold physiological indicators.
                    f0_val = float(current_feats["f0_mean"])
                    jitter_val = float(current_feats["jitter_percent"])
                    shimmer_val = float(current_feats["shimmer_percent"])
                    hnr_val = float(current_feats["hnr_db"])

                    is_pitch_calm = (f0_val <= 145.0) or (160.0 <= f0_val <= 215.0)
                    is_jitter_calm = (jitter_val <= 2.2)
                    is_hnr_clean = (hnr_val >= 10.0)
                    is_shimmer_calm = (shimmer_val <= 12.0)
                    calm_matches = sum([is_pitch_calm, is_jitter_calm, is_hnr_clean, is_shimmer_calm])

                    if model_score >= self.stress_threshold and is_pitch_calm and is_jitter_calm and calm_matches >= 3:
                        # Neural model raised a stress alarm, but vocal folds are physically relaxed:
                        # Dampen false alarm and clamp into CALM range (25 - 35 points)
                        stress_score = round(float(np.clip(model_score * 0.40, 25.0, 35.0)), 1)
                        scoring_method = "acoustic_safeguard_hybrid"
                        explanations.append(
                            f"Bộ lọc sinh lý âm học phổ quát phát hiện các chỉ số thanh quản hoàn toàn thư giãn "
                            f"(F0={f0_val:.1f}Hz, Jitter={jitter_val:.2f}%, HNR={hnr_val:.1f}dB); "
                            f"đã tự động triệt tiêu cảnh báo giả từ mạng nơ-ron và xác nhận trạng thái CALM."
                        )
                    else:
                        stress_score = round(model_score, 1)
                        scoring_method = "trained_hybrid_model"

                explanations.append(
                    f"Mô hình phân loại giọng nói dự đoán nhãn {prediction['predicted_label']}."
                )
            elif baseline is not None:
                average_z = max(0.0, float(np.mean(z_scores)))
                stress_score = round(
                    100.0 / (1.0 + np.exp(-1.2 * (average_z - 1.2))), 1
                )
                confidence = None
                scoring_method = "personal_baseline_rules"
                explanations.append(
                    "Điểm stress dùng quy tắc so sánh với baseline cá nhân, chưa được hiệu chuẩn xác suất."
                )
            else:
                stress_score = None
                confidence = None
                scoring_method = "uncalibrated"
                explanations.append(
                    "Chưa có mô hình stress đã huấn luyện hoặc baseline cá nhân; không thể kết luận mức stress."
                )

            if stress_score is None:
                status = "UNCALIBRATED"
            elif stress_score >= self.high_stress_threshold:
                status = "HIGH_STRESS"
            elif stress_score >= self.stress_threshold:
                status = "MODERATE_STRESS"
            else:
                status = "CALM"

            return {
                "request_id": request_id,
                "processing_time_ms": int((time.time() - start_time) * 1000),
                "speaker": speaker_info,
                "emotion": emotion,
                "assessment": {
                    "status": status,
                    "stress_score": stress_score,
                    "confidence": confidence,
                    "method": scoring_method,
                },
                "acoustic_breakdown": breakdown,
                "explanations": explanations,
            }
        finally:
            if temp_wav and os.path.exists(temp_wav):
                os.remove(temp_wav)
