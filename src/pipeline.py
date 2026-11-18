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
    """Produce emotion predictions and calibrated vocal arousal/stress proxy indicators."""

    def __init__(
        self,
        baseline_dir: str = "data/baseline_profiles",
        device: Optional[str] = None,
        sample_rate: int = 16000,
        enable_enhancer: bool = False,
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
        speaker_similarity_threshold: float = 0.50,
        speaker_identification_threshold: float = 0.70,
        stress_threshold: float = 50.0,
        high_stress_threshold: float = 70.0,
        zscore_alert_threshold: float = 1.5,
        max_audio_seconds: float = 300.0,
    ):
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.sample_rate = int(sample_rate)
        self.baseline_dir = os.path.abspath(baseline_dir)
        os.makedirs(self.baseline_dir, exist_ok=True)
        self.enable_enhancer = bool(enable_enhancer)
        self.speaker_similarity_threshold = float(speaker_similarity_threshold)
        self.speaker_identification_threshold = float(speaker_identification_threshold)
        self.stress_threshold = float(stress_threshold)
        self.high_stress_threshold = float(high_stress_threshold)
        self.zscore_alert_threshold = float(zscore_alert_threshold)
        self.max_audio_seconds = float(max_audio_seconds)

        # Only load MetricGAN+ when explicitly enabled to avoid unused startup memory
        self.enhancer: Optional[AudioEnhancer] = None
        if self.enable_enhancer:
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

        self.enable_emotion = bool(enable_emotion)
        self.emotion_model = None
        self.emotion_load_error = None
        if self.enable_emotion:
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
        self.stress_load_error = None
        if stress_hparams and stress_checkpoint_dir:
            try:
                self.stress_model = StressModel(
                    hparams_path=stress_hparams,
                    checkpoint_dir=stress_checkpoint_dir,
                    device=self.device,
                )
            except FileNotFoundError as exc:
                self.stress_load_error = str(exc)
                logger.info("No trained stress checkpoint is available yet")
            except Exception as exc:
                self.stress_load_error = str(exc)
                logger.exception("Could not load the configured stress model")

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
            enable_enhancer=config.get("enable_enhancer", False),
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
                "speaker_similarity_threshold", 0.50
            ),
            speaker_identification_threshold=config.get(
                "speaker_identification_threshold", 0.70
            ),
            stress_threshold=config.get("stress_threshold", 50.0),
            high_stress_threshold=config.get("high_stress_threshold", 70.0),
            zscore_alert_threshold=config.get("zscore_alert_threshold", 1.5),
            max_audio_seconds=config.get("max_audio_seconds", 300.0),
        )

    @property
    def ready(self) -> bool:
        """Return True only when configured inference models are loaded and ready."""
        emotion_ok = (not self.enable_emotion) or (self.emotion_model is not None)
        stress_ok = self.stress_model is not None
        return bool(emotion_ok and stress_ok)

    def _read_audio_universal(self, audio_path: str, min_seconds: float = 0.5):
        """Read audio file (WAV, FLAC, OGG, MP3, AAC, M4A, WebM) into float32 mono with bounded streaming decode."""
        if not os.path.isfile(audio_path):
            raise FileNotFoundError(f"Audio file not found: {audio_path}")

        # 1. Try native libsndfile first
        try:
            info = sf.info(audio_path)
            if info.samplerate <= 0:
                raise ValueError("Invalid audio header metadata")

            # Only reject early if it's an uncompressed format (WAV, FLAC, AIFF) where header is mathematically exact.
            if info.format in ("WAV", "FLAC", "AIFF") and info.frames > 0:
                est_header_duration = info.frames / float(info.samplerate)
                if est_header_duration < min_seconds:
                    raise ValueError(f"Audio must be at least {min_seconds:g} seconds long")
                if est_header_duration > self.max_audio_seconds:
                    raise ValueError(
                        f"Audio exceeds the {self.max_audio_seconds:g} second limit ({est_header_duration:.1f}s > {self.max_audio_seconds:g}s)"
                    )

            audio, sample_rate = sf.read(
                audio_path, dtype="float32", always_2d=True
            )
            actual_duration = len(audio) / float(sample_rate)
            if actual_duration < min_seconds:
                raise ValueError(f"Audio must be at least {min_seconds:g} seconds long")
            if actual_duration > self.max_audio_seconds:
                raise ValueError(
                    f"Audio exceeds the {self.max_audio_seconds:g} second limit ({actual_duration:.1f}s > {self.max_audio_seconds:g}s)"
                )
            mono = audio.mean(axis=1)
            return mono, sample_rate, True
        except ValueError as val_exc:
            if "actual_duration" in locals() and "second" in str(val_exc):
                raise
        except Exception:
            pass

        # 2. Fallback to PyAV with streaming frame-by-frame sample cap
        try:
            import av

            with av.open(audio_path) as container:
                if not container.streams.audio:
                    raise ValueError("Audio file contains no audio streams")
                stream = container.streams.audio[0]
                sample_rate = int(stream.rate or 16000)

                # Do not rely on container.duration metadata: for VBR MP3/AAC recordings,
                # container duration is frequently miscalculated as 50 minutes due to low-bitrate
                # silence in initial frames. The streaming decode loop below strictly enforces
                # max_audio_seconds sample-by-sample without memory overhead.

                mono_chunks = []
                decoded_samples = 0
                max_allowed_samples = int(np.ceil(self.max_audio_seconds * sample_rate))

                for frame in container.decode(stream):
                    arr = frame.to_ndarray()
                    if arr.ndim == 2:
                        # Planar audio is (channels, samples); packed single-row is (1, samples * channels)
                        channels = int(stream.channels or arr.shape[0] or 1)
                        if arr.shape[0] == 1 and channels > 1 and arr.shape[1] % channels == 0:
                            arr = arr.reshape(-1, channels).T
                        mono_frame = arr.mean(axis=0).astype(np.float32)
                    else:
                        mono_frame = arr.astype(np.float32).ravel()

                    if np.issubdtype(arr.dtype, np.integer):
                        mono_frame = mono_frame / float(np.iinfo(arr.dtype).max)

                    decoded_samples += int(mono_frame.size)
                    if decoded_samples > max_allowed_samples:
                        est_sec = decoded_samples / float(sample_rate)
                        raise ValueError(
                            f"Audio exceeds the {self.max_audio_seconds:g} second limit ({est_sec:.1f}s > {self.max_audio_seconds:g}s)"
                        )
                    mono_chunks.append(mono_frame)

                if not mono_chunks:
                    raise ValueError("Audio file contains no decodable audio frames")
                mono = np.concatenate(mono_chunks, axis=0)
                actual_duration = len(mono) / float(sample_rate)
                if actual_duration < min_seconds:
                    raise ValueError(f"Audio must be at least {min_seconds:g} seconds long")
                return mono, sample_rate, False
        except ValueError as val_exc:
            if "second" in str(val_exc) or "decodable" in str(val_exc):
                raise
            raise ValueError(f"Unsupported or invalid audio file: {val_exc}") from val_exc
        except Exception as av_exc:
            raise ValueError(f"Unsupported or invalid audio file: {av_exc}") from av_exc

    def _validate_and_prepare_audio(self, audio_path: str, min_seconds: float = 0.5):
        mono, sample_rate, is_native = self._read_audio_universal(
            audio_path, min_seconds=min_seconds
        )
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
        if not is_native or sample_rate != self.sample_rate or self.enhancer is not None:
            descriptor, temp_wav_path = tempfile.mkstemp(
                prefix="standard_16k_", suffix=".wav"
            )
            os.close(descriptor)
            sf.write(temp_wav_path, waveform.numpy(), self.sample_rate)
            if self.enhancer is not None:
                self.enhancer.clean_audio(
                    input_path=temp_wav_path, output_path=temp_wav_path
                )
                enhanced_audio, _ = sf.read(temp_wav_path, dtype="float32", always_2d=True)
                waveform = torch.from_numpy(enhanced_audio.mean(axis=1).copy())
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

    def has_baseline(self, user_id: str) -> bool:
        self._validate_user_id(user_id)
        profile_path = os.path.join(self.baseline_dir, f"{user_id}_profile.npz")
        return os.path.isfile(profile_path)

    def delete_baseline(self, user_id: str) -> bool:
        """Delete an existing user baseline profile if present."""
        self._validate_user_id(user_id)
        profile_path = os.path.join(self.baseline_dir, f"{user_id}_profile.npz")
        if os.path.isfile(profile_path):
            os.remove(profile_path)
            return True
        return False

    def register_baseline(
        self, user_id: str, calm_audio_path: str, overwrite: bool = False
    ) -> Dict[str, Any]:
        """Register a vocal baseline for a stable, non-silent recording with full acoustic validity."""
        self._validate_user_id(user_id)
        profile_path = os.path.join(self.baseline_dir, f"{user_id}_profile.npz")
        if os.path.isfile(profile_path) and not overwrite:
            raise FileExistsError(
                f"Baseline profile for '{user_id}' already exists. Set overwrite=True to replace it."
            )

        waveform, effective_wav, temp_wav = self._validate_and_prepare_audio(
            calm_audio_path, min_seconds=1.0
        )
        try:
            extracted = self.acoustic_extractor.extract_features_with_quality(effective_wav)
            metrics = extracted["metrics"]
            quality = extracted["quality"]

            if not quality["all_valid"] or quality["voiced_fraction"] < 0.20:
                missing = [k for k, v in quality["valid_flags"].items() if not v]
                raise ValueError(
                    "Baseline recording does not contain enough clean voiced speech "
                    f"(voiced_fraction={quality['voiced_fraction']:.2f}, invalid={missing})"
                )

            speaker_emb = self.calibrator.extract_embedding(effective_wav)
            acoustic_feats = {
                "f0_mean": float(metrics["f0_mean"]),
                "f0_std": float(metrics["f0_std"]),
                "jitter_percent": float(metrics["jitter_percent"]),
                "shimmer_percent": float(metrics["shimmer_percent"]),
                "hnr_db": float(metrics["hnr_db"]),
            }

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
                "overwritten": os.path.isfile(profile_path) and bool(overwrite),
                "baseline_features": acoustic_feats,
                "quality": quality,
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

    def load_all_baselines(self) -> Dict[str, Dict[str, Any]]:
        """Load all valid baseline profiles in baseline_dir for speaker auto-identification."""
        profiles: Dict[str, Dict[str, Any]] = {}
        if not os.path.isdir(self.baseline_dir):
            return profiles
        for filename in sorted(os.listdir(self.baseline_dir)):
            if filename.endswith("_profile.npz"):
                uid = filename[:-12]
                try:
                    prof = self.load_baseline(uid)
                    if prof is not None:
                        profiles[uid] = prof
                except Exception:
                    logger.warning("Skipping invalid baseline profile: %s", filename)
        return profiles

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
        iemocap_mapping = {
            "neu": "neutral",
            "ang": "angry",
            "hap": "happy",
            "sad": "sad",
        }
        mapped_label = iemocap_mapping.get(str(label), str(label)) if label is not None else None

        return {
            "label": str(label) if label is not None else None,
            "standard_label": mapped_label,
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
        identify_speaker: bool = False,
    ) -> Dict[str, Any]:
        """Analyze a speech recording with acoustic quality gating and honest arousal proxy framing."""
        start_time = time.time()
        request_id = f"req_{uuid.uuid4().hex[:8]}"
        waveform, effective_wav, temp_wav = self._validate_and_prepare_audio(audio_path)
        analyzed_duration = round(float(waveform.numel()) / float(self.sample_rate), 2)
        try:
            extracted = self.acoustic_extractor.extract_features_with_quality(effective_wav)
            raw_metrics = extracted["metrics"]
            quality = extracted["quality"]

            if raw_metrics["f0_mean"] is None:
                raise ValueError("Audio does not contain enough voiced speech")

            speaker_info: Dict[str, Any] = {
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
            elif identify_speaker:
                all_profiles = self.load_all_baselines()
                if all_profiles:
                    matched_uid, best_score = self.calibrator.find_best_match(
                        effective_wav,
                        all_profiles,
                        threshold=self.speaker_identification_threshold,
                    )
                    if matched_uid is not None:
                        user_id = matched_uid
                        baseline = all_profiles[matched_uid]
                        speaker_info = {
                            "user_id": matched_uid,
                            "verified": True,
                            "similarity_score": round(best_score, 3),
                            "identified_automatically": True,
                        }
                    else:
                        speaker_info["verified"] = False
                        speaker_info["similarity_score"] = round(best_score, 3)

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
            metric_specs = [
                ("f0_mean_hz", "f0_mean", "f0_mean", "f0_std", 1),
                ("jitter_percent", "jitter_percent", "jitter_mean", "jitter_std", 1),
                ("shimmer_percent", "shimmer_percent", "shimmer_mean", "shimmer_std", 1),
                ("hnr_db", "hnr_db", "hnr_mean", "hnr_std", -1),
            ]

            if baseline is not None:
                for output_key, raw_key, baseline_key, std_key, direction in metric_specs:
                    val = raw_metrics[raw_key]
                    center = float(baseline[baseline_key])
                    if val is None:
                        breakdown[output_key] = {
                            "value": None,
                            "baseline": round(center, 3),
                            "z_score": None,
                        }
                        continue

                    value = float(val)
                    deviation = max(float(baseline[std_key]), 1e-6)
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
                    "f0_mean_hz": {"value": raw_metrics["f0_mean"], "baseline": None, "z_score": None},
                    "jitter_percent": {"value": raw_metrics["jitter_percent"], "baseline": None, "z_score": None},
                    "shimmer_percent": {"value": raw_metrics["shimmer_percent"], "baseline": None, "z_score": None},
                    "hnr_db": {"value": raw_metrics["hnr_db"], "baseline": None, "z_score": None},
                }

            if not quality["all_valid"]:
                missing_metrics = [k for k, v in quality["valid_flags"].items() if not v]
                explanations.append(
                    f"Chất lượng âm học không đủ để đo các chỉ số: {', '.join(missing_metrics)}; "
                    "không áp dụng bộ lọc âm học hoặc mô hình lai."
                )

            if self.stress_model is not None and quality["all_valid"]:
                valid_feats = {k: float(v) for k, v in raw_metrics.items() if v is not None}
                feature_vec = self.acoustic_extractor.get_feature_vector(
                    effective_wav, features=valid_feats
                )
                prediction = self.stress_model.predict(
                    waveform, feature_vec, sample_rate=self.sample_rate
                )
                model_score = prediction["stress_probability"] * 100.0
                confidence = round(prediction["confidence"], 3)

                if baseline is not None and speaker_info.get("verified") is not False:
                    avg_z = float(np.mean(z_scores)) if z_scores else 0.0
                    if avg_z <= 1.0:
                        dampening = 0.85 + 0.15 * max(-1.0, min(1.0, avg_z))
                        calibrated_score = model_score * dampening
                    else:
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
                    # Uncalibrated Acoustic Safeguard: ONLY runs when ALL 4 physical measurements
                    # succeeded (non-null, positive, voiced_fraction >= 0.30) and all 4 lie within
                    # normal physiological limits.
                    f0_val = float(raw_metrics["f0_mean"])
                    f0_std_val = float(raw_metrics.get("f0_std", 0.0))
                    jitter_val = float(raw_metrics["jitter_percent"])
                    shimmer_val = float(raw_metrics["shimmer_percent"])
                    hnr_val = float(raw_metrics["hnr_db"])

                    # Broadened F0 range (75.0-320.0 Hz) to cover both male (85-180Hz) and female (165-300Hz),
                    # with pitch stability check (f0_std <= 35.0 Hz) to prevent stressed voices slipping through.
                    is_pitch_calm = (75.0 <= f0_val <= 320.0) and (f0_std_val <= 35.0)
                    is_jitter_calm = 0.05 <= jitter_val <= 2.2
                    is_shimmer_calm = 0.10 <= shimmer_val <= 12.0
                    is_hnr_clean = hnr_val >= 10.0
                    is_voiced_sufficient = quality["voiced_fraction"] >= 0.30

                    if (
                        model_score >= self.stress_threshold
                        and is_voiced_sufficient
                        and is_pitch_calm
                        and is_jitter_calm
                        and is_shimmer_calm
                        and is_hnr_clean
                    ):
                        stress_score = round(float(np.clip(model_score * 0.40, 25.0, 35.0)), 1)
                        scoring_method = "acoustic_safeguard_hybrid"
                        explanations.append(
                            f"Các chỉ số âm học thanh quản đo được nằm trong ngưỡng ổn định "
                            f"(F0={f0_val:.1f}Hz, Jitter={jitter_val:.2f}%, Shimmer={shimmer_val:.2f}%, HNR={hnr_val:.1f}dB); "
                            f"điểm ước lượng từ mô hình cảm xúc đã được điều chỉnh giảm (chỉ báo thực nghiệm, chưa có baseline cá nhân)."
                        )
                    else:
                        stress_score = round(model_score, 1)
                        scoring_method = "trained_hybrid_model"

                explanations.append(
                    f"Mô hình phân loại (huấn luyện trên nhãn cảm xúc RAVDESS) dự đoán nhóm '{prediction['predicted_label']}'."
                )
            elif baseline is not None and z_scores:
                average_z = max(0.0, float(np.mean(z_scores)))
                stress_score = round(
                    100.0 / (1.0 + np.exp(-1.2 * (average_z - 1.2))), 1
                )
                confidence = None
                scoring_method = "personal_baseline_rules"
                explanations.append(
                    "Điểm ước lượng dùng quy tắc so sánh với baseline cá nhân, chưa được hiệu chuẩn xác suất."
                )
            elif not quality["all_valid"]:
                stress_score = None
                confidence = None
                scoring_method = "insufficient_acoustic_quality"
            else:
                stress_score = None
                confidence = None
                scoring_method = "uncalibrated"
                explanations.append(
                    "Chưa có mô hình đã huấn luyện hoặc baseline cá nhân; không thể ước lượng mức độ căng thẳng."
                )
            # Acoustic Safeguard for Emotion Classification:
            # IEMOCAP is an English acting model that often misclassifies calm, steady,
            # low-F0 speaking voices (common in Vietnamese conversation) as 'sad' due to
            # cross-lingual acoustic shift. When all physical glottal measurements confirm
            # a calm, stable physiological state, calibrate 'sad' to 'neutral' with honest XAI.
            if emotion is not None and quality.get("all_valid"):
                f0_val = float(raw_metrics.get("f0_mean") or 0.0)
                f0_std_val = float(raw_metrics.get("f0_std") or 0.0)
                jitter_val = float(raw_metrics.get("jitter_percent") or 0.0)
                shimmer_val = float(raw_metrics.get("shimmer_percent") or 0.0)
                hnr_val = float(raw_metrics.get("hnr_db") or 0.0)

                is_pitch_calm = (75.0 <= f0_val <= 320.0) and (f0_std_val <= 35.0)
                is_jitter_calm = 0.05 <= jitter_val <= 2.2
                is_shimmer_calm = 0.10 <= shimmer_val <= 12.0
                is_hnr_clean = hnr_val >= 10.0
                is_voiced_sufficient = quality.get("voiced_fraction", 0.0) >= 0.30

                if (
                    emotion.get("label") == "sad"
                    and is_pitch_calm
                    and is_jitter_calm
                    and is_shimmer_calm
                    and is_hnr_clean
                    and is_voiced_sufficient
                ):
                    raw_emotion_label = emotion.get("label")
                    raw_conf = emotion.get("confidence", 1.0)
                    emotion["original_model_label"] = raw_emotion_label
                    emotion["original_confidence"] = raw_conf
                    emotion["label"] = "neu"
                    emotion["standard_label"] = "neutral"
                    emotion["adjusted_by_safeguard"] = True

                    # Calibrate probability distribution so 'neu' is dominant
                    old_probs = emotion.get("probabilities", {})
                    sad_weight = float(old_probs.get("sad", 1.0))
                    neu_weight = float(old_probs.get("neu", 0.0))
                    ang_weight = float(old_probs.get("ang", 0.0))
                    hap_weight = float(old_probs.get("hap", 0.0))

                    new_neu = round(neu_weight + sad_weight * 0.85, 4)
                    new_sad = round(sad_weight * 0.15, 4)
                    new_probs = {
                        "neu": new_neu,
                        "sad": new_sad,
                        "ang": ang_weight,
                        "hap": hap_weight,
                    }
                    total_p = sum(new_probs.values()) or 1.0
                    emotion["probabilities"] = {
                        k: round(v / total_p, 4) for k, v in new_probs.items()
                    }
                    emotion["confidence"] = emotion["probabilities"]["neu"]

                    explanations.append(
                        f"Bộ bảo vệ âm học (Acoustic Safeguard) đã hiệu chỉnh nhãn cảm xúc từ '{raw_emotion_label}' sang 'neutral' "
                        f"(Bình thường / Điềm đạm) do các chỉ số thanh học ổn định (F0={f0_val:.1f}Hz, F0 std={f0_std_val:.1f}Hz, HNR={hnr_val:.1f}dB), "
                        f"loại bỏ độ lệch do mô hình IEMOCAP tiếng Anh nhận nhầm âm vực trầm tự nhiên."
                    )

            if scoring_method == "insufficient_acoustic_quality":
                status = "INSUFFICIENT_AUDIO"
            elif stress_score is None:
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
                    "construct": "acted_emotion_arousal_proxy",
                    "analyzed_duration_seconds": analyzed_duration,
                },
                "acoustic_breakdown": breakdown,
                "acoustic_quality": quality,
                "explanations": explanations,
            }
        finally:
            if temp_wav and os.path.exists(temp_wav):
                os.remove(temp_wav)
