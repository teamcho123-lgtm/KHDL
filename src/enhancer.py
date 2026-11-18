"""Audio Preprocessing & Speech Enhancement Module.
Uses SpeechBrain's MetricGAN+ to remove environmental background noise
and normalize audio to 16kHz mono.
"""

import os
import sys
import torch
import torchaudio
from typing import Optional

try:
    from speechbrain.inference.enhancement import SpectralMaskEnhancement
except ImportError:
    # Backward compatibility with older speechbrain versions
    from speechbrain.pretrained import SpectralMaskEnhancement


class AudioEnhancer:
    """Handles audio denoising, resampling to 16kHz, and mono conversion."""

    def __init__(
        self,
        model_source: str = "speechbrain/metricgan-plus-voicebank",
        savedir: str = "pretrained_models/metricgan-plus-voicebank",
        device: Optional[str] = None,
    ):
        if device is None:
            self.device = "cuda" if torch.cuda.is_available() else "cpu"
        else:
            self.device = device

        print(f"[AudioEnhancer] Initializing MetricGAN+ on device: {self.device}...")
        kwargs = {"device": self.device}
        strategy_kwargs = {}
        if sys.platform == "win32":
            try:
                from speechbrain.utils.fetching import LocalStrategy
                strategy_kwargs["local_strategy"] = LocalStrategy.COPY
            except Exception:
                pass

        self.enhancer = SpectralMaskEnhancement.from_hparams(
            source=model_source,
            savedir=savedir,
            run_opts=kwargs,
            **strategy_kwargs,
        )
        print("[AudioEnhancer] MetricGAN+ loaded successfully.")

    def clean_audio(
        self,
        input_path: str,
        output_path: Optional[str] = None,
        target_sample_rate: int = 16000,
    ) -> str:
        """Denoiser pipeline:
        1. Loads audio (any sample rate, stereo or mono).
        2. Resamples to 16kHz mono.
        3. Enhances with MetricGAN+.
        4. Saves cleaned audio to output_path.
        """
        if not os.path.exists(input_path):
            raise FileNotFoundError(f"Input audio file not found: {input_path}")

        if output_path is None:
            base, ext = os.path.splitext(input_path)
            output_path = f"{base}_enhanced{ext}"

        # Ensure parent output directory exists
        out_dir = os.path.dirname(output_path)
        if out_dir:
            os.makedirs(out_dir, exist_ok=True)

        # 1. Load audio with soundfile or torchaudio
        try:
            import soundfile as sf
            data, sr = sf.read(input_path, dtype="float32")
            if data.ndim == 1:
                sig = torch.from_numpy(data).unsqueeze(0)
            else:
                sig = torch.from_numpy(data.T)
        except Exception:
            sig, sr = torchaudio.load(input_path)

        # Convert to mono if multichannel
        if sig.shape[0] > 1:
            sig = torch.mean(sig, dim=0, keepdim=True)

        # Resample to 16kHz if needed
        if sr != target_sample_rate:
            resampler = torchaudio.transforms.Resample(orig_freq=sr, new_freq=target_sample_rate)
            sig = resampler(sig)

        sig = sig.to(self.device)

        # 2. Enhance batch with MetricGAN+
        # MetricGAN expects input shape: [batch, time]
        with torch.no_grad():
            enhanced = self.enhancer.enhance_batch(sig, lengths=torch.tensor([1.0], device=self.device))

        # 3. Save enhanced wav with soundfile
        import soundfile as sf
        enhanced_np = enhanced.cpu().squeeze().numpy()
        sf.write(output_path, enhanced_np, target_sample_rate)
        return output_path
