"""Real-time wrapper around chatterbox/vc.py's S3Gen voice conversion.

Mirrors the public surface of `infer.lib.rtrvc.RVC` so that `vc_engine` can
swap one for the other without changes. The streaming approach is
naive — the entire padded buffer is re-tokenized and re-decoded each call,
relying on vc_engine's SOLA crossfade to stitch consecutive chunks together.

Pipeline (per infer() call):
   16 kHz input wav
     ─► (optional librosa pitch-shift, if f0_up_key != 0)
     ─► S3Tokenizer        (16 kHz audio → discrete S3 content tokens, 25 Hz)
     ─► flow.inference     (tokens + ref embedding → 24 kHz mel-spectrogram)
     ─► HiFiGAN (mel2wav)  (mel → 24 kHz waveform)
     ─► tail-trim to exactly `return_length * 240` samples
   24 kHz output wav

The reference voice is a "ref_dict" (speaker-encoder x-vector + a prompt mel
+ prompt tokens) computed ONCE at startup from a target voice wav (or loaded
from a precomputed conds.pt). It does not need to be recomputed per chunk.

Differences from rtrvc:
- No faiss index / hubert features / per-token feature search. Chatterbox
  encodes content via S3 tokens internally, and timbre is supplied via the
  ref_dict rather than a .pth checkpoint of speaker-specific weights.
- f0 detection is not used inside the pipeline (the internal
  ConvRNNF0Predictor in HiFiGAN derives F0 from mels). `rmvpe` is the only
  f0 method accepted; other values raise.
- Pitch / formant shift are not natively supported by chatterbox. `key`
  (semitones) is applied via librosa pitch-shift on the 16 kHz input when
  non-zero; `formant` is currently ignored (warns once).
"""
import logging
import os
import traceback
from pathlib import Path
from typing import Optional

import librosa
import numpy as np
import torch
import torch.nn.functional as F
from safetensors.torch import load_file

from configs.config import Config

# S3Gen = the full token-to-wave model (CFM flow → HiFiGAN). It carries the
# tokenizer (`s3gen.tokenizer`), the flow decoder (`s3gen.flow`), and the
# vocoder (`s3gen.mel2wav`) all in one nn.Module.
from chatterbox.models.s3gen import S3Gen, S3GEN_SR  # 24000
from chatterbox.models.s3tokenizer import S3_SR      # 16000


def printt(strr, *args):
    # Tiny print helper kept identical to rtrvc's so log lines look the same.
    if len(args) == 0:
        print(strr)
    else:
        print(strr % args)


def _resolve_model_dir(pth_path: str) -> Path:
    """Accept either a directory or a file path; return the model directory.

    vc_engine builds pth_path as ``./models/<name>/model.pth`` by convention,
    so we just walk up to the parent if a file path was given.
    """
    p = Path(pth_path)
    if p.is_dir():
        return p
    return p.parent


class Chatterbox:
    """Streaming voice-conversion wrapper for chatterbox's S3Gen."""

    # ------------------------------------------------------------------ #
    # Construction / model loading
    # ------------------------------------------------------------------ #
    def __init__(
        self,
        key,                # pitch shift in semitones (rtrvc: f0_up_key)
        formant,            # formant shift in semitones (no-op here)
        pth_path,           # model dir, or a path inside it
        index_path,         # reference-voice WAV path (rtrvc had a faiss .index here)
        index_rate,         # ignored (no faiss index)
        n_cpu,              # ignored (no harvest workers)
        inp_q,              # ignored (no harvest workers)
        opt_q,              # ignored (no harvest workers)
        config: Config,
        last_chatterbox: Optional["Chatterbox"] = None,
    ) -> None:
        try:
            # ---- 1. Plain config / device bookkeeping --------------------
            self.config = config
            self.device = config.device
            self.is_half = config.is_half
            # Several chatterbox submodules (notably the CFM decoder) are not
            # all fp16-safe end-to-end, so we silently downgrade to fp32.
            if self.is_half:
                logging.warning(
                    "(rtchatterbox) is_half requested but chatterbox runs in fp32; "
                    "ignoring."
                )
                self.is_half = False

            # Pitch / formant shift values are stored but only `key` is honored
            # (applied later in infer()).
            self.f0_up_key = key
            self.formant_shift = formant
            if formant != 0:
                logging.warning(
                    "(rtchatterbox) formant shift is not supported by chatterbox; "
                    "value %s ignored.",
                    formant,
                )

            # ---- 2. API-parity attributes (no functional effect) ---------
            # vc_engine still passes these positionally; keep them around so
            # we don't break the caller, but they have no role here.
            self.n_cpu = n_cpu
            self.inp_q = inp_q
            self.opt_q = opt_q
            self.index_path = index_path
            self.index_rate = index_rate

            # ---- 3. Locate model files -----------------------------------
            self.pth_path: str = pth_path
            self.model_dir = _resolve_model_dir(pth_path)

            # ---- 4. Load (or reuse) the heavy S3Gen network --------------
            # The S3Gen weights are big and slow to load. On a hot reload
            # (e.g. user changed pitch_shift), `last_chatterbox` is the
            # previous instance and we just steal its already-loaded model.
            if last_chatterbox is None:
                self.s3gen = self._load_s3gen(self.model_dir, self.device)
            else:
                self.s3gen = last_chatterbox.s3gen

            # Sample rates exposed to vc_engine. tgt_sr is what infer() emits;
            # vc_engine reads it to size its output buffers.
            self.tgt_sr = S3GEN_SR        # 24 kHz output
            self.model_input_sr = S3_SR   # 16 kHz model input

            # ---- 5. Resolve the reference voice (ref_dict) ---------------
            # ref_dict carries the speaker identity. Order of preference:
            #   a) reuse from last_chatterbox if path inputs are unchanged
            #   b) build from the wav at index_path
            #   c) load conds.pt next to the model weights
            #   d) fall back to reference.wav / ref.wav in the model dir
            self.ref_dict: Optional[dict] = None
            if (
                last_chatterbox is not None
                and last_chatterbox.pth_path == self.pth_path
                and last_chatterbox.index_path == self.index_path
            ):
                self.ref_dict = last_chatterbox.ref_dict
            else:
                self._load_reference(index_path, self.model_dir)

            if self.ref_dict is None:
                raise RuntimeError(
                    f"No reference voice found. Provide a reference wav via "
                    f"index_path or place conds.pt in {self.model_dir}."
                )

            # ---- 6. Optional rmvpe carry-over ----------------------------
            # rmvpe is loaded lazily on demand (see _ensure_rmvpe). If the
            # previous instance already loaded it, reuse it.
            if last_chatterbox is not None and hasattr(last_chatterbox, "model_rmvpe"):
                self.model_rmvpe = last_chatterbox.model_rmvpe
        except Exception:
            printt(traceback.format_exc())
            raise

    @staticmethod
    def _load_s3gen(model_dir: Path, device) -> S3Gen:
        """Instantiate S3Gen and load the safetensors weight file into it.

        `strict=False` is intentional: the upstream checkpoint includes a few
        keys (mel filter buffers etc.) that we don't always carry, and missing
        them is harmless.
        """
        weights_path = model_dir / "s3gen.safetensors"
        if not weights_path.exists():
            raise FileNotFoundError(f"Missing chatterbox weights: {weights_path}")
        printt(f"Loading chatterbox S3Gen from {weights_path}")
        s3gen = S3Gen()
        s3gen.load_state_dict(load_file(str(weights_path)), strict=False)
        s3gen.to(device).eval()
        return s3gen

    def _load_reference(self, ref_wav_path: Optional[str], model_dir: Path) -> None:
        """Populate self.ref_dict from one of the supported sources."""
        # (a) explicit reference wav supplied by the caller
        if ref_wav_path and os.path.exists(ref_wav_path) and not os.path.isdir(ref_wav_path):
            self.set_target_voice(ref_wav_path)
            return

        # (b) precomputed conds.pt (chatterbox's "built-in voice" format)
        conds_path = model_dir / "conds.pt"
        if conds_path.exists():
            printt(f"Loading prebuilt chatterbox conditionals from {conds_path}")
            # CUDA-saved tensors must be remapped when loading on cpu/mps.
            map_location = (
                torch.device("cpu") if str(self.device) in ("cpu", "mps") else None
            )
            states = torch.load(str(conds_path), map_location=map_location)
            ref_dict = states["gen"]
            # Move every tensor in the dict to our active device.
            self.ref_dict = {
                k: (v.to(self.device) if torch.is_tensor(v) else v)
                for k, v in ref_dict.items()
            }
            return

        # (c) wav fallback by convention next to the model weights
        for name in ("reference.wav", "ref.wav"):
            wav_path = model_dir / name
            if wav_path.exists():
                self.set_target_voice(str(wav_path))
                return
        # If we reach here, ref_dict stays None and __init__ will raise.

    def set_target_voice(self, wav_fpath: str) -> None:
        """Compute a fresh ref_dict from a target-voice wav.

        chatterbox uses up to 10 s of reference audio; longer clips are
        truncated. embed_ref runs the speaker encoder + builds the prompt
        mels/tokens that the flow decoder will be conditioned on.
        """
        printt(f"Loading chatterbox reference voice from {wav_fpath}")
        s3gen_ref_wav, _ = librosa.load(wav_fpath, sr=S3GEN_SR)
        s3gen_ref_wav = s3gen_ref_wav[: 10 * S3GEN_SR]  # DEC_COND_LEN cap
        self.ref_dict = self.s3gen.embed_ref(
            s3gen_ref_wav, S3GEN_SR, device=self.device
        )

    # ------------------------------------------------------------------ #
    # Live-tweakable parameters (called by vc_engine.set_param)
    # ------------------------------------------------------------------ #
    def change_key(self, new_key):
        # Pitch shift takes effect on the very next infer() call.
        self.f0_up_key = new_key

    def change_formant(self, new_formant):
        if new_formant != 0:
            logging.warning(
                "(rtchatterbox) formant shift not supported; %s ignored.", new_formant
            )
        self.formant_shift = new_formant

    def change_index_rate(self, new_index_rate):
        # No-op: chatterbox has no faiss index. Stored only so vc_engine's
        # parity calls don't fail if it pokes at this attribute.
        self.index_rate = new_index_rate

    # ------------------------------------------------------------------ #
    # rmvpe (lazy; not used in the live pipeline currently)
    # ------------------------------------------------------------------ #
    def _ensure_rmvpe(self):
        # Loaded on first use only — rmvpe.pt is ~180 MB and we don't want to
        # pay that cost unless something actually asks for F0.
        if not hasattr(self, "model_rmvpe"):
            from infer.lib.rmvpe import RMVPE

            printt("Loading rmvpe model")
            self.model_rmvpe = RMVPE(
                "assets/rmvpe/rmvpe.pt",
                is_half=False,
                device=self.device,
                use_jit=self.config.use_jit,
            )

    def get_f0_rmvpe(self, x: torch.Tensor) -> torch.Tensor:
        """Extract F0 with rmvpe at 16 kHz.

        Currently unused inside infer() (chatterbox predicts F0 internally
        from mels), but exposed for callers that may want to use it for
        gating, pitch-tracking display, etc.
        """
        self._ensure_rmvpe()
        return self.model_rmvpe.infer_from_audio(x, thred=0.03)

    # ------------------------------------------------------------------ #
    # The hot path
    # ------------------------------------------------------------------ #
    @torch.inference_mode()
    def infer(
        self,
        input_wav: torch.Tensor,
        block_frame_16k,
        skip_head,
        return_length,
        f0method,
    ) -> torch.Tensor:
        """Run one chunk through the chatterbox pipeline.

        Args mirror rtrvc.RVC.infer:
        - input_wav: full padded buffer at 16 kHz, shape [N]. vc_engine
          maintains the buffer (history + new chunk + lookahead) and hands us
          the whole thing each time.
        - block_frame_16k: size of the latest chunk in 16 kHz samples (unused
          here; we re-process the full buffer every call).
        - skip_head: head frames (10 ms units) to skip in the output (unused;
          tail-trim implicitly drops the head).
        - return_length: number of 10 ms output frames to return.
        - f0method: must be 'rmvpe' (only supported value).
        """
        # ---- 1. Validate f0 method -----------------------------------------
        # Refuse anything except rmvpe. We don't actually invoke rmvpe in the
        # pipeline below, but we want callers to be explicit so a future hookup
        # has a single well-known method to plug in.
        if f0method != "rmvpe":
            raise ValueError(
                f"rtchatterbox only supports f0method='rmvpe' (got {f0method!r})"
            )

        # ---- 2. Optional pitch shift on the input --------------------------
        # librosa.effects.pitch_shift is STFT-based and not cheap. We only
        # take the hit when the user actually asked for a non-zero shift; the
        # common no-shift case stays on the GPU and avoids a CPU round-trip.
        if self.f0_up_key != 0:
            wav_np = input_wav.detach().cpu().numpy().astype(np.float32)
            wav_np = librosa.effects.pitch_shift(
                y=wav_np, sr=S3_SR, n_steps=float(self.f0_up_key)
            )
            wav_16 = torch.from_numpy(wav_np).to(self.device)
        else:
            wav_16 = input_wav.float().to(self.device)

        # ---- 3. Shape + length normalization for the tokenizer -------------
        # S3Tokenizer wants [B, T]. Length must be a multiple of 640 samples
        # (one S3 token = 40 ms @ 16 kHz = 640 samples) or trailing audio
        # gets dropped. We zero-pad the tail to round up.
        if wav_16.dim() == 1:
            wav_16 = wav_16.unsqueeze(0)
        pad = (-wav_16.shape[-1]) % 640
        if pad:
            wav_16 = F.pad(wav_16, (0, pad))

        # ---- 4. Audio → discrete S3 content tokens (25 Hz) -----------------
        # These tokens are intended to be speaker-independent — they encode
        # *what is being said* but not *who is saying it*. Identity comes
        # from ref_dict in step 5.
        s3_tokens, s3_token_lens = self.s3gen.tokenizer(wav_16)

        # ---- 5. Tokens + speaker → 24 kHz waveform -------------------------
        # s3gen.inference() runs the CFM flow (tokens → mels) and then HiFiGAN
        # (mels → wav). The flow internally PREPENDS the prompt tokens/mels
        # from ref_dict and then strips them from the output, so what comes
        # back is aligned to our input.
        # NOTE: `finalize=True` (the default of `inference`) means the flow
        # does NOT drop its trailing 3-token lookahead. We're relying on
        # vc_engine's SOLA crossfade to clean up boundary artifacts instead.
        wav_24, _ = self.s3gen.inference(
            speech_tokens=s3_tokens,
            ref_dict=self.ref_dict,
        )
        # inference() returns shape [B=1, 1, T]; collapse to a flat [T].
        wav_24 = wav_24.squeeze(0).squeeze(0).float()

        # ---- 6. Trim to exactly the length vc_engine expects ---------------
        # vc_engine talks in 10 ms frames at the output sample rate. With
        # tgt_sr=24000, 1 frame = 240 samples. We always return the *tail*
        # of the synthesized audio — that's the part corresponding to the
        # newest input — and pad with leading zeros if the synthesized clip
        # came up short (rare; would mean we got a tiny input buffer).
        out_samples = int(return_length) * (self.tgt_sr // 100)
        if wav_24.shape[0] < out_samples:
            wav_24 = F.pad(wav_24, (out_samples - wav_24.shape[0], 0))
        else:
            wav_24 = wav_24[-out_samples:]

        return wav_24
