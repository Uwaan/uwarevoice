"""VCEngine — synchronous audio processing pipeline."""
import logging

import librosa
import numpy as np
import torch
import torchaudio.transforms as audio_transform
from tools.torchgate import TorchGate
from time import perf_counter
from typing import Tuple
from loopbackinfer import LoopbackInfer #stub

# From RVC v2
def phase_vocoder(a, b, fade_out, fade_in):
    window = torch.sqrt(fade_out * fade_in)
    fa = torch.fft.rfft(a * window)
    fb = torch.fft.rfft(b * window)
    absab = torch.abs(fa) + torch.abs(fb)
    n = a.shape[0]
    if n % 2 == 0:
        absab[1:-1] *= 2
    else:
        absab[1:] *= 2
    phia = torch.angle(fa)
    phib = torch.angle(fb)
    deltaphase = phib - phia
    deltaphase = deltaphase - 2 * np.pi * torch.floor(deltaphase / 2 / np.pi + 0.5)
    w = 2 * np.pi * torch.arange(n // 2 + 1).to(a) + deltaphase
    t = torch.arange(n).unsqueeze(-1).to(a) / n
    result = (
        a * (fade_out**2)
        + b * (fade_in**2)
        + torch.sum(absab * torch.cos(w * t + phia), -1) * window / n
    )
    return result

class VCEngineLoopback:

    def __init__(self) -> None:
        logging.info("Starting up vc_engine...")
        self.vc = None
        self.vc_active = False
        self.input_sr : int = 48000
        self.model_input_sr : int = 16000
        self.model_output_sr : int = 24000
        self.output_sr : int = 48000
        self.input_gate_db : int = -70
        self.input_pitch_shift : int = 0
        self.input_formant_shift : int = 0
        self.chunk_size_s : float = 0.0
        self.chunk_size_samples : int = 0
        self.pad_size_s : float = 0.0
        self.pad_size_samples : int = 0
        self.crossfade_size_s : float = 0.0
        self.crossfade_size_samples : int = 0
        self.sola_search_samples : int = 0
        self.use_output_rms : float = 1.0
        self.use_noise_reduction = False
        self.use_vocoder = False
        self.use_fp32 : bool = False
        self.use_jit : bool = False
        self.infer_device = "cuda:0"
        self.model_name : str = "defaultvoice"
        self.model_filename : str = ""
        self._settings: dict = {}
        self.input_level_db : int = -144

    def configure(self, settings: dict) -> None:
        self._settings = settings

        hot_update = False
        if self.vc_active:
            logging.warning("WARN (vc_engine): Attempting to change settings while voice conversion is running. Attempting reload; this will almost certainly have unintended side effects.")
            hot_update = True
            self.stop_vc()

        self.input_sr = settings.get("input_sr", 48000)
        self.output_sr = settings.get("output_sr", 48000)
        self.input_gate_db = settings.get("input_gate_db", -70)
        self.input_pitch_shift = settings.get("pitch_shift_semitones", 0)
        self.input_formant_shift = settings.get("formant_shift", 0)
        self.chunk_size_s = settings.get("chunk_size_s", 0.1)
        self.pad_size_s = settings.get("chunk_pad_s", 0.5)
        self.crossfade_size_s = settings.get("crossfade_s", 0.05)

        self.use_output_rms = settings.get("use_output_rms", 1.0)
        self.use_noise_reduction = settings.get("use_noise_reduction", False)
        self.use_vocoder = settings.get("use_vocode_smoothing", False)
        self.use_fp32 = settings.get("use_fp32", False)
        self.use_jit = settings.get("use_jit", False)
        self.infer_device = settings.get("inference_device", "cuda:0")
        self.model_name = settings.get("model_name", "defaultvoice")

        self.model_filename = "./models/" + self.model_name

        # These will get set on server start, for reasons
        #self.chunk_size_samples = 0
        #self.pad_size_samples: int = 0
        #self.crossfade_size_samples: int = 0
        #self.sola_search_samples: int = 0

        if hot_update:
            self.start_vc()

    def start_vc(self):
        if self.vc_active:
            logging.warning("WARN (vc_engine): Conversion already active!")
        torch.cuda.empty_cache()

        # Create & configure VC inference object
        self.vc = LoopbackInfer(self.infer_device)

        # Determine I/O channels? Both should be mono for
        # inference & should be mono when using 'client' as
        # an audio device but will likely need to be
        # multiplexed if using server-side output
        #self.num_channels_input, self.num_channels_output = _audio_provider.get_channel_counts()
        #self.num_channels = min(num_channels_input, num_channels_output, 2)

        # Determine and validate source, inference, and target
        # sample rates
        self.model_input_sr = self.vc.get_input_sample_rate()
        self.model_output_sr = self.vc.get_output_sample_rate()

        logging.info(f"(vc_engine) Model is requesting {self.vc.get_data_type()} data " +
                     f"at {self.model_input_sr} in, " +
                     f"{self.model_output_sr} out.")
        logging.info(f"(vc_engine) Generating buffers for {self.pad_size_s}s pad /{self.chunk_size_s}s chunks.")

        # Many functions (RMS matching, SOLA crossfading) get
        # interpolated over fixed-sized chunks that may differ
        # in size from the user-specified inference chunk
        # length. This value will be used to make sure buffers are
        # aligned with these processing chunks.
        self.hop_length = self.input_sr // 100    # is 10 ms
        self.hop_length_resampled = self.model_input_sr // 100
        self.hop_length_output = self.output_sr // 100

        # Determine length of various windows in our buffers
        self.chunk_size_samples = (
            int (np.round(
                self.chunk_size_s * self.input_sr / self.hop_length
            )) * self.hop_length
        )

        # Model has an expected sample rate that we must obey, and
        # it may differ from the sample rate of the input provided
        self.chunk_resampled_samples = int(np.round(
            self.chunk_size_samples * self.hop_length_resampled / self.hop_length))

        self.chunk_output_samples = (
            int (np.round(
                self.chunk_size_s * self.output_sr / self.hop_length_output
            )) * self.hop_length_output
        )

        self.pad_size_samples = (
            int (np.round(
                self.pad_size_s * self.input_sr / self.hop_length
            )) * self.hop_length
        )

        self.crossfade_size_samples = (
                int(np.round(
                    self.crossfade_size_s * self.output_sr / self.hop_length_output
                )) * self.hop_length_output
        )

        # Might rename this, others (RVC, DDSP-SVC) use a "sola"
        # buffer which appears to be Synchronized Overlap-Add
        # (Time-Domain Pitch-Synchronous Overlap-and-Add, TD-PSOLA)
        # which can change the periodicity of a signal to match
        # A with B and then do a linear (?) crossfade with
        # frequencies aligned

        self.sola_frame_size_samples = (
            min(self.crossfade_size_samples, 4 * self.hop_length_output)
        )
        self.sola_search_samples = self.hop_length_output

        # will be used in the future by tools/rvc_for_realtime.py
        self.skip_head = self.pad_size_samples // self.hop_length
        self.return_length = (
                                (self.chunk_output_samples +
                                self.sola_frame_size_samples +
                                self.sola_search_samples)
                             ) // self.hop_length_output

        # Input buffer needs to send enough data for the inferencer
        # to output the correct amount. Sample rate mismatch
        # could interfere with this
        self.sola_frame_input_equiv = (
            int(np.round(
                self.sola_frame_size_samples
                * (self.input_sr / self.output_sr)
                / self.hop_length
            )) * self.hop_length
        )

        self.sola_search_input_equiv = (
            int(np.round(
                self.hop_length_output
                * (self.input_sr / self.output_sr)
                / self.hop_length
            )) * self.hop_length
        )

        # Now move on to creating the arrays that will hold the
        # actual samples

        self.input_frame: torch.Tensor = torch.zeros(
            self.pad_size_samples +
            self.chunk_size_samples +
            self.sola_frame_input_equiv +   #self.sola_frame_size_samples
            self.sola_search_input_equiv,   #self.sola_search_samples,
            device=self.infer_device,
            dtype=torch.float32,
        )

        # Again, this one is for the model's sample rate
        self.input_frame_resampled: torch.Tensor = torch.zeros(
            self.hop_length_resampled * self.input_frame.shape[0] // self.hop_length,
            device=self.infer_device,
            dtype=torch.float32,
        )

        self.input_frame_denoised: torch.Tensor = self.input_frame.clone()

        # Deprecate? It looks like RVC only uses output_frame for output
        # noise reduction, and that isn't being implemented here.
        # This math will need to be revisited if we ever re-add it.
        #self.output_frame_samples = (int(round(
        #    (self.pad_size_samples +
        #     self.chunk_size_samples +
        #     self.crossfade_size_samples +
        #     self.sola_frame_size_samples) *
        #    (self.output_sr / self.input_sr)
        #    ) // self.hop_length) * self.hop_length)
        # Deprecate/ignore this as well for now
        #self.output_frame: torch.Tensor = torch.zeros(
        #    self.output_frame_samples,
        #    device=self.infer_device,
        #    dtype=torch.float32,
        #)

        self.sola_buffer: torch.Tensor = torch.zeros(
            self.sola_frame_size_samples, device=self.infer_device, dtype=torch.float32
        )

        # Took this stuff from RVC v2

        # fade_in_window and fade_out_window will be used later to
        # compute crossfade over the edges of the sola buffer
        self.fade_in_window: torch.Tensor = (
            torch.sin(
                        0.5 * np.pi * torch.linspace(0.0, 1.0,
                        steps=self.sola_frame_size_samples,
                        device=self.infer_device,
                        dtype=torch.float32,
                    )
            ) ** 2      # '**' in this context is "to the power of"
        )
        self.fade_out_window: torch.Tensor = 1 - self.fade_in_window

        # RMS buffer will be used later to (optionally) match the
        # amplitude of the generated voice samples to the amplitude
        # of the input sample
        self.rms_buffer: np.ndarray = np.zeros(4 * self.hop_length, dtype="float32")

        self.noise_reduce_buffer: torch.Tensor = torch.zeros(
            self.sola_frame_input_equiv, device=self.infer_device, dtype=torch.float32)

        self.noise_reducer_fade_in: torch.Tensor = (
            torch.sin(
                0.5 * np.pi * torch.linspace(0.0, 1.0,
                steps=self.sola_frame_input_equiv,
                device=self.infer_device,
                dtype=torch.float32,
                )
            ) ** 2
        )
        self.noise_reducer_fade_out: torch.Tensor = 1 - self.noise_reducer_fade_in

        # Set resample functions on the required I/O
        self.input_resampler = audio_transform.Resample(
            orig_freq=self.input_sr,
            new_freq=self.model_input_sr,
            dtype=torch.float32
        ).to(self.infer_device)

        if self.output_sr != self.model_output_sr:
            #Original: if self.output_sr != self.rvc.tgt_sr:
            self.output_resampler = audio_transform.Resample(
                orig_freq=self.model_output_sr,
                new_freq=self.output_sr,
                dtype=torch.float32
            ).to(self.infer_device)
        else:
            self.output_resampler = None    # Used for checks later

        # Set noise reduction function on the required input
        # TODO: update? TorchSpectralGating has been integrated into
        # github.com/timsainb/noisereduce
        self.noise_reducer = TorchGate(
            sr=self.input_sr, n_fft=4 * self.hop_length, prop_decrease=0.9
        ).to(self.infer_device)

        logging.info(f"(vc_engine) Chunk size is {self.chunk_size_samples} samples "
                     + f"(pad {self.pad_size_samples}, return length {self.return_length}.)")
        self.vc_active = True

        # That's all. Signal that audio can start up now
        #self.start_stream()
        # (Removed because parent will start audiodevice OR network
        # stream after this invocation finishes)

    def on_chunk_received(self, in_chunk: np.ndarray) -> Tuple[np.ndarray, float, float]:
        """Process one audio chunk through the pipeline.

        Returns: (output_chunk, output_level_db, inference_time_ms)
        output_chunk is the same length as the input chunk.
        """
        # Take timestamp at start (will use at end)
        t0 = perf_counter()

        if not self.vc_active:
            logging.warning("WARN (vc_engine): Audio received for processing, but conversion not running!")
            return np.zeros(in_chunk.shape[-1]), 0.0, 0.0

        # Stream from uwavclient should already be mono, but
        # there's a chance that server audio might be >1 channel
        in_chunk = librosa.to_mono(in_chunk.T)

        if in_chunk.shape[0] != self.chunk_size_samples:
            logging.error(f"ERROR (vc_engine): Received invalid chunk size ({in_chunk.shape[0]}/{self.chunk_size_samples}). Stopping inference.")
            self.stop_vc()
            return np.zeros(in_chunk.shape[-1]), 0.0, 0.0

        # TODO: Go through every reference to dtype=np.float32 and
        # add 'if's to account for alternative input sample formats?

        # Preprocessing

        # RVC v2 had this conditional for the next part, which still doesn't
        # make sense to me
        #if self.input_gate_db > -60:

        # rms_buffer contains a bit of the tail of the last received
        # chunk. We calculate the RMS of one (4 * hop) sample long
        # window at a time, but the incoming chunk is probably not
        # aligned with the boundary of that window.
        # Prepend the tail of the last chunk, then trim the front so
        # that the first element in the array is the start of a new
        # window.
        in_chunk = np.concatenate((self.rms_buffer, in_chunk))     # +4 hops - @ 48KHz +1920 samples
        rms_array = librosa.feature.rms(y=in_chunk,
                                        frame_length=4 * self.hop_length,
#                                       hop_length=self.hop_length)[:, 2:]  # [:, 2:] discards first 2 hop?
                                        hop_length=self.hop_length)[:, 3:-1]
        # save the end of this received chunk in a buffer to be used
        # on the next chunk
        self.rms_buffer[:] = in_chunk[-4 * self.hop_length :]
        #in_chunk = in_chunk[(2 * self.hop_length) - (self.hop_length // 2):]  # @ 48KHz -720 samples
        in_chunk = in_chunk[2 * self.hop_length :]   # discards first 2 hop
        mask = (librosa.amplitude_to_db(rms_array, ref=1.0)[0] < self.input_gate_db)
        for i in range(mask.shape[0]):
            if mask[i]:
                in_chunk[i * self.hop_length : (i + 1) * self.hop_length] = 0.0
        #in_chunk = in_chunk[self.hop_length // 2 :]    # Orig - @ 48KHz -240 samples
        #in_chunk = in_chunk[(2 * self.hop_length) + (self.hop_length // 2):]  # -1.5 hops
        in_chunk = in_chunk[2 * self.hop_length :]  # discards first 2 hop again

        # This is somewhat inaccurate, but store first amplitude
        # value so our parent can retrieve it and show it to
        # the user on demand
        self.input_level_db = librosa.amplitude_to_db(rms_array[0, 0], ref=1.0).item()

        # Shift & load the main buffer
        self.input_frame[: -self.chunk_size_samples] = self.input_frame[self.chunk_size_samples :].clone()
        self.input_frame_resampled[: -self.chunk_resampled_samples] = self.input_frame_resampled[
            self.chunk_resampled_samples :].clone()

        self.input_frame[-in_chunk.shape[0] :] = torch.from_numpy(in_chunk).to(self.infer_device)

        # Perform input noise reduction
        if self.use_noise_reduction:
            self.input_frame_denoised[: -self.chunk_size_samples] = self.input_frame_denoised[
                self.chunk_size_samples :].clone()
            # Reuse pre-calculated sola crossfade tensors/window to
            # crossfade the denoiser's output as well. This will make
            # timing somewhat tied to the user-defined crossfade
            denoise_chunk = self.input_frame[-self.sola_frame_input_equiv - self.chunk_size_samples :]
            denoise_chunk = self.noise_reducer(
                denoise_chunk.unsqueeze(0), self.input_frame.unsqueeze(0)
            ).squeeze(0)
            denoise_chunk[: self.sola_frame_input_equiv] *= self.noise_reducer_fade_in
            denoise_chunk[: self.sola_frame_input_equiv] += self.noise_reduce_buffer * self.noise_reducer_fade_out
            self.input_frame_denoised[-self.chunk_size_samples :] = denoise_chunk[
                : self.chunk_size_samples
            ]
            self.noise_reduce_buffer[:] = denoise_chunk[self.chunk_size_samples :]

            # Resample with either noise-reduced input or with original data
            # Taking this in part from RVC v2 - not yet sure why we are
            # increasing the length by 2 * hop_length
            self.input_frame_resampled[-self.chunk_resampled_samples - self.hop_length_resampled :] = (
                self.input_resampler(self.input_frame_denoised[-self.chunk_size_samples - 2 * self.hop_length :]
                                     )[self.hop_length_resampled :])
        else:
            self.input_frame_resampled[
                (in_chunk.shape[0] // self.hop_length + 1) * -self.hop_length_resampled :] = (
                self.input_resampler(self.input_frame[-in_chunk.shape[0] - 2 * self.hop_length :]
                                     )[self.hop_length_resampled :])


        #########################################
        # And infer based on the resampled data #
        output_frame = self.vc.infer(self.input_frame_resampled, self.return_length)
        # !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!! #
        #########################################


        if self.output_resampler is not None:
            output_frame = self.output_resampler(output_frame)

        # The model generates whatever loudness/tone/etc characteristics
        # it wants to generate. Allow the user to replace the volume
        # envelope with that of their input audio, if they prefer.
        # use_output_rms is a sliding scale (0.0 = exactly match
        # input RMS, 1.0 = leave generated audio exactly as-is)
        if self.use_output_rms < 1:
            if self.use_noise_reduction:
                mix_chunk = self.input_frame_denoised[self.pad_size_samples :]
            else:
                mix_chunk = self.input_frame[self.pad_size_samples :]

            # Figure out RMS of input audio (across frames)
            # mix_chunk is at input sample rate; needs to be applied
            # at output sample rate
            in_rms = librosa.feature.rms(
                y=mix_chunk[:].cpu().numpy(),
                frame_length = 4 * self.hop_length,
                hop_length = self.hop_length,
            )
            in_rms = torch.from_numpy(in_rms).to(self.infer_device)
            # Interpolate frames to match length of output
            in_rms = torch.nn.functional.interpolate(
                in_rms.unsqueeze(0),
                size=output_frame.shape[0] + 1,
                mode="linear",
                align_corners=True,
            )[0, 0, :-1]
            # Repeat, figure out RMS of output audio
            out_rms = librosa.feature.rms(
                y=output_frame[:].cpu().numpy(),
                frame_length = 4 * self.hop_length_output,
                hop_length = self.hop_length_output,
            )
            out_rms = torch.from_numpy(out_rms).to(self.infer_device)
            out_rms = torch.nn.functional.interpolate(
                out_rms.unsqueeze(0),
                size=output_frame.shape[0] + 1,
                mode="linear",
                align_corners=True,
            )[0, 0, :-1]
            out_rms = torch.max(out_rms, torch.zeros_like(out_rms) + 1e-3)
            # And transform the volume of the output
            output_frame *= torch.pow(
                in_rms / out_rms,
                torch.tensor(1 - self.use_output_rms))

        # Crossfade current output with previous outputs

        # Synchronized Overlap-Add (Time-Domain Pitch-Synchronous
        # Overlap-and-Add, or TD-PSOLA)
        # This is a feature in RVC v2 (via DDSP-SVC) that took SOME TIME
        # to figure out. It is a method of reducing crossfade artifacts
        # by mathematically evaluating the point in time where the two
        # waveforms synchronize best.
        #
        # This introduces some latency / boundary jitter, but the
        # tradeoff is much smoother crossfading.

        # conv1d needs [batch, channels, length]
        sola_chunk = output_frame[None, None, :self.sola_frame_size_samples + self.sola_search_samples]
        # Correlation point will be at offset = Σ(a . b) / sqrt(Σ(a^2) . Σ(b^2))
        over = torch.nn.functional.conv1d(sola_chunk, self.sola_buffer[None, None, :])
        under = torch.sqrt(
            torch.nn.functional.conv1d(
                sola_chunk**2,
                torch.ones(1, 1, self.sola_frame_size_samples, device=self.infer_device),
            )
            + 1e-8
        )
        offset = torch.argmax(over[0, 0] / under[0, 0])
        output_frame = output_frame[offset:]    # Choppa


        # To ease the transition further, may use a phase vocoder
        # to interpolate any change in pitch over the crossfade
        # period
        if self.use_vocoder:
            output_frame[: self.sola_frame_size_samples] = phase_vocoder(
                self.sola_buffer,
                output_frame[: self.sola_frame_size_samples],
                self.fade_out_window,
                self.fade_in_window,
            )
        else:
            output_frame[: self.sola_frame_size_samples] *= self.fade_in_window
            output_frame[: self.sola_frame_size_samples] += (
                self.sola_buffer * self.fade_out_window
            )

        # And save the tail of this output to use with the next chunk
        # TODO: chunk_size_samples and sola_frame_size_samples were
        # calculated based on input sample rate, not output sample rate.
        # Buffer should be sized for appropriate length based on output
        # SR?
        self.sola_buffer[:] = output_frame[
            self.chunk_output_samples : self.chunk_output_samples + self.sola_frame_size_samples
        ]

        output_level_db = librosa.amplitude_to_db(
            librosa.feature.rms(y=output_frame[:self.hop_length_output].cpu().numpy(),
                                frame_length=self.hop_length_output,
                                hop_length=self.hop_length_output),
            ref=1.0)
        infer_ms = (perf_counter() - t0) * 1000.0

        return (output_frame[: self.chunk_output_samples].cpu().numpy()), output_level_db[0, 0].item(), infer_ms

    def stop_vc(self):
        if self.vc_active:
            self.vc_active = False
