"""ServerAudioIO — sounddevice I/O with asyncio bridge."""

import threading
from typing import Callable, List, Optional

import numpy as np
import sounddevice as sd


class ServerAudioIO:
    def __init__(self) -> None:
        self._input_stream: Optional[sd.InputStream] = None
        self._output_stream: Optional[sd.OutputStream] = None
        self._output_buffer: np.ndarray = np.empty(0, dtype=np.float32)
        self._output_lock = threading.Lock()

    # ------------------------------------------------------------------
    # Public static API
    # ------------------------------------------------------------------

    @staticmethod
    def list_devices():
        input_list = ["client"]
        output_list = ["client"]
        for dev in sd.query_devices():
            if dev["max_input_channels"] > 0:
                input_list.append(dev["name"])
            if dev["max_output_channels"] > 0:
                output_list.append(dev["name"])
        return input_list, output_list

    def get_device_sample_rate(self, device_name: str) -> int:
        idx = self._find_device_index(device_name)
        return int(sd.query_devices(idx)["default_samplerate"])

    # ------------------------------------------------------------------
    # Stream lifecycle
    # ------------------------------------------------------------------

    def start_input(
        self,
        device_name: str,
        sample_rate: int,
        frames_per_buffer: int,
        on_chunk_callback: Callable,
        loop,
    ) -> None:
        if self._input_stream is not None:
            self.stop_input()

        device_idx = self._find_device_index(device_name)

        def _sd_input_callback(indata, frames, time_info, status):
            chunk = indata[:, 0].copy()  # mono, float32
            loop.call_soon_threadsafe(on_chunk_callback, chunk, sample_rate)

        self._input_stream = sd.InputStream(
            device=device_idx,
            samplerate=sample_rate,
            channels=1,
            dtype="float32",
            blocksize=frames_per_buffer,
            latency="low",
            callback=_sd_input_callback,
        )
        self._input_stream.start()

    def start_output(self, device_name: str, sample_rate: int) -> None:
        if self._output_stream is not None:
            self.stop_output()

        device_idx = self._find_device_index(device_name)

        def _sd_output_callback(outdata, frames, time_info, status):
            with self._output_lock:
                n = min(frames, len(self._output_buffer))
                outdata[:n, 0] = self._output_buffer[:n]
                outdata[n:, 0] = 0.0
                self._output_buffer = self._output_buffer[n:]

        self._output_stream = sd.OutputStream(
            device=device_idx,
            samplerate=sample_rate,
            channels=1,
            dtype="float32",
            latency="low",
            callback=_sd_output_callback,
        )
        self._output_stream.start()

    def write_output(self, data: np.ndarray) -> None:
        with self._output_lock:
            self._output_buffer = np.concatenate([self._output_buffer, data])

    def stop_all(self):
        self.stop_input()
        self.stop_output()


    def stop_input(self) -> None:
        if self._input_stream is not None:
            self._input_stream.stop()
            self._input_stream.close()
            self._input_stream = None


    def stop_output(self) -> None:
        if self._output_stream is not None:
            self._output_stream.stop()
            self._output_stream.close()
            self._output_stream = None
        with self._output_lock:
            self._output_buffer = np.empty(0, dtype=np.float32)

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _find_device_index(self, name: str) -> int:
        name_lower = name.lower()
        for idx, dev in enumerate(sd.query_devices()):
            if name_lower in dev["name"].lower():
                return idx
        raise ValueError(f"Audio device not found: {name!r}")
