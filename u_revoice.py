"""u_revoice.py — Main Socket.IO server for real-time voice conversion."""

import argparse
import asyncio
import json
import logging
import os
import signal
import socket
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Optional

import aiohttp.web
import numpy as np
import psutil
import socketio

from server_audio import ServerAudioIO
from vc_engine import VCEngine

# ---------------------------------------------------------------------------
# GPU / NVML setup
# ---------------------------------------------------------------------------
try:
    import pynvml
    pynvml.nvmlInit()
    _NVML_AVAILABLE = True
    try:
        _CUDA_AVAILABLE = pynvml.nvmlDeviceGetCount() > 0
    except Exception:
        _CUDA_AVAILABLE = False
except Exception:
    _NVML_AVAILABLE = False
    _CUDA_AVAILABLE = False

# ---------------------------------------------------------------------------
# Settings
# ---------------------------------------------------------------------------
DEFAULT_SETTINGS = {
    "input_sr": 48000,
    "output_sr": 48000,
    "input_gate_db": -70,
    "pitch_shift_semitones": 0,
    "formant_shift": 0,
    "chunk_size_s": 0.24,
    "chunk_pad_s": 0.5,
    "crossfade_s": 0.05,
    "input_volume": 1.0,
    "output_volume": 1.0,
    "use_output_rms": 1.0,
    "use_noise_reduction": False,
    "use_vocode_smoothing": False,
    "use_fp32": True,
    "use_jit": True,
    "model_name": "defaultvoice",
    "inference_device": "cuda:0",
    "input_device": "client",
    "output_device": "client",
    "rvc_pth_filename": "./models/defaultvoice/default.pth",
    "rvc_idx_filename": "./models/defaultvoice/default.idx",
    "rvc_index_rate": 0.0,
    "num_cpus": 4,  # TODO: Remove? This needs to be passed to RVC core but the only use is for "harvest" F0 scraping
    "f0_method": "rmvpe",
}

DEFAULT_INPUT_DEVICE_LIST = [
    "client"
]

DEFAULT_OUTPUT_DEVICE_LIST = [
    "client"
]

_SETTINGS_DIR = Path(__file__).parent / "settings"


def _client_settings_path(client_id: str) -> Path:
    return _SETTINGS_DIR / client_id / "settings.json"


def _load_settings(client_id: str) -> dict:
    merged = dict(DEFAULT_SETTINGS)
    path = _client_settings_path(client_id)
    logging.info(f"Loading settings for client '{client_id}' from {path}...")
    if path.exists():
        try:
            with open(path) as f:
                on_disk = json.load(f)
            merged.update(on_disk)
        except Exception as e:
            logging.warning(f"Could not load {path}: {e}; using defaults.")
    else:
        logging.warning(f"No saved settings for client '{client_id}'. Using defaults; settings will save on first SERVER_START.")
    return merged


def _save_settings(client_id: str) -> None:
    path = _client_settings_path(client_id)
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w") as f:
            json.dump(_settings, f, indent=2)
    except Exception as e:
        logging.warning(f"Could not save settings for client '{client_id}': {e}")


# ---------------------------------------------------------------------------
# Module-level globals
# ---------------------------------------------------------------------------
_settings: dict = dict(DEFAULT_SETTINGS)
_input_devices: list = list(DEFAULT_INPUT_DEVICE_LIST)
_output_devices: list = list(DEFAULT_OUTPUT_DEVICE_LIST)
_server_running: bool = False
_active_sample_rate: int = 48000
_audio_chunk_buffer = None
_audio_queue: asyncio.Queue = None  # type: ignore[assignment]
_processing_task: Optional[asyncio.Task] = None
_status_task: Optional[asyncio.Task] = None
_dropped_chunks: int = 0
_executor = ThreadPoolExecutor(max_workers=1)
_engine = VCEngine()
_audio_io = ServerAudioIO()
_active_sid: Optional[str] = None
_active_client_id: Optional[str] = None

QUEUE_OVERLOAD_THRESHOLD = 8

# ---------------------------------------------------------------------------
# Socket.IO + aiohttp app
# ---------------------------------------------------------------------------
sio = socketio.AsyncServer(async_mode="aiohttp", cors_allowed_origins="*")
app = aiohttp.web.Application()
sio.attach(app)

# ---------------------------------------------------------------------------
# GPU / system stats helpers
# ---------------------------------------------------------------------------

def _get_gpu_stats() -> dict:
    if not _NVML_AVAILABLE:
        return {"gpu_load": 0, "gpu_mem_used_mib": 0, "gpu_mem_total_mib": 0}
    try:
        handle = pynvml.nvmlDeviceGetHandleByIndex(0)
        util = pynvml.nvmlDeviceGetUtilizationRates(handle)
        mem = pynvml.nvmlDeviceGetMemoryInfo(handle)
        return {
            "gpu_load": util.gpu,
            "gpu_mem_used_mib": mem.used // (1024 * 1024),
            "gpu_mem_total_mib": mem.total // (1024 * 1024),
        }
    except Exception:
        return {"gpu_load": 0, "gpu_mem_used_mib": 0, "gpu_mem_total_mib": 0}


def _get_system_stats() -> dict:
    vm = psutil.virtual_memory()
    return {
        "cpu_percent": psutil.cpu_percent(),
        "mem_used_mib": vm.used // (1024 *f 1024),
        "mem_total_mib": vm.total // (1024 * 1024),
    }


# ---------------------------------------------------------------------------
# Message helper
# ---------------------------------------------------------------------------

async def _emit_message(level: int, message: str, sid=None) -> None:
    """level: 0=info, 1=warn, 2=error; sid=None → broadcast"""
    await sio.emit("MESSAGE", {"message_level": level, "message": message}, to=sid)

# ---------------------------------------------------------------------------
# Server lifecycle
# ---------------------------------------------------------------------------

async def _stop_server() -> None:
    global _server_running, _processing_task, _status_task, _settings, _engine, _audio_chunk_buffer
    _server_running = False
    for task in (_processing_task, _status_task):
        if task is not None:
            task.cancel()
            try:
                await task
            except asyncio.CancelledError:
                pass
    _processing_task = None
    _status_task = None
    if _settings.get("input_device", "client") == "client" or _settings.get("output_device", "client") == "client":
        _audio_io.stop_all()
    if _engine.vc_active:
        _engine.stop_vc()

    _audio_chunk_buffer = None


async def _enqueue_chunk(chunk: np.ndarray, ts: float, sid) -> None:
    global _dropped_chunks, _audio_queue
    if _audio_queue.qsize() >= QUEUE_OVERLOAD_THRESHOLD:
        _dropped_chunks += 1
        await sio.emit("STREAM_OVERLOAD", {"dropped_chunks": _dropped_chunks})
        return
    await _audio_queue.put((chunk, ts, sid))
    _dropped_chunks = 0


def _on_hardware_chunk(chunk: np.ndarray, sample_rate: int) -> None:
    """Called on the event loop via call_soon_threadsafe (sync, not a coroutine)."""
    asyncio.create_task(_enqueue_chunk(chunk, time.time(), sid=None))


async def _route_output(
    output: np.ndarray,
    ts: float,
    sid,
    in_db: float,
    out_db: float,
    infer_ms: float,
) -> None:
    if _settings.get("output_device") == "client":
        payload = {
            "sample_count": output.size,
            "samples": output.tobytes(),
            "timestamp": ts,
            "input_level_db": in_db,
            "output_level_db": out_db,
            "inference_time_ms": infer_ms,
        }
        #logging.info(f"Out: {output.size} samples")
        await sio.emit("AUDIO_OUT", payload, to=sid)
    else:
        _audio_io.write_output(output)
        # We do still want the client to know the state of conversion,
        # though - send a payload without samples
        payload = {
            "sample_count": 0,
            "timestamp": ts,
            "input_level_db": in_db,
            "output_level_db": out_db,
            "inference_time_ms": infer_ms,
        }
        await sio.emit("AUDIO_OUT", payload, to=sid)


async def _audio_processing_loop(origin_sid) -> None:
    # TODO: sid ownership / route to origin if server input + net output
    loop = asyncio.get_event_loop()     # Python >=3.10 needs asyncio.get_running_loop() ?
    while _server_running:
        try:
            chunk, ts, sid = await asyncio.wait_for(_audio_queue.get(), timeout=1.0)
        except asyncio.TimeoutError:
            continue
        except asyncio.CancelledError:
            break
        if _engine.vc_active and not _engine.hot_update:
            output, out_db, infer_ms = await loop.run_in_executor(
                _executor, _engine.on_chunk_received, chunk
            )
            await _route_output(output, ts, sid, _engine.input_level_db, out_db, infer_ms)


async def _status_push_loop() -> None:
    while _server_running:
        payload = {
            "server_running": _server_running,
            **_get_gpu_stats(),
            **_get_system_stats(),
        }
        await sio.emit("SET_STATUS", payload)
        await asyncio.sleep(1.0)

# ---------------------------------------------------------------------------
# Socket.IO event handlers
# ---------------------------------------------------------------------------

@sio.event
async def connect(sid, environ):
    print("Connection event.")
    logging.info(f"Client connected: {sid}")
    if _active_client_id is not None:
        logging.warning(f"Rejecting {sid}: session {_active_sid} is already connected.")
        await _emit_message(2, "Another client is already connected; only one client is permitted at this time.", sid=sid)
        await sio.disconnect(sid)
        return
    await sio.emit("LIST_DEVICES", {"input": _input_devices, "output": _output_devices}, to=sid)


@sio.event
async def disconnect(sid):
    global _active_sid, _active_client_id
    logging.info(f"Client disconnected: {sid}")
    if sid == _active_sid:
        #logging.info(f"Clearing association {_active_sid} <-> {_active_client_id}.")
        _active_sid = None
        _active_client_id = None
    if _engine.vc_active:
        logging.warning("WARN: Client disconnected while conversion active. Stopping engine.")
        await _stop_server()


@sio.on("PING")
async def on_ping(sid, data):
    await sio.emit("PONG", {
        "received_timestamp": data.get("timestamp"),
        "server_timestamp": time.time(),
    }, to=sid)


@sio.on("GET_STATUS")
async def on_get_status(sid, data=None):
    payload = {
        "server_running": _server_running,
        **_get_gpu_stats(),
        **_get_system_stats(),
        "host_name": socket.gethostname(),
    }
    await sio.emit("SET_STATUS", payload, to=sid)


@sio.on("GET_PARAMS")
async def on_get_params(sid, data):
    global _active_sid, _active_client_id, _settings
    client_id = (data or {}).get("client_id")
    if not isinstance(client_id, str) or not client_id:
        await _emit_message(2, "GET_PARAMS missing required 'client_id'.", sid=sid)
        return

    if _active_client_id is not None and _active_sid != sid:
        await _emit_message(2, "Another client is already connected; only one client is permitted at a time.", sid=sid)
        await sio.disconnect(sid)
        return

    _active_sid = sid
    _active_client_id = client_id
    #logging.info(f"Associated sid {sid} with client_id '{client_id}'.")

    _settings = _load_settings(client_id)
    await sio.emit("SET_PARAMS", dict(_settings), to=sid)


@sio.on("GET_MODELS")
async def on_get_models(sid, data=None):
    models_dir = Path(__file__).parent / "models"
    models = [d.name for d in models_dir.iterdir() if d.is_dir()] if models_dir.exists() else []
    await sio.emit("LIST_MODELS", {"models": models}, to=sid)


@sio.on("GET_DEVICES")
async def on_get_devices(sid, data=None):
    #devices = ["client"] + _audio_io.list_devices()
    await sio.emit("LIST_DEVICES", {"input": _input_devices, "output": _output_devices}, to=sid)


@sio.on("UPDATE_PARAM")
async def on_update_param(sid, data):
    if _active_sid != sid or _active_client_id is None:
        await _emit_message(2, "Client must register a client_id via GET_PARAMS before updating parameters.", sid=sid)
        return
    for key, value in data.items():
        if key in _settings:
            logging.info(f"Updating {key} to: {value}")
            _settings[key] = value

            if _server_running and key in ("pitch_shift_semitones",
                                           "formant_shift",
                                           "use_output_rms",
                                           "use_noise_reduction",
                                           "use_vocode_smoothing"):
                _engine.set_param(key, value)
            elif _server_running and key == "input_volume":
                # Clients should be performing their own gating/
                # amplification. We don't know what conversions
                # they might be doing
                if _settings["input_device"] != "client":
                    _engine.set_param(key, value)
            elif _server_running and key == "output_volume":
                # Client should perform its own amplification
                if _settings["output_device"] != "client":
                    _engine.set_param(key, value)
            elif _server_running and key == "input_gate_db":
                # Client should perform its own gating
                if _settings["input_device"] != "client":
                    _engine.set_param(key, value)
            elif _server_running and key in ("chunk_pad_s",
                                             "crossfade_s",
                                             "use_fp32",
                                             "use_jit",
                                             "rvc_index_rate",
                                             "f0_method"):
                _engine.hot_reload(key, value)
            else:
                if _server_running:
                    await _stop_server()
                    await _emit_message(1, f"Param '{key}' changed to '{value}'. Please restart conversion for changes to take effect.", sid=sid)
                    await sio.emit("SERVER_STOPPED", None, to=sid)

        else:
            logging.info(f"Received invalid parameter to update: {key}")
            await _emit_message(2, f"Received invalid parameter to update: {key}", sid=sid)
    # await sio.emit("SET_PARAMS", dict(_settings), to=sid)


@sio.on("MODEL_UNLOAD")
async def on_model_unload(sid, data=None):
    await _emit_message(0, "Model unload requested (No model loaded).", sid=sid)


@sio.on("MODEL_LOAD")
async def on_model_load(sid, data):
    model_name = data.get("model_name", _settings.get("model_name", "defaultvoice"))
    model_dir = Path(__file__).parent / "models" / model_name
    if not model_dir.is_dir():
        await sio.emit("MODEL_LOAD_FAIL", {"model_name": model_name, "reason": "Directory not found"}, to=sid)
        return
    _settings["model_name"] = model_name
    # TODO: Actually load model using _engine, check valid files + success
    #if _engine.load_model(model_name):
    await sio.emit("MODEL_LOAD_SUCCESS", {"model_name": model_name}, to=sid)


@sio.on("SERVER_SAVE")
async def on_server_save(sid, data=None):
    if _active_sid != sid or _active_client_id is None:
        await _emit_message(2, "Client must register a client_id via GET_PARAMS before saving settings.", sid=sid)
        return
    _save_settings(_active_client_id)
    await _emit_message(0, f"Settings saved for client '{_active_client_id}'.", sid=sid)


@sio.on("SERVER_REQUEST_EXIT")
async def on_server_request_exit(sid, data=None):
    await sio.emit("SERVER_EXITING", {}, to=sid)
    await _stop_server()
    await asyncio.sleep(0.3)
    os.kill(os.getpid(), signal.SIGTERM)


@sio.on("SERVER_START")
async def on_server_start(sid, data=None):
    global _server_running, _audio_queue, _processing_task, _status_task

    if _active_sid != sid or _active_client_id is None:
        await _emit_message(2, "Client must register a client_id via GET_PARAMS before starting the server.", sid=sid)
        return

    if _server_running:
        await _emit_message(2, "Server is already running.", sid=sid)
        return

    _server_running = True

    # Fresh queue — discard any leftover chunks from last session
    _audio_queue = asyncio.Queue()
    _processing_task = asyncio.create_task(_audio_processing_loop(origin_sid=sid))
    _status_task = asyncio.create_task(_status_push_loop())

    # Before any start, make sure configuration is current
    _engine.configure(_settings)
    _engine.start_vc()

    # Determine input sample rate
    input_device = _settings.get("input_device", "client")
    output_device = _settings.get("output_device", "client")

    if input_device != "client":
        try:
            input_sr = _audio_io.get_device_sample_rate(input_device)
            if input_sr != _settings.get("input_sr", 0):
                await _emit_message(2, "Invalid sample rate on server input; correcting", sid=sid)
                _settings["input_sr"] = input_sr
        except Exception as e:
            _server_running = False
            await _emit_message(2, f"Failed to get intput device sample rate: {e}", sid=sid)
            return

    if output_device != "client":
        try:
            output_sr = _audio_io.get_device_sample_rate(output_device)
            if output_sr != _settings.get("output_sr", 0):
                await _emit_message(2, "Invalid sample rate on server output; correcting", sid=sid)
                _settings["output_sr"] = output_sr
        except Exception as e:
            _server_running = False
            await _emit_message(2, f"Failed to get output device sample rate: {e}", sid=sid)
            return

    _engine.configure(_settings)

    # Start hardware input if needed
    if input_device != "client":
        try:
            loop = asyncio.get_event_loop()     # Python >=3.10 needs asyncio.get_running_loop() ?
            _audio_io.start_input(
                input_device, input_sr, _engine.chunk_size_samples, _on_hardware_chunk, loop
            )
        except Exception as e:
            _server_running = False
            _audio_io.stop_all()
            await _emit_message(2, f"Failed to open input device: {e}", sid=sid)
            return

    # Start hardware output if needed
    if output_device != "client":
        try:
            _audio_io.start_output(output_device, output_sr)
        except Exception as e:
            _server_running = False
            _audio_io.stop_all()
            await _emit_message(2, f"Failed to open output device: {e}", sid=sid)
            return

    logging.info("Conversion engine started.")
    await sio.emit("SERVER_STARTED", {
        "input_sample_rate": _settings["input_sr"],
        "output_sample_rate": _settings["output_sr"],
        "received_timestamp": (data or {}).get("timestamp"),
        "server_timestamp": time.time(),
    }, to=sid)

    _save_settings(_active_client_id)


@sio.on("SERVER_STOP")
async def on_server_stop(sid, data=None):
    await _stop_server()
    logging.info("Conversion engine stopped.")
    await sio.emit("SERVER_STOPPED", {}, to=sid)


@sio.on("AUDIO_IN")
async def on_audio_in(sid, data):
    global _audio_chunk_buffer, _engine

    if not _engine.vc_active or _settings.get("input_device") != "client" or  _engine.hot_update:
        return

    # Accept audio input of any size (leaves payload size, network
    # chunking, etc to other layers) and split into chunks of
    # appropriate size for the vc engine
    if _audio_chunk_buffer is not None:
        _audio_chunk_buffer = np.concatenate((_audio_chunk_buffer, np.frombuffer(data["samples"], dtype=np.float32)))
    else:
        _audio_chunk_buffer = np.frombuffer(data["samples"], dtype=np.float32)

    #logging.info(f"In buf: {_audio_chunk_buffer.shape[-1]} samples")

    # buffer should be a 1-dimensional array (mono audio) when received
    # over the network, so we can get away with simple comparison
    while _audio_chunk_buffer.shape[-1] >= _engine.chunk_size_samples:
        chunk = _audio_chunk_buffer[:_engine.chunk_size_samples]
        _audio_chunk_buffer = _audio_chunk_buffer[_engine.chunk_size_samples:]
        await _enqueue_chunk(chunk, float(data["timestamp"]), sid)

# ---------------------------------------------------------------------------
# App shutdown hook
# ---------------------------------------------------------------------------

async def _on_shutdown(app):
    await sio.emit("SERVER_EXITING", {}, to=None)    #catch-all
    await _stop_server()

app.on_shutdown.append(_on_shutdown)

# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Uwarevoice server")
    parser.add_argument("--host", default="0.0.0.0")
    parser.add_argument("--port", type=int, default=8765)
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
        force=True,     # Can't find where this is getting set earlier in the
                        # chain, but logging.INFO isn't working without it
    )

    global _input_devices, _output_devices
    logging.info("Getting list of available server audio devices...")
    _input_devices, _output_devices = _audio_io.list_devices()
    logging.info(f"Found {len(_input_devices) - 1} input devices and {len(_output_devices) - 1} output devices.")

    logging.info(f"CUDA available: {_CUDA_AVAILABLE}")
    if _settings.get("inference_device") == "cuda" and not _CUDA_AVAILABLE:
        logging.warning(
            "inference_device is 'cuda' but no CUDA GPU detected!"
        )

    aiohttp.web.run_app(app, host=args.host, port=args.port)

if __name__ == "__main__":
    main()
