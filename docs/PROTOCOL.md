# Uwarevoice event protocol v1.0

## Client to Server:
| Event | Expected Payload | Description |
|-------|---------|-------------|
| GET_PARAMS  | `client_id` (string) | Requests current conversion settings stored by the server. Should always be the first message sent by client once the client is ready to send and receive messages. `client_id` should contain a unique identifier that may be used across sessions to store and recall this client's settings. |
| SERVER_START    | *(none)* | Begin audio conversion using the current settings. |
| SERVER_STOP | *(none)* | Stop audio conversion (client initiated). |
| SERVER_SAVE | *(none)* | Save the current conversion settings to a file associated with the current `client_id` (which must be registered in advance using GET_PARAMS). |
| SERVER_REQUEST_EXIT | *(none)* | Request shutdown/close of all threads, and to have the server exit. |
| GET_STATUS  | *(none)* | Request server's current statistics. |
| GET_MODELS  | *(none)* | Request list of models stored on this server. |
| GET_DEVICES | *(none)* | Request list of audio devices connected to this server. |
| MODEL_UNLOAD    | *(none)* | Request to unload the current model and destroy/clean up any objects. |
| MODEL_LOAD  | `model_name` *(string)* | Request to load the specified model. |
| UPDATE_PARAM | key/value pair *(type varies, may include multiple pairs)* | Set new value for the named setting. *See* **Parameters** section. |
| AUDIO_IN | `samples` *(binary float32[])*<br />`timestamp` *(double)* | Unprocessed audio stream from client. |
| PING | `timestamp` (double) | Ping? |

## Server to Client:
| Event | Expected Payload | Description |
|-------|---------|----------------------|
| SERVER_STARTED  | `input_sample_rate` *(integer)*<br />`output_sample_rate` *(integer)*<br />`received_timestamp` *(double)*<br />`server_timestamp` *(double)* | Notifies that audio conversion has started successfully. |
| SERVER_STOPPED  | *(none)* | Notifies that conversion has stopped. |
| SERVER_EXITING  | *(none)* | Notifies that the server is shutting down/exiting. |
| SET_STATUS | Any combination of:<br />`server_running` *(bool)*<br />`gpu_load` *(int)*<br />`gpu_mem_used_mib` *(int)*<br />`gpu_mem_total_mib` *(int)*<br />`cpu_percent` *(double)*<br />`mem_used_mib` *(int)*<br />`mem_total_mib` *(int)* | Current state of various server statistics. |
| SET_PARAMS | Any combination of key/value pairs *(types vary)* | Set new values for the named settings; *see* **Parameters** section. |
| LIST_MODELS | `models` *(array of strings)*| Contains a list of models currently available on this server / to this client. |
| LIST_DEVICES | `input` *(array of strings)*<br />`output` *(array of strings)* | Contains a list of audio devices currently available to this server. |
| MODEL_LOAD_SUCCESS | `model_name` *(string)* | Notifies that the specified model has been loaded and is ready for use. |
| MODEL_LOAD_FAIL | `model_name` *(string)*<br />`message` *(string)* | Notifies that the specified model has failed to load; with optional error message. |
| STREAM_OVERLOAD | `dropped_chunks` *(int)* | Notifies when audio chunks cannot be processed as fast as they are being sent. |
| MESSAGE | `message_level` *(int)*<br />`message` *(string)* | Info (1)/Warn (2)/Error (3) message for the user. |
| AUDIO_OUT | `timestamp` *(double)*<br />`in_db` *(double)*<br />`out_db` *(double)*<br />`inference_time_ms` *(double)*<br /> Varies/optional:<br />`sample_count` *(int)*<br />`samples` *(binary float32[])* | Notifies of converted audio chunk; contains samples (if applicable). `sample_count` should be 0 if this message is only being sent to notify the client of input/output occurring on server audio device. |
| PONG | `received_timestamp` *(double)*<br />`server_timestamp` *(double)* | Pong! |

## Parameters
| Parameter | Expected Type | Description |
|-------|---------|----------|
| `input_volume` | double | Self-described (Range: 0.0 to 2.0) |
| `output_volume` | double | Self-described (Range: 0.0 to 2.0) |
| `input_gate_db` | int | (Only applies to server-side audio input) The audio level, in dBFS, at which input should be considered silent. (Range: -140 to 0) |
| `input_sr` | int | Input Sample Rate |
| `output_sr` | int | Output Sample Rate |
| `pitch_shift_semitones` | int | Pitch shift to be performed on the input, in semitones (Range: -12 to 12) |
| `formant_shift` | int | Formant (timbre) shift to be performed on the input (Range: -9 to 9)
| `chunk_size_s` | double | Size, in seconds, of each input/output chunk |
| `chunk_pad_s` | double | Size, in seconds, of context/padding that should be included in each inference pass |
| `crossfade_s` |  double | Size, in seconds, of the crossfade between each chunk |
| `use_output_rms` | double | Volume equalization to be applied between input and output (Range: 0.0 to 1.0) |
| `use_noise_reduction` | bool | Whether to include a noise reduction pass on input chunks |
| `use_vocode_smoothing` | bool | Whether to include a smoothing pass on crossfaded output chunks |
| `use_fp32` | bool | Whether to use 32 or 16 bit precision in the inference step (RVC specific) |
| `use_jit` | bool | Whether to use the JIT compiler (RVC specific) |
| `model_name` | string | Self-described |
| `inference_device` | string | Self-described |
| `input_device` | string | Informs server of user's audio device selection |
| `output_device` | string | Informs server of user's audio device selection |
