# AI Radio Station

Real-time AI music generation web app powered by Google's Lyria Realtime API.

## Architecture

```
┌─────────────────┐     WebSocket      ┌─────────────────┐
│                 │◄──────────────────►│                 │
│  Browser        │   JSON messages    │  FastAPI        │
│  (Web Audio)    │   + base64 PCM     │  Server         │
│                 │                    │                 │
└─────────────────┘                    └────────┬────────┘
                                                │
                                                │ gRPC streaming
                                                │
                                       ┌────────▼────────┐
                                       │  Google Lyria   │
                                       │  Realtime API   │
                                       │  (v1alpha)      │
                                       └─────────────────┘
```

## Audio Pipeline

### Backend → Frontend
- **Format**: 16-bit signed PCM, 48kHz sample rate, stereo interleaved
- **Transport**: Base64-encoded chunks over WebSocket JSON messages
- **Chunk size**: Variable, typically ~4096 samples per channel

### Frontend Audio Processing
```
AudioContext (48kHz)
    │
    ▼
BufferSource ──► GainNode ──► AnalyserNode ──► Destination
                (volume)      (FFT data)       (speakers)
```

- **Buffer strategy**: 5-second lookahead buffer before playback starts
- **Scheduling**: Gapless playback via `AudioBufferSourceNode.start(nextStartTime)`
- **Underrun handling**: Automatic rebuffering on buffer underrun detection

### Frequency Visualizer
- **FFT size**: 2048 (1024 frequency bins)
- **Frequency mapping**: Logarithmic scale, 20Hz–16kHz across 64 bars
- **Bin aggregation**: Multiple FFT bins averaged per visual bar for perceptually uniform distribution
- **Color mapping**: HSL-based rainbow spectrum, saturation/lightness modulated by amplitude

## WebSocket Protocol

### Client → Server

```json
{"action": "init", "api_key": "...", "config": {...}}
{"action": "play"}
{"action": "pause"}
{"action": "stop"}
{"action": "update", "config": {"bpm": 120, "density": 0.5, ...}}
{"action": "disconnect"}
```

### Server → Client

```json
{"type": "connected", "config": {...}}
{"type": "playing"}
{"type": "paused"}
{"type": "stopped"}
{"type": "updated", "config": {...}}
{"type": "audio", "data": "<base64 PCM>"}
{"type": "filtered", "reason": "...", "text": "..."}
{"type": "error", "message": "..."}
{"type": "log", "message": "..."}
```

## Lyria API Integration

### Session Lifecycle
```python
client = genai.Client(api_key=key, http_options={"api_version": "v1alpha"})
async with client.aio.live.music.connect(model="models/lyria-realtime-exp") as session:
    await session.set_weighted_prompts(prompts=[...])
    await session.set_music_generation_config(config=...)
    await session.play()
    async for message in session.receive():
        # Process audio chunks
```

### Configuration Parameters

| Parameter | Range | Notes |
|-----------|-------|-------|
| `bpm` | 60–200 | Requires `reset_context()` on change |
| `density` | 0.0–1.0 | 0.1=sparse, 0.5=balanced, 0.9=chaotic |
| `brightness` | 0.0–1.0 | Tonal brightness |
| `guidance` | 1.0–6.0 | Prompt adherence strength |
| `temperature` | 0.0–3.0 | Output randomness |
| `scale` | enum | Musical key, requires `reset_context()` |
| `music_generation_mode` | QUALITY/DIVERSITY | Coherence vs variation tradeoff |

### Prompt Format
Prompts follow the formula: `[Genre] + [Instrumentation] + [Atmosphere]`

Example: `"Techno, TR-909 Drum Machine, 303 Acid Bass, Driving"`

## Requirements

- Python 3.11+
- [uv](https://github.com/astral-sh/uv) package manager
- Google Gemini API key with Lyria access

## Setup

```bash
uv sync
uv run python main.py
```

Open http://localhost:8000

## Dependencies

| Package | Purpose |
|---------|---------|
| `fastapi` | HTTP server + WebSocket |
| `uvicorn` | ASGI server |
| `google-genai` | Lyria API client |
| `websockets` | WebSocket support |
| `python-multipart` | Form data parsing |

## Project Structure

```
ai_tunes/
├── main.py              # FastAPI server, Lyria session management
├── static/
│   └── index.html       # SPA frontend (HTML/CSS/JS)
├── pyproject.toml       # Dependencies
└── uv.lock              # Locked dependencies
```

## License

MIT
