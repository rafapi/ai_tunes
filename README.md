# AI Radio Station

Real-time AI music generation web app powered by Google's Lyria Realtime API.

## Features

- **Real-time music generation** using Google's Lyria model
- **Quick genre selection** with optimized prompts and BPM settings
- **Live controls**: BPM, Density, Brightness, Guidance, Temperature
- **Mix options**: Mute Bass, Mute Drums, Only Bass & Drums
- **Scale selection**: Choose musical key or let the model decide
- **Generation modes**: Quality (coherent) vs Diversity (varied)
- **Audio visualizer** with live waveform display
- **Debug log** for monitoring generation status

## Requirements

- Python 3.11+
- [uv](https://github.com/astral-sh/uv) package manager
- Google Gemini API key with Lyria access

## Setup

1. Install dependencies:
   ```bash
   uv sync
   ```

2. Run the server:
   ```bash
   uv run python main.py
   ```

3. Open http://localhost:8000 in your browser

4. Enter your Gemini API key and click Save

5. Click Play to start generating music

## Getting a Gemini API Key

1. Go to [Google AI Studio](https://aistudio.google.com/apikey)
2. Create or select a project
3. Generate an API key
4. Note: Lyria Realtime is experimental and may require waitlist access

## Tech Stack

- **Backend**: FastAPI + WebSocket
- **Frontend**: Vanilla HTML/CSS/JavaScript
- **Audio**: Web Audio API with 2-second buffer
- **AI**: Google GenAI Python SDK (v1alpha)

## Project Structure

```
ai_tunes/
├── main.py              # FastAPI server + Lyria integration
├── static/
│   └── index.html       # Frontend (HTML/CSS/JS)
├── pyproject.toml       # Python dependencies
├── CLAUDE.md            # Development instructions
├── SCRATCHPAD.md        # Work log and TODO items
└── README.md            # This file
```

## Available Genres

| Genre | BPM | Style |
|-------|-----|-------|
| Ambient | 70 | Atmospheric, peaceful |
| Reggae | 75 | Offbeat, laid back |
| Bossa Nova | 80 | Brazilian jazz |
| Lo-Fi Hip Hop | 85 | Chill, boom bap |
| R&B | 90 | Smooth, soulful |
| Soul | 95 | Motown, warm |
| Funk | 105 | Syncopated, groovy |
| Afrobeat | 110 | Percussive, danceable |
| Synthwave | 118 | Retro 80s |
| Disco | 120 | Funky, dance |
| Pop | 120 | Catchy, bright |
| House | 124 | Four on the floor |
| Techno | 128 | Driving, electronic |
| Drum and Bass | 174 | Fast breakbeats |

## Controls Reference

- **BPM** (60-180): Beats per minute - requires context reset
- **Density** (0-100%): Note/instrument density
- **Brightness** (0-100%): Tonal brightness
- **Guidance** (1-6): How strictly to follow the prompt
- **Temperature** (0-3): Randomness/variety in output
- **Scale**: Musical key - requires context reset
- **Mode**: Quality (coherent) vs Diversity (varied)

## License

MIT
