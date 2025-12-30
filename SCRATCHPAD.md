# AI Radio Station - Development Scratchpad

## Current Status

**Working**: Basic app is functional with real-time music generation via Lyria API.

## Known Issues

### Audio Quality
- [ ] Some crackling/artifacts in audio output
- [ ] Non-smooth transitions between audio chunks
- [x] Increased buffer to 5 seconds (per Lyria guide settling period)
- [ ] Consider implementing AudioWorklet with ring buffer for smoother playback

### Music Generation
- [ ] Some genres still sound disconnected
- [x] Updated prompts to follow Lyria formula: [Genre] + [Instrumentation] + [Atmosphere]
- [x] Using specific instrument names (TR-909, TB-303, etc.) instead of vague descriptions
- [x] Implemented weighted multi-prompt blending for smoother transitions (crossfade)

## Recent Changes

### 2025-12-30 (Session 5)
- **Implemented crossfade transitions** - Smooth genre/prompt transitions using weighted multi-prompts
  - Backend: `_crossfade()` method gradually shifts weights from old to new prompt over 3 seconds
  - Cancels in-flight crossfades when new transition requested (no stacking)
  - Sends `crossfade_start` and `crossfade_end` WebSocket messages for UI feedback
- **Added Transitions toggle** - UI switch between "Instant" and "Smooth" (default)
  - Genre tags and "Update Station" button respect the setting
  - Debug log shows `[crossfade]` when smooth transitions active
- **Implemented Flavor Layers** - 6 toggleable flavor presets that layer weighted prompts
  - Tight Rhythm, Analog Warmth, Melodic, Live Feel, Atmospheric, Punchy Mix
  - Each flavor has enable toggle + weight slider (0.1-0.8)
  - Flavors add to main prompt (not replace) via Lyria weighted prompts API
  - Flavors reset when switching genres
  - Targets: disjointed beats, dull tones, unclear melodies, lack of variance

### 2025-12-29 (Session 4)
- Curated Quick Genres down to Lyria-friendly anchors (cinematic score, Beethoven symphony, ambient focus, lo-fi study, deep house, indie electronic, warehouse techno, festival trance, dub, jazz fusion) with production/use-case descriptors per Scenario + Google DEV guides.
- Added a negative prompt textarea (default "No vocals, no glitch artifacts, no harsh cymbal wash") and append it as an "Avoid" clause on the backend so every generation suppresses common artifacts.
- Switched the default preset to cinematic score (90 BPM, 50% density) so both backend and UI start in one of Lyria's strongest styles.

### 2024-12-29 (Session 3)
- **Replaced Classical with Focus genre** - Ambient style optimized for concentration (BPM 80, density 25%)
- **Real audio visualizer** - Connected AnalyserNode to audio chain for actual frequency data
- **Logarithmic frequency binning** - Proper octave-based mapping (20Hz-16kHz)
  - FFT size 2048 for better bass resolution
  - Each bar represents perceptually equal frequency range
- **Rainbow spectrum colors** - HSL-based coloring that responds to amplitude
- **Larger visualizer** - Height 120px, wider bars (4px), removed "Audio Visualizer" header
- **UI cleanup** - BPM display right-aligned, smoother transitions (0.05s)

### 2024-12-29 (Session 2)
- **Prompt restructure** following Lyria guide formula: `[Genre] + [Instrumentation] + [Atmosphere]`
- **Specific instruments** in prompts: TR-909, TB-303, Jupiter-8, Buchla, etc.
- **Per-genre density settings** added (e.g., Ambient=25%, DnB=70%)
- **Buffer time increased** from 2s to 5s for settling period
- **BPM range extended** to 60-200 (for DnB at 174)
- **Density range** changed to 10-90% (per guide: 0.1=sparse, 0.9=chaotic)
- New genres: Dub, Acid, Trance, Chillout (replaced R&B, Pop, Soul, etc.)

### 2024-12-29 (Session 1)
- Initial implementation complete
- FastAPI backend with WebSocket for real-time streaming
- Frontend with all controls matching reference app
- Audio player with buffer (matching reference implementation)
- Genre presets with BPM settings
- Debug logging for all control changes

## TODO

### High Priority
- [ ] Improve audio playback smoothness (consider AudioWorklet)
- [ ] Test different prompt structures for better rhythm
- [ ] Add error recovery for connection drops

### Medium Priority
- [ ] Add volume control
- [ ] Save user preferences to localStorage
- [ ] Add keyboard shortcuts (space = play/pause)
- [x] Improve visualizer to react to actual audio levels

### Low Priority
- [ ] Add more genre presets
- [ ] Dark mode theme
- [ ] Mobile responsive improvements
- [ ] Add sharing/export functionality

## Technical Notes

### Lyria API
- Uses `client.aio.live.music.connect()` - returns async context manager
- Audio format: 16-bit PCM, 48kHz, stereo (interleaved)
- Prompts sent via `session.set_weighted_prompts()`
- Config sent via `session.set_music_generation_config()`
- BPM and Scale changes require `session.reset_context()`
- Sessions limited to 10 minutes currently

### Lyria Prompting Best Practices (from official guide)
**Formula**: `[Genre Anchor] + [Instrumentation] + [Atmosphere]`

**Good instruments to reference**:
- Drums: TR-909, TR-808, Breakbeat
- Bass: TB-303, Reese Bass, Deep Bass
- Synths: Jupiter-8, Buchla, Moog, Supersaw
- Other: Clavinet, Rhodes Piano, Nylon Guitar

**Good atmosphere words**:
- Driving, Groovy, Hypnotic, Ethereal, Intense
- Mellow, Spacious, Uplifting, Euphoric

**Avoid**: Artist names ("Style of Daft Punk") - use descriptive elements instead

**Density guide**:
- 0.1 = sparse, minimal
- 0.5 = balanced
- 0.9 = chaotic, intense

**Settling period**: Allow 5-10 seconds after stream start for stable generation

### Weighted Multi-Prompts (implemented)
Blend multiple prompts with weights for smooth transitions:
```python
prompts=[
    WeightedPrompt(text="Old Genre", weight=0.3),  # Fading out
    WeightedPrompt(text="New Genre", weight=0.7),  # Fading in
]
```
Cross-fade implemented in `_crossfade()`: 6 steps over 3 seconds (500ms each)

### Audio Playback
- Web Audio API with AudioContext at 48kHz
- Audio chain: Source → GainNode → AnalyserNode → Destination
- 5-second buffer before playback starts (BUFFER_TIME = 5)
- Scheduling uses `nextStartTime` to queue chunks gaplessly
- Buffer underrun detection resets to loading state

### Audio Visualizer
- AnalyserNode with FFT size 2048 (1024 frequency bins)
- Logarithmic frequency mapping: 20Hz-16kHz across 64 bars
- Each bar averages FFT bins for its frequency range
- Rainbow HSL colors based on bar position
- Saturation/lightness increases with amplitude

### Default Settings (optimized for rhythm)
- BPM: 120
- Density: 70%
- Brightness: 50%
- Guidance: 5.0
- Temperature: 1.0
- Mode: Quality

## Ideas for Future

- Multiple simultaneous prompts with weights (like reference MIDI controller app)
- MIDI controller support
- Preset save/load system
- Collaborative listening sessions
- Peak hold with decay on visualizer bars
