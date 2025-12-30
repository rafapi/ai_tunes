"""
AI Radio Station - Real-time music generation powered by Google Lyria
"""

import asyncio
import base64
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Optional

from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
from google import genai
from google.genai import types


# Flavor presets for layered prompt blending
FLAVOR_PRESETS = {
    "tight_rhythm": "Tight quantized drums, locked groove, steady beat",
    "analog_warmth": "Warm analog saturation, tube compression, vinyl texture",
    "melodic": "Clear melodic hooks, memorable motifs, singable themes",
    "live_feel": "Live performance energy, human timing, natural dynamics",
    "atmospheric": "Spacious reverb, evolving pads, ambient textures",
    "punchy_mix": "Punchy transients, clear separation, professional mixdown",
}

# Scale mapping from string to enum
SCALE_MAP = {
    "C_MAJOR_A_MINOR": types.Scale.C_MAJOR_A_MINOR,
    "D_FLAT_MAJOR_B_FLAT_MINOR": types.Scale.D_FLAT_MAJOR_B_FLAT_MINOR,
    "D_MAJOR_B_MINOR": types.Scale.D_MAJOR_B_MINOR,
    "E_FLAT_MAJOR_C_MINOR": types.Scale.E_FLAT_MAJOR_C_MINOR,
    "E_MAJOR_D_FLAT_MINOR": types.Scale.E_MAJOR_D_FLAT_MINOR,
    "F_MAJOR_D_MINOR": types.Scale.F_MAJOR_D_MINOR,
    "G_FLAT_MAJOR_E_FLAT_MINOR": types.Scale.G_FLAT_MAJOR_E_FLAT_MINOR,
    "G_MAJOR_E_MINOR": types.Scale.G_MAJOR_E_MINOR,
    "A_FLAT_MAJOR_F_MINOR": types.Scale.A_FLAT_MAJOR_F_MINOR,
    "A_MAJOR_G_FLAT_MINOR": types.Scale.A_MAJOR_G_FLAT_MINOR,
    "B_FLAT_MAJOR_G_MINOR": types.Scale.B_FLAT_MAJOR_G_MINOR,
    "B_MAJOR_A_FLAT_MINOR": types.Scale.B_MAJOR_A_FLAT_MINOR,
}


class MusicSession:
    """Manages a live music generation session with Lyria."""

    def __init__(self, api_key: str):
        self.client = genai.Client(api_key=api_key, http_options={"api_version": "v1alpha"})
        self.session = None
        self._context_manager = None
        self.is_playing = False
        self._receive_task: Optional[asyncio.Task] = None
        self._websocket = None
        self._crossfade_task: Optional[asyncio.Task] = None
        self._active_prompt: Optional[str] = None  # Track currently playing prompt
        self._ws_send_lock: asyncio.Lock = asyncio.Lock()  # Protect concurrent WebSocket sends
        self.current_config = {
            "prompt": (
                "Cinematic score with 100-piece orchestra, soaring strings, heroic brass fanfares, "
                "thunderous percussion, sweeping crescendos, blockbuster mix crafted for trailer moments"
            ),
            "bpm": 90,
            "density": 0.5,  # Moderate density (0.1=sparse, 0.9=chaotic)
            "brightness": 0.5,
            "guidance": 5.0,  # Higher guidance to follow prompt more closely
            "temperature": 1.0,  # Default per Lyria guide
            "scale": None,  # None means model decides
            "music_generation_mode": "QUALITY",  # QUALITY for more coherent music
            "mute_bass": False,
            "mute_drums": False,
            "only_bass_and_drums": False,
            "negative_prompt": "No vocals, no glitch artifacts, no harsh cymbal wash",
            "flavors": {},  # Active flavor layers: {"tight_rhythm": {"enabled": True, "weight": 0.3}, ...}
        }

    async def connect(self, max_retries: int = 3) -> bool:
        """Establish connection to Lyria with auto-retry on failure."""
        last_error = None

        for attempt in range(max_retries):
            try:
                # connect() returns an async context manager
                self._context_manager = self.client.aio.live.music.connect(
                    model="models/lyria-realtime-exp"
                )
                # Enter the context manager to get the session
                self.session = await self._context_manager.__aenter__()
                # Set initial configuration
                await self._apply_config()
                await self._apply_prompts()
                if attempt > 0:
                    print(f"Connected successfully after {attempt + 1} attempts")
                return True
            except Exception as e:
                last_error = e
                if attempt < max_retries - 1:
                    wait_time = 2 ** (attempt + 1)  # 2s, 4s, 8s
                    print(f"Connection attempt {attempt + 1} failed: {e}")
                    print(f"Retrying in {wait_time}s...")
                    await asyncio.sleep(wait_time)

        print(f"Failed to connect after {max_retries} attempts: {last_error}")
        return False

    async def _apply_config(self) -> None:
        """Apply current music generation config."""
        if not self.session:
            return

        config = types.LiveMusicGenerationConfig(
            bpm=self.current_config["bpm"],
            density=self.current_config["density"],
            brightness=self.current_config["brightness"],
            guidance=self.current_config["guidance"],
            temperature=self.current_config["temperature"],
            mute_bass=self.current_config["mute_bass"],
            mute_drums=self.current_config["mute_drums"],
            only_bass_and_drums=self.current_config["only_bass_and_drums"],
        )

        # Set scale if specified
        scale_key = self.current_config.get("scale")
        if scale_key and scale_key in SCALE_MAP:
            config.scale = SCALE_MAP[scale_key]

        # Set music generation mode
        mode = self.current_config.get("music_generation_mode")
        if mode == "DIVERSITY":
            config.music_generation_mode = types.MusicGenerationMode.DIVERSITY
        elif mode == "QUALITY":
            config.music_generation_mode = types.MusicGenerationMode.QUALITY

        await self.session.set_music_generation_config(config=config)

    async def _apply_prompts(self, prompts_list: list[types.WeightedPrompt] | None = None) -> None:
        """Apply weighted prompts. If prompts_list is None, uses current config prompt + flavors."""
        if not self.session:
            return

        if prompts_list is None:
            # Build main prompt
            prompt_text = self.current_config["prompt"].strip()
            negative_prompt = self.current_config.get("negative_prompt", "").strip()
            if negative_prompt:
                prompt_text = f"{prompt_text}. Avoid: {negative_prompt}"
            prompts_list = [types.WeightedPrompt(text=prompt_text, weight=1.0)]
            self._active_prompt = prompt_text

            # Add active flavor layers
            flavors = self.current_config.get("flavors", {})
            for flavor_key, flavor_data in flavors.items():
                if flavor_data.get("enabled") and flavor_key in FLAVOR_PRESETS:
                    weight = flavor_data.get("weight", 0.3)
                    prompts_list.append(
                        types.WeightedPrompt(text=FLAVOR_PRESETS[flavor_key], weight=weight)
                    )

        await self.session.set_weighted_prompts(prompts=prompts_list)

    async def _safe_send_json(self, data: dict, websocket: WebSocket | None = None) -> bool:
        """Send JSON to websocket with lock to prevent concurrent send collisions.

        Args:
            data: JSON-serializable dict to send
            websocket: Optional websocket to use (defaults to self._websocket)

        Returns False and clears _websocket on failure, signaling disconnect.
        """
        ws = websocket or self._websocket
        if not ws:
            return False
        try:
            async with self._ws_send_lock:
                await ws.send_json(data)
            return True
        except Exception:
            # Client disconnected - clear internal websocket to signal other tasks
            if ws is self._websocket:
                self._websocket = None
            return False

    async def _crossfade(
        self,
        old_prompt: str,
        new_prompt: str,
        duration: float = 3.0,
        steps: int = 6,
    ) -> None:
        """Smoothly crossfade from old prompt to new prompt over duration seconds."""
        step_time = duration / steps
        negative_prompt = self.current_config.get("negative_prompt", "").strip()

        # Add negative prompt suffix to both
        old_full = f"{old_prompt}. Avoid: {negative_prompt}" if negative_prompt else old_prompt
        new_full = f"{new_prompt}. Avoid: {negative_prompt}" if negative_prompt else new_prompt

        try:
            for i in range(steps + 1):
                # Check if session is still valid
                if not self.session or not self._websocket:
                    break

                alpha = i / steps  # 0.0 -> 1.0

                prompts = [
                    types.WeightedPrompt(text=old_full, weight=1.0 - alpha),
                    types.WeightedPrompt(text=new_full, weight=alpha),
                ]
                await self._apply_prompts(prompts)

                # Log progress at key points (using safe send)
                if i == 0:
                    await self._safe_send_json({
                        "type": "crossfade_start",
                        "from": old_prompt[:50],
                        "to": new_prompt[:50],
                    })
                elif i == steps:
                    await self._safe_send_json({"type": "crossfade_end"})

                await asyncio.sleep(step_time)

            # Final state: just the new prompt at full weight
            if self.session:
                self._active_prompt = new_full
                await self._apply_prompts([types.WeightedPrompt(text=new_full, weight=1.0)])

        except asyncio.CancelledError:
            # Crossfade was interrupted by another transition
            raise  # Re-raise so caller knows it was cancelled
        except Exception as e:
            # Transport error or session closed - log and clean up
            print(f"Crossfade error: {e}")
        finally:
            self._crossfade_task = None

    async def update_config(self, websocket: WebSocket, **kwargs) -> None:
        """Update music generation config."""
        needs_reset = False
        old_bpm = self.current_config.get("bpm")
        old_scale = self.current_config.get("scale")
        old_prompt = self.current_config.get("prompt", "")

        # Extract crossfade settings before updating config
        use_crossfade = kwargs.pop("smooth_transition", False)
        crossfade_duration = kwargs.pop("crossfade_duration", 3.0)

        self.current_config.update(kwargs)

        # Check if we need a context reset (BPM or scale changed)
        if kwargs.get("bpm") is not None and kwargs["bpm"] != old_bpm:
            needs_reset = True
        if "scale" in kwargs and kwargs["scale"] != old_scale:
            needs_reset = True

        if self.session:
            try:
                # Apply config changes
                await self._apply_config()

                # Handle prompt changes
                prompt_changed = "prompt" in kwargs and kwargs["prompt"] != old_prompt
                negative_changed = "negative_prompt" in kwargs
                flavors_changed = "flavors" in kwargs

                if prompt_changed or negative_changed or flavors_changed:
                    if use_crossfade and prompt_changed and self._active_prompt:
                        # Cancel any existing crossfade before starting new one
                        await self._cancel_crossfade()

                        # Start new crossfade in background
                        self._crossfade_task = asyncio.create_task(
                            self._crossfade(
                                old_prompt,
                                kwargs["prompt"],
                                duration=crossfade_duration,
                            )
                        )
                    else:
                        # Immediate transition (includes flavor-only changes)
                        await self._apply_prompts()

                # Reset context if needed for BPM/scale changes
                if needs_reset:
                    await self.session.reset_context()
                    await self._safe_send_json({"type": "log", "message": "Context reset"}, websocket)

            except Exception as e:
                print(f"Failed to update config: {e}")
                await self._safe_send_json({"type": "error", "message": str(e)}, websocket)

    async def _receive_loop(self) -> None:
        """Background task to receive and forward audio chunks."""
        try:
            async for message in self.session.receive():
                if not self.is_playing or not self._websocket:
                    break

                if message.server_content:
                    # Handle audio chunks
                    if hasattr(message.server_content, 'audio_chunks') and message.server_content.audio_chunks:
                        for chunk in message.server_content.audio_chunks:
                            audio_data = base64.b64encode(chunk.data).decode()
                            if not await self._safe_send_json({
                                "type": "audio",
                                "data": audio_data,
                            }):
                                self.is_playing = False
                                return

                    # Handle filtered prompts
                    if hasattr(message.server_content, 'filtered_prompt') and message.server_content.filtered_prompt:
                        fp = message.server_content.filtered_prompt
                        await self._safe_send_json({
                            "type": "filtered",
                            "reason": getattr(fp, 'filtered_reason', 'Unknown'),
                            "text": getattr(fp, 'text', ''),
                        })

                # Small yield to prevent blocking
                await asyncio.sleep(0)

        except asyncio.CancelledError:
            pass
        except Exception as e:
            print(f"Receive loop error: {e}")
            await self._safe_send_json({"type": "error", "message": str(e)})

    async def play(self, websocket: WebSocket) -> None:
        """Start playing and streaming audio to websocket."""
        if not self.session:
            return

        # If already playing, just resume
        if self.is_playing:
            return

        self._websocket = websocket
        self.is_playing = True

        try:
            await self.session.play()

            # Start receive loop only if not already running
            if self._receive_task is None or self._receive_task.done():
                self._receive_task = asyncio.create_task(self._receive_loop())

        except Exception as e:
            print(f"Play error: {e}")
            self.is_playing = False
            await self._safe_send_json({"type": "error", "message": str(e)}, websocket)

    async def pause(self) -> None:
        """Pause playback."""
        self.is_playing = False
        if self.session:
            try:
                await self.session.pause()
            except Exception as e:
                print(f"Pause error: {e}")

    async def _cancel_crossfade(self) -> None:
        """Cancel any in-flight crossfade task."""
        if self._crossfade_task and not self._crossfade_task.done():
            self._crossfade_task.cancel()
            try:
                await self._crossfade_task
            except asyncio.CancelledError:
                pass
            self._crossfade_task = None

    async def stop(self) -> None:
        """Stop playback."""
        self.is_playing = False
        await self._cancel_crossfade()
        if self._receive_task and not self._receive_task.done():
            self._receive_task.cancel()
            try:
                await self._receive_task
            except asyncio.CancelledError:
                pass
            self._receive_task = None
        if self.session:
            try:
                await self.session.stop()
            except Exception as e:
                print(f"Stop error: {e}")

    async def close(self) -> None:
        """Close the session."""
        self.is_playing = False
        await self._cancel_crossfade()
        if self._receive_task and not self._receive_task.done():
            self._receive_task.cancel()
            try:
                await self._receive_task
            except asyncio.CancelledError:
                pass
            self._receive_task = None
        if self._context_manager:
            try:
                await self._context_manager.__aexit__(None, None, None)
            except Exception:
                pass
            self._context_manager = None
            self.session = None
        self._websocket = None


# Active sessions by websocket
sessions: dict[int, MusicSession] = {}


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application lifespan handler."""
    yield
    # Cleanup sessions on shutdown
    for session in sessions.values():
        await session.close()


app = FastAPI(title="AI Radio Station", lifespan=lifespan)

# Mount static files
static_path = Path(__file__).parent / "static"
app.mount("/static", StaticFiles(directory=static_path), name="static")


@app.get("/")
async def index():
    """Serve the main page."""
    return FileResponse(static_path / "index.html")


@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    """WebSocket endpoint for real-time music streaming."""
    await websocket.accept()
    session_id = id(websocket)
    session: Optional[MusicSession] = None

    async def send(data: dict) -> None:
        """Send JSON through session lock if available, else direct."""
        if session:
            await session._safe_send_json(data, websocket)
        else:
            await websocket.send_json(data)

    try:
        while True:
            data = await websocket.receive_json()
            action = data.get("action")

            if action == "init":
                api_key = data.get("api_key")
                if not api_key:
                    await send({"type": "error", "message": "API key required"})
                    continue

                session = MusicSession(api_key)
                if data.get("config"):
                    session.current_config.update(data["config"])

                await send({"type": "log", "message": "Connecting to Lyria (may retry if unavailable)..."})
                if await session.connect():
                    sessions[session_id] = session
                    await send({"type": "connected", "config": session.current_config})
                else:
                    await send({"type": "error", "message": "Lyria service unavailable after 3 retries. Please try again later."})

            elif action == "play":
                if session and session.session:
                    await session.play(websocket)
                    await send({"type": "playing"})
                else:
                    await send({"type": "error", "message": "Not connected"})

            elif action == "pause":
                if session:
                    await session.pause()
                    await send({"type": "paused"})

            elif action == "stop":
                if session:
                    await session.stop()
                    await send({"type": "stopped"})

            elif action == "update":
                if session:
                    config_update = data.get("config", {})
                    await session.update_config(websocket, **config_update)
                    await send({"type": "updated", "config": session.current_config})

            elif action == "disconnect":
                if session:
                    await session.close()
                    sessions.pop(session_id, None)
                    session = None
                await websocket.send_json({"type": "disconnected"})  # No session, direct send ok

    except WebSocketDisconnect:
        pass
    except Exception as e:
        print(f"WebSocket error: {e}")
    finally:
        if session_id in sessions:
            await sessions[session_id].close()
            del sessions[session_id]


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
