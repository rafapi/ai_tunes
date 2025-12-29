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
        self.current_config = {
            "prompt": "House, TR-909 Drums, Deep Bass, Groovy",
            "bpm": 124,
            "density": 0.55,  # Moderate density (0.1=sparse, 0.9=chaotic)
            "brightness": 0.5,
            "guidance": 5.0,  # Higher guidance to follow prompt more closely
            "temperature": 1.0,  # Default per Lyria guide
            "scale": None,  # None means model decides
            "music_generation_mode": "QUALITY",  # QUALITY for more coherent music
            "mute_bass": False,
            "mute_drums": False,
            "only_bass_and_drums": False,
        }

    async def connect(self) -> bool:
        """Establish connection to Lyria."""
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
            return True
        except Exception as e:
            print(f"Failed to connect: {e}")
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

    async def _apply_prompts(self) -> None:
        """Apply current prompt as weighted prompts."""
        if not self.session:
            return

        prompt_text = self.current_config["prompt"]
        prompts = [types.WeightedPrompt(text=prompt_text, weight=1.0)]
        await self.session.set_weighted_prompts(prompts=prompts)

    async def update_config(self, websocket: WebSocket, **kwargs) -> None:
        """Update music generation config."""
        needs_reset = False
        old_bpm = self.current_config.get("bpm")
        old_scale = self.current_config.get("scale")

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

                # Apply prompt if it changed
                if "prompt" in kwargs:
                    await self._apply_prompts()

                # Reset context if needed for BPM/scale changes
                if needs_reset:
                    await self.session.reset_context()
                    await websocket.send_json({"type": "log", "message": "Context reset"})

            except Exception as e:
                print(f"Failed to update config: {e}")
                await websocket.send_json({"type": "error", "message": str(e)})

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
                            try:
                                await self._websocket.send_json({
                                    "type": "audio",
                                    "data": audio_data,
                                })
                            except Exception:
                                self.is_playing = False
                                return

                    # Handle filtered prompts
                    if hasattr(message.server_content, 'filtered_prompt') and message.server_content.filtered_prompt:
                        fp = message.server_content.filtered_prompt
                        try:
                            await self._websocket.send_json({
                                "type": "filtered",
                                "reason": getattr(fp, 'filtered_reason', 'Unknown'),
                                "text": getattr(fp, 'text', ''),
                            })
                        except Exception:
                            pass

                # Small yield to prevent blocking
                await asyncio.sleep(0)

        except asyncio.CancelledError:
            pass
        except Exception as e:
            print(f"Receive loop error: {e}")
            if self._websocket:
                try:
                    await self._websocket.send_json({"type": "error", "message": str(e)})
                except Exception:
                    pass

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
            await websocket.send_json({"type": "error", "message": str(e)})

    async def pause(self) -> None:
        """Pause playback."""
        self.is_playing = False
        if self.session:
            try:
                await self.session.pause()
            except Exception as e:
                print(f"Pause error: {e}")

    async def stop(self) -> None:
        """Stop playback."""
        self.is_playing = False
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

    try:
        while True:
            data = await websocket.receive_json()
            action = data.get("action")

            if action == "init":
                api_key = data.get("api_key")
                if not api_key:
                    await websocket.send_json({"type": "error", "message": "API key required"})
                    continue

                session = MusicSession(api_key)
                if data.get("config"):
                    session.current_config.update(data["config"])

                if await session.connect():
                    sessions[session_id] = session
                    await websocket.send_json({"type": "connected", "config": session.current_config})
                else:
                    await websocket.send_json({"type": "error", "message": "Failed to connect to Lyria"})

            elif action == "play":
                if session and session.session:
                    await session.play(websocket)
                    await websocket.send_json({"type": "playing"})
                else:
                    await websocket.send_json({"type": "error", "message": "Not connected"})

            elif action == "pause":
                if session:
                    await session.pause()
                    await websocket.send_json({"type": "paused"})

            elif action == "stop":
                if session:
                    await session.stop()
                    await websocket.send_json({"type": "stopped"})

            elif action == "update":
                if session:
                    config_update = data.get("config", {})
                    await session.update_config(websocket, **config_update)
                    await websocket.send_json({"type": "updated", "config": session.current_config})

            elif action == "disconnect":
                if session:
                    await session.close()
                    sessions.pop(session_id, None)
                    session = None
                await websocket.send_json({"type": "disconnected"})

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
