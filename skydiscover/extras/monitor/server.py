"""
WebSocket + HTTP server for the live monitor dashboard.

Single port, zero extra dependencies beyond Python stdlib + optional websockets.
Uses raw asyncio TCP so it works regardless of websockets version.

HTTP GET /  → serves dashboard.html
WS upgrade  → real-time event stream
"""

import asyncio
import base64
import hashlib
import json
import logging
import os
import queue
import struct
import threading
import time
import urllib.error
import urllib.request
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Any, Dict, List, Optional, Set

logger = logging.getLogger(__name__)

DASHBOARD_PATH = Path(__file__).parent / "dashboard.html"
WS_GUID = "258EAFA5-E914-47DA-95CA-C5AB0DC85B11"


def _ws_accept_key(client_key: str) -> str:
    digest = hashlib.sha1((client_key + WS_GUID).encode()).digest()
    return base64.b64encode(digest).decode()


def _ws_encode_text(text: str) -> bytes:
    """Encode a text frame (server→client, unmasked)."""
    payload = text.encode("utf-8")
    length = len(payload)
    if length < 126:
        header = struct.pack("BB", 0x81, length)
    elif length < 65536:
        header = struct.pack("!BBH", 0x81, 126, length)
    else:
        header = struct.pack("!BBQ", 0x81, 127, length)
    return header + payload


async def _ws_read_frame(reader: asyncio.StreamReader) -> Optional[str]:
    """Read one WebSocket frame; return text payload or None on close/error."""
    try:
        header = await reader.readexactly(2)
    except Exception:
        return None
    opcode = header[0] & 0x0F
    masked = (header[1] & 0x80) != 0
    length = header[1] & 0x7F

    if opcode == 0x8:  # Close
        return None
    if opcode == 0x9:  # Ping — we could reply with pong but we ignore it here
        return None

    if length == 126:
        ext = await reader.readexactly(2)
        length = struct.unpack("!H", ext)[0]
    elif length == 127:
        ext = await reader.readexactly(8)
        length = struct.unpack("!Q", ext)[0]

    if masked:
        mask = await reader.readexactly(4)
        data = bytearray(await reader.readexactly(length))
        for i in range(length):
            data[i] ^= mask[i % 4]
        payload = bytes(data)
    else:
        payload = await reader.readexactly(length)

    if opcode == 0x1:  # Text
        return payload.decode("utf-8", errors="replace")
    return None  # Binary / continuation frames ignored


class MonitorServer:
    """
    Single-port HTTP+WebSocket server for live solution discovery monitoring.

    - GET /  →  dashboard.html
    - WS upgrade  →  event broadcast
    Runs in a daemon thread with its own asyncio event loop.
    """

    def __init__(self, host: str = "127.0.0.1", port: int = 8765, max_solution_length: int = 10000):
        self.host = host
        self.port = port
        self.max_solution_length = max_solution_length

        self._queue: queue.Queue = queue.Queue()

        # In-memory state for reconnecting clients
        self._programs: List[Dict[str, Any]] = []
        self._program_solutions: Dict[str, str] = {}
        self._parent_solutions: Dict[str, str] = {}
        self._best_program_id: Optional[str] = None
        self._best_score: float = -float("inf")
        self._stats: Dict[str, Any] = {}
        self._config_summary: str = ""

        # Per-program summary cache
        self._program_summary_cache: Dict[str, str] = {}

        # Human feedback reader (set via set_feedback_reader)
        self._feedback_reader: Optional[Any] = None

        # AI summary state
        self._summary_model: str = ""
        self._summary_api_key: str = ""
        self._summary_api_base: str = "https://api.openai.com/v1"
        self._summary_top_k: int = 3
        self._summary_interval: int = 0  # 0 = manual only
        self._summary_text: str = ""
        self._summary_generating: bool = False
        self._summary_last_program_count: int = 0
        self._summary_executor: Optional[ThreadPoolExecutor] = None

        self._loop: Optional[asyncio.AbstractEventLoop] = None
        self._thread: Optional[threading.Thread] = None
        self._clients: Set[asyncio.StreamWriter] = set()
        self._stop_event = threading.Event()
        self._ready_event = threading.Event()  # set when TCP port is bound
        self._dashboard_html: Optional[bytes] = None

    def start(self) -> None:
        """Load the dashboard and start the server in a daemon thread."""
        self._load_dashboard()
        self._thread = threading.Thread(target=self._run_loop, daemon=True)
        self._thread.start()
        # Wait until TCP port is actually bound (up to 5s)
        self._ready_event.wait(timeout=5)
        logger.debug(f"Monitor server started → http://localhost:{self.port}/")

    def stop(self) -> None:
        """Signal the server to stop and wait for the thread to finish."""
        self._stop_event.set()
        loop = self._loop
        if loop is not None and not loop.is_closed():
            # Schedule cancellation of all tasks, then stop the loop
            try:
                loop.call_soon_threadsafe(self._cancel_all_tasks)
            except RuntimeError:
                pass  # Loop already closed
        if self._thread:
            self._thread.join(timeout=5)

    def _cancel_all_tasks(self) -> None:
        """Cancel every pending task on the server's event loop, then stop it."""
        loop = self._loop
        if loop is None or loop.is_closed():
            return
        for task in asyncio.all_tasks(loop):
            task.cancel()
        loop.stop()

    def push_event(self, event: Dict[str, Any]) -> None:
        """Enqueue an event for broadcast to all connected WebSocket clients."""
        self._queue.put_nowait(event)

    def set_config_summary(self, summary: str) -> None:
        """Set a human-readable config summary sent to new dashboard clients."""
        self._config_summary = summary

    def set_feedback_reader(self, reader: Any) -> None:
        """Attach a HumanFeedbackReader for dashboard human feedback controls."""
        self._feedback_reader = reader

    def configure_summary(
        self,
        model: str = "gpt-5-mini",
        api_key: str = "",
        api_base: str = "https://api.openai.com/v1",
        top_k: int = 3,
        interval: int = 0,
    ) -> None:
        """Configure the AI summary generator.

        Args:
            model: OpenAI model name (default gpt-5-mini).
            api_key: API key. Falls back to OPENAI_API_KEY env var.
            api_base: API base URL.
            top_k: Number of top programs to include in summary prompt.
            interval: Auto-generate every N new programs (0 = manual only).
        """
        self._summary_model = model
        self._summary_api_key = api_key or os.environ.get("OPENAI_API_KEY", "")
        self._summary_api_base = api_base.rstrip("/")
        self._summary_top_k = top_k
        self._summary_interval = interval
        self._summary_executor = ThreadPoolExecutor(max_workers=1, thread_name_prefix="summary")

        # Set initial placeholder text so UI knows summary is ready
        if not self._summary_text:
            self._summary_text = (
                "Click 'Refresh Summary' to generate an AI summary of the top programs."
            )

        logger.debug(
            f"AI summary configured: model={model}, top_k={top_k}, "
            f"interval={interval or 'manual'}, api_key={'set' if self._summary_api_key else 'MISSING'}"
        )

    def _get_feedback_state(self) -> Dict[str, Any]:
        """Return current human feedback state."""
        if not self._feedback_reader:
            return {
                "human_feedback_enabled": False,
                "feedback_text": "",
                "feedback_active": False,
                "human_feedback_mode": "append",
                "human_feedback_current_prompt": "",
                "human_feedback_history": [],
            }
        text = self._feedback_reader.read()
        return {
            "human_feedback_enabled": True,
            "feedback_text": text,
            "feedback_active": bool(text),
            "human_feedback_mode": self._feedback_reader.mode,
            "human_feedback_current_prompt": self._feedback_reader.get_current_prompt(),
            "human_feedback_history": self._feedback_reader.get_history(),
        }

    def _build_init_state(self) -> Dict[str, Any]:
        """Build the full init_state payload for new/reconnecting WS clients."""
        state = {
            "type": "init_state",
            "programs": self._programs,
            "best_program_id": self._best_program_id,
            "stats": self._stats,
            "config_summary": self._config_summary,
            "summary_enabled": bool(self._summary_model),
            "summary_model": self._summary_model or "",
            "summary_text": self._summary_text,
            "summary_generating": self._summary_generating,
        }
        state.update(self._get_feedback_state())
        return state

    def _load_dashboard(self) -> None:
        try:
            raw = DASHBOARD_PATH.read_text(encoding="utf-8")
        except FileNotFoundError:
            logger.warning(f"Dashboard HTML not found at {DASHBOARD_PATH}")
            raw = "<html><body><h1>Dashboard not found</h1></body></html>"
        # No port injection needed — WS connects to the same host:port
        self._dashboard_html = raw.encode("utf-8")

    def _run_loop(self) -> None:
        self._loop = asyncio.new_event_loop()
        asyncio.set_event_loop(self._loop)
        try:
            self._loop.run_until_complete(self._serve())
        except (RuntimeError, asyncio.CancelledError):
            pass  # Normal on shutdown
        except Exception:
            logger.exception("Monitor server error")
        finally:
            # Drain any remaining cancelled tasks so they don't warn on GC
            try:
                pending = asyncio.all_tasks(self._loop)
                if pending:
                    for t in pending:
                        t.cancel()
                    self._loop.run_until_complete(asyncio.gather(*pending, return_exceptions=True))
            except Exception:
                logger.debug("Error cancelling tasks during stop", exc_info=True)
            try:
                self._loop.close()
            except Exception:
                logger.debug("Error closing event loop", exc_info=True)

    async def _serve(self) -> None:
        # Try configured port, then auto-increment if already in use
        port = self.port
        for attempt in range(10):
            try:
                server = await asyncio.start_server(self._handle_connection, self.host, port)
                break
            except OSError:
                if attempt == 9:
                    raise
                port += 1
        self.port = port
        async with server:
            self._ready_event.set()  # signal that port is bound
            logger.debug(f"Listening on {self.host}:{self.port}")
            consumer = asyncio.create_task(self._consume_queue())
            hb = asyncio.create_task(self._heartbeat())
            try:
                await asyncio.gather(consumer, hb)
            except (asyncio.CancelledError, RuntimeError):
                pass
            finally:
                try:
                    consumer.cancel()
                    hb.cancel()
                except RuntimeError:
                    pass  # Event loop already closed

    async def _handle_connection(
        self, reader: asyncio.StreamReader, writer: asyncio.StreamWriter
    ) -> None:
        """Route an incoming connection to HTTP or WebSocket handler."""
        try:
            # Read HTTP request line + headers
            raw_headers: Dict[str, str] = {}
            request_line = (await reader.readline()).decode("utf-8", errors="replace").strip()
            if not request_line:
                writer.close()
                return

            while True:
                line = (await reader.readline()).decode("utf-8", errors="replace").strip()
                if not line:
                    break
                if ":" in line:
                    k, _, v = line.partition(":")
                    raw_headers[k.strip().lower()] = v.strip()

            is_ws = raw_headers.get("upgrade", "").lower() == "websocket"

            if is_ws:
                await self._handle_ws(reader, writer, raw_headers)
            else:
                await self._handle_http(writer)
        except Exception:
            logger.debug("Connection handler error", exc_info=True)
        finally:
            try:
                writer.close()
            except Exception:
                logger.debug("Error closing writer", exc_info=True)

    async def _handle_http(self, writer: asyncio.StreamWriter) -> None:
        """Serve the dashboard HTML over a plain HTTP GET."""
        html = self._dashboard_html or b""
        resp = (
            "HTTP/1.1 200 OK\r\n"
            "Content-Type: text/html; charset=utf-8\r\n"
            f"Content-Length: {len(html)}\r\n"
            "Connection: close\r\n"
            "\r\n"
        ).encode() + html
        writer.write(resp)
        await writer.drain()

    async def _handle_ws(
        self,
        reader: asyncio.StreamReader,
        writer: asyncio.StreamWriter,
        headers: Dict[str, str],
    ) -> None:
        """Complete the WebSocket handshake and enter the read loop."""
        key = headers.get("sec-websocket-key", "")
        accept = _ws_accept_key(key)
        handshake = (
            "HTTP/1.1 101 Switching Protocols\r\n"
            "Upgrade: websocket\r\n"
            "Connection: Upgrade\r\n"
            f"Sec-WebSocket-Accept: {accept}\r\n"
            "\r\n"
        ).encode()
        writer.write(handshake)
        await writer.drain()

        self._clients.add(writer)
        logger.debug(f"WS client connected ({len(self._clients)} total)")
        try:
            await self._ws_send(writer, json.dumps(self._build_init_state()))
            # Read loop
            while True:
                text = await _ws_read_frame(reader)
                if text is None:
                    break
                await self._handle_client_msg(writer, text)
        except Exception:
            logger.debug("WebSocket handler error", exc_info=True)
        finally:
            self._clients.discard(writer)
            logger.debug(f"WS client disconnected ({len(self._clients)} total)")

    async def _handle_client_msg(self, writer: asyncio.StreamWriter, raw: str) -> None:
        """Dispatch an incoming WebSocket JSON message from a dashboard client."""
        try:
            msg = json.loads(raw)
        except Exception:
            return
        t = msg.get("type")
        if t == "request_full_state":
            await self._ws_send(writer, json.dumps(self._build_init_state()))
        elif t == "request_program_solution":
            pid = msg.get("program_id", "")
            await self._ws_send(
                writer,
                json.dumps(
                    {
                        "type": "program_solution",
                        "program_id": pid,
                        "solution": self._program_solutions.get(pid, "")[
                            : self.max_solution_length
                        ],
                        "parent_solution": self._parent_solutions.get(pid, "")[
                            : self.max_solution_length
                        ],
                    }
                ),
            )
        elif t == "set_feedback":
            text = msg.get("text", "").strip()
            if self._feedback_reader:
                self._feedback_reader.write_from_dashboard(text)
                ack = {
                    "type": "feedback_ack",
                    "feedback_text": text,
                    "feedback_active": bool(text),
                    "human_feedback_mode": self._feedback_reader.mode,
                }
                await self._broadcast(json.dumps(ack))
                logger.debug(f"Human feedback set from dashboard ({len(text)} chars)")
            else:
                await self._ws_send(
                    writer,
                    json.dumps(
                        {
                            "type": "feedback_ack",
                            "feedback_text": "",
                            "feedback_active": False,
                            "error": "Human feedback not enabled",
                        }
                    ),
                )
        elif t == "clear_feedback":
            if self._feedback_reader:
                self._feedback_reader.write_from_dashboard("")
                ack = {
                    "type": "feedback_ack",
                    "feedback_text": "",
                    "feedback_active": False,
                    "human_feedback_mode": self._feedback_reader.mode,
                }
                await self._broadcast(json.dumps(ack))
                logger.debug("Human feedback cleared from dashboard")
        elif t == "request_feedback_state":
            await self._ws_send(
                writer,
                json.dumps(
                    {
                        "type": "feedback_ack",
                        **self._get_feedback_state(),
                    }
                ),
            )
        elif t == "set_human_feedback_mode":
            mode = msg.get("mode", "append")
            if self._feedback_reader:
                self._feedback_reader.set_mode(mode)
                ack = {
                    "type": "human_feedback_mode_ack",
                    "human_feedback_mode": mode,
                }
                await self._broadcast(json.dumps(ack))
                logger.debug(f"Human feedback mode set to: {mode}")
        elif t == "request_system_prompt":
            prompt_text = ""
            if self._feedback_reader:
                prompt_text = self._feedback_reader.get_current_prompt()
            await self._ws_send(
                writer,
                json.dumps(
                    {
                        "type": "system_prompt",
                        "prompt_text": prompt_text,
                    }
                ),
            )
        elif t == "request_human_feedback_history":
            history = []
            if self._feedback_reader:
                history = self._feedback_reader.get_history()
            await self._ws_send(
                writer,
                json.dumps(
                    {
                        "type": "human_feedback_history",
                        "history": history,
                    }
                ),
            )
        elif t == "request_image":
            image_path = msg.get("image_path", "")
            program_id = msg.get("program_id", "")
            if image_path and os.path.exists(image_path):
                try:
                    import base64 as _b64

                    with open(image_path, "rb") as _f:
                        img_data = _b64.b64encode(_f.read()).decode()
                    ext = os.path.splitext(image_path)[1].lstrip(".").lower()
                    mime = {
                        "png": "image/png",
                        "jpg": "image/jpeg",
                        "jpeg": "image/jpeg",
                        "webp": "image/webp",
                        "gif": "image/gif",
                    }.get(ext, "image/png")
                    await self._ws_send(
                        writer,
                        json.dumps(
                            {
                                "type": "image_data",
                                "program_id": program_id,
                                "data_url": f"data:{mime};base64,{img_data}",
                            }
                        ),
                    )
                except Exception as e:
                    logger.warning(f"Failed to serve image {image_path}: {e}")
        elif t == "request_program_summary":
            pid = msg.get("program_id", "")
            await self._generate_program_summary(writer, pid)
        elif t == "request_summary":
            await self._trigger_summary()

    # ─── Queue consumer & broadcast ──────────────────────────

    async def _consume_queue(self) -> None:
        while not self._stop_event.is_set():
            try:
                event = self._queue.get_nowait()
            except queue.Empty:
                await asyncio.sleep(0.05)
                continue

            etype = event.get("type")
            if etype == "new_program":
                p = event.get("program", {})
                # Annotate with human feedback state for replay on reconnect
                if self._feedback_reader:
                    fb = self._feedback_reader.read()
                    p["human_feedback_active"] = bool(fb)
                else:
                    p["human_feedback_active"] = False
                self._programs.append(p)
                pid = p.get("id", "")
                if "full_solution" in event:
                    self._program_solutions[pid] = event["full_solution"]
                if "parent_full_solution" in event:
                    self._parent_solutions[pid] = event["parent_full_solution"]
                # Independent best tracking: compare scores directly
                new_score = p.get("score", 0)
                if not isinstance(new_score, (int, float)):
                    new_score = 0
                if new_score > self._best_score:
                    self._best_score = new_score
                    self._best_program_id = pid
                    event["is_best"] = True
                elif event.get("is_best"):
                    self._best_program_id = pid
                    self._best_score = max(self._best_score, new_score)
                self._stats = event.get("stats", self._stats)

            # Strip full_solution from broadcast (clients request on demand)
            broadcast = {
                k: v for k, v in event.items() if k not in ("full_solution", "parent_full_solution")
            }
            # Include current human feedback status in program events
            if etype == "new_program" and self._feedback_reader:
                fb = self._feedback_reader.read()
                broadcast["feedback_active"] = bool(fb)
                broadcast["feedback_text"] = fb if fb else ""
                broadcast["human_feedback_mode"] = self._feedback_reader.mode
            await self._broadcast(json.dumps(broadcast))

            # Auto-trigger AI summary every N new programs
            if (
                etype == "new_program"
                and self._summary_interval > 0
                and self._summary_model
                and not self._summary_generating
            ):
                count = len(self._programs)
                if count - self._summary_last_program_count >= self._summary_interval:
                    await self._trigger_summary()

    async def _broadcast(self, message: str) -> None:
        if not self._clients:
            return
        dead = set()
        for writer in list(self._clients):
            try:
                await self._ws_send(writer, message)
            except Exception:
                dead.add(writer)
        self._clients -= dead

    async def _ws_send(self, writer: asyncio.StreamWriter, text: str) -> None:
        writer.write(_ws_encode_text(text))
        await writer.drain()

    async def _heartbeat(self) -> None:
        while not self._stop_event.is_set():
            await asyncio.sleep(5)
            if self._clients:
                await self._broadcast(json.dumps({"type": "heartbeat", "timestamp": time.time()}))

    async def _generate_program_summary(self, writer: asyncio.StreamWriter, pid: str) -> None:
        """Generate a crisp LLM summary of what changed in a single program."""
        # Return cached if available
        if pid in self._program_summary_cache:
            await self._ws_send(
                writer,
                json.dumps(
                    {
                        "type": "program_summary",
                        "program_id": pid,
                        "summary": self._program_summary_cache[pid],
                    }
                ),
            )
            return

        # Need API key + model
        if not self._summary_model or not self._summary_api_key:
            await self._ws_send(
                writer,
                json.dumps(
                    {
                        "type": "program_summary",
                        "program_id": pid,
                        "summary": "AI summary not configured.",
                    }
                ),
            )
            return

        # Find program data
        prog = None
        for p in self._programs:
            if p.get("id") == pid:
                prog = p
                break
        if not prog:
            return

        # Build prompt
        code = self._program_solutions.get(pid, prog.get("solution_snippet", ""))
        parent_solution = self._parent_solutions.get(pid, "")
        score = prog.get("score", "?")
        parent_score = prog.get("parent_score")
        label = prog.get("label_type", "unknown")

        delta_str = ""
        if isinstance(score, (int, float)) and isinstance(parent_score, (int, float)):
            d = score - parent_score
            delta_str = f" (delta: {'+' if d >= 0 else ''}{d:.4f})"

        # Truncate code for prompt efficiency
        if len(code) > 2000:
            code = code[:2000] + "\n... (truncated)"
        if len(parent_solution) > 2000:
            parent_solution = parent_solution[:2000] + "\n... (truncated)"

        is_image_mode = prog.get("image_path") is not None

        if is_image_mode:
            system = (
                "You are analyzing one step in an image generation run. "
                "Given the parent generation prompt and the child generation prompt, describe in 1-2 concise bullet points "
                "what specifically changed in the prompt.\n\n"
                "Rules:\n"
                "- Be specific: name style changes, subject modifications, added details\n"
                "- Each bullet under 25 words\n"
                "- Start each bullet with `- `\n"
                "- No headers, no sections — just 1-2 bullets"
            )
        else:
            system = (
                "You are analyzing one step in a solution discovery run. "
                "Given the parent code and the child code, describe in 1-2 concise bullet points "
                "what specifically changed.\n\n"
                "Rules:\n"
                "- Be specific: name algorithms, parameters, structural changes\n"
                "- Each bullet under 25 words\n"
                "- Start each bullet with `- `\n"
                "- No headers, no sections — just 1-2 bullets\n"
                "- Consider the evolution label: exploration = trying new ideas, "
                "exploitation = refining current best, diverge = deliberately different strategy"
            )

        content_label = "prompt" if is_image_mode else "code"
        user_parts = [f"Label: {label}{delta_str}"]
        if parent_score is not None:
            user_parts.append(f"Score: {parent_score} -> {score}")
        else:
            user_parts.append(f"Score: {score} (no parent)")

        if parent_solution:
            user_parts.append(f"\nParent {content_label}:\n```\n{parent_solution}\n```")
        user_parts.append(f"\nNew {content_label}:\n```\n{code}\n```")

        prompt_data = {"system": system, "user": "\n".join(user_parts)}

        # Ensure executor exists
        if not self._summary_executor:
            self._summary_executor = ThreadPoolExecutor(max_workers=1, thread_name_prefix="summary")

        # Run LLM call in executor
        result = ""
        loop = asyncio.get_running_loop()
        try:
            result = await loop.run_in_executor(
                self._summary_executor,
                self._call_program_summary_api,
                prompt_data,
            )
            self._program_summary_cache[pid] = result
        except Exception as e:
            logger.warning(f"Program summary failed for {pid[:8]}: {e}", exc_info=True)
            result = f"Summary unavailable: {e}"

        await self._ws_send(
            writer,
            json.dumps(
                {
                    "type": "program_summary",
                    "program_id": pid,
                    "summary": result or "Summary unavailable (empty response).",
                }
            ),
        )

    def _call_program_summary_api(self, prompt_data: Dict[str, str]) -> str:
        """Call LLM for per-program summary (blocking, runs in executor)."""
        return self._call_llm_api(prompt_data, max_tokens=2048, timeout=120)

    async def _trigger_summary(self) -> None:
        """Trigger async AI summary generation."""
        if not self._summary_model:
            await self._broadcast(
                json.dumps(
                    {
                        "type": "summary_update",
                        "summary_text": "AI summary not configured (no model set).",
                        "summary_generating": False,
                        "summary_enabled": False,
                    }
                )
            )
            return
        if not self._summary_api_key:
            await self._broadcast(
                json.dumps(
                    {
                        "type": "summary_update",
                        "summary_text": "AI summary not configured. Set OPENAI_API_KEY environment variable or summary_api_key in config.",
                        "summary_generating": False,
                        "summary_enabled": False,
                    }
                )
            )
            return
        if self._summary_generating:
            return  # Already in progress

        # Ensure executor exists
        if not self._summary_executor:
            self._summary_executor = ThreadPoolExecutor(max_workers=1, thread_name_prefix="summary")

        self._summary_generating = True
        self._summary_last_program_count = len(self._programs)

        # Notify clients that generation started
        await self._broadcast(
            json.dumps(
                {
                    "type": "summary_update",
                    "summary_text": self._summary_text,
                    "summary_generating": True,
                    "summary_enabled": True,
                }
            )
        )

        try:
            # Build the prompt data from current programs
            top_programs = self._get_top_k_programs()
            if not top_programs:
                self._summary_text = "No scored programs yet. Run some iterations first."
                logger.debug("AI summary skipped: no scored programs")
            else:
                prompt_data = self._build_summary_prompt(top_programs)
                logger.debug(
                    f"AI summary: calling {self._summary_model} with {len(top_programs)} "
                    f"top programs, api_base={self._summary_api_base}"
                )

                # Run the blocking API call in a thread
                loop = asyncio.get_running_loop()
                result = await loop.run_in_executor(
                    self._summary_executor,
                    self._call_llm_api,
                    prompt_data,
                )
                self._summary_text = result or "AI returned empty response."
                logger.debug(f"AI summary generated ({len(self._summary_text)} chars)")
        except Exception as e:
            logger.warning(f"AI summary generation failed: {e}", exc_info=True)
            self._summary_text = f"Summary generation failed: {e}"
        finally:
            self._summary_generating = False

        # Broadcast the result
        await self._broadcast(
            json.dumps(
                {
                    "type": "summary_update",
                    "summary_text": self._summary_text,
                    "summary_generating": False,
                    "summary_enabled": True,
                }
            )
        )

    def _get_top_k_programs(self) -> List[Dict[str, Any]]:
        """Get top-k programs by score across all islands."""
        if not self._programs:
            return []
        scored = [p for p in self._programs if isinstance(p.get("score"), (int, float))]
        scored.sort(key=lambda p: p["score"], reverse=True)

        # Deduplicate by score (keep best per unique score to show diversity)
        seen_scores = set()
        unique = []
        for p in scored:
            key = round(p["score"], 6)
            if key not in seen_scores:
                seen_scores.add(key)
                unique.append(p)
            if len(unique) >= self._summary_top_k:
                break
        # Fall back to just top-k if not enough unique
        if len(unique) < self._summary_top_k:
            unique = scored[: self._summary_top_k]
        return unique

    def _compute_solution_discovery_analysis(self) -> str:
        """Compute evolution progress, improvement patterns, and stagnation analysis."""
        programs = self._programs
        if not programs:
            return ""

        scored = [p for p in programs if isinstance(p.get("score"), (int, float))]
        if not scored:
            return ""

        lines = []
        n = len(scored)

        improvements = 0
        regressions = 0
        total_with_parent = 0
        improvement_deltas = []
        for p in scored:
            parent_score = p.get("parent_score")
            if isinstance(parent_score, (int, float)):
                total_with_parent += 1
                delta = p["score"] - parent_score
                if delta > 0:
                    improvements += 1
                    improvement_deltas.append(delta)
                elif delta < 0:
                    regressions += 1

        if total_with_parent > 0:
            hit_rate = improvements / total_with_parent * 100
            avg_gain = (
                sum(improvement_deltas) / len(improvement_deltas) if improvement_deltas else 0
            )
            lines.append("=== Improvement Rate ===")
            lines.append(
                f"  {improvements}/{total_with_parent} programs improved over parent ({hit_rate:.0f}% hit rate)"
            )
            lines.append(f"  Avg improvement when positive: {avg_gain:+.4f}")

        if n >= 10:
            quarter = max(n // 4, 1)
            early_scores = [p["score"] for p in scored[:quarter]]
            mid_scores = [p["score"] for p in scored[quarter : quarter * 2]]
            recent_scores = [p["score"] for p in scored[-quarter:]]
            early_avg = sum(early_scores) / len(early_scores)
            mid_avg = sum(mid_scores) / len(mid_scores) if mid_scores else early_avg
            recent_avg = sum(recent_scores) / len(recent_scores)

            lines.append("\n=== Score Trend ===")
            lines.append(
                f"  Early avg (first {quarter}): {early_avg:.4f}  |  "
                f"Mid avg: {mid_avg:.4f}  |  "
                f"Recent avg (last {quarter}): {recent_avg:.4f}"
            )
            if recent_avg > mid_avg + 0.001:
                lines.append("  Trend: IMPROVING")
            elif recent_avg < mid_avg - 0.005:
                lines.append("  Trend: REGRESSING")
            elif abs(recent_avg - mid_avg) < 0.001 and n > 30:
                lines.append("  Trend: PLATEAUED")
            else:
                lines.append("  Trend: STABLE")

        if n >= 5:
            best_so_far = -float("inf")
            streak = 0
            longest_streak = 0
            for p in scored:
                if p["score"] > best_so_far:
                    best_so_far = p["score"]
                    streak = 0
                else:
                    streak += 1
                    longest_streak = max(longest_streak, streak)
            lines.append("\n=== Stagnation ===")
            lines.append(
                f"  Current non-improving streak: {streak} iterations  |  "
                f"Longest streak: {longest_streak}"
            )

        islands: Dict[Any, list] = {}
        for p in scored:
            isl = p.get("island")
            if isl is not None:
                islands.setdefault(isl, []).append(p["score"])
        if len(islands) > 1:
            lines.append(f"\n=== Island Diversity ({len(islands)} islands) ===")
            for isl in sorted(islands.keys()):
                scores = islands[isl]
                lines.append(
                    f"  Island {isl}: {len(scores)} programs, "
                    f"best={max(scores):.4f}, avg={sum(scores)/len(scores):.4f}"
                )

        return "\n".join(lines)

    def _build_summary_prompt(self, top_programs: List[Dict[str, Any]]) -> Dict[str, str]:
        """Build the system + user prompt for the summary LLM call."""
        system = (
            "You are an expert analyst monitoring a solution discovery process. "
            "You will be given run statistics, evolution progress data, and the source code "
            "of the top-performing programs from the current run.\n\n"
            "Respond using EXACTLY this markdown structure:\n\n"
            "## Status\n"
            "One sentence: is the search improving, stagnating, or plateauing? "
            "Cite the score trend numbers.\n\n"
            "## Key Techniques\n"
            "Bullet list of the main algorithmic ideas found in the top programs' code. "
            "Be specific — name the techniques (e.g. 'Kalman filter with adaptive Q', "
            "'hexagonal lattice packing', 'exponential moving average').\n\n"
            "## Diversity\n"
            "Are the top programs converging on one approach or exploring different strategies? "
            "One sentence.\n\n"
            "## Recommendation\n"
            "One specific, actionable suggestion grounded in the code. "
            "For example: **try wavelet denoising** — the top programs all use simple "
            "moving averages which limits frequency response.\n\n"
            "Rules:\n"
            "- Use markdown: **bold** for key terms, `- ` for bullets, `##` for sections\n"
            "- Be concise — max 250 words total\n"
            "- Every claim must reference what you see in the actual code"
        )

        # Build user message with stats + solution discovery analysis + top-k programs
        parts = []
        if self._stats:
            parts.append(
                f"Run: {self._config_summary}\n"
                f"Total programs: {self._stats.get('total_programs', len(self._programs))}\n"
                f"Current iteration: {self._stats.get('current_iteration', '?')}\n"
                f"Best score: {self._stats.get('best_score', '?')}\n"
                f"Programs/min: {self._stats.get('programs_per_min', '?')}\n"
                f"Elapsed: {self._stats.get('elapsed_seconds', '?')}s\n"
                f"Iterations since improvement: {self._stats.get('iterations_since_improvement', '?')}"
            )

        # Add solution discovery analysis
        solution_discovery_analysis = self._compute_solution_discovery_analysis()
        if solution_discovery_analysis:
            parts.append(f"\n{solution_discovery_analysis}")

        for i, p in enumerate(top_programs, 1):
            pid = p.get("id", "?")
            code = self._program_solutions.get(pid, p.get("solution_snippet", ""))
            # Truncate code to keep prompt reasonable
            if len(code) > 2000:
                code = code[:2000] + "\n... (truncated)"
            island_str = f", island={p.get('island')}" if p.get("island") is not None else ""
            parts.append(
                f"\n--- Top Program #{i} ---\n"
                f"ID: {pid}\n"
                f"Score: {p.get('score', '?')}\n"
                f"Iteration: {p.get('iteration', '?')}{island_str}\n"
                f"Metrics: {json.dumps(p.get('metrics', {}))}\n"
                f"Code:\n{code}"
            )

        return {"system": system, "user": "\n".join(parts)}

    def _call_llm_api(
        self, prompt_data: Dict[str, str], max_tokens: int = 8192, timeout: int = 180
    ) -> str:
        """Call OpenAI-compatible API (blocking, runs in executor thread)."""
        url = f"{self._summary_api_base}/chat/completions"
        body = json.dumps(
            {
                "model": self._summary_model,
                "messages": [
                    {"role": "system", "content": prompt_data["system"]},
                    {"role": "user", "content": prompt_data["user"]},
                ],
                "max_completion_tokens": max_tokens,
            }
        ).encode("utf-8")

        req = urllib.request.Request(
            url,
            data=body,
            headers={
                "Content-Type": "application/json",
                "Authorization": f"Bearer {self._summary_api_key}",
            },
            method="POST",
        )

        try:
            with urllib.request.urlopen(req, timeout=timeout) as resp:
                data = json.loads(resp.read().decode("utf-8"))
                return data["choices"][0]["message"]["content"].strip()
        except urllib.error.HTTPError as e:
            error_body = e.read().decode("utf-8", errors="replace")[:500]
            raise RuntimeError(f"API error {e.code}: {error_body}") from e
        except Exception as e:
            raise RuntimeError(f"API call failed: {e}") from e
