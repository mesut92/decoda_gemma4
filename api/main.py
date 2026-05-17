"""
FastAPI proxy. Routes /chat by image-presence:

- Image attached  -> Unsloth vision backend (real Gemma-4 vision pass).
- No image        -> Ollama (text-only GGUF, faster, smaller VRAM footprint).

/generate is always text -> Ollama.
/health pings both backends.
"""
import json
import logging
import os
from typing import AsyncGenerator

import httpx
from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, Field

OLLAMA_URL = os.getenv("OLLAMA_URL", "http://ollama:11434")
VISION_URL = os.getenv("VISION_URL", "http://vision:8000")
OLLAMA_MODEL = os.getenv("OLLAMA_MODEL", "gemma4-rico")

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger("gemma4-proxy")

app = FastAPI(title="Gemma 4 RICO Proxy (Ollama + Vision)")


class Message(BaseModel):
    role: str = Field(pattern="^(system|user|assistant)$")
    content: str
    images: list[str] | None = None


class ChatRequest(BaseModel):
    messages: list[Message]
    stream: bool = False
    temperature: float = 0.7
    top_p: float = 0.9
    max_tokens: int = 512
    format: str | dict | None = None


class GenerateRequest(BaseModel):
    prompt: str
    stream: bool = False
    temperature: float = 0.7
    top_p: float = 0.9
    max_tokens: int = 512
    format: str | dict | None = None


@app.get("/health")
async def health():
    async with httpx.AsyncClient(timeout=5.0) as client:
        try:
            r = await client.get(f"{OLLAMA_URL}/api/tags")
            ollama_ok = r.status_code == 200
            names = [m["name"] for m in r.json().get("models", [])] if ollama_ok else []
        except Exception:
            ollama_ok, names = False, []
        try:
            r = await client.get(f"{VISION_URL}/health")
            vision_ok = r.status_code == 200
        except Exception:
            vision_ok = False
    return {
        "ok": ollama_ok and vision_ok,
        "ollama": {"ok": ollama_ok, "model": OLLAMA_MODEL, "available": names},
        "vision": {"ok": vision_ok, "url": VISION_URL},
    }


# ---------- /chat ----------------------------------------------------------

@app.post("/chat")
async def chat(
    payload: str = Form(...),
    image: UploadFile | None = File(None),
):
    raw = await image.read() if image is not None else b""
    if raw:
        return await _chat_vision(payload, image, raw)
    return await _chat_ollama(payload)


async def _chat_vision(payload: str, image: UploadFile, raw: bytes):
    """Forward multipart payload + image to the Unsloth vision backend."""
    log.info("/chat -> vision (image=%dB, %s)", len(raw), image.filename)
    files = {"image": (image.filename or "image", raw, image.content_type or "image/png")}
    data = {"payload": payload}

    # Detect if client requested streaming so we relay it transparently.
    try:
        stream = json.loads(payload).get("stream", False)
    except Exception:
        stream = False

    if stream:
        return StreamingResponse(
            _relay_sse(f"{VISION_URL}/chat", data=data, files=files),
            media_type="text/event-stream",
        )

    async with httpx.AsyncClient(timeout=600.0) as client:
        r = await client.post(f"{VISION_URL}/chat", data=data, files=files)
        r.raise_for_status()
    return r.json()


async def _chat_ollama(payload: str):
    try:
        req = ChatRequest.model_validate_json(payload)
    except Exception as e:
        raise HTTPException(422, f"invalid payload JSON: {e}")
    log.info("/chat -> ollama (no image, %d messages, stream=%s)",
             len(req.messages), req.stream)

    ollama_payload = {
        "model": OLLAMA_MODEL,
        "messages": [m.model_dump(exclude={"images"}) for m in req.messages],
        "stream": req.stream,
        "think": False,
        "options": _options(req),
    }
    if req.format is not None:
        ollama_payload["format"] = req.format

    if req.stream:
        return StreamingResponse(
            _stream_ollama("/api/chat", ollama_payload, key="message"),
            media_type="text/event-stream",
        )
    async with httpx.AsyncClient(timeout=600.0) as client:
        r = await client.post(f"{OLLAMA_URL}/api/chat", json=ollama_payload)
        r.raise_for_status()
    return {"response": r.json()["message"]["content"]}


# ---------- /generate ------------------------------------------------------

@app.post("/generate")
async def generate(req: GenerateRequest):
    log.info("/generate -> ollama (stream=%s)", req.stream)
    payload = {
        "model": OLLAMA_MODEL,
        "prompt": req.prompt,
        "stream": req.stream,
        "options": _options(req),
    }
    if req.format is not None:
        payload["format"] = req.format
    if req.stream:
        return StreamingResponse(
            _stream_ollama("/api/generate", payload, key="response"),
            media_type="text/event-stream",
        )
    async with httpx.AsyncClient(timeout=600.0) as client:
        r = await client.post(f"{OLLAMA_URL}/api/generate", json=payload)
        r.raise_for_status()
    return {"response": r.json()["response"]}


# ---------- helpers --------------------------------------------------------

def _options(req) -> dict:
    return {
        "temperature": req.temperature,
        "top_p": req.top_p,
        "num_predict": req.max_tokens,
    }


async def _stream_ollama(path: str, payload: dict, key: str) -> AsyncGenerator[bytes, None]:
    async with httpx.AsyncClient(timeout=None) as client:
        async with client.stream("POST", f"{OLLAMA_URL}{path}", json=payload) as r:
            async for line in r.aiter_lines():
                if not line:
                    continue
                obj = json.loads(line)
                chunk = (obj.get(key, {}).get("content") if key == "message"
                         else obj.get(key, ""))
                if chunk:
                    yield f"data: {json.dumps({'message': chunk})}\n\n".encode()
                if obj.get("done"):
                    yield f"data: {json.dumps({'is_message_completed': True})}\n\n".encode()
                    return


async def _relay_sse(url: str, data: dict, files: dict) -> AsyncGenerator[bytes, None]:
    """Relay an SSE stream from the vision backend straight to the client."""
    async with httpx.AsyncClient(timeout=None) as client:
        async with client.stream("POST", url, data=data, files=files) as r:
            async for chunk in r.aiter_bytes():
                if chunk:
                    yield chunk
