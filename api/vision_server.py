"""
Direct Unsloth inference server for Gemma 4 E2B + RICO/OASST adapter.

Replaces the previous Ollama proxy. Same /chat, /generate, /health contract,
so existing mobile clients work unchanged — but image input is now real
(base64 PNG/JPG -> PIL -> Gemma vision tokens).

Run locally:
    ADAPTER_PATH=/path/to/gemma4_e2b_rico_adapter \
    uvicorn main:app --host 0.0.0.0 --port 2222
"""
import asyncio
import base64
import io
import json
import logging
import os
import threading
from typing import AsyncGenerator

os.environ.setdefault("PYTORCH_ALLOC_CONF", "expandable_segments:True")
os.environ.setdefault("HF_HUB_DISABLE_XET", "1")

import torch
from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.responses import StreamingResponse
from PIL import Image
from pydantic import BaseModel, Field

ADAPTER_PATH = os.getenv(
    "ADAPTER_PATH",
    "/media/mesut/2Depo/works/gemma4/gemma4_e2b_rico_adapter",
)
MAX_SEQ_LENGTH = int(os.getenv("MAX_SEQ_LENGTH", "2048"))

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger("gemma4-api")

from unsloth import FastModel
from unsloth.chat_templates import get_chat_template
from transformers import TextIteratorStreamer

log.info("Loading adapter from %s ...", ADAPTER_PATH)
model, tokenizer = FastModel.from_pretrained(
    model_name=ADAPTER_PATH,
    max_seq_length=MAX_SEQ_LENGTH,
    load_in_4bit=True,
    load_in_16bit=False,
    full_finetuning=False,
)
tokenizer = get_chat_template(tokenizer, chat_template="gemma-4")
FastModel.for_inference(model)
log.info("Model loaded. VRAM allocated: %.2f GB", torch.cuda.memory_allocated() / 1e9)

# model.generate is not thread-safe; serialize requests with one lock.
_GEN_LOCK = asyncio.Lock()

app = FastAPI(title="Gemma 4 E2B RICO/OASST API")


class Message(BaseModel):
    role: str = Field(pattern="^(system|user|assistant)$")
    content: str
    # Optional list of base64-encoded image bytes (no data: prefix needed).
    images: list[str] | None = None


class ChatRequest(BaseModel):
    messages: list[Message]
    stream: bool = False
    temperature: float = 0.7
    top_p: float = 0.9
    max_tokens: int = 512
    format: str | dict | None = None  # accepted but unused (Ollama compat)


class GenerateRequest(BaseModel):
    prompt: str
    stream: bool = False
    temperature: float = 0.7
    top_p: float = 0.9
    max_tokens: int = 512
    format: str | dict | None = None


@app.get("/health")
async def health():
    return {
        "ok": True,
        "adapter": ADAPTER_PATH,
        "vram_alloc_gb": round(torch.cuda.memory_allocated() / 1e9, 2),
        "vram_total_gb": round(torch.cuda.get_device_properties(0).total_memory / 1e9, 2),
    }


def _b64_to_image(b64: str) -> Image.Image:
    if b64.startswith("data:"):
        b64 = b64.split(",", 1)[1]
    return Image.open(io.BytesIO(base64.b64decode(b64))).convert("RGB")


def _to_multimodal(messages: list[dict]) -> list[dict]:
    """{role, content, images?} -> Gemma 4 multimodal content list."""
    out = []
    for m in messages:
        content = []
        for b in m.get("images") or []:
            content.append({"type": "image", "image": _b64_to_image(b)})
        if m.get("content"):
            content.append({"type": "text", "text": m["content"]})
        out.append({"role": m["role"], "content": content})
    return out


def _build_inputs(messages: list[dict]):
    return tokenizer.apply_chat_template(
        _to_multimodal(messages),
        tokenize=True,
        add_generation_prompt=True,
        return_tensors="pt",
        return_dict=True,
    ).to("cuda")


def _gen_kwargs(req) -> dict:
    sample = req.temperature > 0
    return {
        "max_new_tokens": req.max_tokens,
        "do_sample": sample,
        "temperature": req.temperature if sample else 1.0,
        "top_p": req.top_p,
        "use_cache": True,
    }


async def _stream_response(inputs, gen_kwargs, sse_key: str) -> AsyncGenerator[bytes, None]:
    streamer = TextIteratorStreamer(tokenizer, skip_prompt=True, skip_special_tokens=True)
    thread = threading.Thread(
        target=model.generate,
        kwargs={**inputs, **gen_kwargs, "streamer": streamer},
        daemon=True,
    )
    thread.start()
    loop = asyncio.get_event_loop()
    # streamer is a blocking iterator; pull chunks off the loop's executor.
    while True:
        chunk = await loop.run_in_executor(None, lambda: next(streamer, None))
        if chunk is None:
            break
        if chunk:
            yield f"data: {json.dumps({sse_key: chunk})}\n\n".encode()
    yield f"data: {json.dumps({'is_message_completed': True})}\n\n".encode()


@app.post("/chat")
async def chat(
    payload: str = Form(...),
    image: UploadFile | None = File(None),
):
    """Multipart: payload=ChatRequest JSON, image=optional file."""
    try:
        req = ChatRequest.model_validate_json(payload)
    except Exception as e:
        raise HTTPException(422, f"invalid payload JSON: {e}")
    messages = [m.model_dump(exclude_none=True) for m in req.messages]

    # Attach uploaded image to the LAST user message — matches previous API behavior.
    if image is not None:
        raw = await image.read()
        if raw:
            b64 = base64.b64encode(raw).decode()
            for i in range(len(messages) - 1, -1, -1):
                if messages[i].get("role") == "user":
                    messages[i]["images"] = (messages[i].get("images") or []) + [b64]
                    break

    log.info(
        "/chat: %d messages, stream=%s, image=%s",
        len(messages), req.stream, image is not None and image.filename,
    )

    async with _GEN_LOCK:
        inputs = _build_inputs(messages)
        prompt_len = inputs["input_ids"].shape[1]
        gen_kwargs = _gen_kwargs(req)

        if req.stream:
            return StreamingResponse(
                _stream_response(inputs, gen_kwargs, sse_key="message"),
                media_type="text/event-stream",
            )

        with torch.inference_mode():
            out = model.generate(**inputs, **gen_kwargs)
        text = tokenizer.decode(out[0][prompt_len:], skip_special_tokens=True).strip()
        return {"response": text}


@app.post("/generate")
async def generate(req: GenerateRequest):
    """Single-prompt completion — wraps prompt as one user message."""
    messages = [{"role": "user", "content": req.prompt}]
    async with _GEN_LOCK:
        inputs = _build_inputs(messages)
        prompt_len = inputs["input_ids"].shape[1]
        gen_kwargs = _gen_kwargs(req)

        if req.stream:
            return StreamingResponse(
                _stream_response(inputs, gen_kwargs, sse_key="response"),
                media_type="text/event-stream",
            )

        with torch.inference_mode():
            out = model.generate(**inputs, **gen_kwargs)
        return {"response": tokenizer.decode(out[0][prompt_len:], skip_special_tokens=True).strip()}
