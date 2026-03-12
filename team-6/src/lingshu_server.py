
"""
LingShu-I-8B Server
====================
Wraps the OFFICIAL LingShu-I-8B inference code into a FastAPI server.
Uses vLLM Python API (LLM class) with enforce_eager=True.

Key: This uses vllm.LLM() directly in Python, NOT "vllm serve" CLI.
This avoids the _moe_C.topk_softmax error.

Run: CUDA_VISIBLE_DEVICES=0,1 python lingshu_server.py
"""

import os
import json
import base64
import io
import logging
import dataclasses
from enum import IntEnum, auto
from typing import Dict, List, Tuple, Union, Optional

from PIL import Image
from fastapi import FastAPI
from pydantic import BaseModel
import uvicorn

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# ── Config ──
MODEL_PATH = os.environ.get(
    "LINGSHU_MODEL_PATH",
    "/scratch/rzv4ve/cardioagent/tools/MRI/Lingshu"
)
PORT = int(os.environ.get("LINGSHU_PORT", "8001"))
# Set to 1 if you only have 1 GPU, 2 if you have 2
TP_SIZE = int(os.environ.get("LINGSHU_TP_SIZE", "2"))
GPU_UTIL = float(os.environ.get("LINGSHU_GPU_UTIL", "0.7"))

os.environ["VLLM_WORKER_MULTIPROC_METHOD"] = "spawn"
os.environ["TOKENIZERS_PARALLELISM"] = "false"

app = FastAPI(title="LingShu-I-8B Medical VLM Server")


# ============================================================
# Official LingShu Conversation Template (copied exactly)
# ============================================================

class SeparatorStyle(IntEnum):
    MPT = auto()

@dataclasses.dataclass
class Conversation:
    name: str
    system_template: str = '{system_message}'
    system_message: str = ''
    roles: Tuple[str, str] = ('USER', 'ASSISTANT')
    messages: List[List[str]] = dataclasses.field(default_factory=list)
    sep_style: SeparatorStyle = SeparatorStyle.MPT
    sep: str = '\n'

    def append_message(self, role, message):
        self.messages.append([role, message])

    def get_prompt(self) -> str:
        system_prompt = self.system_template.format(system_message=self.system_message)
        ret = system_prompt + self.sep
        for role, message in self.messages:
            if message:
                if isinstance(message, tuple):
                    message = message[0]
                ret += role + message + self.sep
            else:
                ret += role
        return ret


def process_messages(messages: list) -> dict:
    """
    Official LingShu message processing.
    Converts OpenAI-style messages to vLLM inputs with InternVL template.
    """
    conv = Conversation(
        name='internvl2_5',
        system_template='<|im_start|>system\n{system_message}',
        system_message='You are a helpful medical AI assistant.',
        roles=('<|im_start|>user\n', '<|im_start|>assistant\n'),
        sep_style=SeparatorStyle.MPT,
        sep='<|im_end|>\n',
    )
    conv.messages = []
    imgs = []

    for message in messages:
        role = message.get('role', 'user')
        content = message.get('content', '')

        # Map role to conv role
        if role == 'user':
            conv_role = conv.roles[0]
        elif role == 'assistant':
            conv_role = conv.roles[1]
        elif role == 'system':
            conv.system_message = content if isinstance(content, str) else ''
            continue
        else:
            conv_role = conv.roles[0]

        if isinstance(content, str):
            conv.append_message(conv_role, content)
        elif isinstance(content, list):
            text = ""
            for item in content:
                if isinstance(item, str):
                    text += item
                elif isinstance(item, dict):
                    if item.get('type') == 'text':
                        text += item['text']
                    elif item.get('type') == 'image':
                        # Direct file path (official format)
                        text = text + "\n<IMG_CONTEXT>"
                        image = item['image']
                        if isinstance(image, str):
                            image = Image.open(image).convert("RGB")
                        imgs.append(image)
                    elif item.get('type') == 'image_url':
                        # OpenAI format — decode base64 or file path
                        text = text + "\n<IMG_CONTEXT>"
                        url = item['image_url']['url']
                        img = decode_image(url)
                        if img:
                            imgs.append(img)

            conv.append_message(conv_role, text)

    conv.append_message(conv.roles[1], None)
    prompt = conv.get_prompt()

    mm_data = {}
    if len(imgs) > 0:
        mm_data["image"] = imgs

    return {"prompt": prompt, "multi_modal_data": mm_data}


def decode_image(url: str) -> Optional[Image.Image]:
    """Decode from base64 data URL or file path"""
    try:
        if url.startswith("data:image"):
            b64_data = url.split(",")[1]
            return Image.open(io.BytesIO(base64.b64decode(b64_data))).convert("RGB")
        elif os.path.exists(url):
            return Image.open(url).convert("RGB")
        return None
    except Exception as e:
        logger.error(f"Image decode error: {e}")
        return None


# ============================================================
# vLLM Engine (loaded once at startup)
# ============================================================

llm_engine = None
sampling_params_default = None


def load_model():
    global llm_engine, sampling_params_default
    from vllm import LLM, SamplingParams

    logger.info(f"Loading LingShu-I-8B from {MODEL_PATH}...")
    logger.info(f"  tensor_parallel_size={TP_SIZE}, gpu_memory_utilization={GPU_UTIL}")

    llm_engine = LLM(
        model=MODEL_PATH,
        limit_mm_per_prompt={"image": 4},
        tensor_parallel_size=TP_SIZE,
        enforce_eager=True,          # Critical: avoids CUDA graph compilation errors
        trust_remote_code=True,
        gpu_memory_utilization=GPU_UTIL,
    )

    sampling_params_default = SamplingParams(
        temperature=0.7,
        top_p=1,
        repetition_penalty=1,
        max_tokens=1024,
        stop_token_ids=[],
    )

    logger.info("LingShu-I-8B loaded successfully!")


def generate_response(
    messages: list,
    max_tokens: int = 1024,
    temperature: float = 0.1,
) -> str:
    """Generate response using official LingShu pipeline"""
    from vllm import SamplingParams

    # Process messages using official template
    llm_inputs = process_messages(messages)

    # Create sampling params
    params = SamplingParams(
        temperature=max(temperature, 0.01),
        top_p=1,
        repetition_penalty=1,
        max_tokens=max_tokens,
        stop_token_ids=[],
    )

    # Generate
    outputs = llm_engine.generate([llm_inputs], sampling_params=params)
    return outputs[0].outputs[0].text


# ============================================================
# FastAPI Endpoints (OpenAI-compatible)
# ============================================================

class Message(BaseModel):
    role: str
    content: Union[str, list]

class ChatRequest(BaseModel):
    model: str = "lingshu-8b"
    messages: list[Message]
    max_tokens: int = 1024
    temperature: float = 0.1


@app.get("/v1/models")
def list_models():
    return {
        "object": "list",
        "data": [{"id": "lingshu-8b", "object": "model", "owned_by": "lingshu"}]
    }


@app.post("/v1/chat/completions")
async def chat_completions(request: ChatRequest):
    messages = [msg.model_dump() for msg in request.messages]

    try:
        response_text = generate_response(
            messages=messages,
            max_tokens=request.max_tokens,
            temperature=request.temperature,
        )
    except Exception as e:
        logger.error(f"Generation error: {e}")
        response_text = f"Error during generation: {str(e)}"

    return {
        "id": "chatcmpl-lingshu",
        "object": "chat.completion",
        "model": "lingshu-8b",
        "choices": [{
            "index": 0,
            "message": {"role": "assistant", "content": response_text},
            "finish_reason": "stop",
        }],
    }


@app.get("/health")
def health():
    return {"status": "ok", "model": "LingShu-I-8B", "tp_size": TP_SIZE}


# ============================================================
# Main
# ============================================================

if __name__ == "__main__":
    load_model()
    logger.info(f"Starting LingShu-I-8B server on port {PORT}...")
    uvicorn.run(app, host="0.0.0.0", port=PORT)