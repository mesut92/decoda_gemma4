"""
Merge LoRA adapter with base Gemma 4 E2B and export to GGUF for Ollama / llama.cpp.

Output: ./gemma4_e2b_rico_gguf/unsloth.Q4_K_M.gguf

Run once before `docker compose up` so the ollama-init container has the
GGUF to load.
"""
import os
os.environ["PYTORCH_ALLOC_CONF"] = "expandable_segments:True"
os.environ.setdefault("HF_HUB_DISABLE_XET", "1")

from unsloth import FastModel

ADAPTER_DIR = "gemma4_e2b_rico_adapter"
GGUF_OUT_DIR = "gemma4_e2b_rico_gguf"
QUANT = "q4_k_m"  # Options: q4_k_m (4-bit, recommended), q5_k_m, q8_0, f16

model, tokenizer = FastModel.from_pretrained(
    model_name=ADAPTER_DIR,
    max_seq_length=2048,
    load_in_4bit=True,
    full_finetuning=False,
)

# Unsloth merges LoRA into the base, runs llama.cpp converter + quantizer.
model.save_pretrained_gguf(
    GGUF_OUT_DIR,
    tokenizer,
    quantization_method=QUANT,
)

print(f"\nGGUF written to ./{GGUF_OUT_DIR}/")
