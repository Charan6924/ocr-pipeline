import os
import json
import logging
from dataclasses import dataclass
from pathlib import Path

from PIL import Image
import torch
from transformers import AutoModel, AutoTokenizer
from openai import OpenAI

logging.basicConfig(level=logging.INFO, format="%(levelname)s  %(message)s")
log = logging.getLogger(__name__)

Image.MAX_IMAGE_PIXELS = 5_000_000_000
MODEL_ID = "ucaslcl/GOT-OCR2_0"

CLEAN_PROMPT = """\
You are a specialist in early modern Spanish paleography (16th–18th century).
You will be given raw OCR output from a scanned page of a printed Spanish book.
Your task is to clean the text while preserving the original language exactly.

Rules:
- Replace long-s (ſ) with s
- Replace ç with z
- Rejoin words split by line-end hyphens (e.g. "assis-\\ntir" → "assistir")
- Remove page artifacts: Google Books stamps, decorative border characters
  misread as letters (e.g. leading "T Q X A" on title pages), catchwords,
  running headers, and page numbers
- Do NOT modernise spelling — preserve original orthography (assi, vuestra, etc.)
- Do NOT add or remove words
- Do NOT translate or paraphrase
- Return only the cleaned text, nothing else

Raw OCR:
"""


def load_model(device: str):
    log.info(f"Loading {MODEL_ID} onto {device} …")
    tokenizer = AutoTokenizer.from_pretrained(
        MODEL_ID, trust_remote_code=True
    )
    model = AutoModel.from_pretrained(
        MODEL_ID,
        trust_remote_code=True,
        torch_dtype=torch.bfloat16,
        device_map=device,
    )
    model.eval()
    log.info("Model loaded.")
    return model, tokenizer


def load_openai_client():
    api_key = os.environ.get("OPENAI_API_KEY")
    if not api_key:
        raise EnvironmentError("OPENAI_API_KEY environment variable is not set.")
    return OpenAI(api_key=api_key)


def clean_with_llm(client: OpenAI, raw_text: str, model: str = "gpt-4o-mini"):
    response = client.chat.completions.create(
        model=model,
        messages=[
            {"role": "user", "content": CLEAN_PROMPT + raw_text}
        ],
        temperature=0.0,   
    )
    return response.choices[0].message.content.strip()


def iter_pages(source_dir: Path):
    for page_dir in sorted(source_dir.iterdir()):
        if not page_dir.is_dir():
            continue
        meta_path = page_dir / "metadata.json"
        if not meta_path.exists():
            continue
        meta = json.loads(meta_path.read_text())
        if meta.get("blank"):
            log.info(f"[{meta['page_id']}] is blank")
            continue
        png_path = page_dir / f"{page_dir.name}.png"
        if not png_path.exists():
            log.warning(f"[{page_dir.name}] Image is missing")
            continue
        yield page_dir.name, png_path


def run_inference(source_dir: Path, model, tokenizer, openai_client: OpenAI) -> int:
    pages = list(iter_pages(source_dir))
    log.info(f"Found {len(pages)} pages to transcribe in {source_dir.name}")

    for page_id, png_path in pages:
        log.info(f"[{page_id}] running OCR …")
        try:
            result = model.chat(
                tokenizer,
                str(png_path),
                ocr_type="ocr",
            )
        except Exception as exc:
            log.error(f"[{page_id}] inference failed: {exc}")
            continue

        raw_path = png_path.parent / f"{page_id}.txt"
        raw_path.write_text(result, encoding="utf-8")
        log.info(f"[{page_id}] raw → {raw_path.name} ({len(result)} chars)")

        try:
            cleaned = clean_with_llm(openai_client, result)
            clean_path = png_path.parent / f"{page_id}_cleaned.txt"
            clean_path.write_text(cleaned, encoding="utf-8")
            log.info(f"  [{page_id}] cleaned → {clean_path.name} ({len(cleaned)} chars)")
        except Exception as exc:
            log.error(f"  [{page_id}] LLM cleaning failed: {exc}")

    return len(pages)


@dataclass
class InferConfig:
    source_dir: str = os.environ.get("SOURCE_DIR", "./output/Buendia_-_Instruccion")
    device: str = "cuda"
    openai_model: str = "gpt-4o-mini"  


def main():
    config = InferConfig()
    source_dir = Path(config.source_dir)

    if not source_dir.is_dir():
        log.error(f"Source directory not found: {source_dir}")
        return

    openai_client = load_openai_client()
    model, tokenizer = load_model(config.device)
    n = run_inference(source_dir, model, tokenizer, openai_client)
    log.info(f"Done! {n} pages transcribed from {source_dir.name}")


if __name__ == "__main__":
    main()