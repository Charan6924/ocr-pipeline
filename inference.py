import os
import json
import logging
from dataclasses import dataclass
from pathlib import Path
from PIL import Image
import torch
from transformers import AutoModel, AutoTokenizer

logging.basicConfig(level=logging.INFO, format="%(levelname)s  %(message)s")
log = logging.getLogger(__name__)
Image.MAX_IMAGE_PIXELS = 5_000_000_000
MODEL_ID = "ucaslcl/GOT-OCR2_0"

def load_model(device: str):
    """Load GOT-OCR-2.0 model and tokenizer onto `device`."""
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


def iter_pages(source_dir: Path):
    """
    Yield (page_id, png_path) for every non-blank page in source_dir,
    in page-number order.

    A page is skipped if its metadata.json marks it blank OR if the
    expected .png file does not exist (split artefact, etc.).
    """
    page_dirs = sorted(source_dir.iterdir())
    for page_dir in page_dirs:
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


def run_inference(source_dir: Path, model, tokenizer) -> int:
    """
    Run GOT-OCR-2.0 over all pages in source_dir.
    Returns the number of pages transcribed.
    """
    pages = list(iter_pages(source_dir))
    log.info(f"Found {len(pages)} pages to transcribe in {source_dir.name}")

    for page_id, png_path in pages:
        log.info(f"  [{page_id}] running OCR …")
        try:
            # ocr_type='ocr' → plain text, no markdown formatting
            result = model.chat(
                tokenizer,
                str(png_path),
                ocr_type="ocr",
            )
        except Exception as exc:
            log.error(f"  [{page_id}] inference failed: {exc}")
            continue

        out_path = png_path.parent / f"{page_id}.txt"
        out_path.write_text(result, encoding="utf-8")
        log.info(f"[{page_id}] to {out_path.name} ({len(result)} chars)")

    return len(pages)

@dataclass
class InferConfig:
    source_dir: str = os.environ.get("SOURCE_DIR", "./output/Buendia_-_Instruccion")
    device: str = "cuda"


def main():
    config = InferConfig()
    source_dir = Path(config.source_dir)

    if not source_dir.is_dir():
        log.error(f"Source directory not found: {source_dir}")
        return

    model, tokenizer = load_model(config.device)
    n = run_inference(source_dir, model, tokenizer)
    log.info(f"Done – {n} pages transcribed from {source_dir.name}")


if __name__ == "__main__":
    main()