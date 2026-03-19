"""
evaluate.py
===========
Computes Character Error Rate (CER) between ground truth transcriptions
and OCR output (raw and cleaned).

CER = (insertions + deletions + substitutions) / total ground truth characters
    = Levenshtein edit distance / len(ground_truth)

Ground truth format (docx exported to plain text)
-------------------------------------------------
The docx uses page markers like "PDF p2", "PDF p3" to delimit pages.
Each marker is followed by the transcribed text for that page.

Page ID mapping
---------------
"PDF p2" maps to the preprocessed page IDs for that PDF page number.
For SPREAD sources: PDF page 2 → p002L and p002R
For SINGLE sources: PDF page 2 → p002

Usage
-----
python evaluate.py
"""

import re
import logging
from dataclasses import dataclass, field
from pathlib import Path

logging.basicConfig(level=logging.INFO, format="%(levelname)s  %(message)s")
log = logging.getLogger(__name__)


@dataclass
class EvalConfig:
    # Path to the ground truth plain text file (convert docx with pandoc first)
    # pandoc Buendia_-_Instruccion_transcription.docx -t plain -o ground_truth.txt
    ground_truth_txt: str = "/mnt/vstor/courses/csds312/cxv166/OCR/ground_truth/ground_truth_porcones_1650.txt"
    # Preprocessed output directory for the source being evaluated
    source_dir:       str = "/mnt/vstor/courses/csds312/cxv166/OCR/output/PORCONES.748.6 – 1650"

    # Layout of the source — affects how PDF page numbers map to page IDs
    # "spread" → PDF p2 = p002L + p002R
    # "single" → PDF p2 = p002
    layout:           str = "single"
    eval_raw:         bool = True   # <page_id>.txt
    eval_cleaned:     bool = True   # <page_id>_cleaned.txt

def levenshtein(a: str, b: str) -> int:
    """Compute character-level Levenshtein edit distance."""
    if not a:
        return len(b)
    if not b:
        return len(a)

    prev = list(range(len(b) + 1))
    for i, ca in enumerate(a):
        curr = [i + 1] + [0] * len(b)
        for j, cb in enumerate(b):
            curr[j + 1] = min(
                prev[j + 1] + 1,       # deletion
                curr[j]     + 1,       # insertion
                prev[j] + (ca != cb),  # substitution
            )
        prev = curr
    return prev[-1]


def normalise(text: str) -> str:
    """
    Minimal normalisation applied to both GT and OCR before CER computation.

    - Rejoin line-end hyphens (assis-\ntir → assistir)
    - Collapse whitespace
    - Lowercase

    Spelling is NOT changed — assi, disseño, vuestra etc. are preserved
    so the CER reflects real OCR errors, not orthographic differences.
    """
    # Rejoin hyphenated line breaks
    text = re.sub(r"-\n(\S)", r"\1", text)
    # Collapse all whitespace to single space
    text = " ".join(text.split())
    return text.lower()


def cer(hypothesis: str, reference: str) -> float:
    """
    Character Error Rate = edit_distance(hyp, ref) / len(ref).
    Both strings are normalised before comparison.
    """
    ref = normalise(reference)
    hyp = normalise(hypothesis)
    if not ref:
        return 0.0
    return levenshtein(hyp, ref) / len(ref)


def parse_ground_truth(gt_path: Path) -> dict[int, str]:
    """
    Parse a plain-text ground truth file with "PDF pN" page markers.

    Returns {pdf_page_number: text} dict.
    The header notes block before the first "PDF p" marker is discarded.
    """
    text   = gt_path.read_text(encoding="utf-8")
    chunks = re.split(r"(?m)^PDF\s+p(\d+)\s*$", text)

    pages = {}
    # re.split with a capturing group gives [pre, num, text, num, text, ...]
    it = iter(chunks[1:])   # skip the pre-marker header
    for num_str, content in zip(it, it):
        pages[int(num_str)] = content.strip()

    log.info(f"Ground truth: {len(pages)} pages parsed from {gt_path.name}")
    return pages

def page_ids_for(pdf_num: int, layout: str) -> list[str]:
    """
    Return the page_id(s) that correspond to a PDF page number.

    spread → ["p002L", "p002R"]
    single → ["p002"]
    """
    if layout == "spread":
        return [f"p{pdf_num:03d}L", f"p{pdf_num:03d}R"]
    return [f"p{pdf_num:03d}"]

def read_ocr(source_dir: Path, page_id: str, cleaned: bool) -> str | None:
    """Read raw or cleaned OCR text for a page. Returns None if missing."""
    suffix   = "_cleaned.txt" if cleaned else ".txt"
    txt_path = source_dir / page_id / f"{page_id}{suffix}"
    if not txt_path.exists():
        return None
    return txt_path.read_text(encoding="utf-8").strip()


def evaluate(config: EvalConfig):
    source_dir = Path(config.source_dir)
    gt_path    = Path(config.ground_truth_txt)

    if not gt_path.exists():
        log.error(f"Ground truth file not found: {gt_path}")
        log.error("Convert the docx first: pandoc ground_truth.docx -t plain -o ground_truth.txt")
        return

    gt_pages = parse_ground_truth(gt_path)

    results = []

    for pdf_num, gt_text in sorted(gt_pages.items()):
        page_ids = page_ids_for(pdf_num, config.layout)

        # For spread sources the GT covers both pages — concatenate OCR output
        raw_parts     = []
        cleaned_parts = []

        for page_id in page_ids:
            if config.eval_raw:
                raw = read_ocr(source_dir, page_id, cleaned=False)
                if raw is not None:
                    raw_parts.append(raw)
                else:
                    log.warning(f"  Raw OCR missing for {page_id}")

            if config.eval_cleaned:
                cleaned = read_ocr(source_dir, page_id, cleaned=True)
                if cleaned is not None:
                    cleaned_parts.append(cleaned)
                else:
                    log.warning(f"  Cleaned OCR missing for {page_id}")

        raw_text     = " ".join(raw_parts)
        cleaned_text = " ".join(cleaned_parts)

        raw_cer     = cer(raw_text,     gt_text) if raw_parts     else None
        cleaned_cer = cer(cleaned_text, gt_text) if cleaned_parts else None

        label = "+".join(page_ids)
        results.append((label, gt_text, raw_cer, cleaned_cer))

        log.info(
            f"  PDF p{pdf_num} [{label}]"
            + (f"  raw CER={raw_cer:.3f}"     if raw_cer     is not None else "")
            + (f"  cleaned CER={cleaned_cer:.3f}" if cleaned_cer is not None else "")
        )

    raw_scores     = [r for _, _, r, _ in results if r is not None]
    cleaned_scores = [c for _, _, _, c in results if c is not None]

    print("\n" + "=" * 60)
    print(f"{'Page':<20} {'Raw CER':>10} {'Cleaned CER':>12}")
    print("-" * 60)
    for label, _, raw_c, clean_c in results:
        r = f"{raw_c:.3f}"     if raw_c     is not None else "N/A"
        c = f"{clean_c:.3f}"   if clean_c   is not None else "N/A"
        print(f"{label:<20} {r:>10} {c:>12}")

    print("-" * 60)
    if raw_scores:
        print(f"{'MEAN':<20} {sum(raw_scores)/len(raw_scores):>10.3f}", end="")
    if cleaned_scores:
        print(f" {sum(cleaned_scores)/len(cleaned_scores):>12.3f}", end="")
    print()
    print("=" * 60)


def main():
    config = EvalConfig()
    evaluate(config)


if __name__ == "__main__":
    main()