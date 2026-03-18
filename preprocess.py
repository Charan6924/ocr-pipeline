"""
preprocess.py
=============
Stage 1 preprocessing pipeline for early modern OCR – GOT-OCR-2.0 edition.

Optimised for *printed* Spanish early modern books (16th–18th c.).

Handles two scan layouts (book-level):
  SPREAD  – open-flat scan: each PDF page = two book pages side by side.
  SINGLE  – each PDF page = one book page.
  AUTO    – detect from aspect ratio + gutter valley (default).

Output per book page
--------------------
  <out>/<source>/<page_id>/
      <page_id>.png        – colour crop, lossless PNG, fed to GOT-OCR-2.0
      <page_id>_binary.png – binarised image for QA / debug (not fed to model)
      metadata.json        – deskew angle, scan layout, text-block bbox

Key differences from the TrOCR pipeline
----------------------------------------
GOT-OCR-2.0 is an end-to-end model that reads full page regions.  This
pipeline therefore:
  • Removes line segmentation entirely (no line strips, no lines/ folder).
  • Outputs one colour PNG per book page for the model.
    Colour is preferred: GOT-OCR-2.0 was trained on colour scans, and old
    Spanish printing often uses red/black bi-chrome ink.
  • Saves as lossless PNG to preserve fine strokes (long-s ſ, tildes, etc.).
  • Keeps Sauvola binarisation as a QA artefact only — NOT passed to model.
  • Keeps deskew and text-block crop (still improve accuracy).

Dependencies: pdf2image, Pillow, opencv-python-headless, numpy, scipy
"""

import json
import logging
from dataclasses import dataclass
from enum import Enum, auto
from pathlib import Path

import cv2
import numpy as np
from PIL import Image
from pdf2image import convert_from_path
from scipy.ndimage import rotate as ndimage_rotate

logging.basicConfig(level=logging.INFO, format="%(levelname)s  %(message)s")
log = logging.getLogger(__name__)


class Layout(Enum):
    """
    Controls how each raw PDF page is split into book pages.

    AUTO   – detect automatically from aspect ratio + gutter-valley signal.
             Reliable when sources are consistently portrait (SINGLE) or
             landscape (SPREAD). Safe default.
    SPREAD – each PDF page is two book pages side-by-side (e.g. printed book
             scanned open flat). Pages are split at the detected midpoint.
    SINGLE – each PDF page is one page (e.g. individual sheets photographed
             one page at a time).
    """
    AUTO   = auto()
    SPREAD = auto()
    SINGLE = auto()


@dataclass
class PreprocessConfig:
    # ── I/O ──────────────────────────────────────────────────────────────────
    pdf_path:    str = "/home/cxv166/OCR/Print 2/PORCONES.748.6 – 1650.pdf"
    output_root: str = "./output"
    dpi:         int = 300          # 300 for production, 150–200 for quick tests

    # ── Layout ───────────────────────────────────────────────────────────────
    layout: Layout = Layout.AUTO
    # Thresholds used by AUTO detection (ignored when layout is SPREAD/SINGLE)
    auto_aspect_threshold:   float = 1.4   # w/h ratio above which a spread is suspected
    auto_gutter_search_frac: float = 0.40  # fraction of width scanned for gutter valley
                                           # wide (±20 % either side of centre) because
                                           # Google Books spreads are often off-centre
    auto_gutter_rel_thresh:  float = 0.25  # min_ink/mean_ink below this → gutter confirmed

    # ── Binarisation (QA artefact only, NOT fed to GOT-OCR-2.0) ─────────────
    binarise_block: int   = 51    # Sauvola window size (pixels, must be odd)
    sauvola_k:      float = 0.2   # Sauvola k (0.1 = lighter, 0.5 = darker threshold)

    # ── Marginaliamasking ────────────────────────────────────────────────────
    margin_frac: float = 0.08     # Fraction of page width hard-masked each side

    # ── Blank-page detection ─────────────────────────────────────────────────
    blank_threshold:   float = 0.02  # Pages with < this fraction of dark pixels → blank
    blank_bottom_mask: float = 0.15  # Mask bottom fraction before blank check to
                                     # exclude clasps, scanner beds, watermarks

    # ── Deskew ───────────────────────────────────────────────────────────────
    deskew_max_angle: float = 5.0  # Only correct skew within ±this many degrees

    # ── Output ───────────────────────────────────────────────────────────────
    save_binary: bool = True  # Save _binary.png alongside colour crop (QA artefact)


# ─────────────────────────────────────────────────────────────────────────────
# Step 1 – PDF → PIL images
# ─────────────────────────────────────────────────────────────────────────────

def pdf_to_images(pdf_path: str, dpi: int) -> list[Image.Image]:
    """Convert each PDF page to a high-resolution PIL RGB image via Poppler."""
    log.info(f"Converting PDF → images at {dpi} DPI: {pdf_path}")
    pages = convert_from_path(pdf_path, dpi=dpi)
    log.info(f"  {len(pages)} PDF pages, first page size = {pages[0].size}")
    return pages


# ─────────────────────────────────────────────────────────────────────────────
# Step 2 – Layout detection and page splitting
# ─────────────────────────────────────────────────────────────────────────────

def find_gutter_x(img: Image.Image, config: PreprocessConfig) -> int:
    """
    Locate the binding gutter of a two-page spread and return its x position
    in the original image's pixel coordinates.

    Method
    ------
    Work on a downscaled greyscale copy for speed.  Binarise with Otsu, then
    compute per-column ink density across a search band centred on the image
    midpoint.  The column with the lowest ink density is the gutter valley.

    The search band is intentionally wide (`auto_gutter_search_frac`, default
    0.40 of total width = ±20 % either side of centre) because Google Books
    spreads are often not perfectly centred — one page can be noticeably wider
    than the other depending on how the book was placed on the scanner.

    Falls back to the exact midpoint if no clear valley is found.

    Returns an x coordinate in *original* image pixels.
    """
    w, h = img.size

    small_w = min(w, 1200)
    small_h = int(h * small_w / w)
    scale   = small_w / w                          # pixel → small coordinate
    small   = np.array(img.convert("L").resize((small_w, small_h), Image.LANCZOS))

    _, binary = cv2.threshold(small, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)

    cx       = small_w // 2
    half_win = int(small_w * config.auto_gutter_search_frac / 2)
    b_start  = max(0,       cx - half_win)
    b_end    = min(small_w, cx + half_win)

    col_inks = binary[:, b_start:b_end].sum(axis=0) / (255.0 * small_h + 1e-6)
    valley   = int(col_inks.argmin())              # offset within search band
    gutter_small = b_start + valley                # absolute x in small image
    gutter_orig  = int(gutter_small / scale)       # back to original pixels

    mean_ink = float(col_inks.mean())
    min_ink  = float(col_inks[valley])
    ratio    = (min_ink / mean_ink) if mean_ink > 1e-6 else 1.0

    log.debug(
        f"    gutter: valley_x={gutter_orig} (small={gutter_small}), "
        f"ink_ratio={ratio:.3f}"
    )

    # If the valley is not significantly lower than average the image may not
    # be a true spread; fall back to midpoint so we don't produce absurd crops.
    if ratio >= config.auto_gutter_rel_thresh:
        log.debug("    gutter: no clear valley — using midpoint")
        return w // 2

    return gutter_orig


def detect_layout(img: Image.Image, config: PreprocessConfig) -> Layout:
    """
    Infer whether a raw PDF page is a two-page spread or a single page.

    Two signals are combined:
    1. Aspect ratio — a spread is roughly twice as wide as tall
       (w/h > auto_aspect_threshold).  Portrait pages are immediately SINGLE.
    2. Gutter valley — delegate to find_gutter_x; a deep valley confirms SPREAD.

    Returns Layout.SPREAD or Layout.SINGLE (never Layout.AUTO).
    """
    w, h  = img.size
    ratio = w / h

    if ratio < config.auto_aspect_threshold:
        log.debug(f"    AUTO: ratio={ratio:.2f} → SINGLE")
        return Layout.SINGLE

    # Reuse find_gutter_x: if it returns midpoint the valley was not deep enough
    gutter_x = find_gutter_x(img, config)
    # A spread is confirmed when find_gutter_x found a real valley (it only
    # returns midpoint as fallback when ratio >= threshold, so we re-check).
    small_w  = min(w, 1200)
    small_h  = int(h * small_w / w)
    scale    = small_w / w
    small    = np.array(img.convert("L").resize((small_w, small_h), Image.LANCZOS))
    _, binary = cv2.threshold(small, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    cx        = small_w // 2
    half_win  = int(small_w * config.auto_gutter_search_frac / 2)
    col_inks  = binary[:, max(0, cx - half_win):min(small_w, cx + half_win)].sum(axis=0)
    col_inks  = col_inks / (255.0 * small_h + 1e-6)
    mean_ink  = float(col_inks.mean())
    min_ink   = float(col_inks.min()) if col_inks.size > 0 else 1.0
    gutter_ratio = (min_ink / mean_ink) if mean_ink > 1e-6 else 1.0

    result = Layout.SPREAD if gutter_ratio < config.auto_gutter_rel_thresh else Layout.SINGLE
    log.debug(
        f"    AUTO: ratio={ratio:.2f}, gutter_ratio={gutter_ratio:.3f} → {result.name}"
    )
    return result


def split_spread(
    img: Image.Image, config: PreprocessConfig
) -> tuple[Image.Image, Image.Image]:
    """
    Split a two-page spread at the detected gutter valley → (left, right).

    Uses find_gutter_x rather than the naïve pixel midpoint, so pages that
    are not perfectly centred in the scan are split correctly.
    """
    w, h     = img.size
    gutter_x = find_gutter_x(img, config)
    log.debug(f"    split_spread: gutter_x={gutter_x} (midpoint would be {w // 2})")
    return img.crop((0, 0, gutter_x, h)), img.crop((gutter_x, 0, w, h))


def get_book_pages(
    img: Image.Image, config: PreprocessConfig
) -> list[tuple[str, Image.Image]]:
    """
    Return a list of (label, image) pairs for a single raw PDF page.

    SPREAD → [("L", left_half), ("R", right_half)]
    SINGLE → [("",  full_image)]
    AUTO   → detect first, then delegate

    The label is appended to the PDF page number to form the page_id
    (e.g. "p003L", "p003R" for a spread; "p003" for a single page).
    """
    resolved = config.layout
    if resolved is Layout.AUTO:
        resolved = detect_layout(img, config)

    if resolved is Layout.SPREAD:
        left, right = split_spread(img, config)
        return [("L", left), ("R", right)]
    return [("", img)]


# ─────────────────────────────────────────────────────────────────────────────
# Step 3 – Blank detection
# ─────────────────────────────────────────────────────────────────────────────

def is_blank(img_gray: np.ndarray, config: PreprocessConfig) -> bool:
    """
    Return True if the page is effectively blank (cover, endpaper, etc.).

    The bottom `blank_bottom_mask` fraction is excluded before counting ink
    pixels to prevent clasps, scanner platens, and 'Digitized by Google'
    watermarks from pushing a blank page over the threshold.
    """
    h = img_gray.shape[0]
    trimmed = img_gray[:int(h * (1.0 - config.blank_bottom_mask)), :]
    _, binary = cv2.threshold(trimmed, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    return (binary.sum() / (255.0 * binary.size)) < config.blank_threshold


# ─────────────────────────────────────────────────────────────────────────────
# Step 4 – Deskew
# ─────────────────────────────────────────────────────────────────────────────

def estimate_skew(img_gray: np.ndarray, max_angle: float) -> float:
    """
    Estimate skew angle via probabilistic Hough lines on Canny edges.
    Only lines within ±max_angle of horizontal are considered.
    Returns the median angle in degrees (positive = CCW tilt).
    """
    small  = cv2.resize(img_gray, (img_gray.shape[1] // 2, img_gray.shape[0] // 2))
    edges  = cv2.Canny(small, 50, 150, apertureSize=3)
    lines  = cv2.HoughLinesP(
        edges, rho=1, theta=np.pi / 180, threshold=100,
        minLineLength=small.shape[1] // 5, maxLineGap=20,
    )
    if lines is None:
        return 0.0

    angles = [
        np.degrees(np.arctan2(y2 - y1, x2 - x1))
        for x1, y1, x2, y2 in lines[:, 0]
        if x2 != x1 and abs(np.degrees(np.arctan2(y2 - y1, x2 - x1))) <= max_angle
    ]
    return float(np.median(angles)) if angles else 0.0


def deskew(img_pil: Image.Image, max_angle: float) -> tuple[Image.Image, float]:
    """
    Correct skew. Returns (corrected_image, angle_applied).
    Uses scipy ndimage rotate (preserves dimensions, fills border with white).
    """
    angle = estimate_skew(np.array(img_pil.convert("L")), max_angle)
    if abs(angle) < 0.1:
        return img_pil, 0.0
    log.debug(f"    Deskewing by {angle:.2f}°")
    rotated = ndimage_rotate(np.array(img_pil), -angle, reshape=False, cval=255, order=1)
    return Image.fromarray(rotated.astype(np.uint8)), angle


# ─────────────────────────────────────────────────────────────────────────────
# Step 5 – Text-block detection (marginaliamasking)
# ─────────────────────────────────────────────────────────────────────────────

def detect_text_block(
    img_gray: np.ndarray, margin_frac: float
) -> tuple[int, int, int, int]:
    """
    Find the bounding box of the main text block via projection profiles.

    Left margin masking (margin_frac)
    ----------------------------------
    A hard left margin is masked before searching for x1 to suppress folio
    numbers, catchwords, and other left-edge marginalia that would otherwise
    pull the crop boundary too far left.

    Right and bottom edges are found from the FULL image width/height.
    Applying a symmetric hard right margin was clipping text on pages where
    the scan has little physical margin on the outer edge (common in Google
    Books spread scans after splitting).

    Bottom masking (blank_bottom_mask handled upstream)
    ---------------------------------------------------
    Scanner stamps and 'Digitized by Google' footers are excluded by the
    blank-detection step; they rarely affect the row-projection y2 because
    their ink density is low relative to body text.

    Returns (x1, y1, x2, y2).
    """
    h, w = img_gray.shape
    _, binary = cv2.threshold(img_gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)

    hard_l = int(w * margin_frac)

    # x1: search from hard_l inward so left-margin artefacts don't anchor the crop
    col_norm = binary.sum(axis=0) / (255.0 * h + 1e-6)
    ink_cols_inner = np.where(col_norm[hard_l:] > 0.005)[0]
    x1 = hard_l + (max(0, ink_cols_inner[0] - 20) if len(ink_cols_inner) else 0)

    # x2: search full width — outer text reaches close to the scan edge
    ink_cols_full = np.where(col_norm > 0.005)[0]
    x2 = min(w, (ink_cols_full[-1] + 20) if len(ink_cols_full) else w)

    # y1 / y2: full height projection profile
    row_norm = binary.sum(axis=1) / (255.0 * w + 1e-6)
    ink_rows = np.where(row_norm > 0.003)[0]
    y1 = max(0,     ink_rows[0]  - 30) if len(ink_rows) else 0
    y2 = min(h - 1, ink_rows[-1] + 30) if len(ink_rows) else h

    return int(x1), int(y1), int(x2), int(y2)


# ─────────────────────────────────────────────────────────────────────────────
# Step 6 – Binarisation (Sauvola) — QA artefact only, NOT fed to GOT-OCR-2.0
# ─────────────────────────────────────────────────────────────────────────────

def sauvola_binarise(img_gray: np.ndarray, block_size: int, k: float) -> np.ndarray:
    """
    Sauvola adaptive thresholding.

    Superior to global Otsu for aged paper: handles uneven illumination,
    reverse-side show-through, and foxing. Returns uint8 (0=bg, 255=ink).

    The result is saved as _binary.png for human QA.
    GOT-OCR-2.0 receives the colour crop, not this image.
    """
    if block_size % 2 == 0:
        block_size += 1

    img_f   = img_gray.astype(np.float64)
    mean    = cv2.boxFilter(img_f,      ddepth=-1, ksize=(block_size, block_size))
    sq_mean = cv2.boxFilter(img_f ** 2, ddepth=-1, ksize=(block_size, block_size))
    std     = np.sqrt(np.maximum(sq_mean - mean ** 2, 0))

    threshold = mean * (1.0 + k * (std / 128.0 - 1.0))
    binary    = np.where(img_f < threshold, 255, 0).astype(np.uint8)

    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (2, 2))
    return cv2.morphologyEx(binary, cv2.MORPH_OPEN, kernel)


# ─────────────────────────────────────────────────────────────────────────────
# Step 7 – Per-page orchestration
# ─────────────────────────────────────────────────────────────────────────────

def process_page(
    img_pil:  Image.Image,
    out_dir:  Path,
    page_id:  str,
    config:   PreprocessConfig,
) -> dict:
    """
    Full preprocessing for one book page.

    Pipeline
    --------
    1. Blank detection  → skip if blank.
    2. Deskew           → correct rotational misalignment.
    3. Text-block crop  → remove margins and irrelevant borders.
    4. Save colour PNG  → lossless, fed to GOT-OCR-2.0.
    5. Binarise         → Sauvola, saved as QA artefact only.
    6. Write metadata.json.

    Output files
    ------------
    <page_id>.png        – colour crop fed to GOT-OCR-2.0
    <page_id>_binary.png – binarised QA image (toggle with save_binary)
    metadata.json
    """
    out_dir.mkdir(parents=True, exist_ok=True)
    meta: dict = {
        "page_id":         page_id,
        "blank":           False,
        "deskew_angle":    0.0,
        "text_block_bbox": None,
    }

    img_gray = np.array(img_pil.convert("L"))

    # ── 1. Blank detection ───────────────────────────────────────────────────
    if is_blank(img_gray, config):
        log.info(f"  [{page_id}] BLANK – skipping")
        meta["blank"] = True
        (out_dir / "metadata.json").write_text(json.dumps(meta, indent=2))
        return meta

    # ── 2. Deskew ────────────────────────────────────────────────────────────
    img_deskewed, angle = deskew(img_pil, config.deskew_max_angle)
    meta["deskew_angle"] = round(angle, 3)
    img_gray_desk = np.array(img_deskewed.convert("L"))

    # ── 3. Text-block crop ───────────────────────────────────────────────────
    x1, y1, x2, y2 = detect_text_block(img_gray_desk, config.margin_frac)
    meta["text_block_bbox"] = [x1, y1, x2, y2]
    img_cropped  = img_deskewed.crop((x1, y1, x2, y2))
    gray_cropped = np.array(img_cropped.convert("L"))

    # ── 4. Save colour PNG for GOT-OCR-2.0 ──────────────────────────────────
    #   Lossless PNG preserves fine strokes: long-s (ſ), tildes, abbreviations
    img_cropped.save(out_dir / f"{page_id}.png", "PNG")
    log.info(f"  [{page_id}] Saved colour crop")

    # ── 5. Binarise for QA ───────────────────────────────────────────────────
    if config.save_binary:
        binary = sauvola_binarise(gray_cropped, config.binarise_block, config.sauvola_k)
        Image.fromarray(binary).save(out_dir / f"{page_id}_binary.png")

    (out_dir / "metadata.json").write_text(json.dumps(meta, indent=2))
    return meta


# ─────────────────────────────────────────────────────────────────────────────
# Top-level runner
# ─────────────────────────────────────────────────────────────────────────────

def run(config: PreprocessConfig = PreprocessConfig()) -> list[dict]:
    """
    Process an entire PDF using settings from a PreprocessConfig.

    Usage examples
    --------------
    # Printed book scanned as two-page spreads (Buendia):
    run(PreprocessConfig(
        pdf_path    = "Buendia_-_Instruccion.pdf",
        output_root = "./output",
        dpi         = 300,
        layout      = Layout.SPREAD,
    ))

    # Individual page scans:
    run(PreprocessConfig(
        pdf_path    = "Peticion_1622.pdf",
        output_root = "./output",
        dpi         = 300,
        layout      = Layout.SINGLE,
    ))

    # Auto-detect layout per page (safe default for unknown sources):
    run(PreprocessConfig(pdf_path = "unknown_source.pdf"))
    """
    pdf_path    = Path(config.pdf_path)
    output_root = Path(config.output_root)
    source_name = pdf_path.stem
    output_root.mkdir(parents=True, exist_ok=True)

    log.info(f"=== {pdf_path.name} | DPI={config.dpi} | layout={config.layout.name} ===")
    pages    = pdf_to_images(str(pdf_path), config.dpi)
    all_meta = []
    tally    = {Layout.SPREAD.name: 0, Layout.SINGLE.name: 0}

    for pdf_idx, page_img in enumerate(pages):
        pdf_num       = pdf_idx + 1
        book_pages    = get_book_pages(page_img, config)
        scan_resolved = Layout.SPREAD if len(book_pages) == 2 else Layout.SINGLE
        tally[scan_resolved.name] += 1
        log.info(f"PDF page {pdf_num}/{len(pages)} [{scan_resolved.name}]")

        for label, sub_img in book_pages:
            page_id  = f"p{pdf_num:03d}{label}" if label else f"p{pdf_num:03d}"
            page_dir = output_root / source_name / page_id
            meta     = process_page(sub_img, page_dir, page_id, config)
            meta["layout"] = scan_resolved.name
            all_meta.append(meta)
            log.info(f"  → {page_id}: {'BLANK' if meta['blank'] else 'OK'}")

    summary = {
        "source":        source_name,
        "dpi":           config.dpi,
        "layout_config": config.layout.name,
        "layout_tally":  tally,
        "pdf_pages":     len(pages),
        "book_pages":    len(all_meta),
        "blank_pages":   sum(1 for m in all_meta if m.get("blank")),
    }
    (output_root / source_name / "summary.json").write_text(
        json.dumps(summary, indent=2)
    )
    log.info(f"Done – {summary['book_pages']} book pages processed")
    return all_meta


if __name__ == "__main__":
    run()