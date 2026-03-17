"""
preprocess.py
=============
Stage 1 preprocessing pipeline for early modern OCR – GOT-OCR-2.0 edition.

Optimised for *printed* Spanish early modern books (16th–18th c.).

Handles two scan layouts (book-level):
  SPREAD  – open-flat scan: each PDF page = two book pages side by side.
  SINGLE  – each PDF page = one book page.
  AUTO    – detect from aspect ratio + gutter valley (default).

Handles two text column layouts (page-level):
  ONE_COL  – single column of text.
  TWO_COL  – two columns separated by a gutter (bilingual editions,
             glossaries, legal codices, etc.).
  COL_AUTO – detect per page from vertical projection profile (default).

Key differences from the TrOCR pipeline
----------------------------------------
GOT-OCR-2.0 is an end-to-end model: it reads full page regions and handles
its own internal layout understanding.  It does NOT want pre-segmented line
strips.  This pipeline therefore:

  • Removes line segmentation entirely.
  • Outputs one *colour* PNG per text region (full column or full page).
    Colour is preferred: GOT-OCR-2.0 was trained on colour scans, and old
    Spanish printing often uses red/black bi-chrome ink.
  • Keeps deskew and text-block crop (still improve accuracy).
  • Keeps Sauvola binarisation as a QA artefact and for column detection
    only — it is NOT passed to the model.
  • Saves regions as lossless PNG (not JPEG) to avoid compression artefacts
    on fine strokes such as long-s (ſ), tildes, and abbreviation marks.

Output per book page
--------------------
  <out>/<source>/<page_id>/
      <page_id>_col1.png   – colour crop, column 1 (or full page if 1-col)
      <page_id>_col2.png   – colour crop, column 2 (two-column pages only)
      <page_id>_binary.png – binarised image for QA / debug (not fed to model)
      metadata.json        – col_layout, region bboxes, deskew angle, …

Dependencies: pdf2image, Pillow, opencv-python-headless, numpy, scipy
"""

import json
import logging
from dataclasses import dataclass, field
from enum import Enum, auto
from pathlib import Path

import cv2
import numpy as np
from PIL import Image
from pdf2image import convert_from_path
from scipy.ndimage import rotate as ndimage_rotate

logging.basicConfig(level=logging.INFO, format="%(levelname)s  %(message)s")
log = logging.getLogger(__name__)


# ─────────────────────────────────────────────────────────────────────────────
# Enumerations
# ─────────────────────────────────────────────────────────────────────────────

class Layout(Enum):
    """
    Controls how each raw PDF page is split into book pages.

    AUTO   – detect automatically from aspect ratio + gutter valley signal.
    SPREAD – each PDF page is two book pages side by side (e.g. printed book
             scanned open flat).  Pages are split at the detected midpoint.
    SINGLE – each PDF page is one page.
    """
    AUTO   = auto()
    SPREAD = auto()
    SINGLE = auto()


class ColLayout(Enum):
    """
    Controls how each book page is split into text columns.

    COL_AUTO – detect per page from vertical projection profile (default).
               Reliable for books that consistently use one layout, and also
               handles mixed pages (e.g. title pages vs. body text).
    ONE_COL  – full page is a single text column.
    TWO_COL  – page carries two columns separated by a gutter (e.g. Spanish/
               Latin bilingual editions, legal glossaries).
    """
    COL_AUTO = auto()
    ONE_COL  = auto()
    TWO_COL  = auto()

@dataclass
class PreprocessConfig:
    # ── I/O ──────────────────────────────────────────────────────────────────
    pdf_path:    str = "source.pdf"
    output_root: str = "./output"
    dpi:         int = 300          # 300 for production, 150–200 for quick tests

    # ── Scan layout ──────────────────────────────────────────────────────────
    layout: Layout = Layout.AUTO
    # AUTO thresholds (ignored when layout is SPREAD/SINGLE)
    auto_aspect_threshold:   float = 1.4   # w/h ratio above which a spread is suspected
    auto_gutter_search_frac: float = 0.20  # fraction of width to scan for vertical gutter
    auto_gutter_rel_thresh:  float = 0.25  # min_ink/mean_ink below this → gutter confirmed

    # ── Column layout ────────────────────────────────────────────────────────
    col_layout: ColLayout = ColLayout.COL_AUTO
    # COL_AUTO thresholds (ignored when col_layout is ONE_COL/TWO_COL)
    col_search_band:      float = 0.30  # fraction of page width searched for column gap
                                        # centred on the midpoint (e.g. 0.30 → 35–65%)
    col_gap_rel_thresh:   float = 0.15  # min_ink/mean_ink below this → two columns confirmed
    col_min_width_frac:   float = 0.20  # each detected column must be ≥ this fraction of the
                                        # page width, guards against spurious splits on
                                        # pages with a single centred column or a large initial
    col_padding:          int   = 8     # pixels of horizontal padding added around each column
                                        # crop so that ascenders/descenders are not clipped

    # ── Binarisation (for column detection + QA only, NOT fed to model) ──────
    binarise_block: int   = 51    # Sauvola window size (pixels, must be odd)
    sauvola_k:      float = 0.2   # Sauvola k (0.1 = lighter, 0.5 = darker threshold)

    # ── Marginaliamasking ────────────────────────────────────────────────────
    margin_frac: float = 0.08     # Fraction of page width hard-masked each side

    # ── Blank-page detection ──────────────────────────────────────────────────
    blank_threshold: float = 0.02  # Pages with < this fraction of dark pixels → blank

    # ── Deskew ───────────────────────────────────────────────────────────────
    deskew_max_angle: float = 5.0  # Only correct skew within ±this many degrees

    # ── Output ───────────────────────────────────────────────────────────────
    save_binary: bool = True  # Save _binary.png alongside colour crops (QA artefact)

def pdf_to_images(pdf_path: str, dpi: int) -> list[Image.Image]:
    """Convert each PDF page to a high-resolution PIL RGB image via Poppler."""
    log.info(f"Converting PDF → images at {dpi} DPI: {pdf_path}")
    pages = convert_from_path(pdf_path, dpi=dpi)
    log.info(f"  {len(pages)} PDF pages, first page size = {pages[0].size}")
    return pages


def detect_layout(img: Image.Image, config: PreprocessConfig) -> Layout:
    """
    Infer whether a raw PDF page is a two-page spread or a single page.

    Two signals are combined:
    1. Aspect ratio — spreads are roughly twice as wide as tall (w/h > threshold).
    2. Gutter valley — a bound-book spread has a vertical low-ink band near the
       horizontal midpoint.  The relative ratio (min/mean) tolerates binding
       shadow and near-spine text.

    Returns Layout.SPREAD or Layout.SINGLE (never Layout.AUTO).
    """
    w, h  = img.size
    ratio = w / h

    if ratio < config.auto_aspect_threshold:
        log.debug(f"    AUTO: ratio={ratio:.2f} < {config.auto_aspect_threshold} → SINGLE")
        return Layout.SINGLE

    small_w = min(w, 1200)
    small_h = int(h * small_w / w)
    small   = np.array(img.convert("L").resize((small_w, small_h), Image.LANCZOS))

    _, binary = cv2.threshold(small, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)

    cx        = small_w // 2
    half_win  = int(small_w * config.auto_gutter_search_frac / 2)
    col_inks  = binary[:, max(0, cx - half_win):min(small_w, cx + half_win)].sum(axis=0)
    col_inks  = col_inks / (255.0 * small_h)
    mean_ink  = float(col_inks.mean())
    min_ink   = float(col_inks.min()) if col_inks.size > 0 else 1.0
    gutter_ratio = (min_ink / mean_ink) if mean_ink > 1e-6 else 1.0

    result = Layout.SPREAD if gutter_ratio < config.auto_gutter_rel_thresh else Layout.SINGLE
    log.debug(
        f"    SCAN-AUTO: ratio={ratio:.2f}, gutter_ratio={gutter_ratio:.3f} → {result.name}"
    )
    return result


def split_spread(img: Image.Image) -> tuple[Image.Image, Image.Image]:
    """Split a two-page spread at the horizontal midpoint → (left, right)."""
    w, h = img.size
    mid  = w // 2
    return img.crop((0, 0, mid, h)), img.crop((mid, 0, w, h))


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
        left, right = split_spread(img)
        return [("L", left), ("R", right)]
    return [("", img)]


def is_blank(img_gray: np.ndarray, threshold: float) -> bool:
    """
    Return True if the page is effectively blank (cover, endpaper, etc.).
    Uses Otsu to locate ink pixels; blank when dark-pixel fraction < threshold.
    """
    _, binary = cv2.threshold(img_gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    return (binary.sum() / (255.0 * binary.size)) < threshold


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
    Correct skew.  Returns (corrected_image, angle_applied).
    scipy ndimage rotate preserves dimensions and fills the border with white.
    """
    angle = estimate_skew(np.array(img_pil.convert("L")), max_angle)
    if abs(angle) < 0.1:
        return img_pil, 0.0
    log.debug(f"    Deskewing by {angle:.2f}°")
    rotated = ndimage_rotate(np.array(img_pil), -angle, reshape=False, cval=255, order=1)
    return Image.fromarray(rotated.astype(np.uint8)), angle

def detect_text_block(
    img_gray: np.ndarray, margin_frac: float
) -> tuple[int, int, int, int]:
    """
    Find the bounding box of the main text block via projection profiles,
    masking a hard margin on each side to suppress marginal annotations.

    Returns (x1, y1, x2, y2).
    """
    h, w = img_gray.shape
    _, binary = cv2.threshold(img_gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)

    hard_l = int(w * margin_frac)
    hard_r = int(w * (1 - margin_frac))

    col_norm = binary.sum(axis=0) / (255.0 * h + 1e-6)
    ink_cols = np.where(col_norm[hard_l:hard_r] > 0.005)[0]
    x1 = hard_l + (max(0,              ink_cols[0]  - 20) if len(ink_cols) else 0)
    x2 = hard_l + (min(hard_r - hard_l, ink_cols[-1] + 20) if len(ink_cols) else hard_r - hard_l)

    row_norm = binary.sum(axis=1) / (255.0 * w + 1e-6)
    ink_rows = np.where(row_norm > 0.003)[0]
    y1 = max(0,     ink_rows[0]  - 30) if len(ink_rows) else 0
    y2 = min(h - 1, ink_rows[-1] + 30) if len(ink_rows) else h

    return int(x1), int(y1), int(x2), int(y2)


def sauvola_binarise(img_gray: np.ndarray, block_size: int, k: float) -> np.ndarray:
    """
    Sauvola adaptive thresholding.

    Handles uneven illumination, show-through, and foxing on aged paper.
    Returns uint8 binary (0 = background, 255 = ink).

    NOTE: The result is used only for column detection and saved as a QA
    artefact.  GOT-OCR-2.0 receives the colour crop, not this image.
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


def detect_columns(
    binary: np.ndarray, config: PreprocessConfig
) -> tuple[ColLayout, int | None]:
    """
    Detect whether a (already cropped) page carries one or two text columns.

    Method — vertical projection profile in the central band:
    1. Compute the per-column ink density (fraction of dark pixels per column).
    2. Restrict the search to the central `col_search_band` of the page width
       to avoid false positives from a centred page number or a wide initial.
    3. Find the column with the minimum ink density within that band.
    4. If  min_ink / mean_ink  <  col_gap_rel_thresh  the valley is deep
       enough to indicate a column gutter → TWO_COL.
    5. Also require that each half is ≥ col_min_width_frac of total width,
       ruling out near-edge splits caused by a large decorated initial or a
       centred headline.

    Returns (ColLayout.ONE_COL | TWO_COL, split_x_or_None).
    split_x is given in *page coordinates* (origin = left edge of binary).
    """
    h, w = binary.shape

    band_half  = int(w * config.col_search_band / 2)
    cx         = w // 2
    b_start    = max(0, cx - band_half)
    b_end      = min(w, cx + band_half)

    col_inks   = binary[:, b_start:b_end].sum(axis=0) / (255.0 * h + 1e-6)
    mean_ink   = float(col_inks.mean())
    if mean_ink < 1e-4:          # completely empty strip — treat as one column
        return ColLayout.ONE_COL, None

    valley_idx  = int(col_inks.argmin())
    min_ink     = float(col_inks[valley_idx])
    gap_ratio   = min_ink / mean_ink

    split_x     = b_start + valley_idx      # in page coordinates
    min_col_w   = int(w * config.col_min_width_frac)

    if (
        gap_ratio < config.col_gap_rel_thresh
        and split_x >= min_col_w
        and (w - split_x) >= min_col_w
    ):
        log.debug(
            f"    COL-AUTO: gap_ratio={gap_ratio:.3f} at x={split_x} → TWO_COL"
        )
        return ColLayout.TWO_COL, split_x

    log.debug(f"    COL-AUTO: gap_ratio={gap_ratio:.3f} → ONE_COL")
    return ColLayout.ONE_COL, None


def split_columns(
    img_colour: Image.Image,
    split_x: int,
    padding: int,
) -> tuple[Image.Image, Image.Image]:
    """
    Split a colour page crop into left and right column images.

    A small amount of horizontal padding is added to each crop so that
    ascenders/descenders at the column edges are not accidentally clipped.
    The padding is white-filled where it would fall outside the image.
    """
    w, h  = img_colour.size
    lx2   = min(w, split_x + padding)
    rx1   = max(0, split_x - padding)
    return img_colour.crop((0, 0, lx2, h)), img_colour.crop((rx1, 0, w, h))

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
    4. Binarise         → Sauvola (used for column detection and QA only).
    5. Column detection → ONE_COL or TWO_COL.
    6. Output colour crops per column as lossless PNG for GOT-OCR-2.0.
    7. Write metadata.json.

    Output files
    ------------
    <page_id>_col1.png   – colour crop fed to GOT-OCR-2.0 (column 1 or full)
    <page_id>_col2.png   – colour crop fed to GOT-OCR-2.0 (column 2, if any)
    <page_id>_binary.png – binarised QA image (not fed to the model)
    metadata.json
    """
    out_dir.mkdir(parents=True, exist_ok=True)
    meta: dict = {
        "page_id":      page_id,
        "blank":        False,
        "deskew_angle": 0.0,
        "col_layout":   None,
        "regions":      [],
    }

    img_gray = np.array(img_pil.convert("L"))

    if is_blank(img_gray, config.blank_threshold):
        log.info(f"  [{page_id}] BLANK – skipping")
        meta["blank"] = True
        (out_dir / "metadata.json").write_text(json.dumps(meta, indent=2))
        return meta
    img_deskewed, angle = deskew(img_pil, config.deskew_max_angle)
    meta["deskew_angle"] = round(angle, 3)
    img_gray_desk = np.array(img_deskewed.convert("L"))

    x1, y1, x2, y2   = detect_text_block(img_gray_desk, config.margin_frac)
    meta["text_block_bbox"] = [x1, y1, x2, y2]

    img_cropped  = img_deskewed.crop((x1, y1, x2, y2))
    gray_cropped = np.array(img_cropped.convert("L"))
    binary = sauvola_binarise(gray_cropped, config.binarise_block, config.sauvola_k)
    if config.save_binary:
        Image.fromarray(binary).save(out_dir / f"{page_id}_binary.png")
    if config.col_layout is ColLayout.COL_AUTO:
        resolved_col, split_x = detect_columns(binary, config)
    elif config.col_layout is ColLayout.TWO_COL:
        # Force two-column: find the best split point within the search band
        _, split_x   = detect_columns(binary, config)
        resolved_col = ColLayout.TWO_COL
        if split_x is None:          # detection failed — fall back to midpoint
            split_x = img_cropped.size[0] // 2
    else:
        resolved_col = ColLayout.ONE_COL
        split_x      = None

    meta["col_layout"] = resolved_col.name
    if resolved_col is ColLayout.TWO_COL and split_x is not None:
        col1, col2 = split_columns(img_cropped, split_x, config.col_padding)

        col1_path = out_dir / f"{page_id}_col1.png"
        col2_path = out_dir / f"{page_id}_col2.png"
        col1.save(col1_path, "PNG")
        col2.save(col2_path, "PNG")

        w_full = img_cropped.size[0]
        meta["regions"] = [
            {
                "idx":    0,
                "label":  "col1",
                "bbox":   [0, 0, min(w_full, split_x + config.col_padding),
                           img_cropped.size[1]],
                "file":   f"{page_id}_col1.png",
            },
            {
                "idx":    1,
                "label":  "col2",
                "bbox":   [max(0, split_x - config.col_padding), 0,
                           w_full, img_cropped.size[1]],
                "file":   f"{page_id}_col2.png",
            },
        ]
        log.info(f"  [{page_id}] TWO_COL – split at x={split_x}")

    else:
        col1_path = out_dir / f"{page_id}_col1.png"
        img_cropped.save(col1_path, "PNG")
        meta["regions"] = [
            {
                "idx":   0,
                "label": "col1",
                "bbox":  [0, 0, img_cropped.size[0], img_cropped.size[1]],
                "file":  f"{page_id}_col1.png",
            }
        ]
        log.info(f"  [{page_id}] ONE_COL")

    (out_dir / "metadata.json").write_text(json.dumps(meta, indent=2))
    return meta


def run(config: PreprocessConfig = PreprocessConfig()) -> list[dict]:
    """
    Process an entire PDF using settings from a PreprocessConfig.

    Usage examples
    --------------
    # Printed book scanned as two-page spreads, auto-detect columns:
    run(PreprocessConfig(
        pdf_path    = "Instruccion_Buendia.pdf",
        output_root = "./output",
        dpi         = 300,
        layout      = Layout.SPREAD,
        col_layout  = ColLayout.COL_AUTO,
    ))

    # Bilingual (Spanish/Latin) edition with known two-column layout:
    run(PreprocessConfig(
        pdf_path    = "Fuero_Real_1781.pdf",
        output_root = "./output",
        dpi         = 300,
        layout      = Layout.SPREAD,
        col_layout  = ColLayout.TWO_COL,
    ))

    # Individual page scans, single column (e.g. archival petitions):
    run(PreprocessConfig(
        pdf_path    = "Peticion_1622.pdf",
        output_root = "./output",
        dpi         = 300,
        layout      = Layout.SINGLE,
        col_layout  = ColLayout.ONE_COL,
    ))
    """
    pdf_path    = Path(config.pdf_path)
    output_root = Path(config.output_root)
    source_name = pdf_path.stem
    output_root.mkdir(parents=True, exist_ok=True)

    log.info(
        f"=== {pdf_path.name} | DPI={config.dpi} "
        f"| layout={config.layout.name} "
        f"| col_layout={config.col_layout.name} ==="
    )

    pages    = pdf_to_images(str(pdf_path), config.dpi)
    all_meta = []
    tally    = {
        Layout.SPREAD.name: 0,
        Layout.SINGLE.name: 0,
        ColLayout.ONE_COL.name: 0,
        ColLayout.TWO_COL.name: 0,
    }

    for pdf_idx, page_img in enumerate(pages):
        pdf_num    = pdf_idx + 1
        book_pages = get_book_pages(page_img, config)
        scan_resolved = Layout.SPREAD if len(book_pages) == 2 else Layout.SINGLE
        tally[scan_resolved.name] += 1
        log.info(f"PDF page {pdf_num}/{len(pages)} [{scan_resolved.name}]")

        for label, sub_img in book_pages:
            page_id  = f"p{pdf_num:03d}{label}" if label else f"p{pdf_num:03d}"
            page_dir = output_root / source_name / page_id
            meta     = process_page(sub_img, page_dir, page_id, config)
            meta["scan_layout"] = scan_resolved.name
            all_meta.append(meta)

            col_str = meta.get("col_layout", "BLANK")
            n_reg   = len(meta.get("regions", []))
            tally.get(col_str, None)  # may be None for blank pages
            if col_str in tally:
                tally[col_str] += 1
            log.info(f"  → {page_id}: {col_str}, {n_reg} region(s)")

    summary = {
        "source":        source_name,
        "dpi":           config.dpi,
        "layout_config": config.layout.name,
        "col_layout_config": config.col_layout.name,
        "tally":         tally,
        "pdf_pages":     len(pages),
        "book_pages":    len(all_meta),
        "total_regions": sum(len(m.get("regions", [])) for m in all_meta),
        "blank_pages":   sum(1 for m in all_meta if m.get("blank")),
    }
    (output_root / source_name / "summary.json").write_text(
        json.dumps(summary, indent=2)
    )
    log.info(
        f"Done – {summary['book_pages']} book pages, "
        f"{summary['total_regions']} regions saved"
    )
    return all_meta


if __name__ == "__main__":
    run()