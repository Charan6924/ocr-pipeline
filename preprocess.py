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
Image.MAX_IMAGE_PIXELS = 5_000_000_000

class Layout(Enum):
    AUTO   = auto()
    SPREAD = auto()
    SINGLE = auto()

@dataclass
class PreprocessConfig:
    pdf_path:str = "/mnt/vstor/courses/csds312/cxv166/OCR/Print 2/Covarrubias - Tesoro lengua.pdf"
    output_root:str = "./output"
    dpi:int = 300
    layout: Layout = Layout.AUTO
    auto_aspect_threshold:float = 1.4   
    auto_gutter_search_frac:float = 0.40 
    auto_gutter_rel_thresh:float = 0.25  
    margin_frac: float = 0.08    
    blank_threshold:float = 0.02  
    blank_bottom_mask:float = 0.15 
    deskew_max_angle: float = 5.0  

def pdf_to_images(pdf_path: str, dpi: int) -> list[Image.Image]:
    log.info(f"Converting PDF to images at {dpi} DPI: {pdf_path}")
    pages = convert_from_path(pdf_path, dpi=dpi)
    log.info(f" {len(pages)} PDF pages, first page size = {pages[0].size}")
    return pages

def find_gutter_x(img: Image.Image, config: PreprocessConfig) -> int:
    w, h = img.size
    small_w = min(w, 1200)
    small_h = int(h * small_w / w)
    scale = small_w / w                         
    small = np.array(img.convert("L").resize((small_w, small_h), Image.LANCZOS))
    _, binary = cv2.threshold(small, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)

    cx = small_w // 2
    half_win = int(small_w * config.auto_gutter_search_frac / 2)
    b_start = max(0,       cx - half_win)
    b_end = min(small_w, cx + half_win)

    col_inks = binary[:, b_start:b_end].sum(axis=0) / (255.0 * small_h + 1e-6)
    valley = int(col_inks.argmin())
    gutter_small = b_start + valley
    gutter_orig = int(gutter_small / scale)

    mean_ink = float(col_inks.mean())
    min_ink = float(col_inks[valley])
    ratio = (min_ink / mean_ink) if mean_ink > 1e-6 else 1.0

    log.debug(
        f"gutter: valley_x={gutter_orig} (small={gutter_small}), "
        f"ink_ratio={ratio:.3f}"
    )

    if ratio >= config.auto_gutter_rel_thresh:
        log.debug("gutter: no clear valley — using midpoint")
        return w // 2

    return gutter_orig


def detect_layout(img: Image.Image, config: PreprocessConfig) -> Layout:
    w, h  = img.size
    ratio = w / h

    if ratio < config.auto_aspect_threshold:
        log.debug(f"    AUTO: ratio={ratio:.2f} → SINGLE")
        return Layout.SINGLE
    gutter_x = find_gutter_x(img, config)
    small_w = min(w, 1200)
    small_h = int(h * small_w / w)
    scale = small_w / w
    small = np.array(img.convert("L").resize((small_w, small_h), Image.LANCZOS))
    _, binary = cv2.threshold(small, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    cx = small_w // 2
    half_win = int(small_w * config.auto_gutter_search_frac / 2)
    col_inks = binary[:, max(0, cx - half_win):min(small_w, cx + half_win)].sum(axis=0)
    col_inks = col_inks / (255.0 * small_h + 1e-6)
    mean_ink = float(col_inks.mean())
    min_ink = float(col_inks.min()) if col_inks.size > 0 else 1.0
    gutter_ratio = (min_ink / mean_ink) if mean_ink > 1e-6 else 1.0

    result = Layout.SPREAD if gutter_ratio < config.auto_gutter_rel_thresh else Layout.SINGLE
    log.debug(
        f"AUTO: ratio={ratio:.2f}, gutter_ratio={gutter_ratio:.3f} → {result.name}"
    )
    return result


def split_spread(img: Image.Image, config: PreprocessConfig) -> tuple[Image.Image, Image.Image]:
    w, h = img.size
    gutter_x = find_gutter_x(img, config)
    log.debug(f"split_spread: gutter_x={gutter_x} (midpoint would be {w // 2})")
    return img.crop((0, 0, gutter_x, h)), img.crop((gutter_x, 0, w, h))


def get_book_pages(img: Image.Image, config: PreprocessConfig) -> list[tuple[str, Image.Image]]:
    resolved = config.layout
    if resolved is Layout.AUTO:
        resolved = detect_layout(img, config)

    if resolved is Layout.SPREAD:
        left, right = split_spread(img, config)
        return [("L", left), ("R", right)]
    return [("", img)]

def is_blank(img_gray: np.ndarray, config: PreprocessConfig) -> bool:
    h = img_gray.shape[0]
    trimmed = img_gray[:int(h * (1.0 - config.blank_bottom_mask)), :]
    _, binary = cv2.threshold(trimmed, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    return (binary.sum() / (255.0 * binary.size)) < config.blank_threshold

def estimate_skew(img_gray: np.ndarray, max_angle: float) -> float:
    small = cv2.resize(img_gray, (img_gray.shape[1] // 2, img_gray.shape[0] // 2))
    edges = cv2.Canny(small, 50, 150, apertureSize=3)
    lines = cv2.HoughLinesP(
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
    angle = estimate_skew(np.array(img_pil.convert("L")), max_angle)
    if abs(angle) < 0.1:
        return img_pil, 0.0
    log.debug(f"    Deskewing by {angle:.2f}°")
    rotated = ndimage_rotate(np.array(img_pil), -angle, reshape=False, cval=255, order=1)
    return Image.fromarray(rotated.astype(np.uint8)), angle

def detect_text_block(img_gray: np.ndarray, margin_frac: float) -> tuple[int, int, int, int]:
    h, w = img_gray.shape
    _, binary = cv2.threshold(img_gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)

    hard_l = int(w * margin_frac)

    col_norm = binary.sum(axis=0) / (255.0 * h + 1e-6)
    ink_cols_inner = np.where(col_norm[hard_l:] > 0.005)[0]
    x1 = hard_l + (max(0, ink_cols_inner[0] - 20) if len(ink_cols_inner) else 0)

    ink_cols_full = np.where(col_norm > 0.005)[0]
    x2 = min(w, (ink_cols_full[-1] + 20) if len(ink_cols_full) else w)

    row_norm = binary.sum(axis=1) / (255.0 * w + 1e-6)
    ink_rows = np.where(row_norm > 0.003)[0]
    y1 = max(0,     ink_rows[0]  - 30) if len(ink_rows) else 0
    y2 = min(h - 1, ink_rows[-1] + 30) if len(ink_rows) else h

    return int(x1), int(y1), int(x2), int(y2)


def process_page(img_pil:  Image.Image,out_dir:  Path,page_id:  str,config:   PreprocessConfig) -> dict:
    out_dir.mkdir(parents=True, exist_ok=True)
    meta: dict = {
        "page_id": page_id,
        "blank": False,
        "deskew_angle":0.0,
        "text_block_bbox": None,
    }

    img_gray = np.array(img_pil.convert("L"))
    if is_blank(img_gray, config):
        log.info(f"  [{page_id}] BLANK – skipping")
        meta["blank"] = True
        (out_dir / "metadata.json").write_text(json.dumps(meta, indent=2))
        return meta

    img_deskewed, angle = deskew(img_pil, config.deskew_max_angle)
    meta["deskew_angle"] = round(angle, 3)
    img_gray_desk = np.array(img_deskewed.convert("L"))
    x1, y1, x2, y2 = detect_text_block(img_gray_desk, config.margin_frac)
    meta["text_block_bbox"] = [x1, y1, x2, y2]
    img_cropped = img_deskewed.crop((x1, y1, x2, y2))
    img_cropped.save(out_dir / f"{page_id}.png", "PNG")
    log.info(f"  [{page_id}] Saved colour crop")

    (out_dir / "metadata.json").write_text(json.dumps(meta, indent=2))
    return meta

def run(config: PreprocessConfig = PreprocessConfig()) -> list[dict]:
    pdf_path = Path(config.pdf_path)
    output_root = Path(config.output_root)
    source_name = pdf_path.stem
    output_root.mkdir(parents=True, exist_ok=True)

    log.info(f"=== {pdf_path.name} | DPI={config.dpi} | layout={config.layout.name} ===")
    pages = pdf_to_images(str(pdf_path), config.dpi)
    all_meta = []
    tally = {Layout.SPREAD.name: 0, Layout.SINGLE.name: 0}

    for pdf_idx, page_img in enumerate(pages):
        pdf_num = pdf_idx + 1
        book_pages = get_book_pages(page_img, config)
        scan_resolved = Layout.SPREAD if len(book_pages) == 2 else Layout.SINGLE
        tally[scan_resolved.name] += 1
        log.info(f"PDF page {pdf_num}/{len(pages)} [{scan_resolved.name}]")

        for label, sub_img in book_pages:
            page_id = f"p{pdf_num:03d}{label}" if label else f"p{pdf_num:03d}"
            page_dir = output_root / source_name / page_id
            meta = process_page(sub_img, page_dir, page_id, config)
            meta["layout"] = scan_resolved.name
            all_meta.append(meta)
            log.info(f"{page_id}: {'BLANK' if meta['blank'] else 'OK'}")

    summary = {
        "source": source_name,
        "dpi": config.dpi,
        "layout_config":config.layout.name,
        "layout_tally": tally,
        "pdf_pages": len(pages),
        "book_pages": len(all_meta),
        "blank_pages": sum(1 for m in all_meta if m.get("blank")),
    }
    (output_root / source_name / "summary.json").write_text(
        json.dumps(summary, indent=2)
    )
    log.info(f"Done! {summary['book_pages']} book pages processed")
    return all_meta


if __name__ == "__main__":
    run()