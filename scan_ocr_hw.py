import os
import re
import subprocess
import time
import argparse
from datetime import datetime
from typing import List, Tuple

import msvcrt

import numpy as np
from PIL import Image

import easyocr
import torch
from transformers import TrOCRProcessor, VisionEncoderDecoderModel

from reportlab.pdfgen import canvas
from reportlab.lib.utils import ImageReader


NAPS2_CONSOLE = r"C:\Program Files\NAPS2\NAPS2.Console.exe"
DEVICE = "EPSON ES-50"

OUT_ROOT = r"C:\Scans"
TMP_DIR = os.path.join(OUT_ROOT, "_tmp_scans")  # still used for DEFAULT mode only

SCAN_DPI = 300
SCAN_BITDEPTH = "color"

EASYOCR_LANGS = ["en"]
TROCR_MODEL_NAME = "microsoft/trocr-base-handwritten"

BOX_PAD = 8
DOUBLE_ENTER_WINDOW = 0.60

FEEDER_EMPTY_PATTERNS = [
    r"no pages are in the feeder",
    r"no scanned pages to export",
]

RETRY_ON_EMPTY_FEEDER = True
EMPTY_FEEDER_RETRY_DELAY_SEC = 0.75
EMPTY_FEEDER_MAX_RETRIES = 0  # 0 = infinite

ORIENTATION_ANGLES = [0, 90, 180, 270]


def log(msg: str) -> None:
    ts = datetime.now().strftime("%H:%M:%S")
    print(f"[{ts}] {msg}", flush=True)


def please_wait(what: str) -> None:
    log(f"Please wait... {what}")


def ensure_dirs() -> None:
    os.makedirs(OUT_ROOT, exist_ok=True)
    os.makedirs(TMP_DIR, exist_ok=True)


def drain_kbd() -> None:
    while msvcrt.kbhit():
        _ = msvcrt.getwch()


def wait_command() -> str:
    print()
    log("ENTER = scan/add | double-ENTER = finalize group | q = quit")
    drain_kbd()

    ch = msvcrt.getwch()
    if ch.lower() == "q":
        return "quit"

    if ch in ("\r", "\n"):
        deadline = time.time() + DOUBLE_ENTER_WINDOW
        while time.time() < deadline:
            if msvcrt.kbhit():
                ch2 = msvcrt.getwch()
                if ch2 in ("\r", "\n"):
                    return "finish"
        return "scan"

    return "noop"


def new_group_id() -> str:
    return "scan-" + datetime.now().strftime("%Y%m%d-%H%M%S")


def is_feeder_empty(text: str) -> bool:
    t = (text or "").lower()
    return any(re.search(pat, t) for pat in FEEDER_EMPTY_PATTERNS)


def run_naps2_scan(output_path: str) -> Tuple[int, str, str]:
    cmd = [
        NAPS2_CONSOLE,
        "--noprofile",
        "--driver", "wia",
        "--device", DEVICE,
        "--dpi", str(SCAN_DPI),
        "--bitdepth", SCAN_BITDEPTH,
        "--force",
        "-v",
        "-o", output_path,
    ]
    p = subprocess.run(cmd, capture_output=True, text=True)
    return p.returncode, (p.stdout or ""), (p.stderr or "")


def scan_to_tiff_with_retries(output_path: str) -> None:
    attempts = 0
    while True:
        attempts += 1
        please_wait(f"Scanning (attempt {attempts})...")
        rc, out, err = run_naps2_scan(output_path)

        if out.strip():
            log("NAPS2 output:")
            print(out, end="" if out.endswith("\n") else "\n", flush=True)
        if err.strip():
            log("NAPS2 error:")
            print(err, end="" if err.endswith("\n") else "\n", flush=True)

        if rc != 0:
            raise RuntimeError(f"NAPS2 failed with exit code {rc}")

        if os.path.exists(output_path) and os.path.getsize(output_path) > 0:
            log(f"Scan produced file: {output_path}")
            return

        if is_feeder_empty(out) or is_feeder_empty(err):
            log("Feeder is empty. Load pages.")
            if not RETRY_ON_EMPTY_FEEDER:
                raise RuntimeError("Feeder empty; retry disabled.")
            if EMPTY_FEEDER_MAX_RETRIES and attempts >= EMPTY_FEEDER_MAX_RETRIES:
                raise RuntimeError("Feeder empty; max retries reached.")
            time.sleep(EMPTY_FEEDER_RETRY_DELAY_SEC)
            continue

        raise RuntimeError(f"No output TIFF created at: {output_path}")


def count_tiff_pages(tiff_path: str) -> int:
    count = 0
    with Image.open(tiff_path) as im:
        while True:
            try:
                im.seek(count)
                count += 1
            except EOFError:
                break
    return count


def load_tiff_pages_stream(tiff_path: str):
    with Image.open(tiff_path) as im:
        i = 0
        while True:
            try:
                im.seek(i)
            except EOFError:
                break
            yield im.convert("RGB")
            i += 1


def clamp(v: int, lo: int, hi: int) -> int:
    return max(lo, min(hi, v))


def box_to_xyxy(box) -> Tuple[int, int, int, int]:
    xs = [int(p[0]) for p in box]
    ys = [int(p[1]) for p in box]
    return min(xs), min(ys), max(xs), max(ys)


def crop_with_pad(img: Image.Image, xyxy: Tuple[int, int, int, int], pad: int) -> Image.Image:
    w, h = img.size
    x1, y1, x2, y2 = xyxy
    x1 = clamp(x1 - pad, 0, w)
    y1 = clamp(y1 - pad, 0, h)
    x2 = clamp(x2 + pad, 0, w)
    y2 = clamp(y2 + pad, 0, h)
    return img.crop((x1, y1, x2, y2))


def rotate_img(img: Image.Image, angle: int) -> Image.Image:
    if angle == 0:
        return img
    return img.rotate(angle, expand=True)


def orientation_score(detections) -> float:
    total_chars = 0
    total_conf = 0.0
    for _, text, conf in detections:
        t = (text or "").strip()
        if not t:
            continue
        total_chars += len(t)
        total_conf += float(conf)
    if total_chars == 0:
        return 0.0
    return total_conf * total_chars


def auto_orient_page(easy_reader: easyocr.Reader, img: Image.Image) -> Tuple[Image.Image, int]:
    best_angle = 0
    best_score = -1.0
    best_img = img

    for angle in ORIENTATION_ANGLES:
        test_img = rotate_img(img, angle)
        det = easy_reader.readtext(np.array(test_img), detail=1, paragraph=False)
        score = orientation_score(det)
        log(f"Orientation test angle={angle} score={score:.2f}")
        if score > best_score:
            best_score = score
            best_angle = angle
            best_img = test_img

    if best_score <= 0.0:
        log("Orientation detection inconclusive; keeping original orientation.")
        return img, 0

    log(f"Selected rotation: {best_angle} degrees")
    return best_img, best_angle


@torch.inference_mode()
def trocr_image_to_text(processor: TrOCRProcessor,
                        model: VisionEncoderDecoderModel,
                        device: str,
                        img: Image.Image) -> str:
    if img.mode != "RGB":
        img = img.convert("RGB")
    pixel_values = processor(images=img, return_tensors="pt").pixel_values.to(device)
    generated_ids = model.generate(pixel_values, max_new_tokens=128)
    text = processor.batch_decode(generated_ids, skip_special_tokens=True)[0]
    return text.strip()


def process_page(easy_reader: easyocr.Reader,
                 trocr_processor: TrOCRProcessor,
                 trocr_model: VisionEncoderDecoderModel,
                 trocr_device: str,
                 img: Image.Image) -> Tuple[List[str], List[str]]:
    printed: List[str] = []
    manu: List[str] = []

    detections = easy_reader.readtext(np.array(img), detail=1, paragraph=False)

    for box, text, conf in detections:
        t = (text or "").strip()
        if t:
            printed.append(t)

        crop = crop_with_pad(img, box_to_xyxy(box), BOX_PAD)
        hw = trocr_image_to_text(trocr_processor, trocr_model, trocr_device, crop)
        if hw:
            manu.append(hw)

    return printed, manu


def pdf_from_tiffs(tiff_paths: List[str], out_pdf_path: str) -> int:
    c = canvas.Canvas(out_pdf_path)
    page_num = 0
    try:
        for tpath in tiff_paths:
            log(f"Building PDF from TIFF: {tpath}")
            for img in load_tiff_pages_stream(tpath):
                page_num += 1
                w_px, h_px = img.size
                w_pt = (w_px / SCAN_DPI) * 72.0
                h_pt = (h_px / SCAN_DPI) * 72.0
                c.setPageSize((w_pt, h_pt))
                c.drawImage(ImageReader(img), 0, 0, width=w_pt, height=h_pt)
                c.showPage()
        please_wait("Writing PDF...")
        c.save()
        return page_num
    except Exception:
        try:
            c.save()
        except Exception:
            pass
        raise


def write_group_outputs(group_id: str,
                        tiff_paths: List[str],
                        easy_reader: easyocr.Reader,
                        trocr_processor: TrOCRProcessor,
                        trocr_model: VisionEncoderDecoderModel,
                        trocr_device: str) -> Tuple[str, str, str]:
    doc_dir = os.path.join(OUT_ROOT, group_id)
    os.makedirs(doc_dir, exist_ok=True)

    base = os.path.join(doc_dir, group_id)
    pdf_path = base + ".pdf"
    ocr_path = base + ".ocr.txt"
    manu_path = base + ".manu.txt"

    please_wait(f"Finalizing '{group_id}' (PDF + OCR + handwriting)...")

    ocr_f = open(ocr_path, "w", encoding="utf-8")
    manu_f = open(manu_path, "w", encoding="utf-8")
    c = canvas.Canvas(pdf_path)

    page_num = 0

    try:
        for tpath in tiff_paths:
            log(f"Reading TIFF: {tpath}")
            for img_raw in load_tiff_pages_stream(tpath):
                page_num += 1
                log(f"Page {page_num}: detecting orientation...")
                img, _angle = auto_orient_page(easy_reader, img_raw)

                log(f"Page {page_num}: processing text...")
                printed_lines, manu_lines = process_page(
                    easy_reader, trocr_processor, trocr_model, trocr_device, img
                )

                log(f"Page {page_num}: adding to PDF...")
                w_px, h_px = img.size
                w_pt = (w_px / SCAN_DPI) * 72.0
                h_pt = (h_px / SCAN_DPI) * 72.0
                c.setPageSize((w_pt, h_pt))
                c.drawImage(ImageReader(img), 0, 0, width=w_pt, height=h_pt)
                c.showPage()

                if page_num > 1:
                    ocr_f.write(f"\n--- PAGE {page_num} ---\n")
                    manu_f.write(f"\n--- PAGE {page_num} ---\n")

                if printed_lines:
                    ocr_f.write("\n".join(printed_lines) + "\n")
                if manu_lines:
                    manu_f.write("\n".join(manu_lines) + "\n")

        please_wait("Writing output files...")
        c.save()
        ocr_f.close()
        manu_f.close()

    except Exception:
        try:
            c.save()
        except Exception:
            pass
        try:
            ocr_f.close()
        except Exception:
            pass
        try:
            manu_f.close()
        except Exception:
            pass
        raise

    log("Finalized successfully.")
    log(f"PDF: {pdf_path}")
    log(f"OCR TXT: {ocr_path}")
    log(f"Handwriting TXT: {manu_path}")

    return pdf_path, ocr_path, manu_path


def delete_files(paths: List[str], label: str) -> None:
    for p in paths:
        try:
            os.remove(p)
            log(f"Deleted {label}: {p}")
        except OSError:
            log(f"Could not delete {label} (ok): {p}")


def tiffs_for_group_in_doc_folder(doc_dir: str, group_id: str) -> List[str]:
    tiffs = []
    for f in os.listdir(doc_dir):
        lf = f.lower()
        if not (lf.endswith(".tif") or lf.endswith(".tiff")):
            continue
        if f.startswith(group_id + "-"):
            tiffs.append(os.path.join(doc_dir, f))
    tiffs.sort()
    return tiffs


def write_fast_outputs(group_id: str, tiff_paths_in_doc_dir: List[str]) -> str:
    doc_dir = os.path.join(OUT_ROOT, group_id)
    os.makedirs(doc_dir, exist_ok=True)
    unproc_pdf = os.path.join(doc_dir, f"{group_id}.unproc.pdf")

    please_wait(f"FAST mode: writing {os.path.basename(unproc_pdf)} (no OCR/handwriting)...")
    pages = pdf_from_tiffs(tiff_paths_in_doc_dir, unproc_pdf)
    log(f"FAST mode complete. Wrote {pages} page(s) to: {unproc_pdf}")
    return unproc_pdf


def process_unproc_folder(doc_dir: str,
                          easy_reader: easyocr.Reader,
                          trocr_processor: TrOCRProcessor,
                          trocr_model: VisionEncoderDecoderModel,
                          trocr_device: str) -> None:
    unproc_files = [f for f in os.listdir(doc_dir) if f.endswith(".unproc.pdf")]
    if not unproc_files:
        return

    for fname in unproc_files:
        unproc_path = os.path.join(doc_dir, fname)
        group_id = fname[:-len(".unproc.pdf")]

        tiffs = tiffs_for_group_in_doc_folder(doc_dir, group_id)
        if not tiffs:
            log(f"PROC: found {unproc_path} but no TIFFs inside the same folder; skipping.")
            continue

        please_wait(f"PROC mode: processing {unproc_path} ...")
        write_group_outputs(group_id, tiffs, easy_reader, trocr_processor, trocr_model, trocr_device)

        delete_files([unproc_path], ".unproc.pdf")
        delete_files(tiffs, "TIFF")


def proc_all_folders() -> None:
    ensure_dirs()

    please_wait("Starting PROC mode...")
    log("Loading OCR models...")

    easy_reader = easyocr.Reader(EASYOCR_LANGS, gpu=True)

    trocr_processor = TrOCRProcessor.from_pretrained(TROCR_MODEL_NAME)
    trocr_model = VisionEncoderDecoderModel.from_pretrained(TROCR_MODEL_NAME)

    trocr_device = "cuda" if torch.cuda.is_available() else "cpu"
    trocr_model.to(trocr_device)
    trocr_model.eval()

    log(f"TrOCR device: {trocr_device}")

    for name in os.listdir(OUT_ROOT):
        doc_dir = os.path.join(OUT_ROOT, name)
        if not os.path.isdir(doc_dir):
            continue
        if os.path.abspath(doc_dir) == os.path.abspath(TMP_DIR):
            continue
        process_unproc_folder(doc_dir, easy_reader, trocr_processor, trocr_model, trocr_device)

    log("PROC mode complete.")


def interactive_scan_loop(fast_mode: bool) -> None:
    ensure_dirs()

    please_wait("Starting up...")
    log("Loading models...")

    easy_reader = None
    trocr_processor = None
    trocr_model = None
    trocr_device = None

    if not fast_mode:
        easy_reader = easyocr.Reader(EASYOCR_LANGS, gpu=True)
        trocr_processor = TrOCRProcessor.from_pretrained(TROCR_MODEL_NAME)
        trocr_model = VisionEncoderDecoderModel.from_pretrained(TROCR_MODEL_NAME)
        trocr_device = "cuda" if torch.cuda.is_available() else "cpu"
        trocr_model.to(trocr_device)
        trocr_model.eval()
        log(f"TrOCR device: {trocr_device}")
    else:
        log("FAST mode enabled: saves TIFFs in the scan folder and writes .unproc.pdf (no OCR now).")

    group_id = new_group_id()
    group_tiffs: List[str] = []
    total_pages = 0

    log(f"Output root: {OUT_ROOT}")
    log(f"Temp folder (default mode only): {TMP_DIR}")
    log(f"Current group: {group_id} (outputs will go to its own folder)")

    while True:
        cmd = wait_command()
        if cmd == "noop":
            continue

        if cmd == "scan":
            try:
                if fast_mode:
                    doc_dir = os.path.join(OUT_ROOT, group_id)
                    os.makedirs(doc_dir, exist_ok=True)
                    tiff_path = os.path.join(doc_dir, f"{group_id}-{len(group_tiffs)+1:03d}.tif")
                else:
                    tiff_path = os.path.join(TMP_DIR, f"{group_id}-{len(group_tiffs)+1:03d}.tif")

                scan_to_tiff_with_retries(tiff_path)
                group_tiffs.append(tiff_path)

                please_wait("Counting pages...")
                pages = count_tiff_pages(tiff_path)
                total_pages += pages
                log(f"Scanned {pages} page(s). Group total: {total_pages} page(s).")

            except Exception as e:
                log(f"Scan failed: {e}")
            continue

        if cmd == "finish":
            if not group_tiffs:
                log("No pages in current group; starting a new group.")
                group_id = new_group_id()
                group_tiffs = []
                total_pages = 0
                log(f"Current group: {group_id}")
                continue

            try:
                if fast_mode:
                    write_fast_outputs(group_id, group_tiffs)
                    # Keep TIFFs in doc folder for --proc
                else:
                    write_group_outputs(group_id, group_tiffs, easy_reader, trocr_processor, trocr_model, trocr_device)
                    delete_files(group_tiffs, "TIFF")

                group_id = new_group_id()
                group_tiffs = []
                total_pages = 0
                log(f"Started new group: {group_id} (new folder)")

            except Exception as e:
                log(f"Finalize failed: {e}")
                if not fast_mode:
                    log(f"Leaving temp TIFFs in place: {TMP_DIR}")
            continue

        if cmd == "quit":
            if group_tiffs:
                log("Active group has pages.")
                log("Press ENTER to finalize it now, or type q to quit without saving it.")
                drain_kbd()
                ch = msvcrt.getwch()
                if ch.lower() == "q":
                    log("Quitting without finalizing current group.")
                    break
                if ch in ("\r", "\n"):
                    try:
                        if fast_mode:
                            write_fast_outputs(group_id, group_tiffs)
                        else:
                            write_group_outputs(group_id, group_tiffs, easy_reader, trocr_processor, trocr_model, trocr_device)
                            delete_files(group_tiffs, "TIFF")
                        log("Finalized and quitting.")
                    except Exception as e:
                        log(f"Finalize failed: {e}")
                        if not fast_mode:
                            log(f"Temp TIFFs kept in: {TMP_DIR}")
                    break
            else:
                log("Quitting.")
                break


def parse_args():
    ap = argparse.ArgumentParser(add_help=True)
    ap.add_argument("--fast", action="store_true")
    ap.add_argument("--proc", action="store_true")
    return ap.parse_args()


def main() -> None:
    args = parse_args()

    if args.proc and args.fast:
        raise SystemExit("Use only one mode: --fast OR --proc")

    if args.proc:
        proc_all_folders()
        return

    interactive_scan_loop(fast_mode=args.fast)


if __name__ == "__main__":
    main()
