# Scanner App (Epson ES-50, Windows, GPU OCR)

This is a Windows command-line scanning workflow optimized for bulk document scanning (“vibe-scanning”).

It uses:
- NAPS2 + Windows WIA for scanning (no Epson Scan UI popups)
- EasyOCR (GPU) for printed text
- TrOCR (GPU) for handwriting recognition
- Automatic orientation detection
- A fast/manual workflow designed for scanning hundreds of pages comfortably

Each document is saved in its own folder under `C:\Scans`.

---

## Folder Layout

Each document becomes its own folder:

C:\Scans\
  scan-20260118-142626\
    scan-20260118-142626.pdf
    scan-20260118-142626.ocr.txt
    scan-20260118-142626.manu.txt

In FAST mode, the folder instead contains:

scan-20260118-142626\
  scan-20260118-142626.unproc.pdf
  scan-20260118-142626-001.tif
  scan-20260118-142626-002.tif

Those TIFFs are later consumed by `--proc`.

---

## Requirements

### 1. Windows
- Windows 10 or Windows 11
- Epson ES-50 (or similar WIA-compatible scanner)

### 2. NAPS2
Install NAPS2 (required for scanning):

https://www.naps2.com

Verify NAPS2 sees your scanner:
```
& "C:\Program Files\NAPS2\NAPS2.Console.exe" --listdevices --driver wia
```

You should see `EPSON ES-50`.

### 3. Python
Install Python 3.11 (64-bit):
https://www.python.org/downloads/windows/

During install:
- Check “Add Python to PATH”

Verify:
```
python --version
```

### 4. Virtual Environment
```
python -m venv .venv
.\.venv\Scripts\Activate.ps1
```
If blocked:
```
Set-ExecutionPolicy -Scope CurrentUser -ExecutionPolicy RemoteSigned
```

### 5. Python Dependencies
```
pip install numpy pillow reportlab easyocr transformers sentencepiece accelerate
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121
```

Verify GPU:
```
python - <<EOF
import torch
print(torch.cuda.is_available())
EOF
```

---

## Usage

### Normal Mode
```
python scan_ocr_hw.py
```

ENTER = scan/add  
double-ENTER = finalize document  
q = quit  

Outputs:
- .pdf
- .ocr.txt
- .manu.txt

### FAST Mode
```
python scan_ocr_hw.py --fast
```

- Scans only
- Saves TIFFs
- Produces .unproc.pdf
- No OCR, handwriting, or orientation

### PROC Mode
```
python scan_ocr_hw.py --proc
```

- Processes all folders containing .unproc.pdf
- Runs orientation + OCR + handwriting
- Deletes .unproc.pdf and TIFFs
- Writes final .pdf, .ocr.txt, .manu.txt

---

## Notes
- Uses WIA (not TWAIN) to avoid blocking dialogs
- OCR always runs on original TIFFs
- Designed for large batch scanning
