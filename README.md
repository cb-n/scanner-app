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

### 1. Tested PC Setup
- Windows 11
- Epson ES-50 (WIA-compatible scanner)

### 2. NAPS2
Install NAPS2 (required for scanning):

https://www.naps2.com

Verify NAPS2 sees your scanner:
```
& "C:\Program Files\NAPS2\NAPS2.Console.exe" --listdevices --driver wia
```

You should see `EPSON ES-50`. If not, install the Epson driver too.

### 3. Python setup


#### Create Python 3.11 venv

```bash
# Similar on Mac/Linux; this is Windows Powershell
py -3.11 -m venv .venv

# Activate virtual environment
Set-ExecutionPolicy -Scope CurrentUser -ExecutionPolicy RemoteSigned
.\.venv\Scripts\Activate.ps1

# Should print 3.11.x
python --version
```

#### Install dependencies

```bash
# Force install the GPU (CUDA) version of PyTorch (nightly for Blackwell support)
python -m pip install --upgrade pip
pip uninstall -y torch torchvision torchaudio
pip cache purge
pip install --no-cache-dir --force-reinstall torch torchvision torchaudio --pre --index-url https://download.pytorch.org/whl/nightly/cu128
python -c "import torch; print(torch.__version__); print(torch.cuda.is_available()); print(torch.version.cuda); print(torch.cuda.get_device_name(0) if torch.cuda.is_available() else None)"

# Install other dependencies...
pip install "numpy<2" pillow reportlab easyocr transformers sentencepiece accelerate

# Verify GPU enabled:
python -c "import torch; print(torch.cuda.is_available(), torch.cuda.get_device_name(0) if torch.cuda.is_available() else None)"
```

---

## Usage

### Normal Mode
```
python scanner.py
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
python scanner.py --fast
```

- Scans only
- Saves TIFFs
- Produces .unproc.pdf
- No OCR, handwriting, or orientation

### PROC Mode
```
python scanner.py --proc
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
