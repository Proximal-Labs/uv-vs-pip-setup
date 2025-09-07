# audiobook.py
from pathlib import Path
from gtts import gTTS
from PyPDF2 import PdfReader  # new API

PDF_PATH = Path("name.pdf")
OUT_MP3 = Path("audiobook.mp3")

if not PDF_PATH.exists():
    raise FileNotFoundError(f"{PDF_PATH} not found")

reader = PdfReader(PDF_PATH.open("rb"))
text_parts = []
for page in reader.pages:
    txt = page.extract_text() or ""
    if txt.strip():
        text_parts.append(txt)

full_text = "\n".join(text_parts).strip()
if not full_text:
    raise ValueError("No extractable text found in PDF")

tts = gTTS(full_text, lang="en")
tts.save(str(OUT_MP3))
print(f"Wrote {OUT_MP3.resolve()}")
