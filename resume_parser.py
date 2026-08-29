"""Extract plain text from a resume file (PDF, Word .docx, .txt/.md).

CLI:   .venv\\Scripts\\python resume_parser.py path\\to\\resume.pdf [-o out.txt]
       (default output: data/resume/<name>.txt - data/ is gitignored)

Library:  from resume_parser import extract_text, extract_text_from_bytes

Formats: .pdf via pypdf; .docx via the standard library (a .docx is a zip
holding word/document.xml - paragraphs and table cells are read in document
order, so no python-docx dependency); .txt / .md read as-is. Legacy binary
.doc is not supported - save it as .docx first.

Extraction from PDFs is imperfect by nature: multi-column layouts can
interleave, and bullet glyphs / ligatures come out as odd characters. The
normalizer below fixes the common cases; the CLI prints a preview so the
result can be sanity-checked before it is used to build a lexicon or an
interviewer persona.
"""

import argparse
import io
import re
import sys
import zipfile
from pathlib import Path
from xml.etree import ElementTree

BASE_DIR = Path(__file__).resolve().parent
DEFAULT_OUT_DIR = BASE_DIR / "data" / "resume"

SUPPORTED = (".pdf", ".docx", ".txt", ".md")

# Typographic clean-up: ligatures, private-use bullet glyphs (Wingdings /
# Symbol fonts), fancy quotes and dashes that break token matching later.
_REPLACEMENTS = {
    "ﬁ": "fi", "ﬂ": "fl", "ﬀ": "ff", "ﬃ": "ffi", "ﬄ": "ffl",
    "‘": "'", "’": "'", "“": '"', "”": '"',
    "–": "-", "—": "-", "•": "-", "●": "-", "▪": "-",
    "✦": "-", "➢": "-", " ": " ", "​": "",
}
_PRIVATE_USE = re.compile(r"[-]")
_HYPHEN_BREAK = re.compile(r"(\w)-\n(?=[a-z])")
_W = "{http://schemas.openxmlformats.org/wordprocessingml/2006/main}"


def normalize(text):
    for src, dst in _REPLACEMENTS.items():
        text = text.replace(src, dst)
    text = _PRIVATE_USE.sub("-", text)          # bullet glyphs from symbol fonts
    text = text.replace("\r\n", "\n").replace("\r", "\n")
    # A hyphen at a line end followed by a lowercase continuation is a wrapped
    # hyphenated term far more often than soft hyphenation in a resume (Word
    # does not auto-hyphenate; bullets wrap at real hyphens), so the line is
    # joined and the hyphen KEPT: "scikit-\nlearn" -> "scikit-learn". A hyphen
    # before a capitalised line start (a list item, a name) is left alone.
    text = _HYPHEN_BREAK.sub(r"\1-", text)
    lines = [re.sub(r"[ \t]+", " ", line).strip() for line in text.split("\n")]
    text = "\n".join(lines)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip() + "\n"


def _pdf_text(data):
    from pypdf import PdfReader  # imported lazily: only PDFs need it

    reader = PdfReader(io.BytesIO(data))
    pages = []
    for page in reader.pages:
        pages.append(page.extract_text() or "")
    return "\n\n".join(pages)


def _docx_text(data):
    with zipfile.ZipFile(io.BytesIO(data)) as archive:
        try:
            xml = archive.read("word/document.xml")
        except KeyError as exc:
            raise ValueError("not a Word document (word/document.xml missing)") from exc
    root = ElementTree.fromstring(xml)
    body = root.find(f"{_W}body")
    if body is None:
        return ""
    lines = []

    def paragraph_text(par):
        parts = []
        for node in par.iter():
            if node.tag == f"{_W}t":
                parts.append(node.text or "")
            elif node.tag == f"{_W}tab":
                parts.append("\t")
            elif node.tag == f"{_W}br":
                parts.append("\n")
        return "".join(parts)

    def walk(element):
        for child in element:
            if child.tag == f"{_W}p":
                lines.append(paragraph_text(child))
            elif child.tag == f"{_W}tbl":
                for row in child.iter(f"{_W}tr"):
                    cells = []
                    for cell in row.iter(f"{_W}tc"):
                        cells.append(" ".join(
                            paragraph_text(p) for p in cell.iter(f"{_W}p")).strip())
                    lines.append(" | ".join(c for c in cells if c))
            elif child.tag == f"{_W}sdt":      # content controls wrap paragraphs
                content = child.find(f"{_W}sdtContent")
                if content is not None:
                    walk(content)

    walk(body)
    return "\n".join(lines)


def extract_text_from_bytes(data, filename):
    suffix = Path(filename).suffix.lower()
    if suffix == ".pdf":
        raw = _pdf_text(data)
    elif suffix == ".docx":
        raw = _docx_text(data)
    elif suffix in (".txt", ".md"):
        raw = data.decode("utf-8-sig", errors="replace")
    elif suffix == ".doc":
        raise ValueError("legacy .doc is not supported - save it as .docx (or PDF) first")
    else:
        raise ValueError(f"unsupported file type {suffix!r}; expected one of {SUPPORTED}")
    return normalize(raw)


def extract_text(path):
    path = Path(path)
    return extract_text_from_bytes(path.read_bytes(), path.name)


def main():
    parser = argparse.ArgumentParser(description="Resume file -> plain text")
    parser.add_argument("file", help="resume (.pdf, .docx, .txt, .md)")
    parser.add_argument("-o", "--out", help="output .txt path "
                        "(default: data/resume/<name>.txt, gitignored)")
    parser.add_argument("--print", action="store_true",
                        help="print the full text instead of a preview")
    args = parser.parse_args()

    src = Path(args.file)
    if not src.exists():
        print(f"Not found: {src}")
        return 1
    try:
        text = extract_text(src)
    except ValueError as exc:
        print(f"Cannot parse {src.name}: {exc}")
        return 1

    out = Path(args.out) if args.out else DEFAULT_OUT_DIR / f"{src.stem}.txt"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(text, encoding="utf-8")

    words = len(text.split())
    print(f"Wrote {out} ({words} words, {text.count(chr(10))} lines)")
    if words < 80:
        print("Warning: very little text extracted - a scanned/image PDF has no "
              "text layer; export the resume from its editor as PDF or .docx.")
    if args.print:
        print("\n" + text)
    else:
        preview = "\n".join(text.splitlines()[:15])
        print("\n--- preview (first 15 lines) ---\n" + preview)
    return 0


if __name__ == "__main__":
    sys.exit(main())
