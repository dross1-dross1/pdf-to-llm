#!/usr/bin/env python3
"""
pdf2llm.py

A DIY CLI tool to extract and clean text, tables, and figures from PDFs into LLM-ready Markdown.

Usage:
    pdf2llm input.pdf [-o output.md] [--tables] [--figures] [--ocr]

Dependencies:
    pip install click PyMuPDF camelot-py[cv] pillow pytesseract
Please install the Tesseract OCR engine separately if you plan to use --ocr on Windows e.g. via Chocolatey: `choco install tesseract`.
"""
import os
import re
import sys
import unicodedata
import click

try:
    import fitz  # PyMuPDF
except ImportError:
    click.echo("Error: PyMuPDF is required. Install with `pip install PyMuPDF`.")
    sys.exit(1)


def extract_text(pdf_path: str) -> str:
    """Extract and normalize raw text from each page using PyMuPDF."""
    doc = fitz.open(pdf_path)
    pages = []
    for page in doc:
        text = page.get_text("text") or ""
        text = unicodedata.normalize("NFKC", text)
        pages.append(text)
    return "\n\n".join(pages)


def clean_text(text: str) -> str:
    """Clean and normalize extracted text for LLM-friendly Markdown."""
    text = unicodedata.normalize("NFKC", text)

    # Remove author and metadata blocks
    text = re.sub(r"(?m)^The authors are with[\s\S]*?Digital Object Identifier.*?$", "", text)
    text = re.sub(r"(?m)^E-mail:.*$", "", text)
    text = re.sub(r"(?m)^Date of publication.*$", "", text)
    text = re.sub(r"(?m)^Date of current version.*$", "", text)
    text = re.sub(r"(?m)^Recommended for acceptance.*$", "", text)
    text = re.sub(r"(?m)^For information on obtaining reprints.*$", "", text)
    text = re.sub(r"(?m)^Digital Object Identifier.*$", "", text)
    text = re.sub(r"(?m)^0162-8828.*IEEE.*$", "", text)

    # Remove odd markers and footnotes
    text = text.replace('', '')

    # Remove page headers/footers, volume/issue, date lines, redistrib URLs
    patterns = [
        r"(?m)^\s*\d+\s*$",                  # page numbers alone
        r"(?m)^.*?IEEE TRANSACTIONS ON.*$",    # journal header
        r"(?m)^VOL\..*NO\..*$",              # VOL. X, NO. Y lines
        r"(?m)^#*\s*DECEMBER \d{4}\s*$",    # month-year lines
        r"(?m)^See ht_tp://.*$",               # redistrib notice URLs
    ]
    for pat in patterns:
        text = re.sub(pat, "", text)

    # Convert Abstract line to heading
    text = re.sub(r"(?m)^\s*Abstract\s*[—-].*$", "## Abstract", text)

    # Remove Index Terms line and artifacts
    text = re.sub(r"Index Terms—.*?\n", "", text)
    text = re.sub(r"Ç", "", text)

    # Fix split uppercase words (e.g., 'M ANY' -> 'MANY')
    text = re.sub(r"\b([A-Z])\s+([A-Z]{2,})\b", r"\1\2", text)

    # Keep content between Abstract and References
    start = re.search(r"(?i)^## Abstract", text)
    end = re.search(r"(?i)^References", text)
    if start: text = text[start.start():]
    if end:   text = text[:end.start()]

    # Remove inline markers
    inline = [
        r"Downloaded on .*?UTC.*?\n",
        r"Authorized licensed use limited to:.*?\n",
        r"Manuscript received.*?\n",
        r"\(cid:\d+\)",
        r"\[\d+\]",
        r"\(\d+\)",
    ]
    for pat in inline:
        text = re.sub(pat, "", text)

    # Fix hyphenation at line breaks
    text = re.sub(r"(\w+)-\s*\n\s*(\w+)", r"\1\2", text)

    # Identify and mark section headings
    def hdr(m): return f"\n## {m.group(1).strip()}\n"
    text = re.sub(r"(?m)^([A-Z][A-Z0-9 &]{2,})\s*$", hdr, text)
    text = re.sub(r"(?m)^([A-Z][a-z]+(?:\s+[A-Z][a-z]+)*):?\s*$", hdr, text)

    # Flatten single newlines to spaces
    text = re.sub(r"(?<!\n)\n(?!\n)", " ", text)
    # Collapse multiple blank lines
    text = re.sub(r"\n{3,}", "\n\n", text)

    return text.strip()


def extract_tables(pdf_path: str) -> list:
    """Extract tables as Markdown using Camelot."""
    try:
        import camelot
    except ImportError:
        click.echo("Warning: camelot-py not found; skipping tables.")
        return []

    try:
        tables = camelot.read_pdf(pdf_path, pages="all", flavor="stream")
    except Exception as e:
        click.echo(f"Warning: table extraction failed ({e}); skipping tables.")
        return []

    md_tables = []
    for idx, table in enumerate(tables, start=1):
        df = table.df
        md = df.to_markdown(index=False)
        md_tables.append(f"**Table {idx}:**\n{md}\n")
    return md_tables


def extract_figures(pdf_path: str, output_dir: str, ocr: bool=False) -> list:
    """Extract embedded images and optionally OCR text, skipping gracefully if tesseract is not available."""
    try:
        from PIL import Image
    except ImportError:
        click.echo("Warning: pillow not found; skipping figures.")
        return []

    if ocr:
        try:
            import pytesseract
            pytesseract.get_tesseract_version()
        except ImportError:
            click.echo("Warning: pytesseract not installed; skipping OCR.")
            ocr = False
        except pytesseract.pytesseract.TesseractNotFoundError:
            click.echo("Warning: Tesseract executable not found in PATH; skipping OCR.")
            ocr = False

    os.makedirs(output_dir, exist_ok=True)
    figs_md = []
    doc = fitz.open(pdf_path)
    for page_num in range(len(doc)):
        page = doc[page_num]
        for img_index, img in enumerate(page.get_images(full=True), start=1):
            xref = img[0]
            base_image = doc.extract_image(xref)
            img_bytes = base_image["image"]
            img_ext = base_image.get("ext", "png")
            fname = f"figure_{page_num+1}_{img_index}.{img_ext}"
            outpath = os.path.join(output_dir, fname)
            with open(outpath, "wb") as f:
                f.write(img_bytes)
            md_path = outpath.replace(os.sep, '/')
            md_line = f"![Figure {page_num+1}.{img_index}]({md_path})"
            if ocr:
                try:
                    pil_img = Image.open(outpath)
                    text = pytesseract.image_to_string(pil_img)
                    text = unicodedata.normalize("NFKC", text).strip()
                    if text:
                        md_line += f"\n\n*OCR text:* {text}"
                except Exception as e:
                    click.echo(f"Warning: OCR failed for {fname} ({e}); continuing.")
            figs_md.append(md_line)
    return figs_md

@click.command()
@click.argument("pdf", type=click.Path(exists=True))
@click.option("-o", "--output", type=click.Path(), default=None,
              help="Output Markdown file; prints to stdout if omitted.")
@click.option("--tables/--no-tables", default=False,
              help="Include tables in output.")
@click.option("--figures/--no-figures", default=False,
              help="Include figures in output.")
@click.option("--ocr/--no-ocr", default=False,
              help="Perform OCR on figures and append detected text.")
def main(pdf, output, tables, figures, ocr):
    """Convert a PDF to LLM-ready Markdown."""
    raw = extract_text(pdf)
    cleaned = clean_text(raw)
    parts = [cleaned]

    if tables:
        tbls = extract_tables(pdf)
        if tbls:
            parts.append("\n".join(tbls))
    if figures:
        fig_dir = os.path.splitext(os.path.basename(pdf))[0] + "_figures"
        fmd = extract_figures(pdf, fig_dir, ocr)
        if fmd:
            parts.append("\n".join(fmd))

    out_md = "\n\n".join(part for part in parts if part)
    if output:
        with open(output, 'w', encoding='utf-8') as f:
            f.write(out_md)
        click.echo(f"LLM-ready Markdown written to {output}")
    else:
        click.echo(out_md)

if __name__ == '__main__':
    main()
