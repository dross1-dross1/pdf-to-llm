PDF-to-LLM
===========

A tool for extracting text, tables, and figures from academic PDFs into Markdown for LLM processing.

Requirements
------------
- Python 3.8+
- pip install -r requirements.txt
- Optional: Tesseract OCR for figure text extraction

Usage
-----
Single PDF:
    python code/pdf_to_llm.py pdf/yourfile.pdf -o output/yourfile.md --tables --figures --ocr

Batch all PDFs:
    python code/batch_pdf_to_md.py

All outputs are written to the output/ directory. Figures are saved in output/[pdfname]_figures/. 