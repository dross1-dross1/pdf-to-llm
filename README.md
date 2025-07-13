PDF-to-LLM
=========

A tool for extracting and converting academic PDFs into Markdown for LLM processing.

Directory Structure
-------------------
- code/: Python scripts
- pdf/: Input PDFs
- output/: All generated outputs (Markdown, figures)

Requirements
------------
See requirements.txt for dependencies. Install with:

    pip install -r requirements.txt

Usage
-----

    python code/pdf_to_llm_improved.py pdf/yourfile.pdf -o output/yourfile.md --tables --figures --ocr

Options:
- --tables: Extract tables
- --figures: Extract figures
- --ocr: OCR on figures

Outputs are written to the output/ directory. 