#!/usr/bin/env python3
"""
pdf_to_llm_improved.py

An improved CLI tool to extract and clean text, tables, and figures from PDFs into LLM-ready Markdown.

Usage:
    python pdf_to_llm_improved.py input.pdf [-o output.md] [--tables] [--figures] [--ocr] [--verbose]

Dependencies:
    pip install click PyMuPDF pillow pytesseract pandas tabula-py
Please install the Tesseract OCR engine separately if you plan to use --ocr on Windows e.g. via Chocolatey: `choco install tesseract`.
"""
import os
import re
import sys
import unicodedata
import logging
from typing import List, Dict, Tuple, Optional
from pathlib import Path
import click

try:
    import fitz  # PyMuPDF
except ImportError:
    click.echo("Error: PyMuPDF is required. Install with `pip install PyMuPDF`.")
    sys.exit(1)

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


def setup_logging(verbose: bool = False):
    """Configure logging level based on verbosity."""
    if verbose:
        logging.getLogger().setLevel(logging.DEBUG)
    else:
        logging.getLogger().setLevel(logging.INFO)


def extract_text(pdf_path: str) -> str:
    """Extract and normalize raw text from each page using PyMuPDF."""
    logger.info(f"Extracting text from {pdf_path}")
    doc = fitz.open(pdf_path)
    pages = []
    
    for page_num, page in enumerate(doc, 1):
        logger.debug(f"Processing page {page_num}/{len(doc)}")
        text = page.get_text("text") or ""
        text = unicodedata.normalize("NFKC", text)
        pages.append(text)
    
    doc.close()
    return "\n\n".join(pages)


def clean_text_improved(text: str) -> str:
    """Improved text cleaning for academic papers with better structure preservation."""
    logger.info("Cleaning extracted text")
    
    # Normalize unicode
    text = unicodedata.normalize("NFKC", text)
    
    # Remove common PDF artifacts and metadata
    artifacts = [
        r"Downloaded on .*?UTC.*?\n",
        r"Authorized licensed use limited to:.*?\n",
        r"Manuscript received.*?\n",
        r"\(cid:\d+\)",
        r"\[\d+\]",
        r"\(\d+\)",
        r"Digital Object Identifier.*?\n",
        r"0162-8828.*IEEE.*\n",
        r"See ht_tp://.*\n",
        r"Recommended for acceptance.*\n",
        r"For information on obtaining reprints.*\n",
        r"Date of publication.*\n",
        r"Date of current version.*\n",
        r"E-mail:.*\n",
    ]
    
    for pattern in artifacts:
        text = re.sub(pattern, "", text, flags=re.MULTILINE | re.IGNORECASE)
    
    # Remove page numbers and headers (more flexible)
    text = re.sub(r"(?m)^\s*\d+\s*$", "", text)  # Standalone page numbers
    text = re.sub(r"(?m)^.*?(IEEE|ACM|Springer|Elsevier).*?(TRANSACTIONS|JOURNAL|CONFERENCE).*$", "", text)
    
    # Better section heading detection
    def detect_section_heading(line: str) -> bool:
        """Detect if a line is likely a section heading."""
        line = line.strip()
        if not line:
            return False
        
        # Common academic section patterns
        patterns = [
            r"^[A-Z][A-Z\s&]+$",  # ALL CAPS headings
            r"^[A-Z][a-z]+(?:\s+[A-Z][a-z]+)*:?\s*$",  # Title Case headings
            r"^\d+\.\s+[A-Z]",  # Numbered sections
            r"^[A-Z]\.\s+[A-Z]",  # Lettered sections
        ]
        
        return any(re.match(pattern, line) for pattern in patterns)
    
    # Process text line by line to preserve structure
    lines = text.split('\n')
    cleaned_lines = []
    in_abstract = False
    in_references = False
    
    for i, line in enumerate(lines):
        line = line.strip()
        if not line:
            continue
            
        # Detect section boundaries
        if re.search(r"(?i)^abstract", line):
            in_abstract = True
            cleaned_lines.append("## Abstract")
            continue
        elif re.search(r"(?i)^references", line):
            in_references = True
            cleaned_lines.append("## References")
            continue
        elif re.search(r"(?i)^introduction", line):
            in_abstract = False
            cleaned_lines.append("## Introduction")
            continue
            
        # Skip content before abstract
        if not in_abstract and not in_references:
            continue
            
        # Convert detected headings
        if detect_section_heading(line):
            cleaned_lines.append(f"## {line}")
        else:
            # Fix common text issues
            line = re.sub(r"(\w+)-\s*\n\s*(\w+)", r"\1\2", line)  # Fix hyphenation
            line = re.sub(r"\b([A-Z])\s+([A-Z]{2,})\b", r"\1\2", line)  # Fix split words
            cleaned_lines.append(line)
    
    # Join lines with proper spacing
    result = "\n\n".join(cleaned_lines)
    
    # Final cleanup
    result = re.sub(r"\n{3,}", "\n\n", result)  # Remove excessive newlines
    result = re.sub(r"Ç", "", result)  # Remove specific artifacts
    
    return result.strip()


def extract_tables_improved(pdf_path: str) -> List[str]:
    """Extract tables using multiple methods for better coverage."""
    logger.info("Extracting tables from PDF")
    tables = []
    
    # Try tabula-py first (more reliable for simple tables)
    try:
        import tabula
        logger.debug("Using tabula-py for table extraction")
        dfs = tabula.read_pdf(pdf_path, pages='all', multiple_tables=True)
        
        for idx, df in enumerate(dfs, 1):
            if not df.empty:
                # Clean the dataframe
                df = df.dropna(how='all').dropna(axis=1, how='all')
                if not df.empty:
                    md_table = df.to_markdown(index=False)
                    tables.append(f"**Table {idx}:**\n{md_table}\n")
                    
    except ImportError:
        logger.warning("tabula-py not found, trying camelot-py")
    except Exception as e:
        logger.warning(f"tabula-py failed: {e}")
    
    # Fallback to camelot-py
    if not tables:
        try:
            import camelot
            logger.debug("Using camelot-py for table extraction")
            table_list = camelot.read_pdf(pdf_path, pages="all", flavor="stream")
            
            for idx, table in enumerate(table_list, 1):
                df = table.df
                # Clean the dataframe
                df = df.dropna(how='all').dropna(axis=1, how='all')
                if not df.empty:
                    md_table = df.to_markdown(index=False)
                    tables.append(f"**Table {idx}:**\n{md_table}\n")
                    
        except ImportError:
            logger.warning("Neither tabula-py nor camelot-py found; skipping tables")
        except Exception as e:
            logger.warning(f"camelot-py failed: {e}")
    
    logger.info(f"Extracted {len(tables)} tables")
    return tables


def extract_figures_improved(pdf_path: str, output_dir: str, ocr: bool = False) -> List[str]:
    """Improved figure extraction with better organization and caption preservation."""
    logger.info(f"Extracting figures to {output_dir}")
    
    try:
        from PIL import Image
    except ImportError:
        logger.warning("Pillow not found; skipping figures")
        return []
    
    # Setup OCR if requested
    ocr_available = False
    if ocr:
        try:
            import pytesseract
            pytesseract.get_tesseract_version()
            ocr_available = True
            logger.info("OCR enabled")
        except (ImportError, pytesseract.pytesseract.TesseractNotFoundError):
            logger.warning("OCR requested but Tesseract not available")
    
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    
    figs_md = []
    doc = fitz.open(pdf_path)
    
    # Extract figure captions from text
    captions = extract_figure_captions(doc)
    
    for page_num in range(len(doc)):
        page = doc[page_num]
        images = page.get_images(full=True)
        
        for img_index, img in enumerate(images, 1):
            try:
                xref = img[0]
                base_image = doc.extract_image(xref)
                img_bytes = base_image["image"]
                img_ext = base_image.get("ext", "png")
                
                # Create descriptive filename
                fname = f"figure_{page_num+1}_{img_index}.{img_ext}"
                outpath = os.path.join(output_dir, fname)
                
                # Save image
                with open(outpath, "wb") as f:
                    f.write(img_bytes)
                
                # Create markdown reference
                md_path = outpath.replace(os.sep, '/')
                md_line = f"![Figure {page_num+1}.{img_index}]({md_path})"
                
                # Add caption if available
                caption_key = f"{page_num+1}.{img_index}"
                if caption_key in captions:
                    md_line += f"\n\n*Caption:* {captions[caption_key]}"
                
                # Add OCR text if requested and available
                if ocr_available:
                    try:
                        pil_img = Image.open(outpath)
                        text = pytesseract.image_to_string(pil_img)
                        text = unicodedata.normalize("NFKC", text).strip()
                        if text:
                            md_line += f"\n\n*OCR text:* {text}"
                    except Exception as e:
                        logger.warning(f"OCR failed for {fname}: {e}")
                
                figs_md.append(md_line)
                
            except Exception as e:
                logger.warning(f"Failed to extract image {img_index} from page {page_num+1}: {e}")
    
    doc.close()
    logger.info(f"Extracted {len(figs_md)} figures")
    return figs_md


def extract_figure_captions(doc) -> Dict[str, str]:
    """Extract figure captions from the PDF text."""
    captions = {}
    
    for page_num in range(len(doc)):
        page = doc[page_num]
        text = page.get_text("text")
        
        # Look for figure caption patterns
        patterns = [
            r"Fig\.\s*(\d+\.\d+)[:\s]+(.*?)(?=\n|$)",
            r"Figure\s*(\d+\.\d+)[:\s]+(.*?)(?=\n|$)",
            r"FIG\.\s*(\d+\.\d+)[:\s]+(.*?)(?=\n|$)",
        ]
        
        for pattern in patterns:
            matches = re.finditer(pattern, text, re.IGNORECASE | re.MULTILINE)
            for match in matches:
                fig_num = match.group(1)
                caption = match.group(2).strip()
                if caption:
                    captions[fig_num] = caption
    
    return captions


def validate_extraction(pdf_path: str, output_md: str) -> Dict[str, any]:
    """Validate the quality of the extraction."""
    logger.info("Validating extraction quality")
    
    validation = {
        'text_length': len(output_md),
        'has_abstract': '## Abstract' in output_md,
        'has_introduction': '## Introduction' in output_md,
        'has_tables': '**Table' in output_md,
        'has_figures': '![Figure' in output_md,
        'sections_found': len(re.findall(r'##\s+', output_md)),
    }
    
    # Check for common issues
    issues = []
    if validation['text_length'] < 1000:
        issues.append("Extracted text seems too short")
    if not validation['has_abstract']:
        issues.append("No abstract found")
    if not validation['has_introduction']:
        issues.append("No introduction found")
    
    validation['issues'] = issues
    validation['quality_score'] = sum([
        1 if validation['has_abstract'] else 0,
        1 if validation['has_introduction'] else 0,
        1 if validation['sections_found'] >= 3 else 0,
        1 if validation['text_length'] > 5000 else 0,
    ]) / 4.0
    
    return validation


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
@click.option("--verbose", "-v", is_flag=True, default=False,
              help="Enable verbose logging.")
@click.option("--validate", is_flag=True, default=False,
              help="Validate extraction quality and show report.")
def main(pdf, output, tables, figures, ocr, verbose, validate):
    setup_logging(verbose)
    os.makedirs('pdf', exist_ok=True)
    os.makedirs('output', exist_ok=True)
    logger.info(f"Processing PDF: {pdf}")
    
    try:
        # Extract and clean text
        raw_text = extract_text(pdf)
        cleaned_text = clean_text_improved(raw_text)
        
        if not cleaned_text.strip():
            logger.error("No text extracted from PDF")
            return
        
        parts = [cleaned_text]
        
        # Extract tables if requested
        if tables:
            extracted_tables = extract_tables_improved(pdf)
            if extracted_tables:
                parts.append("\n\n".join(extracted_tables))
        
        # Extract figures if requested
        if figures:
            fig_dir = Path(pdf).stem + "_figures"
            extracted_figures = extract_figures_improved(pdf, str(fig_dir), ocr)
            if extracted_figures:
                parts.append("\n\n".join(extracted_figures))
        
        # Combine all parts
        output_md = "\n\n".join(part for part in parts if part.strip())
        
        # Validate if requested
        if validate:
            validation = validate_extraction(pdf, output_md)
            logger.info("Extraction Validation Report:")
            logger.info(f"  Text length: {validation['text_length']} characters")
            logger.info(f"  Sections found: {validation['sections_found']}")
            logger.info(f"  Has abstract: {validation['has_abstract']}")
            logger.info(f"  Has introduction: {validation['has_introduction']}")
            logger.info(f"  Has tables: {validation['has_tables']}")
            logger.info(f"  Has figures: {validation['has_figures']}")
            logger.info(f"  Quality score: {validation['quality_score']:.2f}")
            
            if validation['issues']:
                logger.warning("Issues found:")
                for issue in validation['issues']:
                    logger.warning(f"  - {issue}")
        
        # Write output
        if output:
            with open(output, 'w', encoding='utf-8') as f:
                f.write(output_md)
            logger.info(f"LLM-ready Markdown written to {output}")
        else:
            click.echo(output_md)
            
    except Exception as e:
        logger.error(f"Error processing PDF: {e}")
        sys.exit(1)


if __name__ == '__main__':
    main() 