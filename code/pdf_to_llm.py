#!/usr/bin/env python3

import os
import re
import sys
import unicodedata
import logging
from typing import List, Dict, Tuple, Optional
from pathlib import Path
import click

try:
    import fitz  # PyMuPDF is still used for images
except ImportError:
    click.echo("Error: PyMuPDF is required for image extraction. Install with `pip install PyMuPDF`.")
    sys.exit(1)

try:
    from pdfminer.high_level import extract_text as pdfminer_extract_text
    from pdfminer.layout import LAParams
except ImportError:
    click.echo("Error: pdfminer.six is required for text extraction. Install with `pip install pdfminer.six`.")
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
    """Extract and normalize raw text from each page using pdfminer.six for better layout analysis."""
    logger.info(f"Extracting text from {pdf_path} using pdfminer.six")
    
    # These parameters are tuned for academic papers, especially the boxes_flow
    laparams = LAParams(
        line_overlap=0.5,
        char_margin=2.0,
        line_margin=0.5,
        word_margin=0.1,
        boxes_flow=0.5,  # Crucial for layout analysis in two-column documents
        detect_vertical=False,
        all_texts=False
    )
    
    text = pdfminer_extract_text(pdf_path, laparams=laparams)
    text = unicodedata.normalize("NFKC", text)
    return text


def clean_text_improved(text: str) -> str:
    """Overhauled text cleaning function focusing on robust heading detection and paragraph reconstruction."""
    
    # De-hyphenate words broken across lines.
    # This is a common artifact from PDF text extraction from two-column layouts.
    text = re.sub(r'([a-zA-Z])-\n([a-zA-Z])', r'\1\2', text)

    # A regex to join lines that are likely part of the same sentence.
    text = re.sub(r'(?<![.!?:])\n(?=[a-z])', ' ', text)
    
    # 1. Pre-process to remove common headers/footers
    lines = text.split('\n')
    cleaned_lines = []
    for line in lines:
        # Remove common academic paper headers/footers
        if re.match(r'^(IEEE|ACM|Springer|Elsevier|VOL\.|NO\.)', line, re.IGNORECASE):
            continue
        if "transactions on pattern analysis and machine intelligence" in line.lower():
            continue
        if "authorized licensed use limited to" in line.lower():
            continue
        if "downloaded on" in line.lower() and "utc from ieee xplore" in line.lower():
            continue
        # Remove lines that are just a page number
        if re.match(r'^\s*\d+\s*$', line):
            continue
        # Remove manuscript metadata
        if re.match(r'Manuscript received.*', line, re.IGNORECASE):
            continue
        if re.match(r'Digital Object Identifier.*', line, re.IGNORECASE):
            continue
        # Remove personal use statements
        if "personal use is permitted" in line.lower():
            continue
        cleaned_lines.append(line)
    text = '\n'.join(cleaned_lines)

    # Fix common equation and symbol character mapping issues
    replacements = {
        'ð': '(', 'Þ': ')', '1⁄4': '=', '': '-', '': ' * ',
        'þ': '->', 'ﬁ': 'fi', 'ﬂ': 'fl', 'ﬃ': 'ffi', 'ﬄ': 'ffl',
        '·': '*', '’': "'", '“': '"', '”': '"', 'Σ': 'sum'
    }
    for old, new in replacements.items():
        text = text.replace(old, new)

    # Initial cleanup of common artifacts
    text = re.sub(r"\(cid:\d+\)", "", text)  # Remove font artifacts
    text = re.sub(r"Ç", "", text) # Remove strange artifacts
    
    # 2. Reconstruct paragraphs. pdfminer.six with good LAParams gives decent paragraph breaks.
    paragraphs = re.split(r'\n\s*\n', text)
    
    processed_paragraphs = []
    in_references = False
    
    for para in paragraphs:
        para = para.strip()
        if not para:
            continue

        # Remove recurring title header (more robustly)
        if "learning without forgetting" in para.lower() and len(para.split()) < 10:
            continue
            
        # 3. Handle References section
        if re.match(r'^(references|bibliography)$', para, re.IGNORECASE):
            in_references = True
            processed_paragraphs.append("## " + para)
            continue
            
        if in_references:
            # Simple but effective reference parsing
            # Split by patterns like "[1]" or "1. " at the beginning of a line.
            refs = re.split(r'\n(?=\[\d+\])|(?<=\]\.)\s*\n(?=[A-Z])|\n(?=\d+\.)', para)
            for ref in refs:
                ref = ref.strip()
                if ref:
                    processed_paragraphs.append("* " + re.sub(r'\s+', ' ', ref))
            continue

        # 4. More conservative heading detection
        lines = para.split('\n')
        first_line = lines[0].strip()
        
        is_heading = False
        # Pattern for numbered headings like "1. Introduction" or "IV. Methodology"
        if re.match(r'^((\d+\.)+|([IVXLCDM]+\.))\s+[A-Z].*', first_line):
            is_heading = True
        # Pattern for short, title-cased or all-caps lines that don't end with a period.
        elif len(first_line.split()) < 8 and not first_line.endswith('.') and (first_line.istitle() or first_line.isupper()):
            # Exclude lines that are likely figure/table captions
            if not re.match(r'^(Fig|Table)\.?', first_line, re.IGNORECASE):
                 is_heading = True
        
        if is_heading:
            processed_paragraphs.append("\n## " + first_line)
            # The rest of the paragraph is treated as content following the heading
            content = ' '.join([line.strip() for line in lines[1:]])
            if content:
                processed_paragraphs.append(content)
        else:
            # Join all lines to form a single paragraph
            full_paragraph = ' '.join(line.strip() for line in lines)
            # De-hyphenate words at line breaks
            full_paragraph = re.sub(r'-\s+', '', full_paragraph)
            # Remove any lingering noise patterns
            full_paragraph = re.sub(r'Authorized licensed use limited to:.*?Restrictions apply\.', '', full_paragraph, flags=re.IGNORECASE)
            processed_paragraphs.append(full_paragraph)
            
    result = '\n\n'.join(processed_paragraphs)
    
    # Final cleanup of excessive newlines
    result = re.sub(r"\n{3,}", "\n\n", result)
    
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
        logger.warning("tabula-py not found, skipping.")
    except Exception as e:
        logger.warning(f"tabula-py failed with a unicode error, which is a known issue with some PDFs: {e}")
    
    # Fallback to camelot-py
    if not tables:
        try:
            import camelot
            # Check for OpenCV, a key dependency for camelot
            try:
                import cv2
            except ImportError:
                logger.error("OpenCV is not installed, which is required by camelot-py. Please run 'pip install opencv-python-headless'")
                return []

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
            logger.warning("camelot-py not found; it may not be installed correctly. Skipping tables.")
        except Exception as e:
            logger.warning(f"camelot-py failed: {e}")
    
    logger.info(f"Extracted {len(tables)} tables")
    return tables


def rect_distance(r1, r2):
    """Calculate the distance between two rectangles."""
    # Horizontal distance
    if r1.x1 < r2.x0:
        dx = r2.x0 - r1.x1
    elif r2.x1 < r1.x0:
        dx = r1.x0 - r2.x1
    else:
        dx = 0
    # Vertical distance
    if r1.y1 < r2.y0:
        dy = r2.y0 - r1.y1
    elif r2.y1 < r1.y0:
        dy = r1.y0 - r2.y1
    else:
        dy = 0
    return (dx**2 + dy**2)**0.5


def extract_figures_improved(pdf_path: str, output_dir: str, ocr: bool = False) -> List[str]:
    """
    Extracts figures and diagrams from a PDF by identifying both image blocks and vector graphics.
    It renders the areas containing these elements to capture them accurately.
    """
    logger.info(f"Extracting figures from {pdf_path} using a more comprehensive method.")

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
            logger.info("OCR enabled for figure extraction")
        except (ImportError, pytesseract.pytesseract.TesseractNotFoundError):
            logger.warning("OCR requested but Tesseract not available.")

    os.makedirs(output_dir, exist_ok=True)
    figs_md = []
    doc = fitz.open(pdf_path)
    fig_count = 0

    for page_num, page in enumerate(doc):
        # 1. Get bboxes for both raster images and vector graphics (drawings)
        image_info = page.get_image_info(xrefs=True)
        image_bboxes = [fitz.Rect(info['bbox']) for info in image_info]
        drawing_bboxes = [path["rect"] for path in page.get_drawings()]
        
        all_bboxes = image_bboxes + drawing_bboxes

        # 2. Merge overlapping or nearby bounding boxes to group elements
        if not all_bboxes:
            continue
        
        merged_bboxes = []
        all_bboxes.sort(key=lambda r: (r.y0, r.x0)) # Sort top-to-bottom, left-to-right

        if all_bboxes:
            current_bbox = all_bboxes[0]
            for bbox in all_bboxes[1:]:
                # Merge if boxes intersect or are close and overlap on one axis
                if current_bbox.intersects(bbox) or rect_distance(current_bbox, bbox) < 20:
                    current_bbox |= bbox  # Union of the two rectangles
                else:
                    merged_bboxes.append(current_bbox)
                    current_bbox = bbox
            merged_bboxes.append(current_bbox)
        
        # 3. Filter out small artifacts that are unlikely to be figures
        final_bboxes = [r for r in merged_bboxes if r.width > 50 and r.height > 50]

        for bbox in final_bboxes:
            fig_count += 1
            
            # Slightly enlarge the bbox to ensure the whole figure is captured
            bbox.x0 -= 10
            bbox.y0 -= 10
            bbox.x1 += 10
            bbox.y1 += 10

            # Render the region at a high DPI for clarity
            pix = page.get_pixmap(clip=bbox.irect, dpi=300)
            
            fname = f"figure_{page_num+1}_{fig_count}.png"
            outpath = os.path.join(output_dir, fname)
            pix.save(outpath)
            
            md_path = outpath.replace(os.sep, '/')
            md_line = f"![Figure {page_num+1}.{fig_count}]({md_path})"

            # 4. Extract nearby text as a potential caption
            caption_bbox = fitz.Rect(bbox.x0, bbox.y1, bbox.x1, bbox.y1 + 50)
            caption_text = page.get_text(clip=caption_bbox, sort=True).strip()
            if caption_text and len(caption_text.split()) > 3: # Basic check for a valid caption
                 md_line += f"\n\n*Potential caption:* {caption_text}"

            if ocr_available:
                try:
                    text = pytesseract.image_to_string(Image.open(outpath))
                    text = unicodedata.normalize("NFKC", text).strip()
                    if text:
                        md_line += f"\n\n*OCR text:*\n```\n{text}\n```"
                except Exception as e:
                    logger.warning(f"OCR failed for {fname}: {e}")
            
            figs_md.append(md_line)

    doc.close()
    logger.info(f"Extracted {len(figs_md)} figures using comprehensive method.")
    return figs_md


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
    os.makedirs('output', exist_ok=True)
    logger.info(f"Processing PDF: {pdf}")
    try:
        raw_text = extract_text(pdf)
        cleaned_text = clean_text_improved(raw_text)
        if not cleaned_text.strip():
            logger.error("No text extracted from PDF")
            return
        parts = [cleaned_text]
        if tables:
            extracted_tables = extract_tables_improved(pdf)
            if extracted_tables:
                parts.append("\n\n".join(extracted_tables))
        if figures:
            fig_dir = os.path.join('output', Path(pdf).stem + "_figures")
            extracted_figures = extract_figures_improved(pdf, fig_dir, ocr)
            if extracted_figures:
                parts.append("\n\n".join(extracted_figures))
        output_md = "\n\n".join(part for part in parts if part.strip())
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