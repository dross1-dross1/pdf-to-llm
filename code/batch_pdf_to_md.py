import os
from pathlib import Path
import subprocess

def main():
    pdf_dir = Path('pdf')
    out_dir = Path('output')
    out_dir.mkdir(exist_ok=True)
    for pdf_file in pdf_dir.glob('*.pdf'):
        out_file = out_dir / (pdf_file.stem + '.md')
        cmd = [
            'python', 'code/pdf_to_llm.py', str(pdf_file), '-o', str(out_file), '--tables', '--figures'
        ]
        subprocess.run(cmd, check=True)

if __name__ == '__main__':
    main() 