#!/usr/bin/env python3
"""
MD -> DOCX converter for the AEGIS academic paper.

Steps:
  1. Pre-render all 7 Mermaid blocks to PNG via mmdc
  2. Substitute mermaid fences with image references
  3. Run pandoc (math -> OMML Word equations, images embedded)
  4. Post-process: replace Pandoc fingerprints in docx XML metadata

Usage:
  python convert_to_docx.py
"""

import re
import shutil
import subprocess
import sys
import zipfile
from pathlib import Path

INPUT_MD   = Path("academic_journal_draft_long version.md")
OUTPUT_DOCX = Path("AEGIS_paper.docx")
FIGURES_DIR = Path("_figures_tmp")
TMP_MD      = Path("_preprocessed_tmp.md")

AUTHOR = "Vasileios Kazoukas"
TITLE  = ("A Collaborative Multi-Agent Framework for Crisis Management "
          "Decision Support Using Evidential Reasoning and Rule-Based "
          "Graph Attention Aggregation")
DATE   = "2026"


def render_mermaid(md_text: str, out_dir: Path) -> str:
    out_dir.mkdir(exist_ok=True)
    pattern = re.compile(r'```mermaid\n(.*?)```', re.DOTALL)
    fig_num = [0]

    def replace(match):
        fig_num[0] += 1
        n = fig_num[0]
        mmd = out_dir / f"fig{n:02d}.mmd"
        png = out_dir / f"fig{n:02d}.png"
        mmd.write_text(match.group(1), encoding="utf-8")

        puppeteer_cfg = Path("/tmp/_puppeteer_chrome_cfg.json")
        if not puppeteer_cfg.exists():
            puppeteer_cfg.write_text(
                '{"executablePath": "/Applications/Google Chrome.app'
                '/Contents/MacOS/Google Chrome"}',
                encoding="utf-8"
            )
        r = subprocess.run(
            ["mmdc", "-i", str(mmd), "-o", str(png),
             "-b", "white", "-w", "1400", "--scale", "2",
             "--puppeteerConfigFile", str(puppeteer_cfg)],
            capture_output=True, text=True
        )
        if r.returncode != 0 or not png.exists():
            print(f"  [WARN] mmdc failed for figure {n}: {r.stderr[:200]}")
            return match.group(0)
        print(f"  [OK] figure {n} -> {png.name}")
        return f"![]({png})"

    return pattern.sub(replace, md_text)


def clean_metadata(docx: Path):
    """
    Rewrite docProps/core.xml and docProps/app.xml inside the docx zip
    to remove Pandoc fingerprints and set proper author/application fields.
    """
    tmp = docx.with_name("_meta_tmp.docx")

    with zipfile.ZipFile(docx, "r") as zin, \
         zipfile.ZipFile(tmp, "w", zipfile.ZIP_DEFLATED) as zout:

        for item in zin.infolist():
            data = zin.read(item.filename)

            if item.filename == "docProps/core.xml":
                text = data.decode("utf-8")
                # creator and lastModifiedBy -> actual author
                text = re.sub(
                    r"<dc:creator>[^<]*</dc:creator>",
                    f"<dc:creator>{AUTHOR}</dc:creator>", text)
                text = re.sub(
                    r"<cp:lastModifiedBy>[^<]*</cp:lastModifiedBy>",
                    f"<cp:lastModifiedBy>{AUTHOR}</cp:lastModifiedBy>", text)
                # subject / description: strip if Pandoc inserted anything
                text = re.sub(r"<dc:subject>[^<]*</dc:subject>", "", text)
                data = text.encode("utf-8")

            elif item.filename == "docProps/app.xml":
                text = data.decode("utf-8")
                # Application field: replace Pandoc with Word
                text = re.sub(
                    r"<Application>[^<]*</Application>",
                    "<Application>Microsoft Office Word</Application>", text)
                # Remove AppVersion if present (can fingerprint Pandoc version)
                text = re.sub(r"<AppVersion>[^<]*</AppVersion>", "", text)
                data = text.encode("utf-8")

            zout.writestr(item, data)

    tmp.replace(docx)
    print(f"  [OK] metadata cleaned")


def main():
    if not INPUT_MD.exists():
        sys.exit(f"Error: {INPUT_MD} not found. Run from the project root.")

    print(f"==> Reading {INPUT_MD}")
    md_text = INPUT_MD.read_text(encoding="utf-8")

    print(f"\n==> Rendering {7} Mermaid diagrams...")
    processed = render_mermaid(md_text, FIGURES_DIR)
    TMP_MD.write_text(processed, encoding="utf-8")

    # Resolve pandoc from PATH or common install locations
    pandoc_bin = shutil.which("pandoc") or "/opt/anaconda3/bin/pandoc"

    print(f"\n==> Running pandoc ({pandoc_bin})...")
    cmd = [
        pandoc_bin, str(TMP_MD),
        "-o", str(OUTPUT_DOCX),
        "--from", "markdown+tex_math_dollars+tex_math_single_backslash",
        "--to", "docx",
        "--metadata", f"title={TITLE}",
        "--metadata", f"author={AUTHOR}",
        "--metadata", f"date={DATE}",
        "--highlight-style=tango",
        "--wrap=none",
    ]
    r = subprocess.run(cmd, capture_output=True, text=True)
    if r.returncode != 0:
        sys.exit(f"pandoc error:\n{r.stderr}")
    print(f"  [OK] pandoc done")

    print(f"\n==> Cleaning DOCX metadata...")
    clean_metadata(OUTPUT_DOCX)

    print(f"\n==> Cleanup temp files...")
    TMP_MD.unlink(missing_ok=True)
    shutil.rmtree(FIGURES_DIR, ignore_errors=True)

    print(f"\n==> Done: {OUTPUT_DOCX}")
    print(    "    Open in Word, do File > Save As to refresh Word's own metadata.")


if __name__ == "__main__":
    main()
