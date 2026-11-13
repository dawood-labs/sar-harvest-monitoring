"""Turn a report template into one self-contained HTML file: every {{figure:name.png}} becomes an inline image.

Why: people forward the report file alone, so it must not depend on a figures folder.
Usage: python src/embed_figures.py <template.html> <out.html> [figures_dir]
"""
from __future__ import annotations

import base64
import re
import sys
from pathlib import Path


def main(template: str, out: str, figures: str = "reports/figures") -> None:
    def inline(m: re.Match) -> str:
        data = base64.b64encode((Path(figures) / m.group(1)).read_bytes()).decode()
        return f'<img alt="{m.group(1)}" src="data:image/png;base64,{data}">'

    Path(out).write_text(re.sub(r"\{\{figure:([^}]+)\}\}", inline, Path(template).read_text()))
    print(f"written {out} ({Path(out).stat().st_size / 1e6:.1f} MB)")


if __name__ == "__main__":
    main(*sys.argv[1:4])
