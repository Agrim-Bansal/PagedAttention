#!/usr/bin/env python3
"""Build NARRATION.md from the module docstrings of talk/sN_*.py.

Pure stdlib (ast + re) — imports nothing from manim / manim_slides, so it can
run without the render venv's heavy deps.
"""

import ast
import re
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
TALK_DIR = ROOT / "talk"
OUT_PATH = ROOT / "NARRATION.md"

# (file stem, class name, act) — from talk/CONTRACT.md's scene table.
SCENES = [
    ("s0_title", "S0Title", "I"),
    ("s1_gpu", "S1GPU", "I"),
    ("s2_transformers", "S2Transformers", "I"),
    ("s3_kvcache", "S3KVCache", "I"),
    ("s4_problem", "S4Problem", "I"),
    ("s5_pagedattention", "S5PagedAttention", "II"),
    ("s6_os_and_why_hard", "S6OSAndWhyHard", "II"),
    ("s7_sharing", "S7Sharing", "III"),
    ("s8_scheduling", "S8Scheduling", "III"),
    ("s9_results", "S9Results", "III"),
    ("s10_ablations", "S10Ablations", "III"),
    ("s11_takeaways", "S11Takeaways", "III"),
]

ACTS = [
    ("I", "Why memory is the bottleneck", ["s0_title", "s1_gpu", "s2_transformers", "s3_kvcache", "s4_problem"]),
    ("II", "PagedAttention", ["s5_pagedattention", "s6_os_and_why_hard"]),
    ("III", "What it buys you", ["s7_sharing", "s8_scheduling", "s9_results", "s10_ablations", "s11_takeaways"]),
]


def get_docstring(stem):
    path = TALK_DIR / f"{stem}.py"
    tree = ast.parse(path.read_text(), filename=str(path))
    doc = ast.get_docstring(tree, clean=True)
    if doc is None:
        raise SystemExit(f"{path} has no module docstring")
    return doc


def extract_narration(doc):
    """Return the text of the docstring from the NARRATION marker onward."""
    match = re.search(r"^NARRATION\s*$", doc, flags=re.MULTILINE)
    if not match:
        raise SystemExit("no NARRATION section found in docstring:\n" + doc[:200])
    return doc[match.end():].strip("\n")


def build():
    lines = []
    lines.append("# PagedAttention Talk — Speaker Narration")
    lines.append("")
    lines.append(
        "Auto-generated from the `NARRATION` section of each scene's module "
        "docstring in `talk/sN_*.py`. Regenerate with `make narration` "
        "(or `python tools/build_narration.py`) after editing any scene. "
        "Do not hand-edit this file."
    )
    lines.append("")
    lines.append("## Contents")
    lines.append("")

    by_stem = {stem: (cls, act) for stem, cls, act in SCENES}

    for act_no, act_title, stems in ACTS:
        lines.append(f"### Act {act_no}: {act_title}")
        lines.append("")
        for stem in stems:
            cls, act = by_stem[stem]
            num = stem.split("_", 1)[0].upper()  # e.g. "S0"
            anchor = f"{num.lower()}-{cls.lower()}"
            lines.append(f"- [{num} {cls}](#{anchor})")
        lines.append("")

    for stem, cls, act in SCENES:
        doc = get_docstring(stem)
        narration = extract_narration(doc)
        num = stem.split("_", 1)[0].upper()
        lines.append(f"## {num} {cls}")
        lines.append("")
        lines.append(narration)
        lines.append("")

    OUT_PATH.write_text("\n".join(lines).rstrip("\n") + "\n")
    print(f"Wrote {OUT_PATH}")


if __name__ == "__main__":
    build()
