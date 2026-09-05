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

# (file stem, class name, act, target minutes) — from talk/CONTRACT.md's scene table.
SCENES = [
    ("s0_title", "S0Title", "I", "1-2"),
    ("s1_transformers", "S1Transformers", "I", "4-5"),
    ("s2_gpu", "S2GPU", "I", "3"),
    ("s3_kvcache", "S3KVCache", "I", "4"),
    ("s4_problem", "S4Problem", "I", "6-7"),
    ("s5_pagedattention", "S5PagedAttention", "II", "7-8"),
    ("s6_os_and_why_hard", "S6OSAndWhyHard", "II", "4-5"),
    ("s7_sharing", "S7Sharing", "III", "5"),
    ("s8_scheduling", "S8Scheduling", "III", "4"),
    ("s9_results", "S9Results", "III", "4-5"),
    ("s10_ablations", "S10Ablations", "III", "2"),
    ("s11_takeaways", "S11Takeaways", "III", "1-2"),
]

ACTS = [
    ("I", "Why memory is the bottleneck", ["s0_title", "s1_transformers", "s2_gpu", "s3_kvcache", "s4_problem"]),
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

    by_stem = {stem: (cls, act, minutes) for stem, cls, act, minutes in SCENES}

    for act_no, act_title, stems in ACTS:
        lines.append(f"### Act {act_no}: {act_title}")
        lines.append("")
        for stem in stems:
            cls, act, minutes = by_stem[stem]
            num = stem.split("_", 1)[0].upper()  # e.g. "S0"
            anchor = f"{num.lower()}-{cls.lower()}"
            lines.append(f"- [{num} {cls} (~{minutes} min)](#{anchor})")
        lines.append("")

    for stem, cls, act, minutes in SCENES:
        doc = get_docstring(stem)
        narration = extract_narration(doc)
        num = stem.split("_", 1)[0].upper()
        lines.append(f"## {num} {cls} (~{minutes} min)")
        lines.append("")
        lines.append(narration)
        lines.append("")

    OUT_PATH.write_text("\n".join(lines).rstrip("\n") + "\n")
    print(f"Wrote {OUT_PATH}")


if __name__ == "__main__":
    build()
