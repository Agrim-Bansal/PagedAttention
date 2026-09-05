# Build contract for the PagedAttention talk deck

Every worker (components author, scene authors, integrator) follows this file exactly.
Spec of *what* each scene shows: `../TALK_PLAN.md`. Paper facts: `../paper_notes/PAPER_REFERENCE.md`.

## Runtime

- Python venv at `../.venv`. Run everything with `../.venv/bin/manim-slides` / `../.venv/bin/python`.
- Manim Community Edition + manim-slides. **No LaTeX anywhere**: never use `Tex`, `MathTex`,
  `Title` (uses Tex), `BulletedList`, `Brace.get_tex`, or `Integer`/`DecimalNumber` with `Tex`
  (plain `Integer`/`DecimalNumber` are fine, they use Text). Formulas are `Text`/`MarkupText`.
- 16:9 (default manim frame: width 14.22, height 8, x in [-7.1, 7.1], y in [-4, 4]).
- Render from the **project root** so `talk` is importable:
  `cd /Users/agrim/Home/PagedAttention && .venv/bin/manim-slides render -ql talk/s5_pagedattention.py S5PagedAttention`
  (use `-qh` for the final). manim-slides writes to `media/` and `slides/`.

## Scene files

- One file per scene: `talk/sN_name.py`, exactly one class, named per the table below,
  subclassing `manim_slides.Slide`. Files must be independently renderable.
- `from talk.theme import *` and `from talk.components import *` at the top (both use
  explicit `__all__`).
- Start each scene with `self.camera.background_color = BG` (theme handles it via
  `apply_theme(self)`; call that first thing in `construct`).
- Beats: call `self.next_slide()` between beats. A "beat" is a pause point where the presenter
  talks. Aim for the beat counts in the table. Never end `construct` mid-animation: the last
  thing on screen must be a stable frame. Use `self.wait(0.3)` after the final animation.
- Use `self.next_slide(loop=True)` only for intentionally looping idle animations.
- Clean up: at the end of a scene, `self.play(FadeOut(*self.mobjects))` is fine but not
  required.
- Every scene file has a module docstring with a **NARRATION** section: for each beat,
  `Beat N — <title>`: 2–5 sentences the presenter can say, plus `[PAUSE]` markers where
  audience interaction happens. This is the speaker script and is compiled later into
  `NARRATION.md`.
- Keep text short. Max ~12 words per line on screen. Titles at top, `font_size=TITLE_SIZE`.
- Numbers on screen must match `PAPER_REFERENCE.md` / TALK_PLAN "Key numbers".
- Test: after writing, render with `-ql` and fix all errors. Then inspect at least the
  last-frame PNGs (`--save_last_frame` on plain manim won't apply; instead open the produced
  mp4 in `media/videos/...` with `ffmpeg -ss <t> -i file.mp4 -frames:v 1 out.png` to grab
  frames, or use `-ql -s` on `manim` for a single frame) and confirm nothing overflows the
  frame or overlaps illegibly.

| File | Class | Act | Target beats | Target minutes |
|---|---|---|---|---|
| s0_title.py | S0Title | I | 3–4 | 1–2 |
| s1_transformers.py | S1Transformers | I | 6–8 | 4–5 |
| s2_gpu.py | S2GPU | I | 4–5 | 3 |
| s3_kvcache.py | S3KVCache | I | 5–7 | 4 |
| s4_problem.py | S4Problem | I | 7–9 | 6–7 |
| s5_pagedattention.py | S5PagedAttention | II | 9–12 | 7–8 |
| s6_os_and_why_hard.py | S6OSAndWhyHard | II | 6–8 | 4–5 |
| s7_sharing.py | S7Sharing | III | 7–9 | 5 |
| s8_scheduling.py | S8Scheduling | III | 6–8 | 4 |
| s9_results.py | S9Results | III | 8–10 | 4–5 |
| s10_ablations.py | S10Ablations | III | 3–4 | 2 |
| s11_takeaways.py | S11Takeaways | III | 3–4 | 1–2 |

## theme.py (exports)

```python
BG = "#0f1117"          # dark background
FG = "#e6e6e6"          # default text
MUTED = "#8a8f98"       # secondary text
ACCENT = "#ff8c42"      # the single accent (orange)
ACCENT2 = "#4cc9f0"     # cool secondary (used sparingly for K vs V, or "vLLM" series)
GOOD = "#5ad175"        # used / good
BAD = "#ef476f"         # wasted / bad
WARN = "#ffd166"        # reserved / warning
BLOCK_FILL = "#1f2633"  # empty block slot fill
BLOCK_STROKE = "#3b4454"
K_COLOR = ACCENT2
V_COLOR = "#b388ff"
Q_COLOR = ACCENT
FONT = "Avenir Next"      # falls back to system sans if missing
TITLE_SIZE = 44
BODY_SIZE = 30
SMALL_SIZE = 22
TINY_SIZE = 16

def apply_theme(scene): ...        # sets background, Text default font/color
def title(text, **kw) -> Text      # top-of-frame title, positioned .to_edge(UP)
def body(text, **kw) -> Text       # body text default size/color
def small(text, **kw) -> Text
def caption(text, **kw) -> Text    # MUTED, TINY_SIZE
def act_checkpoint(scene, act_no, act_title, done=(), current="", upcoming=()):
    # full-screen "where we are" beat: shows Act I/II/III as three pills, highlights the
    # current one, lists done/current/upcoming as small text. Plays in, calls
    # scene.next_slide(), then fades out. Returns None.
```

## components.py (exports and signatures)

All components are `VGroup` subclasses or functions returning `VGroup`, no LaTeX, and expose
the sub-mobjects listed so scenes can animate them. Sizes are in scene units.

```python
class TokenBox(VGroup):
    """One token as a rounded box with the word inside.  .box, .label"""
    def __init__(self, word, color=FG, fill=BLOCK_FILL, width=None, height=0.6, font_size=SMALL_SIZE)

def token_sequence(words, gap=0.12, **kw) -> VGroup   # of TokenBox, arranged RIGHT

FOUR_SCORE = ["Four", "score", "and", "seven", "years", "ago", "our", "fathers", "brought", "forth"]

class KVBlock(VGroup):
    """Fixed-size KV block ("page") of `slots` slots drawn as a row of cells.
    .cells (VGroup of Squares/Rectangles), .labels (VGroup of Text, one per slot, may be empty
    strings), .index_label (Text under/over the block, e.g. 'Block 7'), .fill_slot(i, word,
    color=ACCENT) -> Animation, .clear_slot(i) -> Animation, .set_state(i, state) where state in
    {"empty","filled","reserved","internal","external"} -> Animation (colors: filled=ACCENT,
    reserved=WARN, internal=BAD, external=BAD hatched-ish/darker)."""
    def __init__(self, slots=4, cell=0.75, words=None, index=None, label_pos=DOWN)

class PhysicalMemGrid(VGroup):
    """A column/grid of KVBlocks representing physical GPU memory. .blocks (list of KVBlock),
    .block(i) -> KVBlock, .title (Text). Blocks labeled 0..n-1."""
    def __init__(self, n_blocks=8, slots=4, cols=1, cell=0.6, title="Physical KV blocks")

class BlockTable(VGroup):
    """Logical->physical table. Rows of (logical index, physical block number, #filled).
    .rows (list), .set_row(i, physical, filled) -> Animation, .add_row(physical, filled) ->
    Animation, .highlight_row(i) -> Animation, .title (Text)."""
    def __init__(self, n_rows=4, title="Block table", show_filled=True)

def arrow_map(src, dst, color=MUTED, **kw) -> Arrow   # thin arrow between mobjects (buff 0.1)

class RefCountBadge(VGroup):
    """Small circle with a number, attaches to a KVBlock corner. .set_value(n) -> Animation"""
    def __init__(self, value=1, color=ACCENT2)

class MemoryBar(VGroup):
    """Horizontal stacked bar. segments = [(label, fraction, color), ...]. .segs (VGroup),
    .labels (VGroup), .animate_in() -> Animation. width default 10."""
    def __init__(self, segments, width=10.0, height=0.8, show_pct=True)

class MemoryPie(VGroup):
    """Pie/donut with same segments spec (Fig 1 left). .sectors, .legend"""
    def __init__(self, segments, radius=1.8)

class GPUSchematic(VGroup):
    """Minimal GPU: a chip with a grid of many tiny 'core' squares, plus a VRAM bar beside it.
    .cores (VGroup), .vram (Rectangle), .vram_label, .fill_vram(fraction, color) -> Animation"""
    def __init__(self, cores=(8, 6), width=5.0)

def attention_diagram(tokens, query_index, kv_colors=True) -> VGroup
    """Q/K/V computation visual: row of TokenBoxes, K and V stacks under each, a Q above the
    query token, arrows from Q to each K, a softmax bar row, weighted sum arrow to output.
    Returns VGroup with .tokens, .keys, .values, .query, .arrows, .weights, .output so scenes
    can animate stepwise."""

def bar_chart(categories, series, y_label="", y_max=None, colors=None, width=8, height=4,
              value_labels=False) -> VGroup
    """Grouped bar chart. series = {"name": [values...]}. Native Rectangles on an Axes.
    .axes, .bars (dict name -> VGroup), .legend, .x_labels. Animate with
    scene.play(*[GrowFromEdge(b, DOWN) for b in ...]) or .animate_in()."""

def line_chart(x, series, x_label="", y_label="", x_range=None, y_range=None, colors=None,
               width=8, height=4, log_y=False, markers=True) -> VGroup
    """Multi-series line chart. .axes, .lines (dict), .dots (dict), .legend. .animate_in()."""
```

Charts and axes must use `Text` for labels (pass `axis_config={"include_numbers": False}` and add
`Text` tick labels manually, or use `Axes(..., x_axis_config={"numbers_to_include": ...})` ONLY
if you confirm numbers render with Text via `label_constructor=Text`; `Axes.get_x_axis_label`
default uses MathTex — always pass a `Text` mobject to it).

## Colors convention (keep consistent across scenes)

- Used KV slots: `ACCENT`. Reserved-but-unused: `WARN`. Internal fragmentation: `BAD`.
  External fragmentation: `BAD` at 50% opacity. Empty: `BLOCK_FILL`.
- vLLM series in charts: `ACCENT`. Baselines: `MUTED`, `ACCENT2`, `V_COLOR` (Orca variants).
- Keys: `K_COLOR`, Values: `V_COLOR`, Query: `Q_COLOR`.
- Request A: `ACCENT`, Request B: `ACCENT2` when two requests share a frame.
