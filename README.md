# PagedAttention Talk

A narrated Manim + [manim-slides](https://github.com/jeertmans/manim-slides) deck
explaining Kwon et al., *"Efficient Memory Management for Large Language Model
Serving with PagedAttention"* (SOSP '23) — the paper behind vLLM. Twelve scenes,
grouped into three acts: why memory (not compute) is the bottleneck in LLM
serving, the paper's core idea (page the KV cache like an OS pages memory), and
what it buys you (throughput, sharing, ablations).

## Setup

```
make setup
```

This creates `.venv` and installs `requirements.txt` (`manim==0.21.0`,
`manim-slides==5.6.0`). Equivalent manual steps:

```
python3 -m venv .venv
.venv/bin/pip install --upgrade pip
.venv/bin/pip install -r requirements.txt
```

Requirements:
- **ffmpeg** must be installed and on `PATH` (Manim shells out to it for
  encoding). On macOS: `brew install ffmpeg`.
- **No LaTeX is needed anywhere** — the deck uses Pango `Text`/`MarkupText`
  only, never `Tex`/`MathTex`.
- On macOS, the `pycairo`/`pangocairo` wheels that Manim depends on usually
  install fine from pip. If a build fails, install the system libraries first
  and retry: `brew install cairo pango pkg-config`.

## Rendering, presenting, exporting

All scenes live in `talk/`, one file each (`talk/sN_*.py`), each containing a
single `manim_slides.Slide` subclass. `talk/theme.py` and `talk/components.py`
hold the shared palette/fonts and reusable mobjects (token boxes, KV blocks,
block tables, charts, etc.) — see `talk/CONTRACT.md` for their exact API.

Render every scene's video (high quality, `-qh`, writes to `media/` and
`slides/`):

```
make render
```

Fast draft pass at low quality (`-ql`), useful while iterating:

```
make render-low
```

Live-present the whole deck in order (all 12 scenes, keyboard-driven, opens a
player window — needs `make render`/`render-low` to have produced slides
first):

```
make present
```

which runs:

```
.venv/bin/manim-slides present S0Title S1Transformers S2GPU S3KVCache S4Problem \
  S5PagedAttention S6OSAndWhyHard S7Sharing S8Scheduling S9Results S10Ablations S11Takeaways
```

Presenter keyboard shortcuts (verified against the installed manim-slides
5.6.0 default keymap, `manim_slides/config.py`):

| Key | Action |
|---|---|
| Right arrow | Next slide/beat |
| Space | Play / pause the current animation |
| Left arrow | Previous slide/beat |
| V | Reverse (play the current slide backwards) |
| R | Replay the current slide from the start |
| F | Toggle full screen |
| H | Hide / show mouse cursor |
| Q | Quit |

(`manim-slides present --help` documents the CLI flags, e.g. `--full-screen`,
`--start-at-scene-number`, `--skip-all` for a dry-run smoke test; it does not
print the keymap, which lives in `manim_slides/config.py`'s `Keys` model.)

Export a standalone reveal.js HTML backup (no player app needed, just a
browser) to `dist/pagedattention_talk.html`:

```
make html
```

which runs:

```
.venv/bin/manim-slides convert S0Title S1Transformers S2GPU S3KVCache S4Problem \
  S5PagedAttention S6OSAndWhyHard S7Sharing S8Scheduling S9Results S10Ablations S11Takeaways \
  dist/pagedattention_talk.html --to html \
  -cslide_number=true -ccontrols=true -cprogress=true -ctransition=none \
  -cwidth=1920 -cheight=1080
```

(`-c`/`--config` options are reveal.js settings; see
`manim-slides convert --to html --show-config` for the full list. The export
references its video assets from a sibling `pagedattention_talk_assets/`
folder, so keep the two together if you copy the HTML elsewhere; add
`--one-file` to embed everything in a single, larger HTML file instead.)

### Rendering / editing a single scene

Each scene file is independently renderable — useful while editing one scene
without waiting for all twelve:

```
.venv/bin/manim-slides render -ql talk/s5_pagedattention.py S5PagedAttention   # fast draft
.venv/bin/manim-slides render -qh talk/s5_pagedattention.py S5PagedAttention   # final quality
.venv/bin/manim-slides present S5PagedAttention                                # preview it alone
```

Always render from the project root (`talk` must be importable as a package).

## File layout

```
talk/
  theme.py               # palette, fonts, sizes, apply_theme(), act_checkpoint()
  components.py          # reusable mobjects: TokenBox, KVBlock, BlockTable, charts, ...
  s0_title.py .. s11_takeaways.py   # one Slide subclass per scene, in talk order
  scratch_components.py  # exercises every component, for smoke-testing components.py
  CONTRACT.md            # authoritative API/spec contract for theme/components/scenes
tools/
  build_narration.py     # regenerates NARRATION.md from scene docstrings
Makefile                 # setup / render / render-low / present / html / narration / clean
requirements.txt
NARRATION.md             # generated speaker script — see below
TALK_PLAN.md             # what each scene shows, build plan, key numbers
PLANNING_NOTES.md        # planning/process notes
paper_notes/PAPER_REFERENCE.md   # extracted facts/figures from the paper
vllm.pdf                 # the source paper
```

## Speaker script

`NARRATION.md` is the full speaker script: one section per scene, each beat's
talking points, generated from the `NARRATION` section of that scene's module
docstring. Regenerate it after editing any scene's narration:

```
make narration
```

which runs `tools/build_narration.py` — a small, dependency-light script (only
`ast`/`re`, no manim import) that reads each `talk/sN_*.py`'s docstring and
writes the combined script with a per-act table of contents and target timing
pulled from `talk/CONTRACT.md`'s scene table.

## Cleaning up

```
make clean
```

removes `media/`, `slides/`, `dist/`, and `__pycache__` directories (rendered
video/slide artifacts and the HTML export — everything `make render`/`make
html` produce; source files are untouched).

## Further reading

- `talk/CONTRACT.md` — the binding spec for `theme.py`/`components.py`'s API
  and per-scene structure (beat counts, target minutes, color conventions).
- `TALK_PLAN.md` — what each scene covers, the paper-figure-to-scene mapping,
  and the key numbers that must stay consistent across scenes.
- `PLANNING_NOTES.md` — process/planning notes behind the deck.
- `paper_notes/PAPER_REFERENCE.md` — facts and figures extracted from
  `vllm.pdf`, used as the source of truth for every number shown on screen.
