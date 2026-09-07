# PagedAttention

A live, narrated [Manim](https://www.manim.community/) + [manim-slides](https://github.com/jeertmans/manim-slides) talk on Kwon et al., *"Efficient Memory Management for Large Language Model Serving with PagedAttention"* ([SOSP '23](https://dl.acm.org/doi/10.1145/3600006.3613165)) — the paper behind [vLLM](https://github.com/vllm-project/vllm).

The bottleneck in LLM serving is leftover GPU memory, not FLOPs. Existing systems store each request's KV cache as one contiguous slab reserved to the model's maximum length, so only 20–38% of that memory holds real tokens. vLLM chops the cache into fixed-size blocks, maps them through a block table, and allocates on demand. Waste falls below one block per request; utilization hits **96.3%**; throughput is **2–4×** Orca at the same latency (up to **22×** FasterTransformer).

Twelve scenes, three acts, every paper figure recreated as native animation (not screenshots). You talk over the resting frames; the deck advances on a keypress.

**Watch:** [projects.agrimbansal.com/PagedAttention](https://projects.agrimbansal.com/PagedAttention/)

## The talk

One thread: leftover VRAM decides batch size, which decides throughput. The Lincoln prompt *"Four score and seven years ago our fathers brought forth"* is the running example through the KV-cache, problem, and mechanism scenes (the paper uses it in Figs. 3, 6, and 7).

### Act I — Why memory is the bottleneck

| # | Scene | What it shows | Figures |
|---|-------|---------------|---------|
| S0 | Title | Cost hook (~10× a keyword search) and three-act roadmap | — |
| S1 | GPU | CPU vs GPU, VRAM vs DRAM, weights persist (~26 GB of 40), leftover VRAM = max batch | — |
| S2 | Transformers | Closed box emits one word; open it for Key / Value / Query and softmax attention; prefill vs decode | Eqs. 1–3 |
| S3 | KV cache | Recompute vs keep; 800 KB/token × 2048 = 1.6 GB reserved; ~7 max-length requests in 12 GB | Fig. 1 left |
| S4 | The problem | Contiguous pre-allocation; reserved / internal / external waste; sharing is impossible | Figs. 2, 3 |

### Act II — Page the KV cache

| # | Scene | What it shows | Figures |
|---|-------|---------------|---------|
| S5 | PagedAttention | Change the *reader* first (Eq. 3 → 4, Fig. 5 live), then the manager (block table, on-demand blocks). Waste under one block; 96.3% | Figs. 5–7, 2 |
| S6 | OS analogy | This *is* virtual memory — then why it is not a free port (no GPU MMU; fused kernel; 20–26% slower, still 2–4× end-to-end) | — |

### Act III — What it buys you

| # | Scene | What it shows | Figures |
|---|-------|---------------|---------|
| S7 | Sharing | Copy-on-write parallel sampling, beam-search tree, shared prefix. Three primitives: `fork` / `append` / `free` | Figs. 8–10 |
| S8 | Scheduling | Overcommit, FCFS + preempt newest, all-or-nothing eviction, swap vs drop-and-prefill | Fig. 4 |
| S9 | Results | Paper §6 in order: ShareGPT / Alpaca, latency vs rate, batch size, sharing, translation, chatbot | Figs. 1 right, 11–17 |
| S10 | Ablations | Kernel overhead 20–26%; block-size sweet spot **16** | Fig. 18 |
| S11 | Takeaways | Memory, not a new model. *What else could you page?* | — |

No target duration — take the time the material needs. Math is shown as styled text, not LaTeX.

## Setup

```sh
make setup
```

Creates `.venv` and installs `requirements.txt` (`manim==0.21.0`, `manim-slides==5.6.0`). Manual equivalent:

```sh
python3 -m venv .venv
.venv/bin/pip install --upgrade pip
.venv/bin/pip install -r requirements.txt
```

Requirements:

- **ffmpeg** on `PATH` (Manim shells out to it). macOS: `brew install ffmpeg`.
- **No LaTeX.** The deck uses Pango `Text` / `MarkupText` only — never `Tex` / `MathTex`.
- On macOS, the `pycairo` / `pangocairo` wheels usually install from pip. If a build fails: `brew install cairo pango pkg-config`, then retry.

## Render and present

Run everything from the repository root (`talk` must be importable as a package).

```sh
make render-low    # draft, 480p — iterate here
make render        # final, 1080p
make present       # live player (needs a render first)
```

`make present` fits the Qt window to the current screen. High-quality renders are 1920×1080; manim-slides otherwise opens that size 1:1 (too large for a laptop, and not resizable). True full screen:

```sh
make present PRESENT_ARGS=-F
```

| Key | Action |
|-----|--------|
| Right | Next beat |
| Left | Previous beat |
| Space | Play / pause |
| R | Replay current beat |
| V | Reverse the current beat |
| F | Toggle full screen |
| H | Hide / show cursor |
| Q | Quit |

Start at a later scene with `--start-at-scene-number N`. `manim-slides present --help` lists CLI flags; the keymap lives in `manim_slides/config.py`.

### One scene while editing

```sh
make render-scene FILE=talk/s5_pagedattention.py SCENE=S5PagedAttention QUALITY=l
.venv/bin/manim-slides present S5PagedAttention
```

Use `QUALITY=h` for the 1080p pass.

### Check and inspect

```sh
make verify    # compile talk/ + tools/, regenerate NARRATION.md
make qa        # resting frame of every slide → /tmp/qa/<SceneClass>/
make help      # all targets
```

The QA PNGs are the frames that stay on screen while the presenter talks.

## Export and publish

```sh
make html      # reveal.js backup → dist/pagedattention_talk.html
make video     # one MP4 of every beat → dist/pagedattention.mp4
make pages     # copy dist/ onto gh-pages and push
make all       # render + narration + html
```

Keep `pagedattention_talk.html` next to `pagedattention_talk_assets/` if you copy the export. Add `--one-file` to the convert command to embed everything in a single (larger) HTML file.

`make video` stitches the rendered clips (no live-talk pause holds, silent audio) for YouTube. Render first (`make render`). Upload `dist/pagedattention.mp4` at [youtube.com/upload](https://www.youtube.com/upload).

`make pages` writes `index.html` onto the `gh-pages` branch without leaving `master`. Rebuild with `make html` first if the videos changed.

## Speaker script

`NARRATION.md` is generated from the `NARRATION` section of each scene's module docstring. Do not hand-edit it. After changing spoken text in a `talk/sN_*.py` docstring:

```sh
make narration
```

`[PAUSE]` markers in the script are audience-interaction beats.

## Layout

```
talk/
  s0_title.py … s11_takeaways.py   one Slide subclass per scene, talk order
  theme.py                         palette, fonts, act_checkpoint()
  components.py                    TokenBox, KVBlock, BlockTable, charts, …
  scratch_components.py            exercises every shared mobject
  CONTRACT.md                      binding API / beat-count / color spec
  SCENE_BRIEF.md                   writer + visual-QA notes
tools/
  present.py                       screen-fitting wrapper around manim-slides
  build_narration.py               docstring → NARRATION.md
  last_frames.py                   resting-frame PNGs for visual QA
  build_video.py                   beat clips → dist/pagedattention.mp4
  publish_pages.py                 dist/ → gh-pages
paper_notes/PAPER_REFERENCE.md     extracted facts and figures from the paper
TALK_PLAN.md                       executable narrative spec
NARRATION.md                       generated speaker script
RUNNING.md                         shorter command cheat-sheet
vllm.pdf                           the source paper
```

## Key numbers

Kept consistent across scenes; source of truth is `paper_notes/PAPER_REFERENCE.md`.

| Fact | Value |
|------|--------|
| OPT-13B on A100-40GB | ~65% weights (26 GB), &gt;30% KV cache (12 GB) |
| KV cache per token (OPT-13B) | 800 KB → 1.6 GB at 2048 tokens |
| Token-state utilization | existing 20.4–38.2%; vLLM **96.3%** |
| Throughput vs Orca | **2–4×** at the same latency |
| vs FasterTransformer | up to **22×** request rate |
| Concurrent batch (OPT-13B) | 7 → 30 (ShareGPT), 7 → 132 (Alpaca) |
| Default block size | **16** tokens |
| PagedAttention kernel | ~20–26% slower than FasterTransformer; net still 2–4× |

## Clean

```sh
make clean
```

Removes `media/`, `slides/`, `dist/`, and `__pycache__`. Source files are untouched.

## Further reading

- [`RUNNING.md`](RUNNING.md) — command cheat-sheet
- [`TALK_PLAN.md`](TALK_PLAN.md) — what each scene covers and the figure-to-scene map
- [`talk/CONTRACT.md`](talk/CONTRACT.md) — theme / component API and per-scene beat counts
- [`paper_notes/PAPER_REFERENCE.md`](paper_notes/PAPER_REFERENCE.md) — numbers and figures extracted from `vllm.pdf`
- [`PLANNING_NOTES.md`](PLANNING_NOTES.md) — decisions behind the deck
