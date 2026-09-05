# Build status — PagedAttention talk deck

Last updated: 2026-09-05 (after subagent quota exhaustion; see "Blockers").
Sources of truth: `TALK_PLAN.md` (spec), `PLANNING_NOTES.md` (decisions), `vllm.pdf` (paper).
This file is the resume point for the remaining QA/fix pass.

## Current status (one paragraph)

The full deck exists and renders: 12 manim-slides scenes (S0–S11, 99 slides total), all 18
paper figures recreated natively, no LaTeX, Avenir Next font, dark theme. A complete
1080p60 render, the HTML backup export (`dist/pagedattention_talk.html`), the compiled
speaker script (`NARRATION.md`), `Makefile`, and `README.md` were all produced at ~01:20.
A content review then listed a handful of factual/consistency fixes and narration trims.
The three QA/fix subagents launched for that pass died immediately on the Sonnet API
usage limit. Two of their edits did land (S0 narration, S5 crossfade) but those two files
have NOT been re-rendered since, so `slides/`, `dist/` and `NARRATION.md` are stale for
S0 and S5. Everything else in the pending list below is still open.

## Work done

- **Environment**: `.venv` (Python 3.13, manim 0.21.0, manim-slides 5.6.0, ffmpeg 9.0.1),
  `requirements.txt`, `smoke/smoke_test.py`.
- **Paper notes**: `paper_notes/PAPER_REFERENCE.md` — every figure (1–19), key numbers,
  section summaries, mechanism walkthroughs used by all scene writers.
- **Build contract**: `talk/CONTRACT.md` (runtime rules, scene table, theme constants,
  component API, color conventions) and `talk/SCENE_BRIEF.md` (writer instructions + QA steps).
- **Shared code**: `talk/theme.py` (colors, sizes, font resolution with fallbacks, text
  helpers, `act_checkpoint`), `talk/components.py` (TokenBox, KVBlock, PhysicalMemGrid,
  BlockTable, RefCountBadge, MemoryBar, MemoryPie, GPUSchematic, attention_diagram,
  bar_chart, line_chart), `talk/scratch_components.py` exerciser.
- **Scenes** (file / class / slides / figures):
  | File | Class | Slides | Figures |
  |---|---|---|---|
  | s0_title.py | S0Title | 4 | – |
  | s1_transformers.py | S1Transformers | 7 | – |
  | s2_gpu.py | S2GPU | 6 | – |
  | s3_kvcache.py | S3KVCache | 7 | – |
  | s4_problem.py | S4Problem | 10 | 2, 3 |
  | s5_pagedattention.py | S5PagedAttention | 10 | 5, 6, 7 |
  | s6_os_and_why_hard.py | S6OSAndWhyHard | 11 | – |
  | s7_sharing.py | S7Sharing | 14 | 8, 9, 10, 15 |
  | s8_scheduling.py | S8Scheduling | 9 | 4, 19 |
  | s9_results.py | S9Results | 11 | 11, 12, 13, 2, 14, 15, 16, 17 |
  | s10_ablations.py | S10Ablations | 4 | 18 |
  | s11_takeaways.py | S11Takeaways | 6 | – |
  Each file has a module docstring with a per-beat NARRATION section.
- **Tooling**: `Makefile` (render, render-low, present, html, narration, clean, setup, all),
  `tools/build_narration.py` → `NARRATION.md`, `tools/last_frames.py <Class>` → last frame
  of every slide as PNG under `/tmp/qa/<Class>/` for visual QA.
- **Outputs**: `media/videos/*/1080p60/`, `slides/<Class>.json` + `slides/files/`,
  `dist/pagedattention_talk.html` + 26 MB of assets, `NARRATION.md` (694 lines), `README.md`.
- **Bugs fixed along the way**: `-qh` misparsed by manim-slides (use `-q h`); raw string
  background color crashes `_save_slides` (use `ManimColor`); MemoryBar small-segment label
  squish (`show_pct=False` + captions); KVBlock label overflow (auto-shrink in components);
  S1 leaked placeholder boxes; S3 arrange clobbering positions; S7 orphan Transform and
  missing axes; S8 right-edge overflow.
- **Font change** requested by user: done, `FONT = "Avenir Next"` with fallback chain.
- **Landed from the aborted QA pass** (verified by grep, 2026-09-05 01:24):
  - S0 Beat 4 narration now foreshadows the mechanism ("chop the KV cache into small
    fixed-size blocks and manage them on demand"); no OS/paging words remain in S0 narration.
  - S5 adds `_fill_slot_clean` (FadeOut/FadeIn crossfade instead of Transform) to stop
    transient label crowding in the Fig 6 walkthrough.

## TODO (in order)

Content fixes (all verified still open by grep):
1. **S9** `talk/s9_results.py`: delete the invented `"FasterTransformer": [7.0, 7.0]`
   series in chart4 (Fig 13 only has Orca variants + vLLM), line ~284. Narration line ~39:
   "versus FasterTransformer's 7" → "versus Orca (Max)'s 7" (and the ShareGPT counterpart).
2. **S7** `talk/s7_sharing.py` line 106: prompt `["Four","score","and","seven"]` →
   `FOUR_SCORE[:7]` so the prompt matches the running example (line 153 block words are fine
   as they are the first block).
3. **S3** `talk/s3_kvcache.py` line 19: "Here it is on our running example" → "Back to our
   example".
4. **S10** `talk/s10_ablations.py` line 121: "tokens per page" → "tokens per block".
5. **S0** `talk/s0_title.py` ~line 208: "Act III" label overflows its 0.55-radius circle
   (seen in `/tmp/qa/S0Title/slide_03.png`). Make all three pills identical: e.g. radius 0.7,
   or `label.scale_to_fit_width(0.8 * circle.width)` when too wide.

Narration trims (current word count → target; total should land near 40 min):
| Scene | Now | Target |
|---|---|---|
| S0 | 229 | ≤200 |
| S2 | 414 | ≤330 |
| S4 | 758 | keep (interactive derive) |
| S5 | 834 | keep (core scene) |
| S6 | 569 | ≤480 |
| S7 | 569 | ≤500 |
| S8 | 675 | ≤430 |
| S9 | 730 | ≤520 |
| S10 | 248 | ≤220 |
| S11 | 243 | ≤200 |

Visual QA:
6. Run `.venv/bin/python tools/last_frames.py <Class>` for all 12 classes; inspect every
   PNG for overflow, overlap, leftovers. Known so far: only the S0 Act III pill.
7. Spot-check S5 mid-animation frames of the Fig 6 beats after re-render to confirm the
   crossfade fix.

Rebuild:
8. Re-render every changed scene:
   `.venv/bin/manim-slides render -q h talk/<file>.py <Class>` (at minimum S0 and S5 now;
   plus S3, S7, S9, S10 and any trimmed scene after the fixes above).
9. `make narration` then `make html` to regenerate `NARRATION.md` and `dist/`.
10. Final recap to the user: deliverables, how to present (`make present`, keys: Right/Left
    navigate, Space pause, R replay, F fullscreen, Q quit), what was fixed, what was left.

## Blockers

- Sonnet subagents: "API Error: 400 You have reached your specified API usage limits.
  You will regain access on 2026-10-01 at 00:00 UTC". Remaining work must be done directly
  in the main session (or with a different model). None of it requires subagents.

## Useful commands

```
cd /Users/agrim/Home/PagedAttention
.venv/bin/manim-slides render -q h talk/s0_title.py S0Title     # one scene, HQ
.venv/bin/manim-slides render -q l talk/s0_title.py S0Title     # fast preview
.venv/bin/python tools/last_frames.py S0Title                   # frames → /tmp/qa/S0Title/
make narration && make html                                     # regenerate script + HTML
make present                                                    # live presentation
```
