# Brief for scene writers

You are writing one or two scenes of a narrated Manim animation for a CS-club talk
on the vLLM / PagedAttention paper (Kwon et al., SOSP '23). There is no target
duration — take the time the material needs; do not compress or rush. Read, in this order:

1. `talk/CONTRACT.md` — runtime rules, file/class naming, beat counts, theme + component API.
2. `TALK_PLAN.md` — the narrative spec. Your scene's paragraph under "Narrative + scene
   contents" is your requirement. Also read "Narrative principles" and "Key numbers".
3. `paper_notes/PAPER_REFERENCE.md` — the figures you must recreate (only the sections for
   your figures + key numbers), so the on-screen content is faithful to the paper.
4. `talk/theme.py` and `talk/components.py` — the real API. Read them fully; use the
   components rather than re-inventing boxes/tables/charts. You MAY add small private helpers
   inside your own scene file, but do NOT edit theme.py / components.py / CONTRACT.md /
   TALK_PLAN.md / PLANNING_NOTES.md / paper_notes / other scenes. If a component is missing a
   capability you need, work around it locally in your file and mention it in your report.

Hard rules:
- No LaTeX (no Tex/MathTex/Title/BulletedList). Text/MarkupText only.
- Class name, file name and beat count per the table in CONTRACT.md.
- `apply_theme(self)` first line of `construct`. `self.next_slide()` between beats.
- Every beat must leave a stable, legible, in-frame final frame (this is what the presenter
  stands on while talking). Keep on-screen text terse; the narration carries the words.
- The module docstring must contain the NARRATION section (per beat, as many spoken
  sentences as the idea needs — do not compress for time; `[PAUSE]` markers for
  audience interaction). Write it in a natural speaking voice for a presenter addressing
  CS juniors + peers. Include the paper's exact numbers where relevant.
- Known cosmetic: `MemoryBar` labels for segments under ~10% get squished; for such segments
  pass `show_pct=False` or add your own `caption` text beside the bar.
- `BlockTable.highlight_row` uses Indicate which may not fully revert color; follow with an
  explicit color reset if you need to.
- **`AnimationGroup` z-order trap.** `AnimationGroup(cell.animate…, FadeIn(label))` wraps the
  animated cells in a fresh `Group`; `Scene.play` adds that Group top-level, which pulls the
  cells *above* their sibling labels — the labels then render underneath and vanish. Always
  pass `group=<the parent already in the scene>` to `AnimationGroup` when its sub-animations
  target children of an on-screen VGroup (see `_fill_slot` in `s5_pagedattention.py`).
- **Text baseline trap.** `Text` centres on the glyph bounding box, so `move_to(cell)` puts
  "and" (ascender), "ago" (descender) and "score" (x-height only) at three different
  heights — a row of words looks ragged. Place short labels by baseline: measure the word
  inside an `"Ág" + word + "Ág"` probe and shift by the offset (`_tx` / `_center` in
  `s5_pagedattention.py`). Same reason: never `arrange(..., aligned_edge=LEFT)` rows whose
  left-most item is a digit — `1` is narrower than `0` and the rows shift; lay out on a grid.
- **Uniform label size.** Do not shrink-to-fit each word separately (`scale_to_fit_width`
  per label): "fathers" ends up smaller than "our". Pick one font size per cell width that
  fits the longest word (`_word_fs`) and use it for every cell in the row.

Verification (mandatory before you report):
1. `cd /Users/agrim/Home/PagedAttention && .venv/bin/manim-slides render -ql talk/<file>.py <Class>`
   must exit 0.
2. Extract frames from the produced mp4 in `media/videos/<file>/480p15/<Class>.mp4` at
   several timestamps spanning the whole video (`ffmpeg -loglevel error -y -ss <t> -i <mp4>
   -frames:v 1 /tmp/<Class>_<t>.png`) — at least one frame per beat, biased to the END of each
   beat (the resting frame). View each with the Read tool. Fix any overflow past the frame
   edges, overlapping text, unreadably small text (below ~TINY_SIZE at 480p is a warning
   sign), or leftover mobjects from earlier beats that shouldn't be there. Re-render and
   re-check until clean.
3. Confirm the slide count in `slides/<Class>.json` matches your intended beats.

Report back (concise): file + class, number of beats, total video seconds, which paper figures
you recreated, any deviations from TALK_PLAN, any component limitations you hit, and which
frames you inspected. Do not paste code.
