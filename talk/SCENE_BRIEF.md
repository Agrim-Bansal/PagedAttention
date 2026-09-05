# Brief for scene writers

You are writing one or two scenes of a narrated Manim animation for a ~40-minute CS-club talk
on the vLLM / PagedAttention paper (Kwon et al., SOSP '23). Read, in this order:

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
- The module docstring must contain the NARRATION section (per beat, 2–5 spoken sentences,
  `[PAUSE]` markers for audience interaction). Write it in a natural speaking voice for a
  presenter addressing CS juniors + peers. Include the paper's exact numbers where relevant.
- Known cosmetic: `MemoryBar` labels for segments under ~10% get squished; for such segments
  pass `show_pct=False` or add your own `caption` text beside the bar.
- `BlockTable.highlight_row` uses Indicate which may not fully revert color; follow with an
  explicit color reset if you need to.

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
