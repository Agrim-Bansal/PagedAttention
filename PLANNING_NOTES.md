# Planning notes — how & what we reached

> Decision log for the PagedAttention/vLLM club talk. The executable spec lives in
> `TALK_PLAN.md`; this file records the *requirements gathered* and the *decisions with
> their rationale*, so the reasoning survives chat compaction.

## Requirements gathered

- **Task:** present the vLLM paper (PagedAttention, Kwon et al., SOSP '23) to a CS
  **club** — juniors + peers.
- **Duration:** none. Take the time the material needs; do not compress or rush.
- **Goal / depth:** introduce prerequisites (GPU, attention, why memory is the way it
  is) *only enough to see the relevance and connection*, then lead into motivation;
  spend the bulk on **intuition + mechanism**. **Math is optional.**
- **Deliverable:** a **graphical Manim animation** the presenter **narrates over live**
  (not a static slide deck, not a pre-recorded video).

## Decisions & rationale

1. **Deliverable = Manim CE + manim-slides.** manim-slides pauses on the last frame of
   each beat and advances on keypress → lets the presenter pace to their speech and
   pause for audience interaction. Plain rendered video can't be paced live.
2. **No LaTeX dependency.** Math is optional, so use Pango `Text`/`MarkupText`; avoids
   installing MacTeX/BasicTeX. Can add `MathTex` later if crisp equations are wanted.
3. **One continuous narrative thread**, ordered: *why GPUs are useful* → attention
   & how transformers generate → KV cache → memory → problem → solution. Segues must
   be smooth; the audience likely doesn't know memory architecture, so it's introduced
   only as needed.
4. **OS-paging analogy moved to AFTER the mechanism (S6), not before.** Present
   PagedAttention on its own terms first, *then* reveal "…this is what an OS does." The
   analogy lands harder as a payoff than as scaffolding.
5. **S4 follows paper §3 / §3.1.** S3 still lands on
   "how do you allocate memory for something whose final size is unknown?" S4
   pauses, then reveals the paper’s answer — contiguous tensors because frameworks
   require it, statically pre-allocated to max length — not an invented grow-and-copy
   / linked-list fork. Unhurried, ~20 beats, visual wastes: unused yellow becomes
   empty slots then internal fragmentation; external is a request that does not
   fit. Fig 2 bars land one system at a time; sharing is a second failure, not a
   fourth waste. No compaction beat. Fig 3 uses the paper’s tokens and max lengths
   (A 2048, B 512) and the paper’s timing distinction among reserved / internal /
   external. Fig 2’s Oracle bar is the numerical form of "even if the actual
   length is known a priori."
6. **Novelty is the paper’s tensor contrast, not the word "paging."** Unlike tensors
   in traditional deep learning workloads, KV "dynamically grows and shrinks" and
   "its lifetime and length are not known a priori." Paging / virtual memory stay
   in S6. Sharing stays in S4 as the paper’s **second** inefficiency (12% prompt,
   up to 55% beam search), not as a fourth waste type.
7. **"Why it's not trivial" beat (S6).** First half is the OS reveal on the S5
   picture (relabel in place, demand paging shown — the original arc, with
   in-frame visuals). Then properties, not CUDA internals: no GPU hardware MMU
   (software translation); the attention kernel had to be rewritten to gather
   scattered blocks in one fused pass (an OS pager never touches computation);
   accessed **every token every step** (not a rare page fault) so the kernel is
   20–26% slower, while end-to-end is still 2–4×. Domain policies stay a
   foreshadow (S8/S10). Fig 18(a) stays in S10 as the measurement. No
   warp-per-block / coalesced / per-kernel walkthrough. The presenter
   confirmed the "not a free port" framing; the rewrite keeps that without
   turning S6 into a kernel-implementation lecture.
8. **Keep ALL mechanical details — they are the payoff, not trim candidates:**
   copy-on-write + reference counts, all three sharing cases (parallel sampling, beam
   search, shared prefix), and eviction recovery (swap-to-CPU vs recompute) each get
   real coverage. (Earlier suggestion to trim the system/back half was rejected.)
9. **Q/K/V with the paper’s math (Eqs. 1–3), shown as a transformer box.** The
   persistent S2 visual is an opaque centered transformer: sequence in, a new token
   out, joins, recenters, repeats. Open the box and introduce Eq. 2 **one vector at
   a time** (Key, then Value, then Query) so someone who has never heard
   “transformer” can follow; then softmax attention (Eq. 3, including
   `q^T k / sqrt(d)`). Close it for the growing-KV loop and for prefill vs decode.
   Still no Eq. 4 / multi-head / FFN — those are not load-bearing for PagedAttention.
   “KV cache” as a named object stays in S3.
10. **[A] Recurring concrete example.** Thread the paper's *"Four score and seven years
    ago our fathers…"* tokens through S3 → S4 → S5. The paper itself reuses it across
    Fig 3/6/7, so it ties the KV-cache, problem, and mechanism scenes together on
    familiar tokens. (Adopted.)
11. **[B] Three-act framing + light checkpoints.** Act I "why memory is the
    bottleneck", Act II "the idea: page the KV cache", Act III "the payoffs", with a
    small "where we are" beat at each seam so the talk stays navigable. (Adopted.)
12. **Every paper figure is animated / run through** (all ~18 figures), recreated
    natively as mobjects rather than screenshotted. The presenter explicitly wanted
    *all* the paper's examples shown and run through for comprehension. Added a
    dedicated **ablations scene (S10, Fig 18)** and dataset distributions (Fig 11) to
    honor this.
13. **Defaults kept without extra ceremony:** a cost hook in S0 (LLM request ~10× a
    keyword search; >30% of GPU memory is mostly-wasted KV cache) and a discussion
    prompt to close S11 ("what else could you page?" → prefix caching, LoRA adapters).
14. **S1 is a front-loadable GPU/VRAM primer, not a recap of generation.** Depth:
    sequential-vs-parallel cores, separate CPU DRAM vs GPU VRAM (PCIe is slow),
    weights persist (~26 GB of 40 GB), one request underuses the machine, batching
    as **one hub on W**, leftover VRAM = max batch = throughput. No transformer /
    attention / decode / KV. File order is GPU → transformers → KV cache.
    (The old “same math, then VRAM bar” sketch was too thin.)
15. **S3 is a deep stitch scene, not a name-and-size dump.** S2 lands
    "the growing K/V bundle sits in memory"; S1 lands "leftover VRAM is the
    budget." S3 names that bundle, shows recompute vs cache on the same
    tokens, prefill-write / decode-append, read-all/write-one, ×40 layers,
    factor-by-factor 800 KB, then unknown length → reserve the 2048-slot max
    (unused tail cut off from other requests) → 1.6 GB spoken for, Fig 1 left,
    and ~7 max-length requests in 12 GB — then earns S4's allocation question
    (including that K/V is position-dependent, not a dictionary of words).
    Still no Eq. 4, paging, named fragmentation, or Fig 1-right.
    Target 10–12 beats.

16. **S5 rewrite follows paper §4.1–4.3, not an invented walkthrough.** Fig 6
    prompt is 7 tokens (`"Four score and seven years ago our"`); ① prefill leaves
    one reserved slot in logical 1 (physical 1, filled 3/4); ② stores `"fathers"`
    there (3→4) with no new allocation; ③ allocates physical 3 for `"brought"`.
    Fig 5 is a later snapshot (query `"forth"`) used for Eq. 4, before the Fig 6
    rewind. Algorithm (kernel can fetch scattered blocks) before manager (so we
    may scatter them). Fig 7 places Request B at physical 5 and 2. No OS /
    paging vocabulary — that remains S6.
17. **S5 second rewrite (2026-09-05): depth + natural progression, 18 beats.**
    The 12-beat version was judged not good enough: mechanism diagrams used ~30%
    of the frame, Eq. 4 was a formula string with no computation shown, no
    bridge from S4's slab to "why chop", and no closure back to Fig 5. Changes:
    - **Opens on the paper's implicit "why this order":** the slab is contiguous
      because the attention *operator* reads K, V as one tensor each — the reader
      dictates the layout — so the paper changes the reader (§4.1) before the
      layout (§4.2). This is the pivot the whole scene hangs on.
    - **Eq. 3 → Eq. 4 shown as a regrouping of sums**, then **Fig 5 run live**
      with running softmax denominator + value-weighted numerator, block by block
      (illustrative numbers, labelled), partial block read via fill count, divide
      → o. The kernel's cost (a lookup per block) and payoff are stated here.
    - **Fig 6 gets one extra decode (`"forth"`)** so physical 1, 3, 7 visibly
      *is* the Fig 5 picture — the algorithm reads what the manager leaves behind.
    - **Fig 2's `?` from S4 resolves to 96.3% inside S5** (S4 strip vs vLLM strip
      side by side first), rather than only in S9.
    - **One bridge beat:** one engine iteration (§4.3 global procedure; select →
      allocate → concatenate → one forward pass; leads to S8). A "block size as
      a dial" beat was drafted and cut (2026-09-06): S10 owns that discussion.
      Final count 17 beats.
    - **Visuals use the full frame:** bigger cells, table→physical arrows per
      mapping, Request B has its own block table (Fig 7), A-trio centred until B
      arrives then slides left. Manager cells' word labels were invisible in the
      old render (an `AnimationGroup` z-order bug, fixed; documented in
      `talk/SCENE_BRIEF.md`).
    - **Visual-polish pass (2026-09-06)** after review ("misaligned, uneven,
      overlapping; font sizes uneven"): every word in a cell now uses one font
      size per cell width (sized so `"brought"` fits) instead of per-word
      shrink-to-fit; all cell/table/index text is placed by *baseline* (an
      `"Ág…Ág"` probe measures the offset; `_tx` / `_center` in the scene) so
      "and" / "ago" / "score" sit level; manager stacks and block tables are laid
      out on an explicit grid rather than `arrange(aligned_edge=LEFT)` over
      variable-width digits; Fig 5 scores sit above the fetch ring; Beat 2's
      block-2 ring no longer runs off the frame. Documented in `talk/SCENE_BRIEF.md`.

18. **S7 rewrite (2026-09-06): mechanism scene, not a chart dump.** The first
    S7 was eight conceptual beats stretched across 14 slides, with a wipe
    between every case, a TokenBox cartoon for Fig 9, a toy prefix, a stolen
    Fig 15 bar chart, and no A2 write-in-place. Rewrite:
    - **Pickup is S4’s unpaid bill on the S5 picture**, not a fresh product
      sketch and not a restart of the S6 OS lecture. After copy-on-write
      lands, one line: this is OS `fork`.
    - **Fig 8 is run through** on one physical column (Lincoln prompt, phys
      7 and 1 shared, COW into phys 3, then A2 writes in place). Packed
      blocks stay shared forever; only the last not-yet-full block is copied.
    - **Fig 9 is a k=4 block tree** (shared trunk, one private branch, prune
      to free pool). **Fig 10** uses the paper’s few-shot translation
      sentences, then dissolves into tables pointing at cached prefix blocks.
    - **Measured savings stay in S9** (Fig 15/16). S7 speaks “up to 55%, we
      measure it later.” Landing is `fork` / `append` / `free` (paper §5.2),
      not a slogan slide. Target 10–12 beats.
19. **S8 rewrite (2026-09-06): pressure story, not a figure checklist.** The
    first S8 opened on Fig 4 (org chart), then faded to a new titled slide every
    beat — FCFS as labels, swap as allocator boxes, Fig 19 as invented ms
    values. That is a catalog, not a story. Rewrite (~11 beats, persistent GPU
    KV pool): paging packed leftover VRAM → **overcommit is new** (contiguous
    systems never ran out mid-decode) → pool dry (pause) → FCFS *animated*
    (preempt newest) → partial eviction fails on the same picture (S6
    foreshadow paid off) → sequence-group gang-preempt (S7 callback) → swap to
    CPU RAM (S1 two-pool / PCIe) or drop-and-prefill → block size as the trade
    → Fig 19 as the measurement (shape, not fake precision) → Fig 4 last, as a
    labeled zoom-out of jobs the audience already watched. Cut: worker-internal
    / all-reduce lecture, fade-to-black between policy beats, slogan landing.
    **Unhurried pass (same day):** the 11-beat version still packed too much
    into one pause (C-arrives+overcommit+contiguous; partial-fail+undo+rule;
    swap-intro+copy+cap; drop+tokens+prefill; both Fig 19 curves). Expanded
    to ~17 beats so each claim sits: A then B then C; stranded state before
    the rule; CPU RAM before the copy; tokens remaining before prefill;
    Fig 19 recompute then swap. Same persistent pool; no new content.
    **Cut comparison (same day):** Fig 19 / swap-vs-recompute ablation dropped
    from S8. Swap and drop-then-prefill stay as two recovery mechanisms; no
    which-wins, no small-blocks-vs-one-prefill panels. Fig 4 still last.
    Fig 19 is omitted from the talk (S10 remains Fig 18 only). Target 15 beats.
20. **S9 rewrite (2026-09-06): follow paper §6, not a redesigned gallery.** The
    first S9 dumped disconnected charts with broken axes (bars at negative x,
    every x-sample labeled, all series faded in together). Flattened Fig 11 to
    mean bars, Fig 12 to 13B-only, Fig 14/16 to invented knee-rate / multiplier
    bars, and recapped Fig 2. Rewrite (~13 beats) walks §6.1–§6.5 in order,
    recreates Figs 1-right and 11–17 as drawn (histograms, 2×3 latency grids,
    printed Fig 13/15 bars, Fig 16 as curves, Fig 12(f) exception). Chart
    helpers now place bars via `c2p`, draw dashed grids, and expose
    `fade_frame` / `draw_series` so vLLM is revealed last. Talk colors stay;
    marker shapes follow the paper.

## Explicitly rejected / deferred

- **C — backup "anticipated question" beats** (kernel overhead / block size as
  jump-to slides): **not selected.** (The facts still live in S6/S10.)
- **D — extra prediction moments** (e.g. guess-the-throughput before the reveal):
  **not selected.** S4 keeps one pause on S3’s question, then walks §3.1.
- **S4 invented allocators** (grow-and-copy / linked chunks as on-screen options):
  **rejected** in the rewrite — the paper’s *why* is contiguous tensors + unknown
  length, not GPU memcpy vs pointer-chasing (that belongs, lightly, in S6).
- **Anchor motif** (a recurring visual device across the whole mechanism section):
  **declined** — the recurring *example* [A] serves the continuity role instead.
- **Trimming the back half** (sharing / system): **rejected** — those are the payoff.

## Process note (workflow)

- Working dir held only `vllm.pdf`; there is no codebase, so `/init` (CLAUDE.md for a
  codebase) did not apply — the real ask was a talk plan, now a build.
- The full paper was read to ground every figure and number (see "Key numbers" in
  `TALK_PLAN.md`).
- Plan was iterated over several back-and-forth rounds before these docs were written.

## Next steps

1. **Run `/compact`** to compress the chat (these two md files are the durable source
   of truth).
2. Execute `TALK_PLAN.md` starting at **Step 1 — Environment** (create `.venv`, install
   `manim` + `manim-slides`, smoke-test).
3. Build `theme.py` + `components.py`, then scenes **S0 → S11** incrementally, sharing a
   preview after each for pacing/visual feedback.
