# Planning notes — how & what we reached

> Decision log for the PagedAttention/vLLM club talk. The executable spec lives in
> `TALK_PLAN.md`; this file records the *requirements gathered* and the *decisions with
> their rationale*, so the reasoning survives chat compaction.

## Requirements gathered

- **Task:** present the vLLM paper (PagedAttention, Kwon et al., SOSP '23) to a CS
  **club** — juniors + peers.
- **Duration:** ~30–40 min (settled at ~40 given full figure coverage).
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
3. **One continuous narrative thread**, ordered: attention & how transformers generate
   → *why GPUs are useful* → KV cache → memory → problem → solution. Segues must be
   smooth; the audience likely doesn't know memory architecture, so it's introduced
   only as needed.
4. **OS-paging analogy moved to AFTER the mechanism (S6), not before.** Present
   PagedAttention on its own terms first, *then* reveal "…this is what an OS does." The
   analogy lands harder as a payoff than as scaffolding.
5. **Audience derives the problem themselves (S4, interactive).** Pose "how do you
   allocate memory for something whose final size is unknown?", **pause for real
   answers**, then reveal fragmentation. Chose genuine interaction over animating a
   hypothetical answer.
6. **Novelty beat:** explicitly show **paging was never needed before LLMs** — pre-LLM
   tensors were fixed / known-size, so contiguous allocation was fine. Establishes why
   this is a *new* problem.
7. **"Why it's not trivial" beat (S6).** It is *not* a free port of an OS idea:
   no GPU hardware MMU (all software); the attention **kernel** had to be rewritten to
   gather scattered blocks *efficiently* (an OS pager never touches computation);
   accessed **every token every step** (not a rare page fault); needs domain-specific
   policies (all-or-nothing eviction, gang scheduling, tuned block size). The presenter
   confirmed this framing is exactly the intended emphasis.
8. **Keep ALL mechanical details — they are the payoff, not trim candidates:**
   copy-on-write + reference counts, all three sharing cases (parallel sampling, beam
   search, shared prefix), and eviction recovery (swap-to-CPU vs recompute) each get
   real coverage. (Earlier suggestion to trim the system/back half was rejected.)
9. **Q/K/V with the paper’s math (Eqs. 1–3), shown as a transformer box.** The
   persistent S1 visual is an opaque centered transformer: sequence in, a new token
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
    small "where we are" beat at each seam to keep a ~40-min talk navigable. (Adopted.)
12. **Every paper figure is animated / run through** (all ~18 figures), recreated
    natively as mobjects rather than screenshotted. The presenter explicitly wanted
    *all* the paper's examples shown and run through for comprehension. Added a
    dedicated **ablations scene (S10, Fig 18)** and dataset distributions (Fig 11) to
    honor this.
13. **Defaults kept without extra ceremony:** a cost hook in S0 (LLM request ~10× a
    keyword search; >30% of GPU memory is mostly-wasted KV cache) and a discussion
    prompt to close S11 ("what else could you page?" → prefix caching, LoRA adapters).
14. **S2 is a front-loadable GPU/VRAM primer, not a recap of generation.** Depth:
    sequential-vs-parallel cores, separate CPU DRAM vs GPU VRAM (PCIe is slow),
    weights persist (~26 GB of 40 GB), one request underuses the machine, batching
    as **one hub on W**, leftover VRAM = max batch = throughput. No transformer /
    attention / decode / KV. Written to sit before transformers; file order not
    shuffled in that pass. (The old 3-minute “same math, then VRAM bar” sketch was
    too thin.)
15. **S3 is a deep stitch scene, not a 4-minute name-and-size dump.** S1 lands
    "the growing K/V bundle sits in memory"; S2 lands "leftover VRAM is the
    budget." S3 names that bundle, shows recompute vs cache on the same
    tokens, prefill-write / decode-append, read-all/write-one, ×40 layers,
    factor-by-factor 800 KB, then unknown length → reserve the 2048-slot max
    (unused tail cut off from other requests) → 1.6 GB spoken for, Fig 1 left,
    and ~7 max-length requests in 12 GB — then earns S4's allocation question
    (including that K/V is position-dependent, not a dictionary of words).
    Still no Eq. 4, paging, named fragmentation, or Fig 1-right. Target
    ~9–11 min / 10–12 beats.

## Explicitly rejected / deferred

- **C — backup "anticipated question" beats** (kernel overhead / block size as
  jump-to slides): **not selected.** (The facts still live in S6/S10.)
- **D — extra prediction moments** (e.g. guess-the-throughput before the reveal):
  **not selected.** The S4 interactive derive is the one interactive beat retained.
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
