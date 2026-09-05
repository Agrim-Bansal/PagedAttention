# Talk: PagedAttention / vLLM — narrated Manim animation

> Executable spec for the talk deck. Companion doc: `PLANNING_NOTES.md` (how/why we got here).

## Context

Presenting *"Efficient Memory Management for LLM Serving with PagedAttention"*
(Kwon et al., SOSP '23 — the vLLM paper) to a CS club (juniors + peers), ~40 min.
Deliverable: a **graphical animation the presenter narrates over live**, built in
**Manim (Community Edition)** + **manim-slides** (pauses on the last frame of each
beat, advances on keypress — ideal for self-paced live narration and pausing for real
audience interaction). A reveal.js **HTML export** is a no-dependencies backup.

## Narrative principles

- **Prereqs = only enough to see the relevance and connection.** Don't teach all of
  GPUs / attention / memory. The paper's background on attention + KV cache is good;
  add lightly. Audience likely does **not** know memory architecture — introduce it
  smoothly, only as needed.
- **One continuous thread, smooth segues:** attention & how transformers generate →
  *why GPUs are useful* → KV cache → memory → **problem/motivation the audience derives
  themselves (interactive: pause for answers, then reveal)** → the paper's solution →
  **only then** "…this is what an OS does (paging)."
- **Show it's not trivial:** a dedicated beat on why this is hard and different from OS
  paging and CPU/GPU memory management — not a free port of an OS idea.
- **Novelty:** in the problem section, show **paging was never needed before LLMs** —
  pre-LLM tensors were fixed, known-size, so contiguous allocation was fine.
- **Keep everything; the mechanical details ARE the payoff:** copy-on-write, all
  sharing cases, eviction + recovery (swap vs recompute), preemption.
- Q/K/V at **intuition** level but **with a visual of the actual computation**.
- **[A] One recurring concrete example**: thread the paper's *"Four score and seven
  years ago our fathers…"* tokens through the KV-cache → problem → mechanism scenes
  (the paper reuses it across Fig 3/6/7), so each new idea lands on familiar tokens.
- **[B] Three-act framing + light checkpoints**: Act I (why memory is the bottleneck),
  Act II (the idea: page the KV cache), Act III (the payoffs). A tiny "where we are"
  beat at each seam keeps a long talk navigable.
- **[Figures] Every example/figure in the paper is shown and animated ("run
  through")** — not screenshotted, recreated natively as mobjects. See coverage map.
- No recurring visual "anchor motif"; visuals otherwise look good as drafted.

## Tech decisions

- **Manim CE** (~0.19, supports Py 3.13) + **manim-slides**; live `present`, `convert`
  to HTML backup.
- **No LaTeX**: Pango `Text`/`MarkupText` (math light/optional); formulas as styled
  text. BasicTeX + `MathTex` is a later option if wanted.
- Env: Python 3.13.13 (pyenv), ffmpeg present; **no** manim/manim-slides/LaTeX yet.
  macOS: pip wheels for pycairo/pangocairo usually suffice; fallback
  `brew install cairo pango pkg-config` if a build fails.

## Step 1 — Environment

- `python3 -m venv .venv`; upgrade pip; `pip install manim manim-slides`; pin into
  `requirements.txt`.
- Smoke-test by rendering a sample scene. Fall back to brew system deps if it fails.

## Project structure

```
PagedAttention/
  vllm.pdf
  .venv/                 # step 1
  requirements.txt
  README.md              # render / present / export HTML
  Makefile               # make present | html | render
  talk/
    __init__.py
    theme.py             # palette, fonts, sizes, act-checkpoint helper
    components.py        # reusable mobjects (below)
    s0_title.py          # ACT I — cost hook + roadmap
    s1_transformers.py   # attention + autoregressive generation, Q/K/V intuition + computation visual
    s2_gpu.py            # why GPUs are useful here + limited VRAM
    s3_kvcache.py        # KV cache: what/why, grows per token, memory budget (Fig 1-left) — intro "Four score…"
    s4_problem.py        # interactive derive; fragmentation (Fig 3, Fig 2); "no paging before LLMs"; no sharing
    s5_pagedattention.py # ACT II — blocks, block table, non-contig kernel (Fig 5), decode walkthrough (Fig 6), two-req (Fig 7)
    s6_os_and_why_hard.py# reveal OS-paging analogy AFTER; then why non-trivial / differs from OS & CPU/GPU (ties to Fig 18a)
    s7_sharing.py        # ACT III — COW parallel sampling (Fig 8), beam search (Fig 9), shared prefix (Fig 10)
    s8_scheduling.py     # vLLM architecture (Fig 4); preemption, all-or-nothing eviction, swap vs recompute
    s9_results.py        # datasets (Fig 11), throughput (Fig 1-right,12,13,17), sharing wins (Fig 14,15,16)
    s10_ablations.py     # block size + kernel overhead (Fig 18)
    s11_takeaways.py     # impact + discussion prompt ("what else could you page?")
```

Each `sN_*.py` is one `manim_slides.Slide`. Independent (render/verify one at a time);
`manim-slides present` chains them in order.

## Paper figure coverage map (all animated / run through)

| Fig | Content | Scene |
|-----|---------|-------|
| 1 (left) | Memory budget: 65% weights / >30% KV / other | S3 |
| 1 (right) | KV growth curve, vLLM vs existing | S9 |
| 2 | Memory-waste breakdown bars (20.4–96.3%) | S4 (setup) / S9 (payoff callback) |
| 3 | KV mgmt in existing systems: reserved/internal/external frag — "Four score…" | S4 |
| 4 | vLLM system overview (scheduler / KV manager / workers) | S8 |
| 5 | PagedAttention: query over non-contiguous K/V blocks | S5 |
| 6 | Block-table translation, decode steps ①②③ | S5 |
| 7 | Two requests stored simultaneously | S5 |
| 8 | Parallel sampling + copy-on-write + ref counts | S7 |
| 9 | Beam search block sharing | S7 |
| 10 | Shared prefix (translation) | S7 |
| 11 | ShareGPT / Alpaca input–output length distributions | S9 (eval setup) |
| 12 | Normalized latency vs request rate (OPT-13/66/175B) | S9 |
| 13 | Avg batched requests 7→30 / 7→132 | S9 |
| 14 | Parallel sampling & beam-search results | S9 |
| 15 | Memory saving from sharing (6–55%) | S9 |
| 16 | Shared-prefix translation results | S9 |
| 17 | Chatbot workload results | S9 |
| 18 | Ablations: attention-kernel latency, block size | S10 (18a also referenced in S6) |

The many S9 result charts are recreated as clean native bar/line mobjects and run
through briskly (animated reveal each), grouped so the eval flows as one story rather
than 8 disconnected plots.

## Reusable components (`talk/components.py`)

- `TokenBox` / `token_sequence([...])` — words/tokens as rounded boxes (drives the
  recurring "Four score…" example).
- `attention_diagram(...)` — Q/K/V computation visual (S1, reused in Fig 5).
- `KVBlock(slots, filled)` — fixed-size KV block (the "page").
- `MemoryBar` / `MemoryPie` — VRAM budget + waste breakdown (Fig 1/2/3).
- `BlockTable(rows)` + `PhysicalMemGrid(cols)` — logical→physical mapping (Fig 6/7/8).
- `GPUSchematic()` — minimal cores + VRAM (S2).
- `RefCountBadge`, `arrow_map()` — sharing/COW/eviction (Fig 8/9/10, S8).
- `bar_chart()` / `line_chart()` — native chart helpers for S9/S10.
- `act_checkpoint(title)` — the "where we are" seam beat [B].

## Narrative + scene contents (one continuous thread)

### ACT I — Why memory is the bottleneck

**S0 Title / cost hook + roadmap (~1–2m).** Open with stakes: an LLM request can cost
~10× a keyword search, and >30% of that expensive GPU memory is KV cache — mostly
*wasted*. Roadmap = the three acts.

**S1 Transformers & attention (~4–5m).** Autoregressive generation, one token at a
time, each token attends to all previous. **Q/K/V at intuition level with a visual of
the computation**. Land: *to produce the next token you need the K and V of every
previous token.*

**S2 Why GPUs (~3m).** Huge amounts of the same math → GPUs do massively parallel
math; **batching** amortizes weights. The catch: a GPU has its **own limited VRAM**;
everything lives there.

**S3 KV cache (~4m).** Recomputing past K/V each step is wasteful → **cache them**.
Grows one block/token; different per request; lives in VRAM. **Memory budget
(Fig 1 left)**. Introduce the recurring **"Four score…"** sequence here. Point: KV
size ⇒ how many requests fit ⇒ throughput.

**S4 The problem, derived by the audience (~6–7m).** Interactive: *how do you allocate
memory for something whose final size is unknown?* Pause for answers → reveal the
naive contiguous, reserve-max approach → let them see **reserved / internal / external
fragmentation (Fig 3, on "Four score…")** → only **20–40% used (Fig 2)**. **Novelty
beat: paging was never needed before LLMs** (tensors were fixed-size). Second gap:
**no sharing** across sequences. *(Checkpoint → Act II.)*

### ACT II — The idea: page the KV cache

**S5 PagedAttention (~7–8m, heaviest).** Split KV into fixed **blocks**; **block
table** maps logical→physical; allocate on demand; blocks non-contiguous; kernel
gathers K/V across them **(Fig 5)**. **Decode walkthrough (Fig 6)** on "Four score…" →
~96% utilization. **Two requests at once (Fig 7)**.

**S6 "This is what an OS does" + why it's hard (~4–5m).** Reveal the **virtual-memory /
paging** analogy (blocks=pages, tokens=bytes, requests=processes, block table=page
table). Then why it's non-trivial and different: no GPU hardware MMU → all software;
the attention **kernel** had to be rewritten to gather scattered blocks *fast* (a
normal OS pager never touches computation); accessed **every token every step** (not a
rare page fault, cf. Fig 18a overhead); needs **domain policies** (all-or-nothing
eviction, gang scheduling, tuned block size). *(Checkpoint → Act III.)*

### ACT III — The payoffs

**S7 Sharing (~5m).** **Parallel sampling** shares prompt blocks, diverges via
**copy-on-write + reference counts (Fig 8)**; **beam search (Fig 9)**; **shared prefix
/ system prompt (Fig 10)** — up to ~55% saved. Impossible in contiguous systems.

**S8 Scheduling under pressure (~4m).** **vLLM architecture (Fig 4)**: scheduler + KV
cache manager + workers. FCFS + **preemption**; **all-or-nothing eviction**; recovery
by **swap** (to CPU RAM) or **recompute** (concatenate + one prefill), and the
trade-off.

**S9 Results (~4–5m).** Eval setup + **dataset distributions (Fig 11)**. Throughput
story: **Fig 1-right, 12, 13** (batched 7→30 / 7→132), **chatbot (Fig 17)**. Sharing
wins: **Fig 14, 15, 16**. Callback to the Fig 2 waste bars: 20–40% → 96%.

**S10 Ablations (~2m).** **Block size** sweet spot (16) and **attention-kernel
overhead** (~20–26% slower kernel, net 2–4× win) — **Fig 18**.

**S11 Takeaways + discussion (~1–2m).** Memory management, not a new model, unlocked
serving; vLLM/PagedAttention is the de-facto standard. Close on a discussion prompt:
*what else could you page?* (prefix caching, LoRA adapters — later real vLLM features).

## Build order & verification

1. Step 1 env + smoke test.
2. `theme.py` + `components.py`; render a scratch scene exercising each component +
   the `act_checkpoint` and chart helpers.
3. Build scenes **S0→S11 incrementally**; after each, render a quick preview
   (`manim -pql …`) and share a checkpoint frame/clip so pacing/visuals adjust early.
4. `Makefile` + `README.md`: `make present` (live), `make html` (reveal.js backup),
   `make render` (mp4s).
5. **End-to-end:** `manim-slides render` all scenes high-quality, `manim-slides
   present` the full sequence as a dry run, generate + open the HTML backup standalone.

## Open/assumed defaults (flag, not blocking)

- 16:9, dark background, one accent color — confirm style at the first checkpoint (S0).
  No anchor motif.
- Not selected (skipped): C "backup Q&A beats", D extra prediction moments. The S4
  interactive derive stays.
- Timing lands ~40 min+ given full figure coverage; presenter runs fast through
  prereqs the audience already knows.

## Key numbers to keep accurate (from the paper)

- KV cache ≈ **800 KB per token** for OPT-13B; a single request up to **~1.6 GB**.
- Memory budget on 13B/A100-40GB: **~65% weights, >30% KV cache, rest ephemeral**.
- Existing systems: only **20.4–38.2%** of KV memory holds real token state; vLLM
  reaches **96.3%** (Fig 2).
- Throughput: **2–4×** vs FasterTransformer / Orca; up to **22×** vs FasterTransformer.
- Batched requests, OPT-13B: **7 → 30** (ShareGPT, 2 req/s), **7 → 132** (Alpaca, 30 req/s).
- Sharing memory savings: parallel sampling **6.1–9.8%** (Alpaca) / **16.2–30.5%**
  (ShareGPT); beam search **37.6–55.2%** / **44.3–66.3%**.
- Shared-prefix translation: **1.67×** (1-shot) / **3.58×** (5-shot) vs Orca (Oracle).
- Default **block size = 16**; PagedAttention kernel **~20–26%** slower than
  FasterTransformer, but net end-to-end win.
