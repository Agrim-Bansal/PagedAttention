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
    s1_transformers.py   # transformer-as-box loop; Eqs. 1–3; prefill vs decode; land on “need all K/V”
    s2_gpu.py            # why GPUs are useful here + limited VRAM
    s3_kvcache.py        # stitch S1+S2: name the KV cache, size it, put it in leftover VRAM (Fig 1-left)
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
- `attention_diagram(...)` — Q/K/V computation visual (scratch / optional overlay).
- `TransformerBox` / `TransformerLoop` — S1 spine: opaque transformer, sequence+KV in,
  new token out, append-and-recenter.
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

**S1 Transformers & attention (~6m).** Persistent visual: a centered **transformer
box** (opaque, not literal black). Start closed: words go in, one next word comes
out, joins, recenters. **Eq. 1** after that first loop. Then open the box and
introduce **Eq. 2 one vector at a time** — Key (label / what the token contains),
Value (payload / what you mix in), Query (the question this step asks; only the
newest token). **Eq. 3**: scores `q^T k / sqrt(d)`, softmax, weighted sum of V.
Close it and run another iteration so the K/V bundle visibly grows and nothing is
discarded. **Prefill vs decode** (paper §2: prompt in parallel / compute-bound vs
one-token loop / memory-bound). Land: *to produce the next token the box must be
handed the K and V of every previous token* — that bundle sits in memory between
iterations. Do not name “KV cache” (S3) or Eq. 4 (S5).

**S2 Why GPUs (front-loadable primer; 8 beats).** Written to sit **before**
transformers (scene file order unchanged this pass). No transformer box, attention,
decode/prefill, or KV cache. Neural-net serving is the same giant matrix multiply
against a shared `W`, over and over — the shape a GPU is built for. **CPU vs GPU**
as sequential vs parallel cores. A GPU has its **own VRAM** (not CPU RAM; PCIe is a
slow bridge); cores can only multiply data already there. Model weights **persist**
in VRAM: OPT-13B ≈ 26 GB on an A100 40 GB (~65% gone before any request). One
request is a tiny amount of math against that W, so cores wait on memory. **Batching**
runs many requests through **one hub** on the same W (one load). Leftover VRAM is the
serving budget: it decides the max batch, which decides throughput. Two VRAM regions
only (weights | leftover); later scenes split leftover.

**S3 KV cache (~9–11m, stitch S1+S2).** Persistent transformer loop on
**"Four score…"**. Pickup: the growing K/V bundle still sits in memory, and
leftover VRAM is where it lives. **Recompute** is a triangle of real
`W_K x` / `W_V x` work on those tokens → **keep them, name the KV cache**;
Query is computed fresh and discarded. **Prefill writes** the prompt's pairs
in one pass; **decode appends** one pair per step. **Read-all / write-one**
is why decode is memory-bound. **×40 layers** makes one token expensive;
build **800 KB / token** factor by factor. **Unknown length:** the cache
grows until EOS, so the only number we can bank on is the model's 2048-token
maximum — the whole strip is reserved up front; the unused tail is cut off
from other requests. That reservation **is** 1.6 GB / request (OPT-13B).
Two requests, independent caches, placed in leftover VRAM. **Fig 1 left**
(65% weights / >30% KV / other) plus Table 1 packing: 12 GB ÷ 1.6 GB ≈
**7 requests at max length**. Land on the reserved-max problem plus
position-dependent K/V, then the S4 question: *how do you allocate memory
for something whose final size is unknown?* No named fragmentation, blocks,
paging, or Fig 1-right.

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
