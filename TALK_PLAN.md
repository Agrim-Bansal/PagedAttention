# Talk: PagedAttention / vLLM — narrated Manim animation

> Executable spec for the talk deck. Companion doc: `PLANNING_NOTES.md` (how/why we got here).

## Context

Presenting *"Efficient Memory Management for LLM Serving with PagedAttention"*
(Kwon et al., SOSP '23 — the vLLM paper) to a CS club (juniors + peers).
There is no target duration; take the time the material needs.
Deliverable: a **graphical animation the presenter narrates over live**, built in
**Manim (Community Edition)** + **manim-slides** (pauses on the last frame of each
beat, advances on keypress — ideal for self-paced live narration and pausing for real
audience interaction). A reveal.js **HTML export** is a no-dependencies backup.

## Narrative principles

- **Prereqs = only enough to see the relevance and connection.** Don't teach all of
  GPUs / attention / memory. The paper's background on attention + KV cache is good;
  add lightly. Audience likely does **not** know memory architecture — introduce it
  smoothly, only as needed.
- **One continuous thread, smooth segues:** *why GPUs are useful* → attention & how
  transformers generate → KV cache → memory → **problem/motivation the audience derives
  themselves (interactive: pause for answers, then reveal)** → the paper's solution →
  **only then** "…this is what an OS does (paging)."
- **Show it's not trivial:** a dedicated beat on why this is hard and different from OS
  paging and CPU/GPU memory management — not a free port of an OS idea.
- **Novelty:** in the problem section, use the paper’s contrast: unlike tensors in
  traditional deep learning workloads, the KV cache grows and shrinks and its
  lifetime and length are not known a priori. Do not name paging (that is S6).
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
    s1_gpu.py            # why GPUs are useful + limited VRAM (front-loaded primer)
    s2_transformers.py   # transformer-as-box loop; Eqs. 1–3; prefill vs decode; land on “need all K/V”
    s3_kvcache.py        # stitch S2+S1: name the KV cache, size it, put it in leftover VRAM (Fig 1-left)
    s4_problem.py        # §3.1: contiguous tensors, Fig 3 three wastes, Fig 2, sharing as second failure
    s5_pagedattention.py # ACT II — who demanded contiguity → blocks → Eq. 3→4 → Fig 5 run live → manager → Fig 6 ①②③(+1) → Fig 2 payoff → Fig 7 → engine iteration → landing
    s6_os_and_why_hard.py# reveal OS-paging analogy AFTER; then why non-trivial / differs from OS & CPU/GPU (ties to Fig 18a)
    s7_sharing.py        # ACT III — COW parallel sampling (Fig 8), beam search (Fig 9), shared prefix (Fig 10)
    s8_scheduling.py     # overcommit → FCFS → all-or-nothing → swap / recompute as recoveries; Fig 4 last
    s9_results.py        # §6: Fig 1-right, 11–17 in paper order (setup, histograms, 2x3 latency, batch, sharing, chatbot)
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
| 2 | Memory-waste breakdown bars (20.4–96.3%) | S4 (setup, vLLM as `?`) / S5 (the `?` becomes 96.3) |
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

The S9 result charts are recreated as native bar/line mobjects and run through
in paper §6 order (animated series reveal, vLLM last), not as 8 disconnected plots.

## Reusable components (`talk/components.py`)

- `TokenBox` / `token_sequence([...])` — words/tokens as rounded boxes (drives the
  recurring "Four score…" example).
- `attention_diagram(...)` — Q/K/V computation visual (scratch / optional overlay).
- `TransformerBox` / `TransformerLoop` — S2 spine: opaque transformer, sequence+KV in,
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

**S0 Title / cost hook + roadmap.** Open with stakes: an LLM request can cost
~10× a keyword search, and >30% of that expensive GPU memory is KV cache — mostly
*wasted*. Roadmap = the three acts.

**S1 Why GPUs (8 beats).** Front-loaded primer. No transformer box, attention,
decode/prefill, or KV cache. Neural-net serving is the same giant matrix multiply
against a shared `W`, over and over — the shape a GPU is built for. **CPU vs GPU**
as sequential vs parallel cores. A GPU has its **own VRAM** (not CPU RAM; PCIe is a
slow bridge); cores can only multiply data already there. Model weights **persist**
in VRAM: OPT-13B ≈ 26 GB on an A100 40 GB (~65% gone before any request). One
request is a tiny amount of math against that W, so cores wait on memory. **Batching**
runs many requests through **one hub** on the same W (one load). Leftover VRAM is the
serving budget: it decides the max batch, which decides throughput. Two VRAM regions
only (weights | leftover); later scenes split leftover.

**S2 Transformers & attention (10 beats).** Persistent visual: a centered **transformer
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

**S3 KV cache (stitch S2+S1).** Persistent transformer loop on
**"Four score…"**. Pickup: last scene the growing K/V bundle still sits in memory;
the scene before, leftover VRAM is where it lives. **Recompute** is a triangle of real
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

**S4 The problem (§3 / §3.1).** Unhurried, visual: one paper claim per beat (~20).
Pickup S3’s question; pause; then the paper’s answer — not a brainstorm
of allocators. Existing systems store each request’s KV as **one contiguous tensor**
(frameworks require it) and **statically pre-allocate the max length** (A = 2048,
B = 512 in Fig 3), irrespective of actual input or output. **Fig 3** stays on a
persistent slab: tokens paint in, reserved pulses, unused yellow becomes empty
slots then internal waste (2038 / 507); a third request fails to fit the gap
(external); even-if is a shorter request bouncing off A’s unused pink. Then
**Fig 2** one bar at a time (Max 20.4%, Oracle 38.2%, vLLM as `?`). **Second**
failure: prompt copies (12%), beam search **up to 55%**. No compaction beat.
No paging word. *(Checkpoint → Act II.)*

### ACT II — The idea: page the KV cache

**S5 PagedAttention (17 beats, heaviest).** Paper §4.1 → §4.2 → §4.3, no OS
words, with the connective tissue the paper leaves implicit. **Pickup (1):**
S4's slab; *who* demanded contiguity? The attention operator reads K and V as
one tensor each — the reader dictates the layout, so the paper changes the
reader first. **§4.1 (2–7):** partition into fixed **blocks** (B=16 default,
draw 4; no 2048 reservation); a block is K_j and V_j for B tokens, last block
partial = the only reservation. **Eq. 3 recap** over the flat row: the two
sums over all tokens are what wanted one contiguous K/V. **Eq. 4:** the same
sums grouped by block, nothing approximated. **Fig 5 run live** (two beats):
query `"forth"`, blocks in physical order 1, 2, 0 at unrelated addresses;
fetch block → q·K_j → exp → add into a running denominator and a running
value-weighted numerator; partial block read by fill count; divide → o.
Cost: one lookup per block; payoff: blocks may live anywhere. **§4.2 (8):**
the **KV cache manager** — logical blocks (request's view), block table
(logical→physical, # filled), physical pool (one GPU allocation, equal
blocks, on demand) — full-width layout with table→physical arrows that stays
on screen. **§4.3 Fig 6 (9–12)** on the 7-token prompt: ① prefill maps
logical 0→phys 7 (4) and 1→phys 1 (3, one reserved), ordinary attention; ②
`"fathers"` fills the reserved slot, 3→4, no allocation; ③ last block full →
allocate phys 3 for `"brought"`; one extra decode `"forth"` → physical 1, 3, 7
*is* the Fig 5 picture (loop closes: the kernel reads what the manager leaves
behind). **Payoff (13):** S4 strip vs vLLM strip side by side; waste **< 1
block**; Fig 2 with the vLLM `?` growing to **96.3%**. **Fig 7 (14–15):**
Request B `"it was the best of times"` at phys 5, 2 with its own table,
interleaved with A; same-size blocks ⇒ any free block fits any request; A
finishes → 7, 1, 3 return, six free blocks, no hole needed. **Bridge
(16):** one engine iteration (select batch → allocate → concatenate all
requests' current tokens → one forward pass with PagedAttention; Act I's
batching payoff, leads to S8). Block size is *not* discussed here beyond
"B=16 default" — that dial is S10's. **Landing (17):** kernel + manager
two-column summary, 96.3%, "this picture should look familiar" → S6.

**S6 "This is what an OS does" + why it's hard (~11 beats, unhurried).**
Same arc as before, one idea per pause. Reveal paging on the S5 picture;
relabel in place; show demand paging. Question, then answer. Then properties,
still not CUDA internals: no GPU MMU (kernel walks the table) as its own beat;
kernel rewritten to gather+attend in one fused pass as its own beat; kernel
**20–26% slower** as its own beat; then the **2–4×** end-to-end win. Landing +
checkpoint. Fig 18(a) stays in S10.

### ACT III — The payoffs

**S7 Sharing (~12 beats, mechanism only; Fig 15/16 stay in S9).** Pickup on
the S5 picture (logical 0→phys 7, 1→1, 2→3): Fig 7 interleaved two requests
in one pool, but they still did not share — that is S4’s unpaid bill.
**Parallel sampling (Fig 8)** on that same physical column: rewind to the
7-token prompt, fork A1/A2, both tables map 0→7 (4/4) and 1→1 (3/4), ref
count 2; packed block 7 stays shared forever; A1 writes `"mothers"` via
**copy-on-write** into phys 3 (only the last not-yet-full block is copied);
A2 writes `"fathers"` **in place** (ref now 1). One line: this is OS `fork`.
**Beam search (Fig 9)** as a k=4 block tree, not TokenBoxes: shared trunk,
one private branch, four live heads; then one prune (candidates 0 and 3
drop, private blocks hit ref 0 and return to the free pool, new heads for
survivors). Spoken callback: up to 55%, measured in S9. **Shared prefix
(Fig 10)** in two beats: rest on the paper’s few-shot translation sequences,
then dissolve into cached physical prefix blocks with two request tables
pointing at them (last shared block marked copy-on-write). **Landing:**
`fork` / `append` / `free` compose all three; next is S8 (the free pool
runs dry). No charts.

**S8 Scheduling under pressure (~15 beats).** One persistent GPU KV pool;
one claim per pause. **Pickup:** A lands, then B, interleaved, 3 free.
**C fills the leftovers** — pool full. **Sit:** contiguous S4 slabs reserved
2048 up front, so they never ran out mid-decode; this picture is new.
**A needs one more block** — pause. **Name the scheduler**, then **preempt C**
(newest; oldest last to kick). **Try taking half of C** and *sit on the
stranded leftover*; then restore: **all or none** (S6 payoff). **Sequence
group** gang-preempted (S7). **CPU RAM appears** (S1 two-pool) before any
copy; then **swap** (copy, A continues, cap). **Or drop:** KV vanishes,
tokens remain; **one prefill** rebuilds. Mechanics only — no swap-vs-recompute
comparison, no Fig 19. **Fig 4** last, assembled from the jobs already
watched. Land: contiguous slabs had no way to take memory back.

**S9 Results.** Paper §6 in order (~13 beats). **§6.1 setup** (Table 1, baselines,
normalized latency). **Fig 1-right** (intro claim: KV growth vs batch). **Fig 11**
histograms (ShareGPT / Alpaca). **§6.2 Fig 12** as the 2×3 latency-vs-rate grid
(13B/66B/175B × ShareGPT/Alpaca; linger on 12(f) exception) then **Fig 13**
batched requests (7.00 / 9.81 / 13.62 / 30.42 and 7.00 / 43.24 / 72.75 / 132.44).
**§6.3 Fig 14** parallel then beam (1.3× → 2.3×) then **Fig 15** printed savings.
**§6.4 Fig 16** 1-shot / 5-shot curves (1.67× / 3.58×). **§6.5 Fig 17** chatbot
(~2×, Orca variants cluster). Series drawn last so vLLM’s gap is visible. No
Fig 2 recap.

**S10 Ablations.** **Block size** sweet spot (16) and **attention-kernel
overhead** (~20–26% slower kernel, net 2–4× win) — **Fig 18**.

**S11 Takeaways + discussion.** Memory management, not a new model, unlocked
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
- No target duration. Cover every figure fully; do not rush prereqs or results.

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
