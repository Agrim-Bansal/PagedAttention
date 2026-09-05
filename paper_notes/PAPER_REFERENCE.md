# vLLM / PagedAttention Paper Reference
Source: Kwon et al., "Efficient Memory Management for Large Language Model Serving with PagedAttention", SOSP '23. Extracted from /Users/agrim/Home/PagedAttention/vllm.pdf (16 pages total, no separate appendix — paper ends at Fig. 19 / Conclusion / References on pages 13-16).

All numeric values taken from bar/line charts are read off the plotted axes and are marked **(approx)**; anything quoted directly from body text is exact and is not marked approx.

---

## FIGURES

### Figure 1 (page 1, top right) — "Memory layout when serving an LLM with 13B parameters on NVIDIA A100 40GB"
Two-panel figure.

**Left panel (diagram, not data):** A stacked horizontal/vertical box labeled "NVIDIA A100 40GB" split into three regions:
- "Parameters (26GB, 65%)" — gray box, largest region.
- "KV Cache (>30%)" — red box.
- "Others" — small yellow/gray sliver.
Caption detail: "The parameters (gray) persist in GPU memory throughout serving. The memory for the KV cache (red) is (de)allocated per serving request. A small amount of memory (yellow) is used ephemerally for activation."

**Right panel (DATA figure):** Two stacked/overlaid line charts sharing x-axis "Batch size (# requests)", x-axis range roughly 0 to 40.
- Top chart: y-axis "Memory usage (GB)", range roughly 0–40 GB, dashed horizontal line at ~26 GB annotated "Parameter size". Two series: "Existing systems" (orange, marked with "x") and "vLLM" (blue, circle markers). Existing-systems line rises steeply (linear, roughly proportional to batch size) reaching ~40GB by batch size ~40; vLLM line rises much more slowly/smoothly and saturates near the top only at much larger batch sizes — i.e., vLLM "smooths out the rapid growth curve of KV cache memory."
- Bottom chart: y-axis "Throughput (tok/s)" with dashed reference lines at 1.22 and 0.84 and 0.49 (approx, small labeled tick marks near the curve, exact values printed on chart: "1.22", "0.84", "0.49"). Series is a green line rising with batch size then flattening (diminishing returns) — this is throughput vs. batch size for existing systems, illustrating the point where added batch size stops helping because of memory-caused batch-size ceiling.
Caption (verbatim): "Left: Memory layout when serving an LLM with 13B parameters on NVIDIA A100. The parameters (gray) persist in GPU memory throughout serving. The memory for the KV cache (red) is (de)allocated per serving request. A small amount of memory (yellow) is used ephemerally for activation. Right: vLLM smooths out the rapid growth curve of KV cache memory seen in existing systems [31, 60], leading to a notable boost in serving throughput."

### Figure 2 (page 2, top left) — "Average percentage of memory wastes in different LLM serving systems during the experiment in §6.2"
DATA figure: stacked vertical bar chart. Y-axis "KV cache usage (%)", 0–100, gridlines at 0,20,40,60,80,100. Four bars (x-axis categories): "Orca (Max)", "Orca (Pow2)", "Orca (Oracle)", "vLLM". Each bar is a stack of 4 segments (bottom to top): "Token states" (green), "Reservation" (yellow/orange), "Internal frag." (red), "External frag. & Others" (gray). Legend order given as: Token states (green), Reservation (yellow), Internal frag. (red), External frag. & Others (gray).

Approx segment values read off the bars (percent, stacked, each bar sums to 100):
| System | Token states (green) | Reservation (yellow) | Internal frag. (red) | External frag. & Others (gray) |
|---|---|---|---|---|
| Orca (Max) | 20.4 | 13.3 | 57.3 | 8.9 |
| Orca (Pow2) | 26.8 | 17.9 | 13.6 | 41.6 |
| Orca (Oracle) | 38.2 | 25.2 | — (not present) | 36.6 |
| vLLM | 96.3 | — | — | — (remaining ~3.7 unlabeled) |

(Numbers 20.4, 57.3, 13.3, 8.9, 41.6, 13.6, 17.9, 26.8, 36.6, 25.2, 38.2, 96.3 are all printed directly on the bar segments in the figure, so these are exact, not approximated.) Body text states: "our profiling results in Fig. 2 show that only 20.4%-38.2% of the KV cache memory is used to store the actual token states in the existing systems" and vLLM reaches 96.3% (near-zero waste).

### Figure 3 (page 4, top, full width) — "KV cache memory management in existing systems"
DIAGRAM figure. Two horizontal token-slot sequences (one per request), each slot is a box containing a token or a placeholder.

**Request A** (top row of boxes), prompt = "Four score and seven years ago our fathers brought forth" (10 tokens: Four, score, and, seven, years, ago, our, fathers, brought, forth):
- Sequence of boxes left to right: `Four, score, and, seven, years, ago, our, fathers` shown as the 7 KV-cache states already computed (annotation below: "7 KV cache states for request A's prompt") — note only 7 are marked filled even though 8 words are listed, consistent with "brought" being the current-iteration token.
- Then `brought` box marked as "Request A current iteration" (the token currently being processed / just generated).
- Then `forth`, `<eos>` boxes.
- Then a run of `<resv>` boxes with "..." and a final `<resv>` — annotated "1 slot for generated token" over the first reserved slot and "2 slots future used (reserved)" over the reserved run, and separately "2038 slots never used (Request A's prompt) internal fragmentation" (i.e., internal fragmentation = the large unused reserved region up to the request's 2048 max length).
- After Request A's whole reserved region, there is a gap of empty/blank boxes labeled "External Fragmentation" (in red text) — this is memory in the allocator that is too small/misaligned to be given to any request (external fragmentation, buddy-allocator style waste between the two requests' pre-allocated chunks).

**Request B** (bottom row of boxes), prompt fragment shown = "You only live once":
- Boxes: `You` (marked "3 KV cache states for request B's prompt" together with two more not individually named), `only`, `live` — 3 filled prompt-KV boxes.
- `once` box marked "Request B current iteration".
- Then a run of `<resv>` boxes, "...", final `<resv>`, annotated "1 slot future used (reserved)".
- Annotation "507 slots never used (internal fragmentation)" under Request B's reserved run.

Caption (verbatim): "KV cache memory management in existing systems. Three types of memory wastes – reserved, internal fragmentation, and external fragmentation – exist that prevent other requests from fitting into the memory. The token in each memory slot represents its KV cache. Note the same tokens can have different KV cache when at different positions."

Three waste categories to reproduce: **reserved** (slots held for the token currently being generated / near-future, small, e.g. "1 slot for generated token"), **internal fragmentation** (the large block of pre-allocated-but-never-used slots up to the request's declared/maximum length — 2038 slots for A, 507 for B), **external fragmentation** (gap of memory between two requests' contiguous chunks that is too small/oddly shaped for the allocator to give to any request, shown as gray blank boxes between A and B's rows).

### Figure 4 (page 5, top left) — "vLLM system overview"
DIAGRAM. Boxes and arrows:
- "Scheduler" box (green) at top left, with arrow down to "KV Cache Manager" box (which contains two small grid/table icons labeled "Block tables").
- "KV Cache Manager" box has two children below it: "CPU Block Allocator" and "GPU Block Allocator" (small boxes at bottom left).
- Scheduler has arrows fanning out to the right to a stack of "Worker" boxes: "Worker 0", "Worker 1", "...", "Worker N-1" (blue boxes on right).
- Each Worker box contains two sub-boxes: "Cache Engine" and "Model Shard i" (e.g., "Model Shard 0", "Model Shard 1", ... "Model Shard N-1"), plus a small chip/GPU icon.
Caption: "vLLM system overview."

### Figure 5 (page 5, top right) — "Illustration of the PagedAttention algorithm, where the attention key and values vectors are stored as non-contiguous blocks in the memory"
DIAGRAM. Title above table: "Key and value vectors". A 2-column-ish table with 3 labeled rows/blocks on the right:
- "Block 1": cells `years, ago, our, fathers`
- "Block 2": cells `brought, forth` (only 2 of 4 slots filled, shown yellow/orange highlighted differently from block1 - i.e., partially filled block)
- "Block 0": cells `Four, score, and, seven`
On the left: "Query vector" box containing the token `forth`, with arrows pointing from it to all three blocks (Block 1, Block 2, Block 0) illustrating that the query attends across all non-contiguous physical blocks.
Caption explains blocks are not stored contiguously (Block 1, 2, 0 out of physical order) and the kernel fetches/multiplies against each block separately, e.g. using key vectors of "Four score and seven" for block 0 to compute attention score, etc.

### Figure 6 (page 6, left) — "Block table translation in vLLM"
DIAGRAM, the worked example for §4.3. Body text and figure annotation agree: prompt is **7 tokens**, not 8.

- "Request A" at far left. Prompt: `"Four score and seven years ago our"`. Outputs: `"fathers"` → `"brought"` → …
- "Logical KV blocks" (left), 4 rows:
  - Block 0: `Four, score, and, seven` (4/4)
  - Block 1: `years, ago, our` then `"fathers"` written in step ② (3/4 after ①, 4/4 after ②)
  - Block 2: `brought` in first slot only — allocated in step ③
  - Block 3: empty (dashes; not yet allocated)
- "Block Table" (middle), columns: physical block number | # filled:
  - Logical 0 → physical **7**, filled 4
  - Logical 1 → physical **1**, filled **3→4** (the ② update)
  - Logical 2 → physical **3**, filled 1 (the ③ allocation)
  - Logical 3 → empty
- "Physical KV blocks (on GPU DRAM)" (right), Block 0 through 7 (figure also draws an empty Block 8):
  - Block 1: `years, ago, our, fathers` (after ②)
  - Block 3: `brought` (after ③)
  - Block 7: `Four, score, and, seven`
  - Blocks 0, 2, 4, 5, 6: empty

Numbered steps from §4.3 body text:
  - **① Prefill.** Prompt has 7 tokens. vLLM maps logical 0 and 1 to physical 7 and 1. Prefill uses **conventional** self-attention (not PagedAttention). Stores first 4 tokens in logical 0 and the following 3 in logical 1; the remaining slot is reserved for generation. Does **not** reserve max sequence length.
  - **② First decode.** PagedAttention on physical blocks 7 and 1 generates `"fathers"`. One slot remains in the last logical block, so the new KV is stored there; # filled updates 3→4. **No new physical block.**
  - **③ Second decode.** Last logical block is full, so vLLM allocates a new physical block (**physical 3**) for new logical block 2, stores `"brought"`, records the mapping.

Caption: "Block table translation in vLLM."

The same Lincoln sentence is cut differently in Fig 3 (7 prompt KV states through `"our"`, current token `"brought"`) and Fig 5 (later snapshot, query `"forth"`). Fig 6 is the allocation walkthrough; do not mix the cuts.

### Figure 7 (page 6, right) — "Storing the KV cache of two requests at the same time in vLLM"
DIAGRAM. Request A (left) and Request B (right), each with its own logical KV blocks, sharing one physical column (Block 0–7, plus an empty Block 8 in the figure).

- Request A logical (continues Fig 6 after ③): Block 0 = `Four, score, and, seven`; Block 1 = `years, ago, our, fathers`; Block 2 = `brought` (partial).
- Request B logical: Block 0 = `it, was, the, best`; Block 1 = `of, times` (partial); Block 2 empty.
- Shared physical (paper figure placement):
  - Block 1 = `years, ago, our, fathers` (A)
  - Block 2 = `of, times` (B)
  - Block 3 = `brought` (A)
  - Block 5 = `it, was, the, best` (B)
  - Block 7 = `Four, score, and, seven` (A)
  - Blocks 0, 4, 6 (and 8) empty / free

So A maps logical 0→7, 1→1, 2→3; B maps logical 0→**5**, 1→**2**. Neighboring logical blocks of either request need not be contiguous in GPU memory. Caption: "Storing the KV cache of two requests at the same time in vLLM."

### Figure 8 (page 7, left) — "Parallel sampling example"
DIAGRAM. Two output-sample circles at top: "Sample A1" (left) and "Sample A2" (right), both derived from the same input request (parallel sampling, one prompt, two sampled outputs).
- Center: single "Physical KV blocks" table (rows Block 0..7 roughly). Block 0 = `Four, score, and, seven`; Block 1 = `years, ago, our, fathers` with a "Ref count: 2 -> 1" annotation on Block 1 (shown in red) — meaning physical block 1 starts with reference count 2 (shared by A1 and A2) and after a write it drops to 1 for whichever copy stays.
- Left: "Logical KV blocks" for A1 — Block 0 = `Four, score, and, seven`; Block 1 = `years, ago, our, fathers` -> `mothers` (the diverging last token, shown appended/highlighted, e.g. "mothers" written into a new copy) — arrow labeled "Copy-on-write" pointing from logical block 1 down/across to a newly allocated physical block (physical block 3, holding `years, ago, our, fathers` + the new distinct token, e.g. "mothers").
- Right: "Logical KV blocks" for A2 — Block 0 = `Four, score, and, seven`; Block 1 = `years, ago, our, fathers`.
- Bottom of the physical table: another copy of block 0 content `Four, score, and, seven` shown reused by both.
Mechanism narrated in text: both outputs share the prompt's physical blocks (7 and 1, ref count 2 each). At generation, the two samples diverge in the *last* logical block; when A1 needs to write to its last logical block (block 1) and finds the reference count of the corresponding physical block > 1, it allocates a new physical block (physical block 3), copies the data from physical block 1, and decrements the old block's ref count. When A2 (ref count now 1) writes to physical block 1, it writes in place (no copy needed).
Caption: "Parallel sampling example."

### Figure 9 (page 7, right) — "Beam search example"
DIAGRAM, beam width k=4. Left column: "Beam candidate 0", "Beam candidate 1", "Beam candidate 2", "Beam candidate 3" (4 rows). Each candidate row points into a shared tree of blocks drawn as boxes: "Block 0", "Block 1", "Block 3" shared by all 4 candidates up to a dotted vertical line marking "prior to the iteration illustrated"; after the dotted line, candidates 0-2 share "Block 5" while candidate 3 diverges into its own "Block 2" (marked with an X, i.e., freed/evicted) and "Block 4" (also marked X, freed). After that, new blocks "Block 6" (shared further by candidates 0/1) and "Block 7" (shared further, candidate 2 and 3's new path) appear, and at the rightmost new-generation column: "Block 9", "Block 10", "Block 11", "Block 12" are newly allocated (one per surviving top-4 candidate), with "Block 8" also appearing (marked X, freed) between block 7's children and the new blocks.
Narrated mechanism (text): all 4 candidates share the first block (prompt, block 0). Candidates 0-2 share the first 3 blocks (0,1,3) and diverge at the 4th block; candidate 3 differs from the second block onward. At the next iteration, the top-4 highest-probability candidates all descend from candidates 1 and 2 (original candidates 0 and 3 are no longer in the top-k), so blocks belonging only to candidates 0 and 3 (blocks 2, 4, 8 — text says "blocks 2, 4, 5, 8" reach ref count 0 and are freed) are freed, and new physical blocks (blocks 9-12) are allocated for the new top-4 candidates' new tokens. After this step, all candidates share blocks 0,1,3; candidates 0 and 1 further share block 6; candidates 2 and 3 further share block 7.
Caption: "Beam search example."

### Figure 10 (page 8, left) — "Shared prompt example for machine translation. The examples are adopted from [5]."
DIAGRAM, two side-by-side sequence boxes:
- "Sequence A": "Prompt" region = "Translate English to French:" + example pairs "sea otter => loutre de mer", "peppermint => menthe poivrée", "plush giraffe => girafe en peluche" (this fixed instruction+examples block is the "Shared prefix", highlighted/boxed separately). Then "Task input" = "cheese?" Then "LLM output" (below, separate box) = "fromage".
- "Sequence B": same "Shared prefix" box content (identical text: "Translate English to French: sea otter => loutre de mer / peppermint => menthe poivrée / plush giraffe => girafe en peluche"). "Task input" = "I love you?" (shown as "I love you?" per figure; body just calls it a different task input). "LLM output" = "Je t'aime".
Labeled rows on the left margin: "Shared prefix", "Task input", "Task output" for each sequence.
Caption: "Shared prompt example for machine translation. The examples are adopted from [5]."

### Figure 11 (page 9, right) — "Input and output length distributions of the (a) ShareGPT and (b) Alpaca datasets" — DATA figure
Two histograms (density on y-axis, "# Tokens" on x-axis, range 0 to ~2000 for both).
(a) ShareGPT: y-axis "Density" scaled ×1e-2, visible range roughly 0.0 to 2.0 (ticks at 0.0, 0.5, 1.0, 1.5, 2.0). Legend: "Input (mean: 161.31)" (blue), "Output (mean: 337.99)" (orange). Both distributions are heavily right-skewed (tall spike near 0, long tail out to 2000).
(b) Alpaca: y-axis "Density" scaled ×1e-2, range roughly 0 to 8 (ticks 0,2,4,6,8). Legend: "Input (mean: 19.31)" (blue), "Output (mean: 58.45)" (orange). Even more sharply peaked near 0 with a long thin tail.
Exact means (printed in legend, not approximated): ShareGPT input mean 161.31 tokens, output mean 337.99 tokens; Alpaca input mean 19.31 tokens, output mean 58.45 tokens. Body text: "the ShareGPT dataset has 8.4× longer input prompts and 5.8× longer outputs than the Alpaca dataset, with higher variance."

### Figure 12 (page 10, bottom) — "Single sequence generation with OPT models on the ShareGPT and Alpaca dataset" — DATA figure
6 subplots in 2 rows x 3 columns, all with y-axis "Normalized latency (s/token)" range 0.0–1.0 (dashed gridlines at 0.25/0.5/0.75/1.0-ish, ticks 0.0,0.5,1.0) and x-axis "Request rate (req/s)". Legend common to all: "FasterTransformer" (gray line, x markers), "Orca (Max)" (red, x markers), "Orca (Pow2)" (orange, triangle), "Orca (Oracle)" (green, square), "vLLM" (blue, circle). Each series has a characteristic shape: flat near 0 latency at low request rate, then a knee where latency shoots up toward 1.0+ (queueing blow-up) at some critical request rate — the ranking of systems is by how far right (higher throughput) their knee occurs, with vLLM's knee furthest right in every subplot.

Approx critical/knee request rates (req/s) read off each subplot's x-axis (where curve shoots up toward y=1):
| Subplot | x-axis max shown | FasterTransformer knee | Orca(Max) knee | Orca(Pow2) knee | Orca(Oracle) knee | vLLM knee |
|---|---|---|---|---|---|---|
| (a) OPT-13B,1 GPU,ShareGPT | 0–2.0 | ~0.3 | ~0.5 | ~0.75 | ~1.1 | ~1.9 |
| (b) OPT-66B,4 GPU,ShareGPT | 0–1.0 | ~0.1 | ~0.2 | ~0.4 | ~0.65 | ~0.95 |
| (c) OPT-175B,8 GPU,ShareGPT | 0–2.5 | ~0.5 | ~1.0 | ~1.5 | ~1.8 | ~2.4 |
| (d) OPT-13B,1 GPU,Alpaca | 0–30 | ~5 | ~7 | ~13 | ~20 | ~28 |
| (e) OPT-66B,4 GPU,Alpaca | 0–20 | ~3 | ~5 | ~9 | ~13 | ~18 |
| (f) OPT-175B,8 GPU,Alpaca | 0–~22 | ~4 | ~6 | ~13 | ~16 | ~20 |
(all approx, read visually from curve knees)

Key text numbers (exact, quoted from §6.2): "On the ShareGPT dataset, vLLM can sustain 1.7×-2.7× higher request rates compared to Orca (Oracle) and 2.7×-8× compared to Orca (Max), while maintaining similar latencies... Compared to FasterTransformer, vLLM can sustain up to 22× higher request rates." Fig 12(f) exception: "vLLM's advantage over Orca (Oracle) and Orca (Pow2) is less pronounced" because OPT-175B's large GPU memory + Alpaca's short sequences make Orca baselines less memory-constrained there.

### Figure 13 (page 10, bottom-left) — "Average number of batched requests when serving OPT-13B for the ShareGPT (2 reqs/s) and Alpaca (30 reqs/s) traces" — DATA figure
Two bar charts, y-axis "# Batched requests".
(a) ShareGPT (y-axis 0–35): Orca (Max) = 7.00, Orca (Pow2) = 9.81, Orca (Oracle) = 13.62, vLLM = 30.42. (exact values printed on bars)
(b) Alpaca (y-axis 0–150): Orca (Max) = 7.00, Orca (Pow2) = 43.24, Orca (Oracle) = 72.75, vLLM = 132.44. (exact values printed on bars)
Text: "For example, as shown in Fig. 13a, for OPT-13B vLLM processes 2.2× more requests at the same time than Orca (Oracle) and 4.3× more requests than Orca (Max)."

### Figure 14 (page 11, top) — "Parallel generation and beam search with OPT-13B on the Alpaca dataset" — DATA figure
6 subplots, 2 rows × 3 cols, same axis style as Fig 12 (Normalized latency (s/token) 0.0-1.0 vs Request rate (req/s)); legend: Orca (Max) red, Orca (Pow2) orange, Orca (Oracle) green, vLLM blue (no FasterTransformer here).
Row 1 — parallel generation: (a) parallel size=2, x-axis 0–17ish; (b) parallel size=4, x-axis 0–10ish; (c) parallel size=6, x-axis 0–7ish. Knees move left (lower max throughput) as parallel size increases for all systems, but vLLM's relative advantage grows.
Row 2 — beam search: (d) beam width=2, x-axis 0–17ish; (e) beam width=4, x-axis 0–10ish; (f) beam width=6, x-axis 0–7ish.
Approx knee request rates (req/s):
| Subplot | Orca(Max) | Orca(Pow2) | Orca(Oracle) | vLLM |
|---|---|---|---|---|
| (a) parallel=2 | ~2 | ~7 | ~10 | ~15 |
| (b) parallel=4 | ~1.5 | ~4 | ~6 | ~10 |
| (c) parallel=6 | ~1 | ~2.5 | ~4 | ~6.5 |
| (d) beam=2 | ~2 | ~7 | ~10 | ~16 |
| (e) beam=4 | ~1 | ~4 | ~5.5 | ~9.5 |
| (f) beam=6 | ~1 | ~2.5 | ~4 | ~7 |
(all approx)
Text (exact): "the improvement of vLLM over Orca (Oracle) on OPT-13B and the Alpaca dataset goes from 1.3× in basic sampling to 2.3× in beam search with a width of 6."

### Figure 15 (page 11, right) — "Average amount of memory saving from sharing KV blocks, when serving OPT-13B for the Alpaca trace" — DATA figure
Two bar charts, y-axis "Memory saving (%)".
(a) Parallel sampling, x-axis "# Output sequences" = 2, 4, 6 (0-10% range): values 6.09, 8.53, 9.79 (exact, printed on bars).
(b) Beam search, x-axis "Beam width" = 2, 4, 6 (0-60% range): values 37.56, 53.13, 55.16 (exact, printed on bars).
Text (exact): "We show 6.1%-9.8% memory saving on parallel sampling and 37.6%-55.2% on beam search. In the same experiments with the ShareGPT dataset, we saw 16.2%-30.5% memory saving on parallel sampling and 44.3%-66.3% on beam search."

### Figure 16 (page 12, top) — "Translation workload where the input prompts share a common prefix" — DATA figure
Two subplots, same axis style (Normalized latency 0-1.0 vs Request rate req/s). Legend: Orca (Oracle) green, vLLM blue.
(a) "1-shot prefix prompt" (80-token prefix): x-axis approx 0–50; Orca(Oracle) knee ~20-25 req/s, vLLM knee ~40-45 req/s (approx).
(b) "5-shot prefix prompt" (341-token prefix): x-axis approx 0–50; Orca(Oracle) knee ~20 req/s, vLLM knee ~40 req/s (approx).
Text (exact): "vLLM achieves 1.67× higher throughput than Orca (Oracle) when the one-shot prefix is shared... when more examples are shared (Fig. 16 (b)), vLLM achieves 3.58× higher throughput than Orca (Oracle)." Model used: LLaMA-13B, dataset WMT16 English-to-German translation.

### Figure 17 (page 12, middle-left) — "Performance on chatbot workload" — DATA figure
Single subplot, Normalized latency (s/token) 0.0-1.0 vs Request rate (req/s), x-axis approx 0-0.9. Legend: Orca (Max) red, Orca (Pow2) orange, Orca (Oracle) green, vLLM blue. Orca variants all cluster together with knee around ~0.55-0.6 req/s (approx, described as behaving similarly to each other because they all reserve 1024 tokens for outputs under buddy allocation); vLLM knee further right around ~0.75-0.8 req/s (approx). Text (exact): "Fig. 17 shows that vLLM can sustain 2× higher request rates compared to the three Orca baselines." Uses OPT-13B, ShareGPT-based synthesized chatbot workload, prompt truncated to last 1024 tokens, output capped at 1024 tokens, no KV cache retained across conversation rounds.

### Figure 18 (page 12, right) — "Ablation experiments" — DATA figure, two subplots
(a) "Latency of attention kernels": y-axis "Kernel latency (us)" range ~0-250+ (ticks visible ~0,50,100,150,200,250), x-axis "Context length" with tick values 64, 128, 256. Series: "vLLM (bs 8)" (blue), "vLLM (bs 32)" (orange), "FT (bs 8)" (green), "FT (bs 32)" (red) — bs = batch size, FT = FasterTransformer. All four lines rise with context length; at each context length the two FT lines sit below (lower latency = better) the two vLLM lines, consistent with text's "20-26% higher attention kernel latency" for vLLM vs FasterTransformer. Approx values (us) at context length 256: vLLM bs32 ~185, vLLM bs8 ~110, FT bs32 ~150(approx), FT bs8 ~90(approx).
(b) "End-to-end latency with different block sizes": y-axis "Normalized latency (s/token)" range ~1.0 to ~17.5 (log-ish scale, ticks 1.0,2.5,5.0,7.5,10.0,12.5,15.0,17.5), x-axis "Block size" with tick values 1,2,4,8,16,32,64,128,256. Two series: "ShareGPT" (blue) and "Alpaca" (orange). Both curves are roughly flat/low (near the bottom, ~1-2) for block sizes 8 through ~64, then rise sharply for block size 128 and especially 256 (Alpaca rises earlier/more steeply than ShareGPT since Alpaca sequences are shorter). Text (exact): "In the ShareGPT trace, block sizes from 16 to 128 lead to the best performance. In the Alpaca trace, block size 16 and 32 work well, while larger block sizes significantly degrade performance... vLLM sets its default block size to 16."

### Figure 19 (page 13, top) — "(a) Overhead of recomputation and swapping for different block sizes. (b) Performance when serving OPT-13B with the ShareGPT traces at the same request rate" — DATA figure
(a) "Microbenchmark": y-axis "Time (ms)" range 0-140 (ticks 0,20,40,60,80,100,120,140), x-axis "Block size" ticks 1,2,4,8,16,32,64,128,256 (log scale). Four series: "Recompute" (red, roughly flat horizontal line around ~20-30ms across all block sizes — text: "the overhead of recomputation remains constant across different block sizes"), "Swap in" (orange), "Swap out" (green), "Swap in+out" (blue) — the three swap-related curves all start very high at block size 1 (~130-140ms) and decay steeply as block size increases, flattening out near block size 64-256 to roughly the same low level as recompute or slightly higher. Approx values: at block size 1, Swap in+out ≈ 135ms, Swap out ≈ 90ms(approx), Swap in ≈ 70ms(approx); at block size 256, all curves converge to roughly 10-20ms.
(b) "End-to-end performance": y-axis "Normalized latency (s/token)" range 0-2.5+ (ticks 0,0.5,1.0,1.5,2.0,2.5), x-axis "Block size" ticks 1,2,4,8,16,32,64,128,256. Two series: "Recompute" (red) and "Swap" (blue). Both start high at block size 1 (~2.2-2.5, approx) and decrease as block size grows; Recompute stays flat/low earlier and is generally at or below Swap for small block sizes, while for larger block sizes (64-256) the two curves are close/comparable. Text (exact): "swapping incurs excessive overhead with small block sizes... recomputation remains constant... recomputation is more efficient when the block size is small, while swapping is more efficient when the block size is large, though recomputation overhead is never higher than 20% of swapping's latency. For medium block sizes from 16 to 64, the two methods exhibit comparable end-to-end performance."

---

## KEY NUMBERS (with source section)

**§1 Introduction / Abstract**
- vLLM improves throughput of popular LLMs by 2-4× compared to state-of-the-art systems (FasterTransformer, Orca), with the same level of latency, without affecting model accuracy.
- Improvements more pronounced with longer sequences, larger models, more complex decoding algorithms.

**§1 / §2 Background**
- Processing an LLM request can be 10× more expensive than a traditional keyword query (cited estimate).
- Fig 1 example: 13B-param LLM on NVIDIA A100 40GB — ~65% of memory for model weights, ~30% for dynamic KV cache states, small remainder for activations.

**§3 Memory Challenges**
- KV cache per token for OPT-13B: 800 KB = 2 (key+value) × 5120 (hidden size) × 40 (layers) × 2 (bytes for FP16).
- OPT can generate sequences up to 2048 tokens ⇒ KV cache of one request up to 1.6 GB.
- Fig 2 memory waste %: existing systems' "actual effective memory" (token states) as low as 20.4% (Orca Max) up to 38.2% (Orca Oracle); vLLM reaches 96.3%.
- Beam search KV sharing: "up to 55% memory saving" mentioned in §3 as motivation (later measured precisely in Fig 15 as 55.16%).
- Parallel sampling prompt-KV sharing example: "the KV cache of the prompt part... accounts for 12% of the total KV cache memory in our experiment (§6.3)".

**§4 Method**
- Block size B: fixed number of tokens per KV block; default block size = 16 (chosen in §7.2 ablation).
- vLLM engine implementation: 8.5K lines of Python + 2K lines of C++/CUDA (§5).
- Kernel overhead: PagedAttention kernels incur 20-26% higher attention kernel latency compared to highly-optimized FasterTransformer implementation (§7.1, Fig 18a).
- Copy-on-write example (Fig 8): reference count starts at 2 (shared prompt blocks), drops to 1 after one sample writes its own new physical block.
- Beam search example (Fig 9): beam width k=4; blocks 2,4,5,8 freed (ref count → 0) when original candidates 0 and 3 drop out of top-k; new physical blocks 9-12 allocated.

**§5 Implementation**
- Table 1 (model sizes and server configurations):
| Model size | 13B | 66B | 175B |
|---|---|---|---|
| GPUs | A100 | 4×A100 | 8×A100-80GB |
| Total GPU memory | 40 GB | 160 GB | 640 GB |
| Parameter size | 26 GB | 132 GB | 346 GB |
| Memory for KV cache | 12 GB | 21 GB | 264 GB |
| Max. # KV cache slots | 15.7K | 9.7K | 60.1K |
- Also evaluated LLaMA-13B (for the shared-prefix/translation experiment, Fig 16, Fig 10).
- Fused GPU kernels implemented: (1) fused reshape and block write, (2) fusing block read and attention (adapted from FasterTransformer's kernel, one GPU warp per block for coalesced access, variable-length support), (3) fused block copy (batches copy-on-write copies into a single kernel launch instead of many small cudaMemcpyAsync calls).
- Decoding algorithm support via three primitive methods: fork, append, free.

**§6 Evaluation — datasets (Fig 11)**
- ShareGPT: input mean 161.31 tokens, output mean 337.99 tokens.
- Alpaca: input mean 19.31 tokens, output mean 58.45 tokens.
- ShareGPT has 8.4× longer input prompts and 5.8× longer outputs than Alpaca, with higher variance.
- Request arrival times synthesized via Poisson distribution at varying request rates.
- For most experiments: 1-hour traces; OPT-175B experiments use 15-minute traces due to cost limits.

**§6.2 Basic Sampling**
- ShareGPT: vLLM sustains 1.7×-2.7× higher request rates vs Orca (Oracle), 2.7×-8× vs Orca (Max), at similar latency. Up to 22× higher request rates vs FasterTransformer.
- Fig 13a (OPT-13B, ShareGPT @ 2 req/s): avg batched requests — Orca(Max) 7.00, Orca(Pow2) 9.81, Orca(Oracle) 13.62, vLLM 30.42. vLLM processes 2.2× more concurrent requests than Orca (Oracle), 4.3× more than Orca (Max).
- Fig 13b (Alpaca @ 30 req/s): Orca(Max) 7.00, Orca(Pow2) 43.24, Orca(Oracle) 72.75, vLLM 132.44.
- Fig 12(f) OPT-175B/Alpaca exception: vLLM's advantage smaller because 175B's large GPU memory + Alpaca's short sequences make the workload compute-bound rather than memory-bound.

**§6.3 Parallel Sampling and Beam Search**
- Alpaca, OPT-13B: memory saving from KV block sharing — parallel sampling 6.1%-9.8% (Fig 15a: 6.09, 8.53, 9.79 for 2/4/6 output sequences); beam search 37.6%-55.2% (Fig 15b: 37.56, 53.13, 55.16 for beam widths 2/4/6).
- Same experiments on ShareGPT: 16.2%-30.5% memory saving on parallel sampling, 44.3%-66.3% on beam search.
- Improvement of vLLM over Orca (Oracle) on OPT-13B/Alpaca: 1.3× (basic sampling) up to 2.3× (beam search, width 6).

**§6.4 Shared prefix**
- Model: LLaMA-13B (multilingual). Dataset: WMT16 English-to-German.
- 1-shot prefix (80 tokens): vLLM 1.67× higher throughput than Orca (Oracle).
- 5-shot prefix (341 tokens): vLLM 3.58× higher throughput than Orca (Oracle).

**§6.5 Chatbot**
- OPT-13B, ShareGPT-derived chatbot workload; prompt truncated to last 1024 tokens, output capped at 1024 tokens; no KV cache retained between conversation rounds.
- vLLM sustains 2× higher request rates than the three Orca baselines.

**§7 Ablation Studies**
- §7.1 Kernel microbenchmark: PagedAttention kernels have 20-26% higher attention kernel latency vs FasterTransformer's highly-optimized kernel (extra overhead from block table access, branches, variable-length handling); overhead confined to the attention operator, not other ops like Linear.
- §7.2 Block size: block size 16 gives best/near-best performance on ShareGPT (good range 16-128) and Alpaca (good at 16, 32; degrades badly for larger sizes since Alpaca sequences are shorter than large block sizes). Default block size = 16.
- §7.3 Recompute vs Swap (Fig 19): swapping incurs excessive overhead at small block sizes (many small CPU↔GPU transfers, PCIe bandwidth underused); recomputation overhead constant across block sizes since it doesn't touch KV blocks; recomputation overhead never exceeds 20% of swapping's latency; for medium block sizes (16-64) the two methods are comparable end-to-end.

**§8 Discussion**
- vLLM's paging ideas suit LLM serving specifically because output length is unknown a priori and performance is GPU-memory-bound; the authors note it would likely not help (or could hurt, due to memory-indirection overhead) for workloads with static tensor shapes (e.g., DNN training) or compute-bound non-LLM serving.
- LLM-specific adaptations vs classic OS virtual memory: all-or-nothing eviction policy (whole sequence's blocks evicted together, since all of a sequence's blocks are always accessed together); recomputation as a recovery mechanism (not typically used in OS); fused GPU kernels to offset memory-indirection overhead.

**§9 Related Work**
- Orca comparison: Orca's iteration-level scheduling and vLLM's PagedAttention are complementary — Orca increases GPU utilization via scheduling/interleaving; vLLM increases memory utilization so more requests' working sets fit in memory. vLLM achieves 2-4× speedup vs Orca by reducing fragmentation and enabling sharing.
- FlashAttention is noted as reducing peak memory of attention computation via tiling/kernel optimizations and reducing I/O, distinct from vLLM's block-level *memory management* for online serving.

---

## SECTION SUMMARIES

**§2 Background**
- LLMs are autoregressive Transformers: joint probability factorized token-by-token (Eq. 1); self-attention computes query/key/value per token (Eq. 2) and weighted output via softmax attention (Eq. 3).
- Serving has two phases: the prompt/prefill phase (processes all prompt tokens in parallel, compute-bound, generates first output token) and the autoregressive generation phase (one token at a time, reuses cached K/V from all previous positions, memory-bound, dominates latency).
- KV cache = the cached key/value vectors from earlier tokens, needed to generate each new token; grows and shrinks dynamically per request, unlike static-shaped DNN tensors.
- Batching improves GPU utilization but is complicated by requests arriving at different times and having widely varying lengths; naive batching causes queueing delay or wasteful padding.
- Fine-grained/iteration-level batching (cellular batching, Orca) processes at the iteration level, swapping in/out completed/new requests without waiting for the whole batch, and avoids padding via special kernels.

**§3 Memory Challenges in LLM Serving**
- KV cache size grows quickly with number of requests and can dominate GPU memory (per-token cost example: 800KB for OPT-13B).
- Existing systems store each request's KV cache as one contiguous chunk pre-allocated to the request's maximum possible length, causing internal fragmentation (unused pre-allocated space) and external fragmentation (unusable gaps between differently-sized chunks, as in buddy allocators).
- Complex decoding algorithms (parallel sampling, beam search) create opportunities to share KV cache across sequences/requests, but contiguous-memory storage prevents such sharing in existing systems.
- Unknown/variable input and output lengths force systems into conservative preallocation and complicate scheduling decisions (e.g., when to swap/evict).

**§4.1 PagedAttention**
- Inspired by OS virtual memory/paging: partitions each sequence's KV cache into fixed-size KV blocks instead of one contiguous tensor.
- Attention computation (Eq. 4) is reformulated as a block-wise sum over per-block partial attention scores/outputs, letting the kernel fetch and compute against each block separately regardless of its physical location.
- Enables non-contiguous physical storage of KV blocks, the foundation for eliminating fragmentation and enabling sharing.

**§4.2 KV Cache Manager**
- Analogous to OS virtual memory: logical KV blocks (contiguous from the requester's point of view) map to physical KV blocks (potentially scattered in GPU DRAM) via per-request block tables.
- Block tables store, per logical block, the physical block number and the count of filled positions.
- Physical memory need not be reserved for the whole max sequence length up front — blocks are allocated on demand as new tokens arrive, eliminating nearly all internal fragmentation (waste bounded by less than one block per sequence) and all external fragmentation (all blocks are the same fixed size).
- A CPU block allocator mirrors the GPU block allocator for swap space.

**§4.3 Decoding with PagedAttention and vLLM**
- Walks through single-sequence example (Fig 6): 7-token prompt `"Four score and seven years ago our"` fills logical 0→phys 7 (4/4) and logical 1→phys 1 (3/4, one reserved); first decode stores `"fathers"` in that reserved slot (filled 3→4); second decode allocates physical 3 for logical 2 and stores `"brought"`.
- Global per-iteration procedure: select candidate sequences for the batch, allocate physical blocks for newly required logical blocks, concatenate current-iteration input tokens across prefill and decode requests, run PagedAttention, save new KV into the assigned physical blocks.
- When a request finishes, its blocks are freed back to the pool for other requests.

**§4.4 Application to Other Decoding Scenarios**
- Parallel sampling: multiple outputs share one prompt's KV blocks (reference-counted); copy-on-write only needed on the final logical block when an output first needs to diverge/write into a shared block.
- Beam search: shares KV blocks not just for the prompt but dynamically across surviving beam candidates at every step (process-tree-like sharing); vLLM frees blocks whose reference count drops to zero when candidates are pruned, avoiding the large physical KV-cache copies needed by naive systems.
- Shared prefix: a common system-prompt/prefix's KV blocks can be precomputed and cached by the service provider; new requests map their prefix logical blocks to these cached physical blocks (last shared block marked copy-on-write) and only compute the task-specific suffix.
- Mixed decoding methods: because the logical-to-physical block indirection is hidden from the LLM execution kernel (which only sees physical block IDs), vLLM can batch requests using different decoding methods together, which existing systems cannot do efficiently.

**§4.5 Scheduling and Preemption**
- FCFS scheduling policy across all requests (fairness, no starvation).
- When GPU runs out of physical blocks for new tokens, vLLM must decide which blocks to evict and how to recover them; all sequences within a "sequence group" (e.g., beam candidates of one request) are gang-scheduled/preempted together since they may share memory.
- All-or-nothing eviction policy: evict either all or none of a sequence's blocks (since all of a sequence's blocks are always accessed together, unlike arbitrary OS page eviction heuristics).
- Two recovery mechanisms: swapping (copy evicted blocks to CPU RAM, like OS swap to disk; swap space bounded by total GPU KV-cache physical blocks) and recomputation (recompute KV by re-running one prefill iteration over the original prompt + already-generated tokens as a new "prompt", which can be cheaper than swapping since it's parallelized).

**§4.6 Distributed Execution**
- vLLM supports Megatron-LM-style SPMD tensor-model parallelism; attention heads are partitioned across GPU workers.
- Despite model-parallel execution, all shards process the same input tokens and need the same KV cache positions, so vLLM uses a single centralized KV cache manager shared by all GPU workers.
- Scheduler broadcasts control message (input token IDs + block tables) each step; workers execute and synchronize intermediate results via all-reduce without scheduler coordination; workers report sampled tokens back to the scheduler; workers don't need to synchronize on memory-management info beyond receiving it once per step.

**§5 Implementation**
- End-to-end serving system: FastAPI frontend (OpenAI API-compatible), GPU-based inference engine; 8.5K lines Python + 2K lines C++/CUDA. Supports GPT, OPT, LLaMA model families via PyTorch/Transformers/NCCL.
- Kernel-level optimizations: fused reshape+block write, fused block read+attention (adapted from FasterTransformer, per-block GPU warp for coalesced reads, variable sequence length support), fused block copy (batches copy-on-write copy operations into one kernel launch).
- Decoding-algorithm support built from three primitives: fork (new sequence from existing one), append (add a token), free (delete a sequence) — composed to implement parallel sampling, beam search, prefix sharing, etc.

**§6 Evaluation**
- Models: OPT-13B/66B/175B and LLaMA-13B; hardware: Google Cloud A2 instances with NVIDIA A100 GPUs (see Table 1).
- Baselines: FasterTransformer (highly latency-optimized distributed inference engine, given a custom dynamic-batching scheduler for fair comparison) and three re-implemented variants of Orca (Oracle = knows true output lengths in advance = infeasible upper bound; Pow2 = over-reserves by up to 2×; Max = always reserves up to model's max sequence length, 2048).
- Key metric: normalized latency = mean of each request's end-to-end latency divided by its output length, plotted against request rate; a good system keeps normalized latency low even as request rate increases.
- Across basic sampling, parallel sampling, beam search, shared-prefix translation, and chatbot workloads, vLLM consistently sustains substantially higher request rates than all baselines before normalized latency explodes.

**§7 Ablation Studies**
- Kernel microbenchmark: quantifies PagedAttention's own overhead (20-26% slower attention kernel vs FasterTransformer) in isolation from the system-level batching benefits.
- Block size sweep: shows a sweet spot at 16 that balances GPU parallelism (needs enough tokens per block) against internal fragmentation and reduced sharing opportunities (needs small enough blocks); Alpaca (short sequences) is more sensitive to overly large blocks than ShareGPT.
- Recompute vs swap sweep across block sizes: recomputation cost is independent of block size; swap cost is dominated by small-transfer overhead at small block sizes and improves as block size grows; end-to-end the two are comparable for medium block sizes.

**§8 Discussion**
- The virtual-memory/paging idea generalizes best to workloads with dynamic, a-priori-unknown memory needs and where performance is memory-capacity-bound — true for LLM serving, not necessarily true for DNN training (static tensor shapes) or compute-bound serving.
- vLLM adds LLM-specific twists to classic OS techniques: all-or-nothing eviction (vs arbitrary page eviction), recomputation as a recovery method (not standard in OS), and kernel fusion to hide the cost of non-contiguous/indirected memory access.

**§9 Related Work**
- General model-serving systems (Clipper, TensorFlow Serving, Nexus, InferLine, Clockwork, DVABatch, REEF, Shepherd, AlpaServe) address batching/caching/placement/scheduling generically but miss LLM-specific autoregressive/KV-cache optimization opportunities.
- Specialized Transformer-serving systems use kernel optimizations, advanced batching, model/parameter parallelism/sharing; Orca is the closest prior work (see comparison in Key Numbers above).
- Memory optimization literature for training (swapping, recomputation, and their combination) and inference (FlexGen for offline single-GPU LLM inference, OLLA for tensor lifetime/placement, FlashAttention for reducing attention's peak memory/I-O via tiling) are related but distinct from vLLM's online, block-level KV cache memory management contribution.

**§10 Conclusion**
- Introduces PagedAttention (non-contiguous paged KV cache storage) and vLLM (high-throughput serving system built on it).
- Demonstrates that OS techniques (virtual memory, copy-on-write) can be adapted to manage KV cache and support diverse decoding algorithms.
- Achieves 2-4× throughput improvement over state-of-the-art serving systems.

---

## EXACT MECHANISM DETAILS

**PagedAttention block-wise attention computation (§4.1, Eq. 4, in plain text)**
Given block size B, the key block K_j = the key vectors at positions (j-1)B+1 through jB; value block V_j similarly for values. Standard attention (Eq. 3) is: attention weight a_ij = softmax over t of (q_i^T k_t / sqrt(d)) for t = 1..i; output o_i = sum over j=1..i of a_ij * v_j.
PagedAttention reformulates this per-block: for the j-th KV block,
- A_ij (row vector of attention scores on block j) = exp(q_i^T K_j / sqrt(d)) / [ sum over t=1..⌈i/B⌉ of exp(q_i^T K_t 1 / sqrt(d)) ]
- o_i = sum over j=1..⌈i/B⌉ of V_j A_ij^T
In plain words: the query vector for token i is multiplied against the key vectors within each relevant block separately to get that block's partial (unnormalized) attention scores; a running/global softmax normalizer is computed across all blocks up to token i's block; then the output is accumulated block-by-block by multiplying each block's value vectors by that block's (normalized) attention-score sub-vector and summing across all blocks. This lets the GPU kernel fetch and process each KV block independently and combine results, regardless of the blocks' physical (dis)contiguity.

**KV Cache Manager / block tables (§4.2)**
- Request's KV cache = ordered list of logical KV blocks, filled left-to-right as new tokens are generated; last logical block has some unfilled ("reserved") positions.
- Block table: one entry per logical block, recording (a) the physical block number it currently maps to, and (b) the number of filled positions ("# filled") in that physical block.
- Block engine allocates a contiguous chunk of GPU DRAM (also mirrored on CPU RAM for swap) and divides it into fixed-size physical KV blocks; the manager hands out blocks on demand.
- New physical block is only allocated once all previously allocated blocks for that sequence are completely full — bounds per-sequence internal-fragmentation waste to less than one block.

**Copy-on-write with reference counts (§4.4, Fig 8)**
- When multiple sequences (e.g., parallel samples or beam candidates) share a prompt's physical blocks, each shared physical block carries a reference count equal to the number of sequences currently mapped to it (e.g., 2 for two parallel samples sharing the prompt).
- When a sequence needs to write new KV data into a logical block whose corresponding physical block has reference count > 1, vLLM: (1) allocates a new physical block, (2) instructs the block engine to copy the existing block's data into the new block, (3) decrements the original physical block's reference count by 1, and (4) writes the new data into the newly allocated block (which now has ref count 1, owned solely by the writer).
- If reference count is already 1 (only one owner remains), the write happens in place with no copy.
- This mechanism is only invoked when the newly generated token(s) fall within an old, still-shared block; once a sequence has its own private block going forward, subsequent writes there need no further copying.

**All-or-nothing eviction (§4.5)**
- Because every block belonging to one sequence is always accessed together, standard fine-grained OS page-eviction heuristics (predicting which individual page is accessed furthest in the future) don't apply well.
- vLLM's rule: evict either *all* of a sequence's blocks, or *none* — never a partial subset. Sequences within a sequence group (e.g., all beam candidates of one request, which may share memory) are gang-scheduled: preempted or resumed together as a unit.

**Swap vs Recompute (§4.5, quantified in §7.3 / Fig 19)**
- Swapping: evicted physical blocks are copied to a CPU RAM swap space managed by a CPU block allocator; once GPU runs out of free physical blocks, vLLM selects a set of sequences to evict, stops admitting new requests until all currently-preempted sequences are done being swapped, then evicts and transfers their blocks to CPU. On completion (or when re-scheduled), blocks come back from CPU to GPU. Swap space size is naturally bounded by total GPU KV-cache blocks. Swapping is efficient at large block sizes (fewer, larger transfers use PCIe bandwidth well) but has excessive overhead at small block sizes (many tiny transfers).
- Recomputation: simply discards the evicted KV cache and, when the sequence is rescheduled, recomputes its KV cache in one prefill-style forward pass over the original prompt tokens concatenated with the already-generated output tokens (treated as a new "prompt"). This recomputation is fast because it parallelizes over all positions at once (like the original prefill phase) instead of doing sequential per-token generation. Its overhead is constant regardless of block size (no KV blocks are read/written) and never exceeds ~20% of swapping's latency in the measured microbenchmark; for block sizes 16-64 the two methods perform comparably end-to-end, but recompute wins clearly at small block sizes.

**Distributed execution KV cache sharing (§4.6)**
- Under SPMD tensor-parallel execution (Megatron-LM style), attention is split by head across GPU workers; every shard nonetheless processes the same input token sequence and needs KV cache for the same positions.
- vLLM therefore keeps one single centralized KV cache manager (in the scheduler) shared across all GPU workers/model shards — each worker stores only the KV data for its own subset of attention heads, but all workers use the identical logical-to-physical block mapping distributed by the scheduler each iteration.
- Per iteration: scheduler sends each worker the input token IDs and block table for the batch; workers run the model, reading/writing KV cache per the block table, and synchronize intermediate activations via all-reduce (no scheduler involvement needed for that); workers return sampled tokens to the scheduler at the end of the iteration.

**Fused GPU kernels (§5.1)**
1. Fused reshape and block write: in every Transformer layer, newly computed KV vectors are reshaped into the block-optimized memory layout and written directly to their block-table-specified positions in one fused kernel, minimizing launch overhead.
2. Fusing block read and attention: adapted from FasterTransformer's attention kernel to read KV cache according to the block table and perform attention on the fly in a single kernel; one GPU warp reads each block to ensure coalesced memory access; supports variable sequence lengths within a batch.
3. Fused block copy: batches the (potentially numerous, small) copy-on-write block-copy operations for a step into a single kernel launch instead of many small `cudaMemcpyAsync` calls, to avoid per-call overhead.

**Decoding-primitive API (§5.2)**
- `fork`: creates a new sequence from an existing one (used to spin up parallel-sampling output sequences or beam-search children sharing the parent's blocks).
- `append`: appends a newly generated token (and its KV) to a sequence.
- `free`: deletes a sequence (and releases/decrements reference counts on its blocks) once it meets a stopping condition.
- All supported decoding algorithms (parallel sampling, beam search, shared-prefix) are implemented by composing these three primitives.
