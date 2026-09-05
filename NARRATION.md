# PagedAttention Talk — Speaker Narration

Auto-generated from the `NARRATION` section of each scene's module docstring in `talk/sN_*.py`. Regenerate with `make narration` (or `python tools/build_narration.py`) after editing any scene. Do not hand-edit this file.

## Contents

### Act I: Why memory is the bottleneck

- [S0 S0Title (~1-2 min)](#s0-s0title)
- [S1 S1Transformers (~4-5 min)](#s1-s1transformers)
- [S2 S2GPU (~3 min)](#s2-s2gpu)
- [S3 S3KVCache (~4 min)](#s3-s3kvcache)
- [S4 S4Problem (~6-7 min)](#s4-s4problem)

### Act II: PagedAttention

- [S5 S5PagedAttention (~7-8 min)](#s5-s5pagedattention)
- [S6 S6OSAndWhyHard (~4-5 min)](#s6-s6osandwhyhard)

### Act III: What it buys you

- [S7 S7Sharing (~5 min)](#s7-s7sharing)
- [S8 S8Scheduling (~4 min)](#s8-s8scheduling)
- [S9 S9Results (~4-5 min)](#s9-s9results)
- [S10 S10Ablations (~2 min)](#s10-s10ablations)
- [S11 S11Takeaways (~1-2 min)](#s11-s11takeaways)

## S0 S0Title (~1-2 min)

Beat 1 — Title card
This talk is about "Efficient Memory Management for LLM Serving with PagedAttention" —
Kwon et al., SOSP '23, the paper behind vLLM. If you've used an LLM API, this is the
system idea that made it fast and cheap. [PAUSE] Show of hands: who's wondered why LLM
APIs are so cheap?

Beat 2 — The cost hook
Here's a fact that surprised me: a single LLM request can cost roughly ten times what a
keyword search costs. A search is basically a lookup; an LLM request runs a
many-billion-parameter network, one step per output token. That gap is why serving
efficiency matters. [PAUSE]

Beat 3 — The memory hook
A big chunk of that cost is memory, not compute. On a GPU serving a large model, over 30%
of GPU memory goes to the "KV cache" — we'll build that up shortly. The catch: existing
systems only put 20 to 40% of it to use — that waste is this paper's target.

Beat 4 — Roadmap
Here's the shape of the talk, in three acts. Act I: why memory — not compute — is the
real bottleneck. Act II: the paper's core idea — chop the KV cache into small fixed-size
blocks and manage them on demand. Act III: the payoffs — the actual speedups and sharing
tricks. Let's start with the bottleneck.

## S1 S1Transformers (~4-5 min)

Beat 1 — Autoregressive generation
Here's how an LLM actually produces text: it does not write a whole sentence at once.
It looks at everything so far, predicts a single next token, appends that token to the
sequence, and repeats. Prompt in, one token out, append, repeat — that loop is the
entire generation process. Watch it run twice: from "Four score and seven" it produces
"years", then from "Four score and seven years" it produces "ago". [PAUSE] Notice: every
step re-reads the whole sequence so far.

Beat 2 — Q, K, V intuition
To decide what comes next, the model turns every token into three vectors. A Query:
"what am I looking for right now?" A Key: "what do I contain, that others might look
for?" And a Value: "what I'll actually contribute if someone attends to me." Only the
newest token needs a fresh Query; every token — old and new — carries a Key and a
Value.

Beat 3 — Query meets every Key
The newest token's Query gets compared against the Key of every token in the sequence,
including its own. Each comparison produces one number, a raw "score" — query dot key —
that says roughly how relevant that earlier token is to what we're looking for right
now. [PAUSE] Which earlier token do you think gets the highest score here?

Beat 4 — Softmax turns scores into weights
Those raw scores get squashed by softmax into weights that are all positive and sum to
one — a probability distribution over "how much attention to pay to each earlier
token." Bigger score, bigger slice of attention.

Beat 5 — Weighted sum → output → next token
Now take every token's Value vector, scale it by its attention weight, and add them all
up. That weighted sum of values is the output of this step, and it's what the model
turns into the next predicted token. Here, that's "years" — which gets appended right
back onto the sequence.

Beat 6 — One step later: the stacks grow
Run the loop again to produce "ago": the sequence is now one token longer, so there's
one more Key and one more Value in play than last time. Every single generation step
adds exactly one new K and one new V — and none of the old ones ever get thrown away,
because the next step still needs to compare against them too.

Beat 7 — The landing point
So here's the load-bearing fact for the rest of this talk: to produce the very next
token, you need the Key and the Value of every previous token, not just the most recent
one. That K/V state has to sit somewhere and stick around for the entire request. Keep
an eye on those colored stacks — that's exactly what we'll come back to.

## S2 S2GPU (~3 min)

---------
Beat 1 — Same math, everywhere.
Every decoding step we just saw is, under the hood, the same handful of
matrix multiplies applied over and over: one token in, one token's worth of
math against every weight matrix in the model. Now imagine that happening
for every layer, and — once we start batching requests — for many tokens at
once. It's not complicated math. It's just an enormous amount of *identical*
math, fanned out over and over. [PAUSE] That "same operation, many times"
shape is exactly what a GPU is built for.

Beat 2 — CPU vs GPU.
A CPU has a handful of big, fast cores — great at doing one complicated
thing quickly, one after another. A GPU flips that trade: thousands of
small, simple cores. Give it one huge matrix multiply and it slices the
work into thousands of tiny pieces and runs them all at the same time, one
piece per core. Our "do the same multiply-and-add for every token" workload
maps almost perfectly onto that grid of cores. [PAUSE]

Beat 3 — Batching amortizes the weights.
Here's the trick that makes serving efficient: the model's weights are
identical for every request — request 1, request 2, request N all multiply
against the exact same matrices. So instead of loading those weights once
per request, we load them once and run many requests' tokens through them
in the same pass. Throughput becomes a question of how many requests we can
batch together into one pass over the same weights. More batching, more
throughput — for free, almost.

Beat 4 — The catch: VRAM is limited.
Almost for free. Here's the catch: the GPU doesn't borrow the computer's
regular memory — it has its own memory, called VRAM, and it's finite. An
A100 GPU, for example, has 40GB of it. Anything the model touches while it
runs — the weights, the intermediate activations, and any per-request state
we want to keep around — has to fit inside that 40GB. The weights alone for
a 13-billion-parameter model like OPT-13B are about 26GB. That's already
almost two-thirds of an A100's memory, gone, before we've served a single
request. [PAUSE]

Beat 5 — Landing.
So: weights take a big, fixed bite out of VRAM the moment the model loads.
Whatever is left over is what we have to work with for everything else —
and that leftover space is what decides how many requests we can actually
batch together at once.

## S3 S3KVCache (~4 min)

---------
Beat 1 — Recomputing is wasteful.
Remember: to generate the next token, the model needs the key and value
vectors of every token that came before it. The naive way to get those is
to just recompute them — at every single decoding step, run every previous
token back through the model to rebuild its K and V. Step 2 recomputes
token 1's K/V. Step 3 recomputes tokens 1 and 2's. Step 10 recomputes nine
tokens' worth, all over again, just to add one new token. [PAUSE] That
triangle of repeated work only grows as the sequence gets longer.

Beat 2 — Cache them instead.
So don't recompute — cache them. The first time we compute a token's K and
V, we keep them around, and every later step just reads them back and adds
one new pair for the newest token. This is "the KV cache": for a given
request, one row that grows by exactly one K,V pair per generated token.
Back to our example — "Four score and seven years ago our
fathers" — each token box with its cached K (blue) and V (purple) sitting
right beneath it.

Beat 3 — Per request, and it lives in VRAM.
Every request gets its own cache — it's a per-request structure, not
shared. Two requests running at once means two separate caches, growing
independently, side by side. And remember where all of this has to live:
in the GPU's own limited VRAM, right alongside the model's weights.

Beat 4 — It's surprisingly big.
Here's the number that makes this matter: for OPT-13B, one token's K and V
together cost about 800 kilobytes. That comes from 2 — one for K, one for V
— times 5120, the hidden size, times 40 layers, times 2 bytes per value in
FP16. [PAUSE] Multiply that out over a full 2048-token request and you get
roughly 1.6 gigabytes — for a single request's cache.

Beat 5 — The memory budget.
Put it on the same picture as the weights: on a 13-billion-parameter model
serving on an A100's 40GB, about 65% of that memory is the model's
parameters, more than 30% is KV cache, and a small remainder is other,
short-lived activation memory. Weights are fixed the moment the model
loads. KV cache is the only part of this picture that grows and shrinks
while we're serving.

Beat 6 — Landing: KV cache decides batch size.
That flexible region — the KV cache slice of the budget — is the one part
we get to spend. How many requests' KV caches we can fit into it is exactly
how many requests we can batch together, which is exactly our throughput.
[PAUSE] So the question becomes: how do we actually lay all of these
per-request, growing caches out in that memory?

## S4 S4Problem (~6-7 min)

---------
Beat 1 — The question.
How do you allocate memory for something whose final size is unknown? A
request's output length is unknown until the model itself emits an
end-of-sequence token — there is no header, no content-length field, nothing
that tells you in advance how long the answer will be.
[PAUSE]
Take a second — how would you design this? (Typical answers to react to:
"guess and reallocate as you go" — reallocating a growing tensor is exactly
what causes copies and stalls; "use a linked list of small chunks" — closer
to the real answer, but on a GPU kernel indirection is expensive; "just
reserve the maximum possible length" — this is in fact what every serving
system did before this paper, and it is our next beat.)

Beat 2 — Reserve the maximum.
Systems like FasterTransformer and Orca solve "unknown final size" the
simplest possible way: they pre-allocate one contiguous chunk of GPU memory
sized to the model's maximum sequence length — for OPT, that is 2048 slots —
the moment a request arrives. Request A claims its 2048-slot strip up front,
whether it ends up needing 10 tokens or 2000.

Beat 3 — Fig. 3, zoomed in on Request A.
Let's put real tokens on this strip: "Four score and seven years ago our" —
seven prompt tokens already have KV cache computed, shown filled. "brought"
is the current iteration — the token being generated right now. A couple of
slots just past it are reserved for the immediate next tokens — idle right
now, but nobody else can borrow them. And then: 2038 slots, allocated the
instant the request arrived, that this request will never touch if it stops
early. That's internal fragmentation — memory that belongs to a request but
holds nothing.

Beat 4 — Request B arrives, and a second waste appears.
Request B, "You only live once," gets its own reserved strip the same way:
3 tokens filled, "once" as the current iteration, a couple of reserved
slots, and 507 slots of its own internal fragmentation. But look at the gap
the allocator leaves between A's chunk and B's chunk — it's real free
memory, but it's the wrong shape for any other request's reservation to fit
into. That gap is external fragmentation, and it's dead until both A and B
finish.

Beat 5 — Zoom out: Fig. 2, the whole KV region.
Across a real serving run, the paper measured this precisely. Orca that
always reserves the max: only 20.4% of its KV memory is holding actual
token state — the rest is reservation, internal fragmentation, and external
fragmentation. Even Orca's best-case, oracle-knows-the-future variant only
reaches 38.2%. [PAUSE] So: across existing systems, only 20 to 40% of the
KV memory you paid for is doing any work. What do you think the system in
this paper achieves? We'll come back to that number.

Beat 6 — This problem is new.
Here's something worth sitting with: paging was never needed before large
language models. A pre-LLM deep learning tensor — a batch of images, say —
has a fixed, known shape before you ever run the model: 32 images, 3
channels, 224 by 224 pixels. Contiguous allocation is perfect for that,
there is nothing to fragment. The KV cache is different in kind: its length
grows one token at a time, per request, and nobody — not the system, not
the model — knows where it stops until it stops.

Beat 7 — The second gap: no sharing.
Contiguous allocation has a second, quieter cost. If two requests share the
same prompt — say, two parallel samples of one question — a contiguous
system stores that identical prompt's KV cache twice, once per request,
because each request owns one indivisible chunk. There is no way to point
two requests at the same physical memory when memory is handed out as
monolithic strips. Every duplicate prompt is wasted memory that a smarter
layout wouldn't need to pay for at all.

Beat 8 — Summary.
So we have four distinct wastes stacked on top of each other: reservation
for tokens not yet generated, internal fragmentation from guessing a max
length wrong, external fragmentation between requests, and duplicated
memory because identical content can't be shared. All four come from one
design choice: forcing each request's KV cache into one contiguous block.
Put simply — memory, not compute, caps how many requests a GPU can serve at
once.

Beat 9 — Seam to Act II.
[act_checkpoint — presenter narrates while it plays]

## S5 S5PagedAttention (~7-8 min)

Beat 1 — The idea in one line
Here's the paper's fix, in one sentence: instead of one growing, contiguous slab
of memory per request, chop each request's KV cache into small fixed-size
chunks called blocks. The paper's default block size is 16 tokens; to keep the
pictures readable I'll draw blocks of 4. Watch our running example, "Four score
and seven years ago our fathers brought forth," get sliced into three blocks of
four tokens each — the last one only half full. [PAUSE] Note that block size is
fixed once and for all, chosen ahead of time — it does not depend on how long
any particular request turns out to be.

Beat 2 — Logical vs. physical blocks
Here's the key trick: blocks don't have to live next to each other in memory.
Each request has a logical view — its blocks in order, 0, 1, 2 — but those
logical blocks can be scattered anywhere in physical GPU memory. A block table
records, for each logical block, which physical block it actually lives in, and
how many of its slots are currently filled. And blocks are handed out lazily:
vLLM only grabs a new physical block once the current last block is completely
full. That laziness is what kills internal fragmentation.

Beat 3 — The kernel: attention over scattered blocks (Fig. 5)
So if the blocks are scattered, how does attention even work? This is the
paper's Figure 5. The query vector for the newest token, "forth," still has to
attend to every earlier token — but those tokens now live in three separate,
non-contiguous physical blocks. The PagedAttention kernel just fetches each
block on its own, computes a partial attention score against it, and combines
the partial results at the end — that's the paper's Equation 4, block-by-block
attention instead of one contiguous sweep. Non-contiguous memory stops being a
problem the moment your kernel is written to expect it.

Beat 4 — Decode walkthrough: prefill (Fig. 6)
Let's walk through this exactly the way the paper does, step by step. Prompt:
"Four score and seven years ago our fathers," eight tokens. Step ①, prefill, packs the first
four tokens, "Four score and seven," into logical block 0, which lands on
physical block 7; the remaining four tokens, "years ago our fathers," go into logical
block 1 on physical block 1. The block table now
has two rows: logical 0 to physical 7, four filled; logical 1 to physical 1,
four filled.

Beat 5 — Step ② — block table grows
First decode step generates "brought." Logical block 1 is completely
full, so vLLM allocates a brand-new logical block 2, mapped to a fresh physical
block 3, and adds a new row to the block table. This is the only moment a new
block ever gets allocated: exactly when the previous one is completely full.

Beat 6 — Step ③ — fill the new block
Second decode step generates "forth," landing in physical block 3 right next to
"brought" — filled count ticks from one to two.

Beat 7 — The punchline on waste
Notice the pattern: at any
moment, at most one block per request is partially empty. Everything else is
either completely full or not yet allocated. That's why the paper measures up
to 96.3% of KV cache memory actually holding real token state — remember that
question mark from the waste chart earlier? This is the answer.

Beat 8 — Two requests at once (Fig. 7)
None of this is special to one request. Here's Request B, "it was the best of
times," arriving while Request A is still running. Its logical blocks land on
physical blocks 2 and 4 — completely interleaved with Request A's blocks 7, 1,
and 3. Blocks 0, 5, and 6 are still sitting free, available to whichever
request needs them next. Physical layout has nothing to do with logical order
anymore.

Beat 9 — Freeing blocks
When a request finishes, every physical block it was using goes straight back
to the free pool — instantly, no matter where in memory those blocks happened
to be. There's no need to find a same-sized contiguous hole for the next
request, because blocks are always the same fixed size. That's external
fragmentation eliminated entirely; the only fragmentation left is at most one
partial block per live request.

Beat 10 — Recap
So: fixed-size blocks, a block table mapping logical to physical, allocation
only on demand, and a kernel built from the ground up to read scattered blocks
and combine the results. Four ideas, and together they take KV cache
utilization from roughly 20 to 40 percent up to 96.3%.

## S6 S6OSAndWhyHard (~4-5 min)

---------
Beat 1 — The reveal.
Look at what we just built: logical blocks for a request, a block table that
maps them to physical slots, and a pool of physical blocks on the GPU. [PAUSE]
Take a step back and squint at this picture. Logical blocks that get mapped,
on demand, onto scattered physical blocks through a table... this is exactly
what an operating system does when it manages a process's memory. We just
reinvented virtual memory paging — for the KV cache.

Beat 2 — The same picture, OS vocabulary.
Same picture, new labels. What we called a block, the OS calls a page. What
we called a token, the OS calls a byte. What we called a request, the OS
calls a process. And our block table is just a page table, mapping a
process's virtual pages onto physical RAM. It's the same idea, one layer
lower in the stack.

Beat 3 — Paging, for anyone who skipped the OS class.
Here's the whole idea in one breath: every process gets to believe it has
one big, contiguous chunk of memory. Underneath, the page table quietly
scatters that memory across whatever physical frames happen to be free.
Frames get handed out only when the process actually touches that page — not
up front — so there's no need to reserve a giant contiguous region, and no
external fragmentation between processes.

Beat 4 — So was this just copying the OS?
So, fair question: did we just copy fifty-year-old operating systems ideas
onto a GPU and call it a paper? [PAUSE] No. The idea transfers, but making it
work on a GPU, for attention, required real systems work that a textbook page
table never has to do.

Beat 5 — Reason one and two: no MMU, and the kernel itself changes.
First: your CPU has dedicated hardware for this — a memory management unit
that walks page tables and a TLB that caches recent translations, all in
silicon, off the critical path. The GPU has none of that for our purposes;
vLLM does every logical-to-physical translation in software, inside the
kernel, on every access. Second, and bigger: an OS page fault handler never
touches your program's computation — it just finds the page and hands control
back. Here, the computation *is* the memory access. The attention kernel
itself had to be rewritten to gather scattered KV blocks fast: a fused
reshape-and-write kernel, a fused block-read-and-attention kernel, and a
fused block-copy kernel for copy-on-write.

Beat 6 — Reason three and four: this isn't a rare fault, and OS policy doesn't fit.
Third: a page fault is a rare event — maybe once every few thousand
instructions. Our "fault" happens on *every token, every step*: attention
reads every block, every time. That's why the PagedAttention kernel itself
runs about 20 to 26 percent slower than FasterTransformer's kernel on
contiguous memory — Figure 18(a) in the paper. And yet the end-to-end system
is 2 to 4 times faster, because the memory efficiency gain swamps that
per-kernel cost. [PAUSE] Fourth: the OS doesn't know our workload. We needed
domain-specific policies it never had — all-or-nothing eviction of a whole
sequence's blocks at once, gang-scheduling sequences that share blocks, and
tuning the block size itself. More on those shortly.

Beat 7 — Landing.
So: an OS idea, re-engineered for a workload the OS was never designed for.

Beat 8 — Checkpoint.

## S7 S7Sharing (~5 min)

---------
Beat 1 — Parallel sampling, the setup.
A common trick for better outputs: take one prompt and sample several
completions from it — "give me 2 tries at an answer." Each sample, call them
A1 and A2, is really its own sequence with its own KV cache. But they all
start from the exact same prompt. [PAUSE] In a contiguous-memory system,
that means copying the prompt's entire KV cache once per sample. With
blocks, we don't have to.

Beat 2 — Same physical blocks, two logical views.
Here's Figure 8 from the paper. A1's block table and A2's block table both
map their logical blocks 0 and 1 to the *same* physical blocks. No copy has
happened — both samples are just pointers into one shared region. Each
shared physical block carries a reference count: here, 2, because two
logical blocks point at it.

Beat 3 — Copy-on-write.
Block 0 is completely full — four out of four prompt tokens — so nobody
ever needs to write to it again; it can stay shared forever. Block 1 is
different: it still has one open slot. When A1 generates its next token and
tries to write into that slot, vLLM checks the reference count, sees it's 2,
and refuses to write in place. Instead: allocate a fresh physical block,
copy the old block's contents into it, write A1's new token into the copy,
and decrement the original block's count to 1. [PAUSE] A2 keeps using the
original block, untouched. Notice the cost: only that *last*, not-yet-full
block is ever copied. Every earlier, fully-packed block just stays shared.

Beat 4 — Beam search shares a tree.
Beam search shares even more aggressively. Multiple candidate beams don't
just share the prompt — they share prefixes of each other's generated
tokens too, because at every step they're extensions of a shared history.
Picture it as a tree: one shared trunk, branching into candidates, each
candidate touching only the new blocks it needed to extend the sequence.

Beat 5 — And it changes every step.
Now watch what happens when a candidate falls out of the top-k and gets
dropped. Every physical block that only it was using has its reference
count drop to zero — and those blocks go straight back to the free pool,
instantly, without any bulk copy. [PAUSE] The sharing pattern is being
recomputed, cheaply, at every single decoding step.

Beat 6 — Shared prefix / system prompt.
This same trick shows up in production systems today: a fixed instruction
or a handful of few-shot examples that every request in a workload shares.
vLLM computes that shared prefix's KV blocks exactly once, caches them, and
every new request's block table just points at those cached blocks. Only
the request-specific suffix — the actual question — needs new computation
and new blocks.

Beat 7 — What it's worth.
On the paper's numbers: parallel sampling saves 6.1 to 9.8 percent of memory
on Alpaca, and 16.2 to 30.5 percent on ShareGPT. Beam search saves far more
— 37.6 to 55.2 percent on Alpaca, 44.3 to 66.3 percent on ShareGPT. [PAUSE]
None of this is possible in a contiguous-memory system — there, "sharing" a
block would mean the copies have to physically exist somewhere, defeating
the point.

Beat 8 — Landing.
Two small mechanisms — reference counts and copy-on-write — turn a memory
layout trick into free, automatic sharing across requests.

## S8 S8Scheduling (~4 min)

---------
Beat 1 — The vLLM system overview (Fig 4).
Zoom out to how vLLM is actually organized. A central Scheduler decides which
requests run in each step. It talks to a KV Cache Manager, which is the one
place that owns every block table in the system — logical block to physical
block, for every request. Underneath, a CPU Block Allocator and a GPU Block
Allocator hand out physical blocks on each side. And on the right, a row of
Workers — Worker 0 through Worker N-1 — each on its own GPU, each running a
shard of the model plus a Cache Engine that moves blocks around exactly when
the scheduler tells it to. [PAUSE] One brain, many hands.

Beat 2 — Running out of room.
In normal operation this is boring: requests arrive, the scheduler admits
them, blocks get allocated on demand as each one generates tokens, and the
batch grows. But GPU memory is finite, so eventually the free block pool runs
dry — some request's next token needs a new block, and there isn't one.
[PAUSE] Which request gives up its memory, and what do we do with it?

Beat 3 — Policy: first-come, first-served.
vLLM's answer is deliberately simple: first-come, first-served, with
preemption. Requests are served oldest-first; when the scheduler must free
space, it preempts the most recently arrived request first. That guarantees
no request starves waiting behind an endless stream of newer arrivals — the
one that's been running longest is the last one ever kicked out.

Beat 4 — All-or-nothing eviction.
Here's a detail that only makes sense once you remember how attention works:
every generated token reads every block of a sequence, every single step. So
evicting half a sequence's blocks buys you nothing — the sequence still can't
run without the other half. vLLM evicts a sequence's blocks all at once, and
if several sequences share blocks — say, all the beams of one request — the
whole gang goes together. An OS, by contrast, happily evicts individual
pages one at a time, because a process only touches a handful of them per
instruction.

Beat 5 — Recovery option 1: swapping.
Once a sequence is preempted, vLLM needs to get its blocks out of the way
without losing them, and there are two ways to do that. The first is
swapping: the CPU block allocator copies the evicted blocks into ordinary CPU
RAM, over PCIe, and swaps them back in later when the request is
rescheduled. The cost is PCIe bandwidth. And notice the swap space can never
grow unbounded — it's capped by exactly the GPU's total KV block capacity,
since that's the most that could ever be evicted at once.

Beat 6 — Recovery option 2: recomputation.
The second option is more radical: just drop the evicted blocks entirely.
When the request is rescheduled, concatenate its original prompt with all
the tokens it had already generated, and treat that whole thing as one new
prompt — run a single prefill pass over it. That rebuilds the entire KV
cache in one parallel pass instead of one slow token at a time, so it's
often cheaper than it sounds.

Beat 7 — So which one wins? [PAUSE]
It depends on block size. In the paper's microbenchmark, recomputation's
overhead is essentially flat no matter the block size — it never touches a
KV block. Swapping is expensive at small block sizes, because it means many
tiny PCIe transfers, but gets cheaper as blocks get larger and transfers get
bigger. They cross over somewhere in the medium range, roughly block size 16
to 64, where end-to-end performance is comparable either way. These numbers
are approximate, read off the paper's Figure 19.

Beat 8 — Landing.
Put together, this is the payoff of paging: preemption, all-or-nothing
eviction, swap, recompute — none of that exists in a system where memory is
one fixed contiguous slab per request. Paging gives you a knob the
contiguous systems never had: you can take memory back, and give it back
later.

## S9 S9Results (~4-5 min)

---------
Beat 1 — Setup.
We evaluated vLLM on OPT-13B, 66B and 175B, and on LLaMA-13B, on real
Google Cloud A100 servers — one A100 for 13B, four for 66B, eight 80GB
A100s for 175B. The baselines are FasterTransformer, NVIDIA's
latency-optimized serving engine, and three re-implemented variants of
Orca: Oracle, which cheats by knowing each request's true output length in
advance; Pow2, which rounds its reservation up to the next power of two;
and Max, which always reserves the model's full 2048-token maximum.
[PAUSE] The metric is normalized latency — latency per output token —
plotted against request rate. A system is "better" if it keeps latency flat
out to a higher request rate before it falls over.

Beat 2 — Two workloads.
We test on two datasets with very different shapes. ShareGPT — real
multi-turn chat logs — averages 161 tokens of input and 338 tokens of
output. Alpaca — short instruction-following prompts — averages only 19
tokens in and 58 out. That makes ShareGPT's prompts 8.4 times longer and
its outputs 5.8 times longer than Alpaca's, which means ShareGPT puts far
more pressure on the KV cache.

Beat 3 — The headline result.
Here's OPT-13B on both datasets: normalized latency versus request rate.
Every system is flat at low request rate, then hits a knee and blows up as
queueing delay takes over. vLLM's knee sits far to the right of everyone
else's. On ShareGPT, vLLM sustains 1.7 to 2.7 times the request rate of
Orca (Oracle) and 2.7 to 8 times that of Orca (Max), at the same latency —
and up to 22 times FasterTransformer. [PAUSE] Same model, same GPU, just
better memory management.

Beat 4 — Why: more requests fit in the batch.
Here's the mechanism underneath that curve. At a fixed request rate, we
counted how many requests are actually batched together at once. On
ShareGPT, vLLM batches 30 requests on average versus Orca (Oracle)'s 13.6
— 2.2 times more. On Alpaca, with its shorter sequences, vLLM batches 132
versus Orca (Max)'s 7. More requests batched per GPU pass is exactly
the memory savings from Act II showing up as throughput.

Beat 5 — Callback: the waste bars, closing the loop.
Remember this chart from Act I? Orca (Max) actually stores tokens in only
20.4% of its reserved KV memory; Orca (Pow2), 26.8%; Orca (Oracle), even
knowing the future, only 38.2%. vLLM: 96.3%. [PAUSE] That's the whole
story of this talk in one bar chart — turning wasted reservation into
actual throughput.

Beat 6 — Harder decoding: parallel sampling and beam search.
Complex decoding makes memory sharing even more valuable, and vLLM's
advantage grows with it. On Alpaca with OPT-13B, going from plain sampling
to beam search of width 6, vLLM's edge over Orca (Oracle) widens from 1.3
times to 2.3 times — because more candidates sharing memory means more for
vLLM's block-level sharing to exploit.

Beat 7 — Why: memory actually saved by sharing.
Directly measuring the KV blocks vLLM shares instead of duplicating: on
Alpaca, parallel sampling saves 6.1 to 9.8% of memory, and beam search
saves 37.6 to 55.2% — beam search shares far more because candidates share
almost their whole prefix. On ShareGPT, with its longer sequences, sharing
is even bigger: 16.2 to 30.5% for parallel sampling, 44.3 to 66.3% for beam
search.

Beat 8 — Shared prefixes: translation.
One more sharing case: a common system prompt shared across every request.
For LLaMA-13B doing English-to-German translation with a few-shot prefix,
vLLM gets 1.67 times the throughput of Orca (Oracle) with a short, one-shot
prefix, and 3.58 times with a longer, five-shot prefix — the more prefix
there is to share, the bigger vLLM's advantage.

Beat 9 — A harder case: chatbot.
Now the least favorable setting: a chatbot workload with long, truncated
1024-token prompts on ShareGPT-style multi-turn conversations. vLLM still
sustains about 2 times the request rate of the Orca baselines — but notice
the three Orca variants now cluster together. With prompts this long, there
just isn't much slack left for any reservation strategy to get right, so
Oracle, Pow2, and Max converge.

Beat 10 — Summary.
Across every workload we tried, vLLM delivers 2 to 4 times the throughput
of Orca at the same latency, and up to 22 times FasterTransformer's — with
zero changes to the model itself. [PAUSE] All of that came from managing
memory better.

## S10 S10Ablations (~2 min)

---------
Beat 1 — The kernel itself is slower.
Before we celebrate, let's be honest about the cost. PagedAttention's own
attention kernel has to look up a block table and read key/value data from
scattered, non-contiguous memory locations, instead of one clean contiguous
strip. The paper measures this in isolation: across batch sizes and context
lengths, vLLM's attention kernel runs about 20 to 26% slower than
FasterTransformer's tightly hand-optimized kernel. [PAUSE] That's a real,
measurable cost — but attention is only one operator in a whole forward
pass, so this slowdown barely dents end-to-end latency.

Beat 2 — Picking the block size.
The other knob is block size: how many tokens live in one page. Make blocks
too small — say, 1 or 2 tokens — and the kernel can't batch its memory
reads efficiently; it loses the GPU parallelism it needs. Make blocks too
large and you're back to the old problem: internal fragmentation grows, and
fewer requests get to share a block. On ShareGPT's long, varied prompts,
anything from 16 to 128 tokens per block works well. Alpaca's prompts are
much shorter, so it degrades past 32. [PAUSE] That's why vLLM ships block
size 16 as its default — it's the sweet spot for both.

Beat 3 — Landing.
So the trade is explicit: a slightly slower attention kernel, in exchange
for a dramatically better memory story. Net result, end to end: still that
2 to 4 times throughput win we saw in the results.

## S11 S11Takeaways (~1-2 min)

---------
Beat 1 — Three takeaways.
Three things worth carrying out of this talk. One: the bottleneck in LLM
serving was never the model's math — it was memory, specifically the KV
cache. Two: vLLM's fix is almost embarrassingly simple in hindsight — fixed
size blocks, a block table indirection, and on-demand allocation get you to
about 96% memory utilization, and sharing between requests basically falls
out for free. Three: this is an OS idea, but re-engineered for GPUs — a
software address-translation layer, a new fused attention kernel to pay for
that indirection, and new eviction and recovery policies suited to how LLMs
actually behave.

Beat 2 — Impact.
The practical result: vLLM became the de-facto open-source engine for
serving large language models, and PagedAttention's block-based KV cache
design has been adopted well beyond the original project, across the
industry.

Beat 3 — Discussion prompt.
So here's the question I want to leave you with: paging worked for the KV
cache. What else in an LLM serving system could you page? [PAUSE] A few
things people have actually tried since this paper: caching and reusing
shared prompt prefixes across different requests automatically; paging
LoRA adapters themselves, so many fine-tuned variants can be swapped in and
out like blocks; splitting prefill and decode onto different machines;
and offloading colder KV blocks from GPU memory to CPU RAM or disk.

Beat 4 — Thanks and questions.
Thanks for listening. Happy to take questions.
