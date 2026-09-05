# PagedAttention Talk — Speaker Narration

Auto-generated from the `NARRATION` section of each scene's module docstring in `talk/sN_*.py`. Regenerate with `make narration` (or `python tools/build_narration.py`) after editing any scene. Do not hand-edit this file.

## Contents

### Act I: Why memory is the bottleneck

- [S0 S0Title (~1-2 min)](#s0-s0title)
- [S1 S1Transformers (~6 min)](#s1-s1transformers)
- [S2 S2GPU (~6 min)](#s2-s2gpu)
- [S3 S3KVCache (~9-11 min)](#s3-s3kvcache)
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
Look at the two resources. The GPU still has compute to spare — cores sitting idle.
Memory is packed: there is no room to batch more requests. That 10x cost is memory,
not math. [PAUSE] And the kicker: over 30% of GPU memory is the KV cache, which we'll
build up shortly. Existing systems only put 20 to 40% of it to use — that waste is
this paper's target.

Beat 4 — Roadmap
Here's the shape of the talk, in three acts. Act I: why memory — not compute — is the
real bottleneck. Act II: the paper's core idea — chop the KV cache into small fixed-size
blocks and manage them on demand. Act III: the payoffs — the actual speedups and sharing
tricks. Let's start with the bottleneck.

## S1 S1Transformers (~6 min)

Beat 1 — One word at a time
You do not need the word "transformer" yet. A language model does not write a
sentence in one go. It takes the words so far, runs them through a box, and the
box emits one next word. That word is appended, the sequence recenters, and the
loop runs again. Watch "Four score and seven" produce "years". [PAUSE] In the
paper that loop is Equation 1: the joint probability of a sentence is just the
product of these next-word guesses.

Beat 2 — Key: a label for each token
Open the box. Each token is first a vector x_i — a list of numbers that stands
for that word in this layer. A learned matrix W_K multiplies it: k_i equals
W_K x_i. That is the Key. Think of it as a label on the token: what this word
contains, and how later steps will find it. Every token gets one. [PAUSE]

Beat 3 — Value: what the token will add
A second matrix produces the Value: v_i equals W_V x_i. If a later step decides
this token matters, it is the Value that actually gets mixed into the output —
the payload, not the label. Key is how you find it; Value is what you take.
Every token now carries both. [PAUSE]

Beat 4 — Query: the question this step asks
A third vector, only for the token we are writing from. q_i equals W_Q x_i —
the Query. It is the question this step is asking: what should I look up to
choose the next word? Only the newest token needs a fresh Query. Every earlier
token just sits there with its Key and Value. [PAUSE]

Beat 5 — Query against every Key
Equation 3 starts with a score. The Query is compared to every Key from position
1 through i — including its own, never a future token. s_j equals q_i transpose
k_j over square root of d. The numbers here are illustrative. [PAUSE] Which
token do you think scores highest?

Beat 6 — Softmax: scores become weights
Those scores become attention weights by softmax: a_ij equals exp of the score,
divided by the sum of those exps from t equals 1 to i. The weights are positive
and they sum to one. Bigger score, bigger slice of attention — a budget to spend
across the tokens so far.

Beat 7 — Mix the Values, write the next word
Spend that budget on Values: o_i equals the sum from j equals 1 to i of a_ij
v_j. That mixture is what the box turns into the next word. Here that's "ago",
which exits, joins the sequence, and recenters. Close the box — same machine as
beat 1, we just know what's inside.

Beat 8 — Nothing is thrown away
Run the loop once more, box closed. Every old Key and Value stays on the input;
the new token arrives with one new pair; nothing is thrown away. Next iteration
the box will consume the full sequence and every previous K and V again.

Beat 9 — Prefill vs decode
Serving splits this into two phases. Prefill: the whole prompt enters at once,
in parallel, and the box writes K and V for every prompt token plus the first
output token — compute-bound. Decode: the loop we have been watching — one new
token per pass, re-reading the growing K/V bundle every time — memory-bound, and
that is the phase that dominates latency.

Beat 10 — The landing point
So the load-bearing fact: to emit the next token, the box must be handed the Key
and the Value of every previous token, not just the most recent one. That growing
bundle sits in memory between iterations — at every layer, every decode step, for
the entire request. That is the state the rest of this talk is about.

## S2 S2GPU (~6 min)

---------
Beat 1 — Same math, over and over.
Neural-net serving is not fancy one-off logic. Under the hood it is the same
matrix multiply, again and again, against a huge shared weight matrix W.
One small input, one giant W, one small output — and that pattern repeats
across the whole model. [PAUSE] That "same operation, many times" shape is
exactly what a GPU is built for.

Beat 2 — CPU vs GPU.
A CPU has a handful of big, fast cores. Great at one complicated thing at a
time. Watch them light up in sequence. A GPU flips the trade: thousands of
small, simple cores. Give it a pile of identical multiplies and it runs them
all at once, one piece per core. [PAUSE] Our workload — the same multiply,
many times — maps almost perfectly onto that grid.

Beat 3 — Two memory pools.
Here is the catch that matters for serving. The GPU does not borrow the
computer's regular RAM. It has its own memory, called VRAM, sitting next to
the cores. CPU DRAM and GPU VRAM are two separate pools, joined by a PCIe
link that is slow compared to on-device memory. Cores can only multiply data
that is already in VRAM. If it is still on the CPU side, the GPU is waiting.

Beat 4 — Weights persist in VRAM.
So the model's weights have to live in VRAM for the whole time we are
serving. For OPT-13B on an A100, that is about 26 gigabytes of a 40-gigabyte
card. Sixty-five percent of the GPU's memory is gone the moment the model
loads — before we have served a single request. [PAUSE] Those weights stay
there. They do not come and go per request.

Beat 5 — One request vs 26 GB.
Now send in one request. The cores have to stream that whole 26 GB of
weights to produce one small result. Most of the time they are waiting on
memory, not multiplying. A single request is a tiny amount of math against a
huge W, so the machine looks idle even though VRAM is already packed.
Serving one-at-a-time wastes the GPU.

Beat 6 — Batching: one hub, many requests.
The fix is batching. The weights are identical for every request, so we load
W once and send many requests through the same multiply. Watch: every
request arrow ends at one point on W, and every result arrow starts from
that same point. One pass over the weights, N results. [PAUSE] Throughput
becomes a question of how many requests we can pack into that one pass.

Beat 7 — Leftover VRAM is the budget.
Almost for free — except leftover VRAM is finite. Weights already took
26 GB. The empty slice at the top is all we have for live request state.
Some requests fit; the rest bounce off. We cannot batch more than that
leftover space can hold. [PAUSE]

Beat 8 — Landing.
So: leftover VRAM decides the maximum batch, and the maximum batch decides
throughput. That is the resource this talk is about. Everything that follows
is about how we spend that leftover slice.

## S3 S3KVCache (~9-11 min)

---------
Beat 1 — Pickup.
Last scene, leftover VRAM was the serving budget. The scene before that, the
box needed the Key and Value of every previous token to write the next word.
That bundle is still sitting here — watch the Query on "years" look across
every Key. We are going to name this bundle, size it, and put it in that
leftover slice. [PAUSE]

Beat 2 — Recompute is a triangle of real work.
Suppose we throw the Keys and Values away after each step. To write the next
word, we would rebuild them: k equals W_K x, v equals W_V x, for every
previous token, every time. Step 2 rebuilds token 1. Step 3 rebuilds 1 and 2.
Step 5 rebuilds four tokens just to add one new pair. [PAUSE] That triangle
only grows. The sentence gets longer; the wasted multiply gets worse.

Beat 3 — Keep them: the KV cache.
So don't throw them away. The first time we compute a token's Key and Value,
we write them down and keep them. The next step reads that store and computes
only the new pair. Query is different: it is made fresh for the newest token
and discarded — only K and V persist. This store is the KV cache.

Beat 4 — Prefill writes; decode appends.
Serving fills that cache in two phases. Prefill: the whole prompt enters at
once, in parallel, and the box writes a K,V pair for every prompt token in
one pass. Decode: the loop we have been watching — one new pair appended per
step. The cache is the state that survives between those decode steps.

Beat 5 — Read all, write one.
Look at one decode step closely. To write the next word, the cores must read
every cached pair — the whole row — and then write exactly one new pair on
the end. Read all, write one. That is why decode is memory-bound, and why
this object will eat leftover VRAM as the sentence grows.

Beat 6 — Every layer has its own copy.
And it is not one row. The model has many layers — forty, for OPT-13B — and
each layer keeps its own Keys and Values for every token. What looks like a
single strip is forty copies stacked. That is the first reason one token is
expensive.

Beat 7 — 800 KB, built in public.
Here is the arithmetic, one factor at a time. One vector is 5120 numbers in
FP16 — two bytes each — about 10 kilobytes. Times two, because each token
stores a Key and a Value: about 20 kilobytes. Times forty layers: 800
kilobytes per token. That is the cost of one word. [PAUSE] How many words
does this request need?

Beat 8 — Unknown length, so reserve the max.
We do not know. The cache grows one token at a time until the model emits
stop — there is no content-length. The only number we can bank on is the
model's maximum, 2048. So the whole strip gets reserved up front. The tokens
we have actually written sit on the left. Everything past them is reserved
and cut off from every other request, whether we ever fill it or not. [PAUSE]

Beat 9 — That reserved strip is 1.6 GB.
Now the last multiply means something. 800 kilobytes times 2048 reserved
slots is about 1.6 gigabytes — not "how big this request is," but how much
VRAM one request has spoken for. Used or not, that whole strip is gone from
the leftover budget. [PAUSE]

Beat 10 — Per request, in leftover VRAM.
Every request owns its own reserved strip. Request A is "Four score…";
request B is "it was the best…" — they grow independently, different lengths,
not shared. Both of them have to live in leftover VRAM, beside the 26
gigabytes of weights that never move.

Beat 11 — Fig 1 left, and how many fit.
That leftover slice is the paper's Figure 1: on a 13B model and an A100
40-gigabyte card, about 65 percent is weights, more than 30 percent is KV
cache — 12 gigabytes — and a sliver is other, short-lived activations.
Weights are fixed. KV is the only region that grows and shrinks. 12 gigabytes
divided by 1.6 gigabytes reserved is about seven requests at maximum length.
An eighth does not fit.

Beat 12 — Two hard properties.
Two facts make this object awkward to place. First: we reserved the max
because the length was unknown — most of that strip may never fill. Second:
the same word at a different position has a different Key and Value — this
is a timeline, not a dictionary of words. Leftover VRAM is spent on these
growing rows; how we lay them out is the batch size. [PAUSE] So: how do you
allocate memory for something whose final size is unknown?

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
