# PagedAttention Talk — Speaker Narration

Auto-generated from the `NARRATION` section of each scene's module docstring in `talk/sN_*.py`. Regenerate with `make narration` (or `python tools/build_narration.py`) after editing any scene. Do not hand-edit this file.

## Contents

### Act I: Why memory is the bottleneck

- [S0 S0Title](#s0-s0title)
- [S1 S1GPU](#s1-s1gpu)
- [S2 S2Transformers](#s2-s2transformers)
- [S3 S3KVCache](#s3-s3kvcache)
- [S4 S4Problem](#s4-s4problem)

### Act II: PagedAttention

- [S5 S5PagedAttention](#s5-s5pagedattention)
- [S6 S6OSAndWhyHard](#s6-s6osandwhyhard)

### Act III: What it buys you

- [S7 S7Sharing](#s7-s7sharing)
- [S8 S8Scheduling](#s8-s8scheduling)
- [S9 S9Results](#s9-s9results)
- [S10 S10Ablations](#s10-s10ablations)
- [S11 S11Takeaways](#s11-s11takeaways)

## S0 S0Title

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
tricks. Let's start with the GPU.

## S1 S1GPU

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

## S2 S2Transformers

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

## S3 S3KVCache

---------
Beat 1 — Pickup.
Last scene, the box needed the Key and Value of every previous token to write
the next word. The scene before that, leftover VRAM was the serving budget.
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

## S4 S4Problem

---------
Beat 1 — The question.
Last scene left us with a question: how do you allocate memory for a KV cache
whose final size is unknown? A request has no content-length. It grows until
the model emits end-of-sequence. Sit with that. How would you lay this object
down in leftover VRAM? [PAUSE]

Beat 2 — One contiguous tensor.
Existing systems all give the same answer. They store each request's KV cache
as one contiguous tensor. Not because that is a good fit for a growing cache —
because that is what deep learning frameworks require. An operator wants a
contiguous chunk. So the serving system hands it one. [PAUSE]

Beat 3 — Unlike a traditional tensor.
That choice was fine for the tensors in traditional deep learning workloads.
Those have a fixed, known shape before you ever run the model: allocate once,
contiguous is perfect, there is nothing to fragment. The KV cache is different
in kind. It dynamically grows and shrinks as the model generates tokens, and
its lifetime and length are not known a priori. Nobody — not the system, not
the model — knows where it stops until it stops. [PAUSE]

Beat 4 — Pre-allocate the maximum.
So FasterTransformer and Orca do the conservative thing. They statically
allocate a contiguous chunk to the request's maximum possible sequence length,
irrespective of the actual input or the eventual output. Request A is given
2048 slots — OPT's max — the moment it arrives. Used or not, that whole strip
is spoken for. [PAUSE]

Beat 5 — Request B is smaller, still a slab.
The paper's Figure 3 also has a second request. Request B is allowed a maximum
of 512, not 2048. Same rule, different size: one contiguous chunk, reserved up
front. Two slabs of different lengths, sitting in the same leftover VRAM.
Remember 512 — it is how 507 unused slots will show up in a moment. [PAUSE]

Beat 6 — Fig. 3, the prompt.
Here is Figure 3, on our running example. Seven KV cache states for request A's
prompt, already computed: "Four score and seven years ago our." Each box is
that token's Key and Value — not the word itself. [PAUSE]

Beat 7 — Current iteration.
"brought" is the current iteration: the token being generated right now. Its
slot is live. It is not waste. It is the work this step is doing. [PAUSE]

Beat 8 — Reserved.
Two slots past it are reserved for tokens this request will actually generate
— "forth", and end-of-sequence. The paper marks "1 slot for generated token"
and "2 slots future used." Reserved memory is eventually used. But it occupies
space for the entire request's duration, space that could otherwise have gone
to other requests. [PAUSE]

Beat 9 — Internal fragmentation.
The rest of the 2048 is still sitting there — empty slots, never written.
That is internal fragmentation. Watch them fill the remainder of A's slab.
Two thousand and thirty-eight slots never used. Pure waste. We only realize
it after sampling finishes and we know the request stopped early. [PAUSE]

Beat 10 — Request B.
Request B, same treatment, max 512. Prompt "You only live," current "once,"
one reserved slot, and then the same empty-slot cascade: 507 never used.
Same internal-fragmentation story, smaller slab. [PAUSE]

Beat 11 — External fragmentation.
The hole between the two chunks is free memory with the wrong shape. Watch
another request try to land there. It does not fit. That is external
fragmentation: known before we even serve, and it will never hold generated
tokens. The three wastes together keep other requests out of the GPU. [PAUSE]

Beat 12 — Even if you knew the length.
Even if the actual length were known a priori, that unused pink still belongs
to A for the whole lifetime. A shorter request cannot borrow it. Knowing the
future does not break the slab. [PAUSE]

Beat 13 — Fig. 2, Orca (Max).
Figure 2 measures this on a real serving run. Orca that always reserves the
max — the policy we just watched — puts actual token state in only 20.4
percent of its KV memory. 57.3 percent is internal fragmentation. 13.3 percent
is reservation. 8.9 percent is external fragmentation and other. [PAUSE]

Beat 14 — Orca (Pow2).
Round the reservation up to a power of two instead. Internal fragmentation
drops. External fragmentation blows up to 41.6 percent. You moved the waste;
you did not remove it. Token state is still only 26.8 percent. [PAUSE]

Beat 15 — Orca (Oracle).
Give the system an oracle: every output length in advance. Internal
fragmentation goes to zero. You still only reach 38.2 percent token state.
Unknown length was never the whole problem. Contiguity is. [PAUSE]

Beat 16 — What does this paper reach?
Across existing systems, only 20.4 to 38.2 percent of the KV memory you paid
for stores actual token states. The paper's system is the last bar. We will
come back to that number. [PAUSE]

Beat 17 — Second failure: two copies.
Contiguous chunks have a second cost. Existing systems cannot share, because
each sequence's KV cache is a separate contiguous space. Watch the prompt
copy: identical prefix, two full copies. [PAUSE]

Beat 18 — Twelve percent, and more in beam search.
In the paper's experiment the prompt was 12 percent of total KV — paid once
per sample, not once per prompt. In beam search, sharing could save up to
55 percent. A contiguous system cannot point two sequences at the same
physical memory. [PAUSE]

Beat 19 — One cause.
One design. Each request's KV cache is a contiguous chunk, pre-allocated to a
maximum length. Three wastes, and sharing is impossible. Memory, not compute,
caps how many requests the GPU can serve. [PAUSE]

Beat 20 — Seam to Act II.
[act_checkpoint — presenter narrates while it plays]

## S5 S5PagedAttention

Beat 1 — Who demanded contiguity?
Here is where Act I left us: request A owns one contiguous chunk, ten slots
in use, two thousand and thirty-eight reserved and never used. Before we fix
it, ask why the chunk had to be contiguous in the first place. It is not a
property of memory. It is a property of the code that reads the memory. The
attention operator, like most operators in PyTorch or TensorFlow, takes K as
one matrix and V as one matrix, and it wants each of them laid out back to
back. The reader dictates the layout. [PAUSE] So the paper's first move is
not a new allocator. It is a new reader: an attention kernel that does not
need the whole row side by side. Change the reader, and the layout is free.

Beat 2 — Partition into fixed-size blocks
Step one of the new reader: partition the KV cache of each sequence into
fixed-size blocks. The paper's default is sixteen tokens per block; I will
draw four so it fits on a slide. Our ten tokens become three blocks: four,
four, and a last block that is half full. Notice what is missing: the two
thousand reserved slots. We have not allocated them. Nothing about the block
size depends on how long this request will turn out to be.

Beat 3 — A block is K and V for B tokens
Zoom in on one block. Each slot is not a word — it is that token's Key
vector and Value vector, the pair we sized at 800 kilobytes per token in the
KV-cache scene. One block holds B of those pairs, packed left to right. The
paper writes K sub j for the keys of block j and V sub j for its values. The
last block of a sequence may have empty slots; those are reserved for the
tokens this request has not generated yet, and that is the only reservation
we will allow: less than one block.

Beat 4 — Attention as we left it (Eq. 3)
Now the computation. Recall Equation 3 from the transformer scene. The
query for the newest token — here, "forth" — is dotted with every key to
get a score. Softmax turns the scores into weights: each exponentiated
score divided by the sum of all of them. Then the output is the weighted sum
of every value. Look at the two sums. Both run over every previous token,
one through i. That single long sum is the reason existing kernels wanted
K and V as one contiguous matrix each: one pass, one pointer, one stride.

Beat 5 — The same sums, grouped by block (Eq. 4)
Here is the whole trick, and it is arithmetic you learned in primary school:
addition does not care how you group the terms. Split the sum over tokens
into a sum over blocks of a sum within each block. Per block j, the query
times that block's keys gives a small vector of scores, A sub i j. The
softmax denominator is the sum of the exponentiated scores across all the
blocks. And the output is the sum over blocks of V sub j times A sub i j.
That is Equation 4. Nothing was approximated. It is the same attention,
computed one block at a time and accumulated.

Beat 6 — Fig. 5: fetch a block, score it, accumulate
Figure 5, run live. The query is "forth". Its keys and values live in three
blocks that are not next to each other in memory — Block 1, Block 2, Block 0,
in that physical order. The kernel keeps two running totals: the softmax
denominator, and the value-weighted numerator. It fetches Block 0 — "Four
score and seven" — multiplies the query against those four keys, exponentiates,
adds the four numbers into the denominator, and adds the four weighted values
into the numerator. [PAUSE] Illustrative numbers, of course. But watch the
pattern: one block in, two totals updated, nothing else touched.

Beat 7 — Blocks 1 and 2, then divide
Block 1: "years ago our fathers". Same operation; the totals grow. Block 2
has only two tokens, "brought" and "forth" — the kernel handles the partial
block by reading its fill count. Totals grow again. Now divide numerator by
denominator, and that is o, the attention output for this step. Exactly what
Equation 3 would have produced. The kernel fetched three blocks from three
unrelated addresses and never needed them to be adjacent. That is
PagedAttention. It costs a lookup per block — we will quantify that cost
later — and it buys the freedom to put blocks anywhere in GPU memory.

Beat 8 — So who decides where blocks live? The KV cache manager
If the kernel can read a block from anywhere, someone has to decide where
each block goes. That is the KV cache manager, and it has three parts.
On the left, the request's own view: a series of logical blocks, filled from
left to right as tokens arrive, always contiguous from the request's point
of view. On the right, GPU memory: one large allocation carved into
fixed-size physical blocks, a pool that every request draws from on demand.
In the middle, the piece that connects them: a block table, one per request,
with one row per logical block, recording which physical block it lives in
and how many slots are filled. This layout stays on screen while we run a
request through it.

Beat 9 — Fig. 6, step ①: prefill
The paper reuses our sentence but starts the prompt at "our", so we can
watch two words get generated. Seven tokens: "Four score and seven years ago
our". vLLM does not reserve two thousand and forty-eight slots. It reserves
exactly the blocks the prompt needs: two. Logical block 0 gets the first
four tokens and is mapped to physical block 7. Logical block 1 gets the
remaining three and is mapped to physical block 1. Three of four filled; the
last slot is reserved for generation. Prefill itself runs ordinary
self-attention — every prompt token is known, so there is nothing to page —
and the resulting K and V are written into blocks 7 and 1.

Beat 10 — Step ②: first decode, no new block
First decode step. The query is the newest token. PagedAttention runs over
physical blocks 7 and 1 — exactly the block-by-block loop from Figure 5 —
and the model emits "fathers". Where does its K and V go? The last logical
block still has a free slot, so it goes there. The block table's fill count
ticks from three to four. No new physical block. Nothing else moved.

Beat 11 — Step ③: the last block is full — allocate
Second decode step. Logical block 1 is full, so vLLM opens logical block 2,
asks the pool for any free physical block — it gets physical 3 — and stores
"brought" there. The block table gains a row: logical 2 maps to physical 3,
one filled. This is the only moment memory is allocated: when every
previous block is completely full, and then exactly one block. Compare that
to S4, where the whole two thousand and forty-eight slots were claimed
before the first token.

Beat 12 — One more step, and you have seen this picture
One more decode: "forth" lands in the second slot of physical block 3, and
the fill count goes to two. Now look at the three physical blocks this
request is using, top to bottom: block 1 holds "years ago our fathers",
block 3 holds "brought forth", block 7 holds "Four score and seven". That is
Figure 5. The scattered blocks the kernel was reading in Beat 6 are simply
the state the manager arrives at by allocating on demand. The algorithm and
the manager are two halves of one design.

Beat 13 — All waste, less than one block
Put S4's picture and this one side by side. Same ten tokens. S4: ten slots
used, two thousand and thirty-eight reserved and never used — internal
fragmentation, plus a reserved run, plus external holes between chunks.
vLLM: ten slots used, two slots empty in the last block. Because blocks are
filled left to right and a new one is allocated only when all previous
blocks are full, all memory waste for a request is confined to less than
one block. [PAUSE] Remember the question mark on the Figure 2 chart? Here is
the answer. Existing systems: 20.4 to 38.2 percent of KV memory holding real
token state. vLLM: 96.3 percent. Same model, same GPU.

Beat 14 — Fig. 7: a second request shares the pool
None of this is special to one request. Request B arrives: "it was the best
of times". Its logical block 0 is mapped to physical 5; its partial logical
block 1 to physical 2. Look at the pool now: A's blocks at 7, 1, 3 and B's
at 5, 2, interleaved. Neighboring logical blocks of either request are not
adjacent in GPU memory, and it does not matter. Because every physical block
is the same size, any free block fits any request. There is no such thing as
a hole that is too small.

Beat 15 — A finishes: blocks return to the pool
Request A finishes. Its three physical blocks — 7, 1, and 3 — go straight
back to the free pool, wherever they sat. Request B is untouched. Six blocks
are free, and every one of them is usable by whoever comes next, without
finding a contiguous hole and without compaction. External fragmentation has
nothing to fragment.

Beat 16 — One iteration of the engine
Zooming out, here is what vLLM does on every single decode iteration.
First, pick which sequences to run this step — that is the scheduler, and
we will spend a scene on it. Second, allocate physical blocks for any
logical blocks that became necessary. Third, concatenate the current tokens
of all those requests into one flat sequence: the whole prompt for a request
in prefill, one token for each request in decode. Fourth, run the model once
over that sequence; the attention layers use PagedAttention to read each
request's blocks through its block table, and the new keys and values are
written into their assigned physical blocks. One forward pass serves every
request in the batch — the batching payoff from Act I, now with a batch that
fits.

Beat 17 — Landing
So, two halves. A kernel that computes attention one block at a time, so
blocks need not be contiguous. A manager that therefore places blocks
anywhere, on demand, through a per-request block table. Together they take
KV-cache utilization from roughly twenty to forty percent up to 96.3
percent. [PAUSE] Next scene: look at this picture once more — logical
blocks, a table, physical blocks. It should look very familiar.

## S6 S6OSAndWhyHard

---------
Beat 1 — Pickup.
Look at what we just built. Same Lincoln sentence, same mapping as the last
scene. Logical 0 lives in physical 7. Logical 1 in physical 1. Logical 2 in
physical 3. A request, a block table, a pool of GPU blocks. Sit with this
picture — we are going to look at it again. [PAUSE]

Beat 2 — The reveal.
Take a step back and squint. Logical blocks mapped, on demand, onto
scattered physical blocks through a table... this is exactly what an
operating system does when it manages a process's memory. We just
reinvented virtual memory paging — for the KV cache. [PAUSE]

Beat 3 — OS vocabulary.
Same picture, new labels. A block is a page. A token is a byte. A request
is a process. And the block table is a page table: virtual pages onto
physical frames. Same idea, one layer lower in the stack. [PAUSE]

Beat 4 — Demand paging, shown.
The process believes it has one contiguous chunk — pages 0, 1, 2 in order,
on the left. Underneath, those pages sit in whatever frames were free: 7,
1, and 3, not next to each other. Frames are handed out when the process
actually touches them, not reserved up front. No giant slab. No external
fragmentation. [PAUSE]

Beat 5 — The question.
So, fair question. Did we just copy fifty-year-old operating systems ideas
onto a GPU and call it a paper? Sit with that. [PAUSE]

Beat 6 — The answer.
The idea transfers. The engineering does not. Making paging work on a GPU,
for attention, required real systems work that a textbook page table never
has to do. Two reasons. We will take them one at a time.

Beat 7 — No GPU MMU.
First: your CPU has dedicated hardware for this — a memory management unit
and a TLB, in silicon, off the critical path. The GPU has none of that for
our purposes. Every logical-to-physical translation happens in software,
inside the kernel, on every access. This table is walked by the kernel
itself. [PAUSE]

Beat 8 — The kernel is the pager.
Second: an OS page-fault handler never touches your program's computation.
It finds the page and hands control back. Here the computation is the
memory access. The attention kernel had to be rewritten to gather those
scattered KV blocks and attend in one fused pass. An OS pager never
rewrites your program. [PAUSE]

Beat 9 — The kernel is slower.
And that rewrite is not free. A page fault is rare. Ours happens on every
token, every step: attention reads every block, every time. That's why the
PagedAttention kernel itself runs about 20 to 26 percent slower than
FasterTransformer's kernel on contiguous memory. [PAUSE]

Beat 10 — And yet.
And yet the end-to-end system is 2 to 4 times faster. Leftover VRAM now
holds a bigger batch. The memory win swamps the per-kernel cost. We will
measure that kernel overhead after the results. [PAUSE]

Beat 11 — Landing.
So: an OS idea, re-engineered for a workload the OS was never designed for.
A new kernel. New policies still to come — because every block of a
sequence is always touched together, and OS-style per-page eviction does
not apply.

## S7 S7Sharing

---------
Beat 1 — Pickup.
Look at the mapping we just built. Same Lincoln sentence: logical 0 lives in
physical 7, logical 1 in physical 1, logical 2 in physical 3. Figure 7 put a
second request into that same pool — interleaved, not shared. Two sequences
still owned two copies of everything. [PAUSE] Act I already named the bill:
a contiguous KV cache cannot point two sequences at the same physical
memory. Prompt copies were 12 percent of the cache. Beam search could save
up to 55 percent. We now have a block table. We can pay that bill.

Beat 2 — Parallel sampling, the product.
A common trick for better outputs: one prompt, several tries. "Give me two
samples." Call them A1 and A2. Each is its own sequence with its own KV
cache, but they start from the exact same prompt. [PAUSE] In a contiguous
system that means copying the prompt's entire KV cache, once per sample.
With blocks, we rewind to just after prefill — and we do not copy.

Beat 3 — Same physical blocks, two logical views.
Figure 8. A1's block table and A2's block table both map logical 0 to
physical 7 and logical 1 to physical 1. No copy has happened — both samples
are pointers into one shared region. Each shared physical block carries a
reference count: here, 2, because two logical blocks point at it.

Beat 4 — Only the last block can move.
Physical 7 is packed. Four out of four prompt tokens. Nobody ever writes
there again, so it can stay shared forever. Physical 1 is different: one
open slot, reserved for the next token. That is the only place a write can
happen. [PAUSE]

Beat 5 — Copy-on-write.
A1 samples a different continuation and tries to write "mothers" into that
open slot. vLLM checks the reference count, sees 2, and refuses to write in
place. Allocate a fresh physical block — physical 3 — copy the three shared
tokens into it, write "mothers" into the copy, retarget A1's table, and
decrement the original block's count to 1. [PAUSE] A2 still points at the
original, untouched. The cost: only that last, not-yet-full block is copied.
Every packed block just stays shared.

Beat 6 — Write in place.
Now A2 writes "fathers" into the original block. The reference count is
already 1, so the write happens in place. No copy. [PAUSE] This is the same
trick an operating system uses when you fork a process: share the pages,
copy only the one that someone actually writes. Two samples, one prompt's
worth of KV, plus a single last-block copy.

Beat 7 — Beam search shares a tree.
Beam search uses the same mechanism more aggressively. A beam of width 4
keeps four candidate sequences at every step. They do not just share the
prompt — they share prefixes of each other's generated tokens, because
every candidate is an extension of a shared history. Picture a tree: one
shared trunk, a private branch for the candidate that diverged early, and
four live heads. Each head is only the new block that candidate needed.

Beat 8 — One prune step.
Watch the next iteration. Candidates 0 and 3 fall out of the top 4. Every
physical block that only they were using has its reference count drop to
zero — and those blocks go straight back to the free pool, with no bulk
copy. New blocks are allocated for the surviving heads. [PAUSE] The sharing
pattern is being recomputed, cheaply, at every single decoding step. This
is the case that can save up to 55 percent of KV memory. We will measure
that in the results.

Beat 9 — Shared prefix, the paper's example.
Same trick, now across different requests. A production translation service
ships the same few-shot instruction with every call: "Translate English to
French," plus three examples — sea otter, peppermint, plush giraffe.
Sequence A asks for "cheese?" and gets "fromage." Sequence B asks "I love
you?" and gets "Je t'aime." Look at how much of each request is identical.
[PAUSE]

Beat 10 — Cached blocks, two tables.
vLLM computes that shared prefix's KV blocks exactly once, caches them, and
every new request's block table just points at those cached blocks. The
last shared block is marked copy-on-write. Only the task-specific suffix —
the actual question — needs new computation and new blocks.

Beat 11 — Three primitives.
None of this is three special features. The engine exposes three operations.
Fork: create a new sequence from an existing one, share its blocks, bump
the reference counts. Append: write a new token; copy-on-write if the block
is still shared. Free: drop a sequence, decrement, reclaim at zero.
Parallel sampling is fork then append. Beam search is fork, append, and
free, every step. A shared prefix is a fork onto a cache that was computed
once. One API.

Beat 12 — Landing.
Reference counts and copy-on-write pay Act I's bill: sequences can now
point at the same physical memory. Next: what happens when that free pool
is empty.

## S8 S8Scheduling

---------
Beat 1 — Pickup: request A.
Look at leftover VRAM — the slice we have been spending all talk. Request A
is live. Four blocks, scattered, allocated on demand. Three still free.
Paging put those blocks wherever there was room. Sit with this picture.
[PAUSE]

Beat 2 — Request B joins.
Request B draws from the same pool. Its blocks sit interleaved with A's —
not a contiguous slab, just whatever was free. Sharing from last scene
packed this leftover even tighter. Three slots remain. The GPU is fuller
than it has ever been in this talk. It is still finite.

Beat 3 — C fills the leftovers.
A third request arrives. Any free block fits anyone, so C lands in those
three leftover slots, interleaved with A and B. The pool is full. We have
now admitted more work than we can finish if everyone keeps growing.
[PAUSE]

Beat 4 — This picture is new.
Contiguous systems never faced this. They reserved two thousand and
forty-eight slots the moment a request arrived — the model's maximum —
so they never ran out mid-decode. They bounced new work at the door.
Paging created the ability to overcommit. That is why we suddenly need
a policy for taking memory back.

Beat 5 — The pool runs dry.
A generates another token and needs one more block. There isn't one.
[PAUSE] Which request gives up its memory? And what do we do with it?

Beat 6 — Name the scheduler.
The answer starts with a job. Someone has to pick who runs this step,
and who gives up space when the pool is empty. That is the scheduler.
We will watch it work.

Beat 7 — Preempt the newest.
The policy is deliberately simple: first-come, first-served, with
preemption. Serve oldest first. When someone has to leave, evict the
newest arrival — here, C. A has been running longest, so A is the last
request we ever kick. A stream of new arrivals cannot starve the ones
already on the GPU. [PAUSE]

Beat 8 — Try taking half.
Suppose we only take two of C's three blocks, so A can continue. A
gets a block. One of C's blocks is free. One is still C's. Sit with
that leftover. [PAUSE] Every decode step reads every block of a
sequence — that is why the PagedAttention kernel walks the whole table,
every token. C cannot run on one leftover block. Those two freed blocks
bought A a step and stranded C.

Beat 9 — All or none.
Put them back. The rule is all of C's blocks, or none of them. An
operating system will happily evict a single page. We cannot. That is
the policy the OS scene promised: every block of a sequence is always
touched together. [PAUSE]

Beat 10 — Sequence groups.
And if C were two beam candidates sharing a trunk — the copy-on-write
picture from last scene — kicking only one of them would strand the
shared blocks. Reference count two: both still need that memory. So
the scheduler treats a sequence group as one unit. Beams of the same
request are gang-preempted, and gang-resumed, together. Sharing couples
their lifetimes.

Beat 11 — The other pool.
C is leaving. Before we move anything: remember the two memory pools
from the GPU scene. GPU VRAM, and ordinary CPU RAM, joined by PCIe —
the slow bridge. Swap space lives on that other side. Watch it appear.
[PAUSE]

Beat 12 — Swap.
Copy C's blocks across the bridge. GPU slots empty. A takes one and
continues. Swap space cannot grow forever. It is capped at the GPU's
total KV capacity, because that is the most we could ever evict at
once. The cost is PCIe bandwidth.

Beat 13 — Or drop, keep the tokens.
Or we do something an OS almost never does: throw the KV cache away.
Undo the copy. Drop C's blocks. Keep the tokens — the original prompt,
plus whatever C had already generated. The words are cheap. The cache
was the expensive object. [PAUSE]

Beat 14 — One prefill.
When C is rescheduled, concatenate those tokens and treat the whole
string as one new prompt. One prefill pass rebuilds the entire cache
in parallel, not one slow decode step at a time.

Beat 15 — Fig. 4, and the knob.
Zoom out. Same three jobs we just watched. The scheduler picks who
runs and who is preempted. It talks to a KV cache manager that owns
every block table — and under that, a GPU allocator for the leftover
pool, and a CPU allocator for the swap strip. Workers, one GPU each,
run the step they are told to run. Contiguous slabs had no way to
take memory back. Paging is the knob.

## S9 S9Results

---------
Beat 1 — Setup (§6.1).
Same models, same GPUs, Google Cloud A100 machines. OPT-13B on one A100,
66B on four, 175B on eight 80-gigabyte A100s — Table 1 in the paper.
LLaMA-13B is for the translation experiment later. The baselines are
FasterTransformer, NVIDIA's latency-optimized engine, given a dynamic
batching scheduler so the comparison is fair; and three re-implementations
of Orca. Oracle cheats: it knows each request's true output length in
advance — an infeasible upper bound. Pow2 over-reserves by up to two
times. Max always reserves the model's 2048-token maximum. [PAUSE]
The metric is normalized latency: each request's end-to-end latency
divided by its output length, then the mean of that, plotted against
request rate. A system is better if it keeps that number low out to a
higher request rate. Arrivals are Poisson; traces are one hour, fifteen
minutes for 175B because of cost.

Beat 2 — Fig 1-right, the intro claim.
Here is Figure 1 from page 1, the right panel — the claim the evaluation
is about to measure. Existing systems: KV cache memory explodes with
batch size and hits the 40-gigabyte wall. vLLM smooths that growth, so
the batch can keep growing, and throughput keeps rising. The rest of
this scene is section 6 putting numbers on that picture.

Beat 3 — Fig 11, two workloads.
Two datasets. ShareGPT is real multi-turn chat: input mean 161.31 tokens,
output mean 337.99, long right tail out to two thousand. Alpaca is short
instruction-following: input mean 19.31, output mean 58.45, a much
sharper peak. ShareGPT's prompts are 8.4 times longer and its outputs
5.8 times longer, with higher variance — more pressure on the KV cache.

Beat 4 — Fig 12(a), OPT-13B ShareGPT.
Figure 12: normalized latency versus request rate. Watch FasterTransformer
fall over first, then Orca Max, Pow2, Oracle — and vLLM keeps going.
On ShareGPT, vLLM sustains 1.7 to 2.7 times the request rate of Orca
Oracle and 2.7 to 8 times Orca Max, at similar latency — and up to 22
times FasterTransformer. [PAUSE] Same model, same GPU.

Beat 5 — Fig 12 ShareGPT row, 13B / 66B / 175B.
The same ranking at 66B and 175B. Every subplot, vLLM's knee sits furthest
right. This is not a 13B trick.

Beat 6 — Fig 12 Alpaca row, and the 12(f) exception.
Alpaca, shorter sequences, same axes, higher request rates. Same ranking
on 13B and 66B. Now 175B — panel (f). The paper's exception: vLLM's
advantage over Oracle and Pow2 is less pronounced, because 175B has so
much GPU memory and Alpaca's sequences are so short that the workload
is less memory-bound. Memory management helps most when memory is the
constraint. [PAUSE]

Beat 7 — Fig 13, why the knee moved.
Here is the mechanism, Figure 13, OPT-13B. Average number of requests
actually in the batch. ShareGPT at 2 requests per second: Orca Max 7.00,
Pow2 9.81, Oracle 13.62, vLLM 30.42 — 2.2 times Oracle, 4.3 times Max.
Alpaca at 30 per second: 7.00, 43.24, 72.75, 132.44. More of leftover
VRAM is real token state, so more requests fit in one pass. [PAUSE]

Beat 8 — Fig 14, parallel sampling.
Section 6.3. Parallel generation on Alpaca, OPT-13B — two, four, then
six samples per prompt. No FasterTransformer here. Knees move left as
you pay for more sequences, but vLLM's relative gap grows, because the
samples share prompt blocks.

Beat 9 — Fig 14, beam search.
Beam search, width 2, 4, 6. Same story, more sharing. The paper's
sentence: vLLM's improvement over Orca Oracle on OPT-13B and Alpaca
goes from 1.3 times in basic sampling to 2.3 times at beam width 6.
[PAUSE]

Beat 10 — Fig 15, memory actually saved.
Directly measuring the KV blocks shared instead of copied, Alpaca,
OPT-13B. Parallel sampling: 6.09, 8.53, 9.79 percent. Beam search:
37.56, 53.13, 55.16 percent — beam shares almost the whole prefix.
On ShareGPT, with longer sequences, the paper saw 16.2 to 30.5 percent
for parallel sampling and 44.3 to 66.3 percent for beam search.

Beat 11 — Fig 16, shared prefix.
LLaMA-13B, WMT16 English to German, a few-shot prefix shared across
every request. One-shot, 80 tokens: vLLM 1.67 times the throughput of
Orca Oracle. Five-shot, 341 tokens: 3.58 times. The more prefix there
is to share, the bigger the win. [PAUSE]

Beat 12 — Fig 17, chatbot.
The least favorable case: a chatbot workload, ShareGPT-style multi-turn,
prompt truncated to the last 1024 tokens, output capped at 1024, no KV
kept across turns. The three Orca variants cluster — they all reserve
about 1024 under buddy allocation, so Oracle, Pow2, and Max converge.
vLLM still sustains about 2 times their request rate.

Beat 13 — Landing.
Across basic sampling, parallel sampling, beam search, shared prefix,
and chatbot: 2 to 4 times the throughput of Orca at the same latency,
up to 22 times FasterTransformer, no change to the model. Gains are
larger with longer sequences, larger models, and more complex decoding —
exactly what those figures showed. [PAUSE] Next: the kernel itself is
slower. Section 7.

## S10 S10Ablations

---------
Beat 1 — The kernel itself is slower.
The OS scene already said it: PagedAttention's attention kernel is slower
on its own. Here is the measurement. The kernel has to look up a block
table and read key/value data from scattered, non-contiguous memory
instead of one clean contiguous strip. Across batch sizes and context
lengths, it runs about 20 to 26% slower than FasterTransformer's tightly
hand-optimized kernel. [PAUSE] That's a real cost — but attention is only
one operator in a whole forward pass, so this slowdown barely dents
end-to-end latency.

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

## S11 S11Takeaways

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
