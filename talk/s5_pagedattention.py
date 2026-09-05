"""S5 — PagedAttention and the KV cache manager (Act II core).

Paper §4.1 → §4.2 → §4.3, in the paper's order and on its own terms. No
operating-system, virtual-memory, or "page" vocabulary: that reveal is S6.
The paper's word is "block".

Why this order: S4 ended on "each request's KV cache is one contiguous,
pre-allocated chunk". The chunk is contiguous because the attention operator
reads K and V as one tensor each. So the paper first changes the *reader*
(PagedAttention, §4.1), and only then is free to change the *layout* (the KV
cache manager, §4.2), and finally walks one request through it (§4.3).

Figures recreated: 5 (run live), 6 (①②③ plus one extra step that closes the
loop back to Fig 5), 7, and the vLLM bar of Fig 2 as the payoff callback.

NARRATION

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
"""

import math

from manim import (
    DOWN,
    LEFT,
    ORIGIN,
    RIGHT,
    UP,
    AnimationGroup,
    Brace,
    FadeIn,
    FadeOut,
    GrowFromEdge,
    LaggedStart,
    Line,
    Rectangle,
    RoundedRectangle,
    SurroundingRectangle,
    Transform,
    VGroup,
)
from manim_slides import Slide

from talk.components import (
    FOUR_SCORE,
    KVBlock,
    arrow,
    shoot,
)
from talk.theme import (
    ACCENT,
    ACCENT2,
    BAD,
    BG,
    BLOCK_FILL,
    BLOCK_STROKE,
    FG,
    GOOD,
    K_COLOR,
    MUTED,
    Q_COLOR,
    SMALL_SIZE,
    TINY_SIZE,
    TITLE_SIZE,
    V_COLOR,
    WARN,
    apply_theme,
    body,
    caption,
    formula,
    small,
    text,
    title,
)

# Sizes ---------------------------------------------------------------------
CELL = 0.52          # manager (Fig 6/7) cells
FIG5_CELL = 0.76     # Fig 5 cells
SLAB_W, SLAB_H = 0.92, 0.50

# Fig 5 / Eq. 4 — illustrative scores, labeled on screen as such.
_SCORES = {
    0: [0.4, 0.5, 0.3, 0.6],
    1: [0.5, 0.7, 0.4, 1.5],
    2: [1.2, 2.2],
}

W0 = ["Four", "score", "and", "seven"]
W1 = ["years", "ago", "our"]
BW0 = ["it", "was", "the", "best"]
BW1 = ["of", "times"]


# ---------------------------------------------------------------------------
# Small helpers
# ---------------------------------------------------------------------------


def _fit(mob, max_w):
    if mob.width > max_w:
        mob.scale_to_fit_width(max_w)
    return mob


def _foot(s, color=MUTED, size=SMALL_SIZE):
    t = text(s, font_size=size, color=color)
    _fit(t, 13.0)
    t.to_edge(DOWN, buff=0.30)
    return t


# --- baseline-consistent text -------------------------------------------------
# Manim's Text centres on the glyph bounding box, so "and" / "ago" / "score"
# land at three different heights when each is move_to'd the same point. _tx
# measures the word inside an "Ág…Ág" probe (full ascender + descender box)
# and remembers the vertical offset; _center places by that offset instead.


def _tx(content, **kw):
    t = text(content, **kw)
    t.dy = 0.0
    if content:
        probe = text("Ág" + content + "Ág", **kw)
        n = len(probe.submobjects)
        if n == len(t.submobjects) + 4:
            inner = VGroup(*probe.submobjects[2:n - 2])
            t.dy = inner.get_center()[1] - probe.get_center()[1]
    return t


def _center(mob, point):
    """move_to, but keep the text baseline where a full-height word would sit."""
    mob.move_to(point)
    mob.shift(UP * getattr(mob, "dy", 0.0))
    return mob


_LONGEST = "brought"
_FS_CACHE = {}


def _word_fs(avail_w, base=TINY_SIZE):
    """One font size per cell width: the largest at which every word fits."""
    key = (round(avail_w, 3), base)
    if key not in _FS_CACHE:
        probe = text(_LONGEST, font_size=base)
        _FS_CACHE[key] = base * min(1.0, avail_w / probe.width)
    return _FS_CACHE[key]


def _chip(letter, color, side=0.24, fs=13):
    sq = RoundedRectangle(
        width=side, height=side, corner_radius=0.04,
        fill_color=color, fill_opacity=0.95, stroke_width=0,
    )
    lab = _tx(letter, font_size=fs, color=BG)
    _center(lab, sq.get_center())
    g = VGroup(sq, lab)
    g.sq = sq
    return g


def _slab_cell(word, fill, w=SLAB_W, h=SLAB_H, fs=None, txt_color=BG, uniform_w=None):
    """Rounded cell with a word. All cells of width `uniform_w` (default w)
    share one font size, so a row of words never looks uneven."""
    box = RoundedRectangle(
        width=w, height=h, corner_radius=0.08,
        fill_color=fill, fill_opacity=1.0, stroke_color=BLOCK_STROKE, stroke_width=1.5,
    )
    if fs is None:
        fs = _word_fs(0.84 * (uniform_w or w), base=15)
    lbl = _tx(str(word), font_size=fs, color=txt_color)
    _fit(lbl, w * 0.9)
    _center(lbl, box.get_center())
    g = VGroup(box, lbl)
    g.box, g.label = box, lbl
    return g


def _tok(word, w, h, fs, fill=BLOCK_FILL, color=FG):
    """TokenBox equivalent with a baseline-aligned label. .box, .label"""
    box = RoundedRectangle(
        corner_radius=0.1, width=w, height=h,
        fill_color=fill, fill_opacity=1.0, stroke_color=BLOCK_STROKE, stroke_width=2,
    )
    lbl = _tx(str(word), font_size=fs, color=color)
    _center(lbl, box.get_center())
    g = VGroup(box, lbl)
    g.box, g.label = box, lbl
    return g


def _tok_row(words, w, h, fs, gap):
    return VGroup(*[_tok(x, w, h, fs) for x in words]).arrange(RIGHT, buff=gap)


def _fill_slot(block, i, word, color=ACCENT, txt_color=FG):
    """Crossfade a KVBlock cell label and recolor the cell."""
    new_label = _tx(str(word), font_size=_word_fs(0.88 * block.cell), color=txt_color)
    _center(new_label, block.cells[i].get_center())
    old_label = block.labels[i]
    block.labels.submobjects[i] = new_label
    # group=block: otherwise AnimationGroup wraps the cell in a new Group that
    # Scene.play adds top-level, pulling the cells above their labels (labels
    # then render underneath the filled squares and vanish).
    return AnimationGroup(
        block.cells[i].animate.set_fill(color, opacity=1.0),
        FadeOut(old_label),
        FadeIn(new_label),
        group=block,
    )


def _fill_words(block, words, color=ACCENT, start=0, txt_color=FG):
    return AnimationGroup(
        *[_fill_slot(block, start + i, w, color, txt_color) for i, w in enumerate(words)],
        lag_ratio=0.12,
        group=block,
    )


def _prefilled(slots, cell, words, color, index_text=None, txt_color=BG):
    blk = KVBlock(slots=slots, cell=cell, index=None)
    fs = _word_fs(0.88 * cell)
    for i, w in enumerate(words):
        blk.cells[i].set_fill(color, opacity=1.0)
        lbl = _tx(str(w), font_size=fs, color=txt_color)
        _center(lbl, blk.cells[i].get_center())
        blk.labels.submobjects[i] = lbl
    return blk


# ---------------------------------------------------------------------------
# Manager pieces (Fig 6 / 7)
# ---------------------------------------------------------------------------

IDX_W = 0.34  # fixed-width index column left of every row (rows stay aligned)


class _Row(VGroup):
    """Index label + KVBlock. Cells are centred on x=0; the index sits in a
    fixed-width column so rows with different digit widths line up."""

    def __init__(self, index, slots=4, cell=CELL, color=MUTED):
        super().__init__()
        self.kv = KVBlock(slots=slots, cell=cell, index=None)
        self.kv.move_to(ORIGIN)
        self.idx = _tx(str(index), font_size=TINY_SIZE, color=color)
        _center(self.idx, self.kv.cells.get_left() + LEFT * IDX_W / 2)
        self.add(self.idx, self.kv)
        self.cells = self.kv.cells

    def fill(self, i, word, color=ACCENT):
        return _fill_slot(self.kv, i, word, color, txt_color=BG)

    def fill_many(self, words, color=ACCENT, start=0):
        return _fill_words(self.kv, words, color, start, txt_color=BG)

    def reserve(self, i):
        return self.kv.cells[i].animate.set_fill(WARN, opacity=0.55)

    def clear_all(self, n):
        return AnimationGroup(*[_fill_slot(self.kv, i, "", BLOCK_FILL) for i in range(n)], group=self.kv)


class _Stack(VGroup):
    def __init__(self, n, head_lines, cell=CELL, head_color=FG, ghost_from=None, gap=0.10):
        super().__init__()
        self.rows = [_Row(i, cell=cell) for i in range(n)]
        # rows are built with cells centred on x=0; stack them without re-aligning
        # on the (variable-width) index labels
        for i, r in enumerate(self.rows):
            r.shift(DOWN * i * (cell + gap))
        self.body = VGroup(*self.rows)
        self.head = text(head_lines, font_size=18, color=head_color, line_spacing=0.9)
        _fit(self.head, self.body.width + 0.9)
        self.head.next_to(self.body, UP, buff=0.18)
        self.add(self.head, self.body)
        self.ghost_from = ghost_from
        if ghost_from is not None:
            for r in self.rows[ghost_from:]:
                r.set_opacity(0.18)

    def row(self, i):
        return self.rows[i]


class _Table(VGroup):
    """Block table: logical | physical | # filled. Rows exist from the start ('–')."""

    def __init__(self, n=4, head_str="Block table", head_color=FG):
        super().__init__()
        # explicit grid: column centres and row centres; every entry is placed
        # with _center so digits, dashes and headers share one baseline per row
        col_x = [0.0, 1.0, 2.05]
        row_h = 0.42
        head_y = 0.0
        self.grid = (col_x, [head_y - 0.46 - i * row_h for i in range(n)])
        self.head = text(head_str, font_size=18, color=head_color)
        cols = VGroup()
        for x, s in zip(col_x, ("logical", "physical", "# filled")):
            c = _tx(s, font_size=TINY_SIZE, color=MUTED)
            _center(c, [x, head_y, 0])
            cols.add(c)
        self.cols = cols
        self.phys = []
        self.fill = []
        self.rows = VGroup()
        for i in range(n):
            y = self.grid[1][i]
            lg = _center(_tx(str(i), font_size=18, color=MUTED), [col_x[0], y, 0])
            ph = _center(_tx("–", font_size=18, color=MUTED), [col_x[1], y, 0])
            fl = _center(_tx("–", font_size=18, color=MUTED), [col_x[2], y, 0])
            row = VGroup(lg, ph, fl)
            self.phys.append(ph)
            self.fill.append(fl)
            self.rows.add(row)
        self.frame = RoundedRectangle(
            width=(col_x[-1] - col_x[0]) + 1.15,
            height=(head_y - self.grid[1][-1]) + 0.62,
            corner_radius=0.10,
            stroke_color=BLOCK_STROKE,
            stroke_width=1.5,
            fill_opacity=0,
        )
        self.frame.move_to([(col_x[0] + col_x[-1]) / 2, (head_y + self.grid[1][-1]) / 2, 0])
        self.head.next_to(self.frame, UP, buff=0.12)
        self.add(self.frame, self.head, cols, self.rows)

    def _cell_point(self, i, col):
        # current position of the grid point (the table may have moved)
        ref = self.rows[i][0]  # logical index never changes
        return [self.cols[col].get_center()[0], ref.get_center()[1] - ref.dy, 0]

    def set_mapping(self, i, physical, filled, color=FG):
        new_p = _center(_tx(str(physical), font_size=18, color=color), self._cell_point(i, 1))
        new_f = _center(_tx(str(filled), font_size=18, color=color), self._cell_point(i, 2))
        old_p, old_f = self.phys[i], self.fill[i]
        self.phys[i], self.fill[i] = new_p, new_f
        row = self.rows[i]
        row.submobjects[1] = new_p
        row.submobjects[2] = new_f
        return AnimationGroup(FadeOut(old_p), FadeIn(new_p), FadeOut(old_f), FadeIn(new_f), group=self)

    def set_filled(self, i, filled, color=FG):
        new_f = _center(_tx(str(filled), font_size=18, color=color), self._cell_point(i, 2))
        old_f = self.fill[i]
        self.fill[i] = new_f
        self.rows[i].submobjects[2] = new_f
        return AnimationGroup(FadeOut(old_f), FadeIn(new_f), group=self)

    def row_right(self, i):
        p = self.frame.get_right().copy()
        p[1] = self.rows[i].get_center()[1]
        return p

    def clear_all(self):
        anims = []
        for i in range(len(self.rows)):
            anims.append(self.set_mapping(i, "–", "–", MUTED))
        return AnimationGroup(*anims, group=self)


def _map_arrow(table, i, dram_row, color):
    end = dram_row.idx.get_left().copy()
    return arrow(table.row_right(i), end, color=color, buff=0.05, stroke_width=1.8)


# ---------------------------------------------------------------------------
# Fig 2 mini stacked bar (payoff callback)
# ---------------------------------------------------------------------------


def _stack_bar(name, parts, h=2.6, w=1.05):
    outline = Rectangle(width=w, height=h, stroke_color=BLOCK_STROKE, stroke_width=2)
    segs = VGroup()
    labels = VGroup()
    y = outline.get_bottom()[1]
    cx = outline.get_center()[0]
    for frac, color, opacity, lab in parts:
        sh = h * frac
        rect = Rectangle(width=w, height=max(sh, 1e-4), fill_color=color, fill_opacity=opacity, stroke_width=0)
        rect.move_to([cx, y + sh / 2, 0])
        segs.add(rect)
        if lab and sh >= 0.22:
            t = text(lab, font_size=13, color=FG)
            _fit(t, w * 0.88)
            t.move_to(rect.get_center())
            labels.add(t)
        y += sh
    name_t = text(name, font_size=14, color=FG, line_spacing=1.0)
    _fit(name_t, w + 0.5)
    name_t.next_to(outline, DOWN, buff=0.16)
    g = VGroup(outline, segs, labels, name_t)
    g.outline, g.segs, g.seg_labels, g.name_t = outline, segs, labels, name_t
    return g


# ---------------------------------------------------------------------------
# Scene
# ---------------------------------------------------------------------------


class S5PagedAttention(Slide):
    def construct(self):
        apply_theme(self)

        # ==================================================================
        # Beat 1 — Who demanded contiguity?
        # ==================================================================
        heading = title("Why was it one contiguous chunk?")

        cells = VGroup(*[_slab_cell(w, ACCENT) for w in FOUR_SCORE])
        cells.arrange(RIGHT, buff=0.06)
        resv = _slab_cell("2038 slots reserved · never used", BAD, w=2.9, fs=13, txt_color=FG, uniform_w=SLAB_W)
        slab = VGroup(cells, resv).arrange(RIGHT, buff=0.06)
        _fit(slab, 13.0)
        slab.move_to(UP * 1.35)
        slab_lab = caption("request A, as existing systems store it (S4)")
        slab_lab.next_to(slab, UP, buff=0.16)

        self.play(FadeIn(heading), run_time=0.4)
        self.play(FadeIn(slab_lab), LaggedStart(*[FadeIn(c) for c in cells], lag_ratio=0.05), FadeIn(resv), run_time=1.0)

        kernel = RoundedRectangle(
            width=6.6, height=1.25, corner_radius=0.14,
            fill_color=BLOCK_FILL, fill_opacity=1.0, stroke_color=BLOCK_STROKE, stroke_width=2.5,
        )
        kernel.move_to(DOWN * 1.15)
        k_title = small("attention operator", color=FG)
        k_sub = formula(
            "reads  K = [k<sub>1</sub> … k<sub>n</sub>]   and   V = [v<sub>1</sub> … v<sub>n</sub>]   as one tensor each",
            font_size=17, color=MUTED,
        )
        _fit(k_sub, kernel.width - 0.4)
        k_inner = VGroup(k_title, k_sub).arrange(DOWN, buff=0.12).move_to(kernel.get_center())
        kernel_g = VGroup(kernel, k_inner)

        # bracket from the kernel up to the whole used region of the slab
        up_arrows = VGroup()
        for i in (0, 4, 9):
            a = arrow(kernel.get_top() + RIGHT * ((i - 4.5) * 0.35), cells[i].get_bottom(), color=MUTED, buff=0.08)
            up_arrows.add(a)
        self.play(FadeIn(kernel_g), run_time=0.45)
        self.play(*[shoot(a) for a in up_arrows], run_time=0.6)

        why = small("frameworks require contiguous tensors  →  the reader dictates the layout", color=FG)
        _fit(why, 13.0)
        why.next_to(kernel, DOWN, buff=0.42)
        self.play(FadeIn(why), run_time=0.4)

        pivot = _foot("change the reader, and the layout is free", color=ACCENT)
        self.play(FadeIn(pivot), run_time=0.4)
        self.wait(0.3)
        self.next_slide()

        # ==================================================================
        # Beat 2 — Partition into fixed-size blocks
        # ==================================================================
        heading = self._retitle(heading, "Partition the KV cache into fixed-size blocks",
                                FadeOut(kernel_g), FadeOut(up_arrows), FadeOut(why), FadeOut(pivot), FadeOut(slab_lab))
        self._dump(VGroup(kernel_g, up_arrows, why, pivot, slab_lab))

        # drop the reservation, spread the tokens
        self.play(FadeOut(resv, shift=RIGHT * 0.4), run_time=0.45)
        self._dump(resv)
        TOK_W, TOK_GAP = 1.04, 0.09
        tokens = _tok_row(FOUR_SCORE, TOK_W, 0.62, 20, TOK_GAP)
        # block 2 has two empty slots to the right: centre the 12-slot span, not the 10 words
        tokens.move_to(UP * 1.0 + LEFT * (TOK_W + TOK_GAP))
        self.play(Transform(cells, tokens), run_time=0.7)
        self.remove(cells)
        self.add(tokens)

        def _ring(group, color):
            return SurroundingRectangle(group, buff=0.10, color=color, stroke_width=3, corner_radius=0.08)

        r0 = _ring(VGroup(*tokens[:4]), ACCENT)
        r1 = _ring(VGroup(*tokens[4:8]), ACCENT)
        r2 = RoundedRectangle(width=r0.width, height=r0.height, corner_radius=0.08,
                              stroke_color=WARN, stroke_width=3, fill_opacity=0)
        r2.move_to([tokens[8].get_left()[0] - 0.10 + r2.width / 2, tokens[8].get_center()[1], 0])
        empty = VGroup(*[
            RoundedRectangle(corner_radius=0.1, width=TOK_W, height=0.62, fill_opacity=0,
                             stroke_color=BLOCK_STROKE, stroke_width=2).next_to(tokens[9], RIGHT, buff=TOK_GAP * (k + 1) + TOK_W * k)
            for k in range(2)
        ])
        for e in empty:
            e.set_stroke(opacity=0.5)
        l0 = small("Block 0", color=ACCENT).next_to(r0, DOWN, buff=0.18)
        l1 = small("Block 1", color=ACCENT).next_to(r1, DOWN, buff=0.18)
        l2 = small("Block 2  ·  2 / 4", color=WARN).next_to(r2, DOWN, buff=0.18)
        self.play(FadeIn(r0), FadeIn(l0), run_time=0.35)
        self.play(FadeIn(r1), FadeIn(l1), run_time=0.35)
        self.play(FadeIn(r2), FadeIn(l2), FadeIn(empty), run_time=0.35)

        gone = VGroup(
            body("block size B is fixed, chosen once, up front", font_size=24),
            small("paper default B = 16 tokens  ·  drawn here with B = 4", color=MUTED),
            small("no slots reserved for the 2048-token maximum", color=GOOD),
        ).arrange(DOWN, buff=0.20)
        gone.move_to(DOWN * 1.55)
        self.play(LaggedStart(*[FadeIn(g) for g in gone], lag_ratio=0.3), run_time=0.9)
        self.wait(0.3)
        self.next_slide()

        # ==================================================================
        # Beat 3 — A block is K and V for B tokens
        # ==================================================================
        heading = self._retitle(heading, "A block is K and V for B tokens",
                                FadeOut(gone), FadeOut(r1), FadeOut(l1), FadeOut(r2), FadeOut(l2), FadeOut(l0), FadeOut(empty))
        self._dump(VGroup(gone, r1, l1, r2, l2, l0, empty))

        rest = VGroup(*tokens[4:])
        self.play(rest.animate.set_opacity(0.18), run_time=0.35)
        zoom = VGroup(*tokens[:4])
        zoom_target = zoom.copy().scale(1.35).move_to(UP * 1.35 + LEFT * 2.2)
        r0_target = _ring(zoom_target, ACCENT)
        self.play(Transform(zoom, zoom_target), Transform(r0, r0_target), FadeOut(rest), run_time=0.7)
        self._dump(rest)

        kchips, vchips = VGroup(), VGroup()
        for tok in zoom:
            k = _chip("k", K_COLOR, side=0.42, fs=18)
            v = _chip("v", V_COLOR, side=0.42, fs=18)
            k.next_to(tok, DOWN, buff=0.22)
            v.next_to(k, DOWN, buff=0.08)
            kchips.add(k)
            vchips.add(v)
        self.play(LaggedStart(*[FadeIn(k, shift=DOWN * 0.1) for k in kchips], lag_ratio=0.12), run_time=0.6)
        self.play(LaggedStart(*[FadeIn(v, shift=DOWN * 0.1) for v in vchips], lag_ratio=0.12), run_time=0.6)

        brace_k = Brace(kchips, RIGHT, buff=0.18, color=K_COLOR)
        brace_v = Brace(vchips, RIGHT, buff=0.18, color=V_COLOR)
        kj = formula("K<sub>j</sub> = ( k<sub>(j−1)B+1</sub> , … , k<sub>jB</sub> )", font_size=22, color=K_COLOR)
        vj = formula("V<sub>j</sub> = ( v<sub>(j−1)B+1</sub> , … , v<sub>jB</sub> )", font_size=22, color=V_COLOR)
        kj.next_to(kchips, RIGHT, buff=0.55)
        vj.next_to(vchips, RIGHT, buff=0.55)
        kj.align_to(kchips, DOWN).shift(UP * 0.02)
        vj.align_to(vchips, DOWN).shift(UP * 0.02)
        self.play(FadeIn(kj), FadeIn(vj), run_time=0.5)

        b_brace = Brace(zoom, UP, buff=0.22, color=MUTED)
        b_lab = small("B tokens", color=MUTED).next_to(b_brace, UP, buff=0.08)
        self.play(FadeIn(b_brace), FadeIn(b_lab), run_time=0.4)

        notes = VGroup(
            small("one slot = one token's Key and Value  (800 KB per token for OPT-13B)", color=FG),
            small("stored per layer and head; every layer sees the same block structure", color=MUTED),
            small("last block may be partial: those slots are reserved — the only reservation, < 1 block", color=WARN),
        ).arrange(DOWN, buff=0.16, aligned_edge=LEFT)
        for n in notes:
            _fit(n, 13.0)
        notes.arrange(DOWN, buff=0.16, aligned_edge=LEFT)
        notes.to_edge(DOWN, buff=0.35)
        self.play(LaggedStart(*[FadeIn(n) for n in notes], lag_ratio=0.3), run_time=0.9)
        self.wait(0.3)
        self.next_slide()

        # ==================================================================
        # Beat 4 — Eq. 3 over the flat row
        # ==================================================================
        self.play(FadeOut(*self.mobjects), run_time=0.4)
        heading = self._retitle(None, "Attention as we left it  (Eq. 3)")

        row = _tok_row(FOUR_SCORE, 1.02, 0.56, 18, 0.08)
        row.move_to(UP * 0.55 + RIGHT * 0.55)
        krow, vrow = VGroup(), VGroup()
        for tok in row:
            k = _chip("k", K_COLOR, side=0.34, fs=15).next_to(tok, UP, buff=0.14)
            v = _chip("v", V_COLOR, side=0.34, fs=15).next_to(tok, DOWN, buff=0.14)
            krow.add(k)
            vrow.add(v)
        krow_lab = small("keys", color=K_COLOR).next_to(krow, LEFT, buff=0.22)
        vrow_lab = small("values", color=V_COLOR).next_to(vrow, LEFT, buff=0.22)
        row_lab = small("tokens 1 … i", color=MUTED).next_to(row, LEFT, buff=0.22)

        q_box = VGroup(caption('query  ·  "forth"'), _chip("q", Q_COLOR, side=0.42, fs=18)).arrange(DOWN, buff=0.08)
        q_box.next_to(krow, UP, buff=0.50)
        q_box.set_x(row[-1].get_center()[0])

        self.play(FadeIn(row), FadeIn(row_lab), run_time=0.5)
        self.play(FadeIn(krow), FadeIn(vrow), FadeIn(krow_lab), FadeIn(vrow_lab), run_time=0.5)
        self.play(FadeIn(q_box), run_time=0.35)

        fan = VGroup(*[arrow(q_box[1].get_bottom(), k.get_top(), color=Q_COLOR, buff=0.05, stroke_width=1.2) for k in krow])
        self.play(LaggedStart(*[shoot(a) for a in fan], lag_ratio=0.04), run_time=0.8)

        eq3 = VGroup(
            formula("a<sub>ij</sub>  =  exp( q<sub>i</sub><sup>T</sup> k<sub>j</sub> / √d )  /  Σ<sub>t = 1 … i</sub> exp( q<sub>i</sub><sup>T</sup> k<sub>t</sub> / √d )", font_size=22),
            formula("o<sub>i</sub>  =  Σ<sub>j = 1 … i</sub>  a<sub>ij</sub> v<sub>j</sub>", font_size=22, color=V_COLOR),
        ).arrange(DOWN, buff=0.18, aligned_edge=LEFT)
        eq3.move_to(DOWN * 1.75)
        _fit(eq3, 12.8)
        self.play(FadeIn(eq3), run_time=0.5)

        emph = _foot("two sums over every previous token — one pass over one contiguous K, one contiguous V", color=FG)
        self.play(FadeIn(emph), run_time=0.4)
        self.wait(0.3)
        self.next_slide()

        # ==================================================================
        # Beat 5 — same sums grouped by block (Eq. 4)
        # ==================================================================
        heading = self._retitle(heading, "The same sums, grouped by block  (Eq. 4)", FadeOut(emph), FadeOut(fan))
        self._dump(VGroup(emph, fan))

        g0 = VGroup(*row[:4], *krow[:4], *vrow[:4])
        g1 = VGroup(*row[4:8], *krow[4:8], *vrow[4:8])
        g2 = VGroup(*row[8:], *krow[8:], *vrow[8:])
        rings = VGroup(
            SurroundingRectangle(g0, buff=0.09, color=ACCENT, stroke_width=2.5, corner_radius=0.08),
            SurroundingRectangle(g1, buff=0.09, color=ACCENT, stroke_width=2.5, corner_radius=0.08),
            SurroundingRectangle(g2, buff=0.09, color=WARN, stroke_width=2.5, corner_radius=0.08),
        )
        ring_labs = VGroup(
            formula("block 0  →  K<sub>0</sub>, V<sub>0</sub>", font_size=TINY_SIZE, color=MUTED).next_to(rings[0], DOWN, buff=0.10),
            formula("block 1  →  K<sub>1</sub>, V<sub>1</sub>", font_size=TINY_SIZE, color=MUTED).next_to(rings[1], DOWN, buff=0.10),
            formula("block 2  →  K<sub>2</sub>, V<sub>2</sub>", font_size=TINY_SIZE, color=MUTED).next_to(rings[2], DOWN, buff=0.10),
        )
        self.play(LaggedStart(*[FadeIn(r) for r in rings], lag_ratio=0.2), FadeIn(ring_labs), run_time=0.8)

        eq4 = VGroup(
            formula(
                "A<sub>ij</sub>  =  exp( q<sub>i</sub><sup>T</sup> K<sub>j</sub> / √d )  /  Σ<sub>t = 1 … ⌈i/B⌉</sub> exp( q<sub>i</sub><sup>T</sup> K<sub>t</sub> 1 / √d )",
                font_size=22,
            ),
            formula("o<sub>i</sub>  =  Σ<sub>j = 1 … ⌈i/B⌉</sub>  V<sub>j</sub> A<sub>ij</sub><sup>T</sup>", font_size=22, color=V_COLOR),
        ).arrange(DOWN, buff=0.18, aligned_edge=LEFT)
        eq4.move_to(eq3.get_center())
        _fit(eq4, 12.8)
        gloss = VGroup(
            caption("A_ij : the scores of query i against the keys of block j  ·  a row vector of length B"),
            caption("sum over tokens  =  sum over blocks of a sum within the block  ·  nothing approximated"),
        ).arrange(DOWN, buff=0.08)
        gloss.to_edge(DOWN, buff=0.28)
        self.play(FadeOut(eq3), FadeIn(eq4), run_time=0.6)
        self._dump(eq3)
        self.play(FadeIn(gloss), run_time=0.4)
        self.wait(0.3)
        self.next_slide()

        # ==================================================================
        # Beat 6 — Fig 5: fetch a block, score, accumulate (block 0)
        # ==================================================================
        self.play(FadeOut(*self.mobjects), run_time=0.4)
        heading = self._retitle(None, "Fig. 5  ·  fetch a block, score it, accumulate")

        # Query panel (left)
        q_frame = RoundedRectangle(width=2.3, height=1.35, corner_radius=0.12, stroke_color=Q_COLOR,
                                   stroke_width=2.5, fill_color=BLOCK_FILL, fill_opacity=1.0)
        q_tok = _tok("forth", 1.3, 0.52, SMALL_SIZE, fill=Q_COLOR, color=BG)
        q_inner = VGroup(_chip("q", Q_COLOR, side=0.3, fs=15), q_tok).arrange(DOWN, buff=0.10).move_to(q_frame.get_center())
        q_lab = caption("query vector").next_to(q_frame, DOWN, buff=0.10)
        query = VGroup(q_frame, q_inner, q_lab).move_to(LEFT * 4.55 + UP * 1.55)

        # Accumulator panel (left, below)
        acc_frame = RoundedRectangle(width=3.1, height=2.0, corner_radius=0.12, stroke_color=BLOCK_STROKE,
                                     stroke_width=2, fill_color=BLOCK_FILL, fill_opacity=1.0)
        acc_frame.move_to(LEFT * 4.55 + DOWN * 1.1)
        acc_head = caption("running totals").next_to(acc_frame.get_top(), DOWN, buff=0.12)
        den_lab = small("Σ exp", color=FG)
        den_val = text("0.0", font_size=SMALL_SIZE, color=Q_COLOR)
        den_row = VGroup(den_lab, den_val).arrange(RIGHT, buff=0.30)
        num_lab = small("Σ exp · v", color=FG)
        num_track = Rectangle(width=1.5, height=0.30, stroke_color=BLOCK_STROKE, stroke_width=1.5, fill_opacity=0)
        num_bar = Rectangle(width=1e-3, height=0.30, fill_color=V_COLOR, fill_opacity=0.95, stroke_width=0)
        num_bar.move_to(num_track.get_left(), aligned_edge=LEFT)
        num_row = VGroup(num_lab, VGroup(num_track, num_bar)).arrange(RIGHT, buff=0.30)
        acc_rows = VGroup(den_row, num_row).arrange(DOWN, buff=0.30, aligned_edge=LEFT)
        acc_rows.move_to(acc_frame.get_center() + DOWN * 0.12)
        acc = VGroup(acc_frame, acc_head, acc_rows)

        # Blocks (right), physical order 1, 2, 0 — Fig 5
        def _fig5_block(idx, words, color, n_fill):
            blk = _prefilled(4, FIG5_CELL, words, color)
            lab = small(f"Block {idx}", color=MUTED).next_to(blk.cells, LEFT, buff=0.28)
            g = VGroup(lab, blk)
            g.blk, g.lab = blk, lab
            g.n_fill = n_fill
            return g

        fb1 = _fig5_block(1, ["years", "ago", "our", "fathers"], ACCENT, 4)
        fb2 = _fig5_block(2, ["brought", "forth"], ACCENT, 2)
        fb2.blk.cells[2].set_fill(WARN, opacity=0.55)
        fb2.blk.cells[3].set_fill(WARN, opacity=0.55)
        fb0 = _fig5_block(0, ["Four", "score", "and", "seven"], ACCENT, 4)
        blocks = VGroup(fb1, fb2, fb0).arrange(DOWN, buff=0.95, aligned_edge=LEFT)
        blocks.move_to(RIGHT * 2.85 + DOWN * 0.30)
        kv_head = small("key and value blocks  ·  not adjacent, not in order", color=MUTED)
        kv_head.move_to(UP * 2.62)
        kv_head.set_x(blocks.get_center()[0])
        # addresses to make "scattered" concrete
        addr = VGroup(
            caption("@ 0x1c00").next_to(fb1.blk.cells, RIGHT, buff=0.22),
            caption("@ 0x7a80").next_to(fb2.blk.cells, RIGHT, buff=0.22),
            caption("@ 0x0340").next_to(fb0.blk.cells, RIGHT, buff=0.22),
        )

        self.play(FadeIn(query), run_time=0.4)
        self.play(FadeIn(kv_head), FadeIn(blocks), FadeIn(addr), run_time=0.55)
        self.play(FadeIn(acc), run_time=0.4)

        illus = caption("illustrative numbers")
        illus.next_to(acc_frame, DOWN, buff=0.10)
        self.play(FadeIn(illus), run_time=0.25)

        running_den = [0.0]
        total_den = sum(math.exp(s) for k in _SCORES for s in _SCORES[k])

        def _process_block(fbk, idx, step_label):
            anims_in = []
            fetch = arrow(q_frame.get_right(), fbk.blk.cells.get_left(), color=Q_COLOR, buff=0.08, stroke_width=1.8)
            ring = SurroundingRectangle(fbk.blk.cells, buff=0.05, color=Q_COLOR, stroke_width=2.5, corner_radius=0.06)
            # scores sit on one baseline, clear of the ring (ring top = cells top + 0.05)
            score_y = fbk.blk.cells.get_top()[1] + 0.05 + 0.20
            step = formula(step_label, font_size=TINY_SIZE, color=Q_COLOR)
            step.next_to(fbk.blk.cells, UP, buff=0.50)
            step.align_to(fbk.blk.cells, LEFT)
            self.play(shoot(fetch), FadeIn(ring), run_time=0.5)
            scores = VGroup()
            for i, s in enumerate(_SCORES[idx]):
                t = _tx(f"{s:.1f}", font_size=14, color=Q_COLOR)
                _center(t, [fbk.blk.cells[i].get_center()[0], score_y, 0])
                scores.add(t)
            self.play(FadeIn(scores), FadeIn(step), run_time=0.4)
            exps = VGroup()
            for i, s in enumerate(_SCORES[idx]):
                e = math.exp(s)
                t = _tx(f"{e:.1f}", font_size=14, color=WARN)
                _center(t, [fbk.blk.cells[i].get_center()[0], score_y, 0])
                exps.add(t)
            step2 = caption("exp", color=WARN).next_to(step, RIGHT, buff=0.3)
            self.play(Transform(scores, exps), FadeIn(step2), run_time=0.45)
            # accumulate
            running_den[0] += sum(math.exp(s) for s in _SCORES[idx])
            new_den = text(f"{running_den[0]:.1f}", font_size=SMALL_SIZE, color=Q_COLOR)
            new_den.move_to(den_val.get_left(), aligned_edge=LEFT)
            new_bar = Rectangle(width=max(1.5 * running_den[0] / total_den, 1e-3), height=0.30,
                                fill_color=V_COLOR, fill_opacity=0.95, stroke_width=0)
            new_bar.move_to(num_track.get_left(), aligned_edge=LEFT)
            fly = VGroup(*[t.copy() for t in scores])
            self.play(
                fly.animate.move_to(den_val.get_center()).set_opacity(0),
                Transform(den_val, new_den),
                Transform(num_bar, new_bar),
                run_time=0.7,
            )
            self.remove(fly)
            self.play(FadeOut(fetch), FadeOut(ring), FadeOut(step), FadeOut(step2), run_time=0.25)
            self._dump(VGroup(fetch, ring, step, step2))
            return scores

        s0 = _process_block(fb0, 0, "q<sup>T</sup> K<sub>0</sub> / √d")
        self.wait(0.3)
        self.next_slide()

        # ==================================================================
        # Beat 7 — blocks 1, 2, then divide
        # ==================================================================
        heading = self._retitle(heading, "Fig. 5  ·  blocks 1 and 2, then divide")
        s1 = _process_block(fb1, 1, "q<sup>T</sup> K<sub>1</sub> / √d")
        partial = caption("partial block: # filled = 2, kernel reads only those", color=WARN)
        partial.next_to(fb2.blk.cells, DOWN, buff=0.10).align_to(fb2.blk.cells, LEFT)
        self.play(FadeIn(partial), run_time=0.3)
        s2 = _process_block(fb2, 2, "q<sup>T</sup> K<sub>2</sub> / √d")

        out = RoundedRectangle(width=2.35, height=0.9, corner_radius=0.10, fill_color=V_COLOR, fill_opacity=0.95, stroke_width=0)
        out_lab = VGroup(
            text("o  =  Σ exp·v  /  Σ exp", font_size=17, color=BG),
            text("attention output for this step", font_size=12, color=BG),
        ).arrange(DOWN, buff=0.08)
        _fit(out_lab, out.width - 0.16)
        out_lab.move_to(out.get_center())
        out_g = VGroup(out, out_lab)
        out_g.next_to(acc_frame, RIGHT, buff=0.35)
        out_g.set_y(acc_rows.get_center()[1])
        div_arrow = arrow(acc_frame.get_right(), out.get_left(), color=V_COLOR, buff=0.06)
        self.play(FadeOut(illus), shoot(div_arrow), FadeIn(out_g), run_time=0.6)
        self._dump(illus)

        land7 = VGroup(
            small("same output as Eq. 3  ·  three blocks, three unrelated addresses, never adjacent", color=FG),
            small("cost: one block-table lookup per block   ·   payoff: blocks may live anywhere", color=MUTED),
        ).arrange(DOWN, buff=0.10)
        for l in land7:
            _fit(l, 13.0)
        land7.arrange(DOWN, buff=0.10)
        land7.to_edge(DOWN, buff=0.28)
        self.play(FadeIn(land7), run_time=0.45)
        self.wait(0.3)
        self.next_slide()

        # ==================================================================
        # Beat 8 — the KV cache manager: three objects
        # ==================================================================
        self.play(FadeOut(*self.mobjects), run_time=0.4)
        heading = self._retitle(None, "Who decides where a block lives?  The KV cache manager")

        logical = _Stack(4, "Request A\nlogical KV blocks", head_color=ACCENT, ghost_from=2)
        table = _Table(4, "Block table  (A)", head_color=ACCENT)
        dram = _Stack(8, "Physical KV blocks\n(GPU DRAM)", head_color=FG, gap=0.08)
        logical_b = _Stack(3, "Request B\nlogical KV blocks", head_color=ACCENT2, ghost_from=2)
        table_b = _Table(3, "Block table  (B)", head_color=ACCENT2)
        b_col = VGroup(logical_b, table_b).arrange(DOWN, buff=0.35)

        manager = VGroup(logical, table, dram, b_col)
        manager.arrange(RIGHT, buff=0.55, aligned_edge=UP)
        _fit(manager, 13.4)
        manager.next_to(heading, DOWN, buff=0.28)
        # align heads on the same baseline
        for part in (logical, table, dram, logical_b):
            part.head.align_to(dram.head, UP)
        # B column stays hidden until Fig 7; until then, centre the A trio and
        # slide it left when B arrives.
        trio = VGroup(logical, table, dram)
        trio_dx = -trio.get_center()[0]
        trio.shift(RIGHT * trio_dx)

        self.play(FadeIn(logical), run_time=0.45)
        cap_l = text("what the request sees:\ncontiguous, filled left → right", font_size=TINY_SIZE, color=MUTED, line_spacing=0.9)
        cap_l.next_to(logical.body, DOWN, buff=0.28)
        cap_l.align_to(logical.body, LEFT)
        self.play(FadeIn(cap_l), run_time=0.3)

        self.play(FadeIn(dram), run_time=0.45)
        cap_d = text("one GPU allocation,\ncarved into equal blocks:\na shared pool", font_size=TINY_SIZE, color=MUTED, line_spacing=0.9)
        cap_d.next_to(dram.body, RIGHT, buff=0.35)
        _fit(cap_d, max(6.9 - cap_d.get_left()[0], 1.0))
        cap_d.next_to(dram.body, RIGHT, buff=0.35)
        self.play(FadeIn(cap_d), run_time=0.3)

        self.play(FadeIn(table), run_time=0.45)
        cap_t = text("one per request:\nlogical → physical, and # filled", font_size=TINY_SIZE, color=MUTED, line_spacing=0.9)
        cap_t.next_to(table.frame, DOWN, buff=0.28)
        self.play(FadeIn(cap_t), run_time=0.3)

        foot8 = _foot("physical blocks are handed out on demand — not reserved for the maximum length", color=FG)
        self.play(FadeIn(foot8), run_time=0.35)
        self.wait(0.3)
        self.next_slide()

        # ==================================================================
        # Beat 9 — Fig 6 ① prefill
        # ==================================================================
        heading = self._retitle(heading, "Fig. 6  ·  ①  prefill: 7 tokens, 2 blocks",
                                FadeOut(cap_l), FadeOut(cap_d), FadeOut(cap_t), FadeOut(foot8))
        self._dump(VGroup(cap_l, cap_d, cap_t, foot8))

        prompt = VGroup(
            small('prompt:  "Four score and seven years ago our"', color=FG),
            caption("prefill runs ordinary self-attention — every prompt token is known"),
        ).arrange(DOWN, buff=0.08)
        prompt.to_edge(DOWN, buff=0.28)
        self.play(FadeIn(prompt), run_time=0.3)

        self.play(logical.row(0).fill_many(W0, ACCENT), run_time=0.8)
        self.play(table.set_mapping(0, 7, 4, ACCENT), run_time=0.35)
        arr_a0 = _map_arrow(table, 0, dram.row(7), ACCENT)
        self.play(shoot(arr_a0), dram.row(7).fill_many(W0, ACCENT), run_time=0.8)

        self.play(logical.row(1).fill_many(W1, ACCENT), logical.row(1).reserve(3), run_time=0.7)
        self.play(table.set_mapping(1, 1, 3, ACCENT), run_time=0.35)
        arr_a1 = _map_arrow(table, 1, dram.row(1), ACCENT)
        self.play(shoot(arr_a1), dram.row(1).fill_many(W1, ACCENT), dram.row(1).reserve(3), run_time=0.8)

        resv_note = caption("1 slot reserved for generation  ·  nothing else reserved", color=WARN)
        resv_note.to_edge(DOWN, buff=0.28)
        self.play(FadeOut(prompt), FadeIn(resv_note), run_time=0.35)
        self._dump(prompt)
        self.wait(0.3)
        self.next_slide()

        # ==================================================================
        # Beat 10 — ② first decode
        # ==================================================================
        heading = self._retitle(heading, 'Fig. 6  ·  ②  first decode → "fathers"  ·  no new block', FadeOut(resv_note))
        self._dump(resv_note)

        q10 = _chip("q", Q_COLOR, side=0.34, fs=15)
        q10.next_to(dram.head, RIGHT, buff=0.40)
        aq7 = arrow(q10.get_bottom(), dram.row(7).cells.get_right() + LEFT * 0.02, color=Q_COLOR, buff=0.06, stroke_width=1.5)
        aq1 = arrow(q10.get_bottom(), dram.row(1).cells.get_right() + LEFT * 0.02, color=Q_COLOR, buff=0.06, stroke_width=1.5)
        self.play(FadeIn(q10), run_time=0.25)
        self.play(shoot(aq7), shoot(aq1), run_time=0.5)
        att = caption("PagedAttention over physical blocks 7 and 1 — the Fig. 5 loop", color=Q_COLOR)
        att.to_edge(DOWN, buff=0.28)
        self.play(FadeIn(att), run_time=0.3)

        out_tok = _tok("fathers", 1.15, 0.46, 16, fill=ACCENT, color=BG)
        out_tok.next_to(q10, RIGHT, buff=0.30)
        self.play(FadeIn(out_tok, shift=RIGHT * 0.1), run_time=0.3)
        self.play(
            logical.row(1).fill(3, "fathers", ACCENT),
            dram.row(1).fill(3, "fathers", ACCENT),
            table.set_filled(1, 4, ACCENT),
            FadeOut(out_tok, target_position=dram.row(1).cells[3]),
            run_time=0.7,
        )
        self.play(FadeOut(q10), FadeOut(aq7), FadeOut(aq1), FadeOut(att), run_time=0.3)
        self._dump(VGroup(q10, aq7, aq1, att, out_tok))
        no_alloc = _foot("# filled  3 → 4   ·   the reserved slot is used   ·   no allocation", color=GOOD)
        self.play(FadeIn(no_alloc), run_time=0.3)
        self.wait(0.3)
        self.next_slide()

        # ==================================================================
        # Beat 11 — ③ allocate
        # ==================================================================
        heading = self._retitle(heading, 'Fig. 6  ·  ③  last block full → allocate physical 3 → "brought"', FadeOut(no_alloc))
        self._dump(no_alloc)

        full_ring = SurroundingRectangle(logical.row(1).cells, buff=0.06, color=WARN, stroke_width=2.5, corner_radius=0.05)
        full_cap = caption("full", color=WARN).next_to(full_ring, RIGHT, buff=0.10)
        self.play(FadeIn(full_ring), FadeIn(full_cap), run_time=0.35)
        self.play(logical.row(2).animate.set_opacity(1.0), FadeOut(full_ring), FadeOut(full_cap), run_time=0.4)
        self._dump(VGroup(full_ring, full_cap))

        # ask the pool: flash free blocks, pick 3
        free_rings = VGroup(*[SurroundingRectangle(dram.row(i).cells, buff=0.05, color=GOOD, stroke_width=1.5, corner_radius=0.05)
                              for i in (0, 2, 3, 4, 5, 6)])
        self.play(FadeIn(free_rings), run_time=0.3)
        self.play(FadeOut(VGroup(*[free_rings[i] for i in (0, 1, 3, 4, 5)])), run_time=0.3)
        self.play(table.set_mapping(2, 3, 1, ACCENT), run_time=0.35)
        arr_a2 = _map_arrow(table, 2, dram.row(3), ACCENT)
        self.play(shoot(arr_a2), FadeOut(free_rings[2]), run_time=0.45)
        self._dump(free_rings)
        self.play(logical.row(2).fill(0, "brought", ACCENT), dram.row(3).fill(0, "brought", ACCENT), run_time=0.6)

        alloc_note = _foot("the only moment memory is allocated: previous blocks completely full → exactly one more block", color=FG)
        self.play(FadeIn(alloc_note), run_time=0.3)
        self.wait(0.3)
        self.next_slide()

        # ==================================================================
        # Beat 12 — one more step; this is Fig 5
        # ==================================================================
        heading = self._retitle(heading, 'One more decode → "forth"  ·  you have seen this picture', FadeOut(alloc_note))
        self._dump(alloc_note)
        self.play(
            logical.row(2).fill(1, "forth", ACCENT),
            dram.row(3).fill(1, "forth", ACCENT),
            table.set_filled(2, 2, ACCENT),
            run_time=0.6,
        )
        fig5_rings = VGroup(*[SurroundingRectangle(dram.row(i).cells, buff=0.06, color=Q_COLOR, stroke_width=2.5, corner_radius=0.05)
                              for i in (1, 3, 7)])
        tags = VGroup(
            caption("Block 1", color=Q_COLOR).next_to(fig5_rings[0], RIGHT, buff=0.12),
            caption("Block 2", color=Q_COLOR).next_to(fig5_rings[1], RIGHT, buff=0.12),
            caption("Block 0", color=Q_COLOR).next_to(fig5_rings[2], RIGHT, buff=0.12),
        )
        self.play(LaggedStart(*[FadeIn(r) for r in fig5_rings], lag_ratio=0.2), FadeIn(tags), run_time=0.7)
        loop_note = VGroup(
            small("physical 1, 3, 7  =  the scattered blocks the kernel read in Fig. 5", color=FG),
            caption("the algorithm reads what the manager leaves behind — two halves of one design"),
        ).arrange(DOWN, buff=0.08)
        loop_note.to_edge(DOWN, buff=0.28)
        self.play(FadeIn(loop_note), run_time=0.4)
        self.wait(0.3)
        self.next_slide()

        # ==================================================================
        # Beat 13 — waste < 1 block; Fig 2 payoff
        # ==================================================================
        manager_visible = VGroup(logical, table, dram, arr_a0, arr_a1, arr_a2)
        self.play(FadeOut(fig5_rings), FadeOut(tags), FadeOut(loop_note), FadeOut(manager_visible), run_time=0.45)
        self._dump(VGroup(fig5_rings, tags, loop_note))
        heading = self._retitle(heading, "All waste for a request: less than one block")

        # S4 strip vs vLLM strip
        def _strip(label_str, used, tail_parts, y):
            lab = small(label_str, color=FG)
            used_cells = VGroup(*[_slab_cell(w, ACCENT, w=0.68, h=0.42) for w in used])
            used_cells.arrange(RIGHT, buff=0.04)
            parts = [used_cells]
            for txt, color, w, op in tail_parts:
                parts.append(_slab_cell(txt, color, w=w, h=0.42, fs=13, txt_color=FG))
                parts[-1].box.set_fill(opacity=op)
            strip = VGroup(*parts).arrange(RIGHT, buff=0.04)
            g = VGroup(lab, strip).arrange(DOWN, buff=0.12, aligned_edge=LEFT)
            _fit(g, 12.6)
            g.move_to(UP * y)
            g.set_x(-6.5 + g.width / 2)
            return g

        s4_strip = _strip("existing systems (S4)", FOUR_SCORE,
                          [("2038 slots · internal fragmentation", BAD, 4.2, 1.0),
                           ("external", BAD, 1.1, 0.5)], 2.3)
        vl_strip = _strip("vLLM", FOUR_SCORE, [("2 reserved", WARN, 1.3, 0.55)], 1.25)
        self.play(FadeIn(s4_strip), run_time=0.5)
        self.play(FadeIn(vl_strip), run_time=0.5)
        bound = small("filled left → right, a new block only when all previous are full  ⇒  waste < 1 block per request", color=GOOD)
        _fit(bound, 12.8)
        bound.next_to(vl_strip, DOWN, buff=0.22).set_x(0)
        self.play(FadeIn(bound), run_time=0.4)
        self.wait(0.2)

        # Fig 2 callback: 4 bars, vLLM grows from '?'
        bar_h, bar_w = 1.85, 0.95
        b_max = _stack_bar("Orca\n(Max)", [(0.204, GOOD, 1, "20.4"), (0.133, WARN, 1, "13.3"), (0.573, BAD, 1, "57.3"), (0.089, BAD, 0.5, "")], bar_h, bar_w)
        b_pow = _stack_bar("Orca\n(Pow2)", [(0.268, GOOD, 1, "26.8"), (0.179, WARN, 1, "17.9"), (0.136, BAD, 1, "13.6"), (0.416, BAD, 0.5, "41.6")], bar_h, bar_w)
        b_ora = _stack_bar("Orca\n(Oracle)", [(0.382, GOOD, 1, "38.2"), (0.252, WARN, 1, "25.2"), (0.366, BAD, 0.5, "36.6")], bar_h, bar_w)
        b_q = _stack_bar("vLLM", [(1.0, MUTED, 0.35, "")], bar_h, bar_w)
        qmark = text("?", font_size=TITLE_SIZE, color=FG).move_to(b_q.outline.get_center())
        b_v = _stack_bar("vLLM", [(0.963, GOOD, 1, "96.3"), (0.037, MUTED, 0.6, "")], bar_h, bar_w)
        bars = VGroup(b_max, b_pow, b_ora, b_q).arrange(RIGHT, buff=0.42, aligned_edge=DOWN)
        bars.move_to(DOWN * 2.05 + LEFT * 2.6)
        b_v.move_to(b_q, aligned_edge=DOWN)
        qmark.move_to(b_q.outline.get_center())
        fig2_lab = caption("Fig. 2  ·  share of KV memory holding real token state (%)")
        fig2_lab.next_to(bars, UP, buff=0.18)
        self.play(FadeIn(bars), FadeIn(qmark), FadeIn(fig2_lab), run_time=0.6)

        stat = text("96.3%", font_size=54, color=GOOD, weight="BOLD")
        stat_cap = VGroup(
            small("of KV memory is real token state", color=FG),
            small("existing systems: 20.4 – 38.2%", color=MUTED),
            caption("same model, same GPU — only the memory manager changed"),
        ).arrange(DOWN, buff=0.08)
        stat_g = VGroup(stat, stat_cap).arrange(DOWN, buff=0.14)
        stat_g.move_to(DOWN * 2.0 + RIGHT * 3.4)
        self.play(
            FadeOut(qmark),
            LaggedStart(*[GrowFromEdge(s, DOWN) for s in b_v.segs], lag_ratio=0.1),
            FadeIn(b_v.seg_labels),
            run_time=0.8,
        )
        self.play(FadeIn(stat_g), run_time=0.5)
        self.wait(0.3)
        self.next_slide()

        # ==================================================================
        # Beat 14 — Fig 7: request B
        # ==================================================================
        self.play(FadeOut(s4_strip), FadeOut(vl_strip), FadeOut(bound), FadeOut(bars), FadeOut(b_v), FadeOut(fig2_lab), FadeOut(stat_g), run_time=0.4)
        self._dump(VGroup(s4_strip, vl_strip, bound, bars, b_v, fig2_lab, stat_g, qmark, b_q))
        heading = self._retitle(heading, "Fig. 7  ·  a second request draws from the same pool")
        self.play(FadeIn(manager_visible), run_time=0.5)
        # slide the A trio back to its Fig 7 position to make room for B
        self.play(manager_visible.animate.shift(LEFT * trio_dx), run_time=0.5)

        self.play(FadeIn(logical_b), FadeIn(table_b), run_time=0.45)
        b_prompt = small('request B:  "it was the best of times"', color=ACCENT2)
        b_prompt.to_edge(DOWN, buff=0.28)
        self.play(FadeIn(b_prompt), run_time=0.3)

        self.play(logical_b.row(0).fill_many(BW0, ACCENT2), run_time=0.7)
        self.play(table_b.set_mapping(0, 5, 4, ACCENT2), run_time=0.3)
        arr_b0 = arrow(table_b.frame.get_left() + UP * (table_b.rows[0].get_center()[1] - table_b.frame.get_center()[1]),
                       dram.row(5).cells.get_right(), color=ACCENT2, buff=0.06, stroke_width=1.8)
        self.play(shoot(arr_b0), dram.row(5).fill_many(BW0, ACCENT2), run_time=0.7)

        self.play(logical_b.row(1).fill_many(BW1, ACCENT2), logical_b.row(1).reserve(2), logical_b.row(1).reserve(3), run_time=0.6)
        self.play(table_b.set_mapping(1, 2, 2, ACCENT2), run_time=0.3)
        arr_b1 = arrow(table_b.frame.get_left() + UP * (table_b.rows[1].get_center()[1] - table_b.frame.get_center()[1]),
                       dram.row(2).cells.get_right(), color=ACCENT2, buff=0.06, stroke_width=1.8)
        self.play(shoot(arr_b1), dram.row(2).fill_many(BW1, ACCENT2), dram.row(2).reserve(2), dram.row(2).reserve(3), run_time=0.7)

        interleave = _foot("A at 7, 1, 3  ·  B at 5, 2  ·  free: 0, 4, 6   —   every block is the same size, so any free block fits any request", color=FG)
        self.play(FadeOut(b_prompt), FadeIn(interleave), run_time=0.35)
        self._dump(b_prompt)
        self.wait(0.3)
        self.next_slide()

        # ==================================================================
        # Beat 15 — A finishes
        # ==================================================================
        heading = self._retitle(heading, "Request A finishes  ·  its blocks return to the pool", FadeOut(interleave))
        self._dump(interleave)

        done = text("done ✓", font_size=30, color=GOOD, weight="BOLD")
        done.move_to(logical.body.get_center())
        self.play(FadeIn(done, scale=1.3), run_time=0.3)
        self.play(
            dram.row(7).clear_all(4), dram.row(1).clear_all(4), dram.row(3).clear_all(2),
            logical.row(0).clear_all(4), logical.row(1).clear_all(4), logical.row(2).clear_all(2),
            FadeOut(arr_a0), FadeOut(arr_a1), FadeOut(arr_a2),
            table.clear_all(),
            run_time=0.9,
        )
        self.play(logical.animate.set_opacity(0.25), table.animate.set_opacity(0.25), run_time=0.4)
        free_rings2 = VGroup(*[SurroundingRectangle(dram.row(i).cells, buff=0.05, color=GOOD, stroke_width=1.5, corner_radius=0.05)
                               for i in (0, 1, 3, 4, 6, 7)])
        self.play(LaggedStart(*[FadeIn(r) for r in free_rings2], lag_ratio=0.08), run_time=0.6)
        free_note = _foot("6 free blocks, at 6 unrelated addresses  ·  all usable by the next request  ·  no contiguous hole, no compaction", color=GOOD)
        self.play(FadeIn(free_note), run_time=0.3)
        self.wait(0.3)
        self.next_slide()

        # ==================================================================
        # Beat 16 — one engine iteration (§4.3, global procedure)
        # ==================================================================
        self.play(FadeOut(*self.mobjects), run_time=0.4)
        heading = self._retitle(None, "Every decode iteration, vLLM does this")

        def _step_box(num, head_str, sub_str, color):
            box = RoundedRectangle(width=3.0, height=1.55, corner_radius=0.12, fill_color=BLOCK_FILL, fill_opacity=1.0,
                                   stroke_color=color, stroke_width=2.2)
            n = text(num, font_size=14, color=color)
            h = small(head_str, color=FG)
            _fit(h, 2.7)
            s = caption(sub_str)
            _fit(s, 2.7)
            inner = VGroup(n, h, s).arrange(DOWN, buff=0.10).move_to(box.get_center())
            return VGroup(box, inner)

        steps = VGroup(
            _step_box("①  scheduler", "pick the sequences", "which requests run this step", FG),
            _step_box("②  KV manager", "allocate blocks", "for newly needed logical blocks", ACCENT),
            _step_box("③  concatenate", "one flat sequence", "prefill: whole prompt · decode: 1 token", FG),
            _step_box("④  model", "forward + PagedAttention", "read via block tables, write new K, V", Q_COLOR),
        ).arrange(RIGHT, buff=0.32)
        _fit(steps, 13.2)
        steps.move_to(UP * 1.35)
        step_arrows = VGroup(*[arrow(steps[i].get_right(), steps[i + 1].get_left(), color=MUTED, buff=0.05) for i in range(3)])
        self.play(LaggedStart(*[FadeIn(s) for s in steps], lag_ratio=0.2), run_time=0.9)
        self.play(*[shoot(a) for a in step_arrows], run_time=0.5)

        # concrete batch row
        seq = VGroup()
        for w, c in (("brought", ACCENT), ("times", ACCENT2), ("You", V_COLOR), ("only", V_COLOR), ("live", V_COLOR), ("once", V_COLOR)):
            seq.add(_slab_cell(w, c, w=1.4, h=0.5, fs=17))
        seq.arrange(RIGHT, buff=0.08).move_to(DOWN * 0.9)
        seq_labs = VGroup(
            caption("A · decode", color=ACCENT).next_to(seq[0], DOWN, buff=0.12),
            caption("B · decode", color=ACCENT2).next_to(seq[1], DOWN, buff=0.12),
            caption("C · prefill (whole prompt)", color=V_COLOR).next_to(VGroup(*seq[2:]), DOWN, buff=0.12),
        )
        seq_head = caption("the batch this iteration, as one input sequence").next_to(seq, UP, buff=0.14)
        self.play(FadeIn(seq_head), LaggedStart(*[FadeIn(s) for s in seq], lag_ratio=0.08), FadeIn(seq_labs), run_time=0.8)

        one_pass = VGroup(
            small("one forward pass serves every request in the batch", color=FG),
            small("each request's attention reads its own blocks through its own block table", color=MUTED),
            caption("Act I's batching payoff — with a batch that now fits in memory"),
        ).arrange(DOWN, buff=0.10)
        one_pass.to_edge(DOWN, buff=0.30)
        self.play(FadeIn(one_pass), run_time=0.45)
        self.wait(0.3)
        self.next_slide()

        # ==================================================================
        # Beat 17 — landing
        # ==================================================================
        self.play(FadeOut(*self.mobjects), run_time=0.4)
        heading = self._retitle(None, "PagedAttention  +  KV cache manager")

        def _half(head_str, color, lines):
            box = RoundedRectangle(width=6.1, height=2.6, corner_radius=0.14, fill_color=BLOCK_FILL, fill_opacity=1.0,
                                   stroke_color=color, stroke_width=2.5)
            h = body(head_str, font_size=26, color=color)
            # one font size for both boxes: lines are written to fit at 20 px
            ls = VGroup(*[text(l, font_size=20, color=FG) for l in lines]).arrange(DOWN, buff=0.16, aligned_edge=LEFT)
            inner = VGroup(h, ls).arrange(DOWN, buff=0.28).move_to(box.get_center())
            return VGroup(box, inner)

        left = _half("the kernel  (§4.1)", Q_COLOR, [
            "attention computed one block at a time",
            "Eq. 4: the same sums, grouped by block",
            "⇒ blocks need not be contiguous",
        ])
        right = _half("the manager  (§4.2–4.3)", ACCENT, [
            "logical → physical, one table per request",
            "allocate a block only when the last is full",
            "⇒ waste < 1 block · any free block fits",
        ])
        halves = VGroup(left, right).arrange(RIGHT, buff=0.5)
        halves.move_to(UP * 0.55)
        plus = text("+", font_size=40, color=MUTED).move_to(halves.get_center())
        self.play(FadeIn(left), run_time=0.45)
        self.play(FadeIn(plus), FadeIn(right), run_time=0.45)

        land = VGroup(
            text("96.3%", font_size=44, color=GOOD, weight="BOLD"),
            small("of KV memory holding real token state  ·  from roughly 20–40%  ·  same model, same GPU", color=FG),
            caption("next: look at this picture once more — it should look familiar"),
        ).arrange(DOWN, buff=0.12)
        _fit(land, 13.0)
        land.to_edge(DOWN, buff=0.35)
        self.play(FadeIn(land), run_time=0.5)
        self.wait(0.3)
        self.next_slide()

    # ------------------------------------------------------------------
    def _dump(self, mob):
        if mob is None:
            return
        family = list(mob.get_family())
        self.remove(mob, *family)
        mob.set_opacity(0)
        mob.shift(LEFT * 40)

    def _retitle(self, old, new_str, *anims, run_time=0.5):
        new = title(new_str)
        if new.width > 13.4:
            new.scale_to_fit_width(13.4)
            new.to_edge(UP)
        extra = list(anims)
        if old is not None:
            extra = [FadeOut(old, shift=UP * 0.18)] + extra
            self.play(FadeIn(new, shift=DOWN * 0.08), *extra, run_time=run_time)
            self._dump(old)
        else:
            self.play(FadeIn(new), *extra, run_time=run_time)
        return new
