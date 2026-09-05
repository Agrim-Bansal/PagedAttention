"""S4 — The problem (Act I). Paper §3 / §3.1: contiguous pre-allocation.

Unhurried: one paper claim per beat. No minute cap.

NARRATION
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
"""

from manim import (
    DOWN,
    LEFT,
    ORIGIN,
    PI,
    RIGHT,
    UP,
    Circumscribe,
    Create,
    DashedVMobject,
    FadeIn,
    FadeOut,
    GrowFromCenter,
    GrowFromEdge,
    Indicate,
    LaggedStart,
    Line,
    Rectangle,
    RoundedRectangle,
    Square,
    SurroundingRectangle,
    Transform,
    TransformFromCopy,
    VGroup,
    Write,
    there_and_back,
)
from manim_slides import Slide

from talk.theme import *
from talk.components import *


CELL_W = 0.92
CELL_H = 0.50
CELL_FS = 13
CELL_BUFF = 0.06


def _legend_swatch(label, color, opacity=1.0):
    sq = Square(side_length=0.22, fill_color=color, fill_opacity=opacity, stroke_width=0)
    txt = caption(label)
    txt.next_to(sq, RIGHT, buff=0.10)
    return VGroup(sq, txt)


def _frag_legend():
    entries = [
        _legend_swatch("token states", ACCENT),
        _legend_swatch("current iteration", ACCENT2),
        _legend_swatch("reserved", WARN),
        _legend_swatch("internal frag.", BAD),
        _legend_swatch("external frag.", BAD, opacity=0.5),
    ]
    row = VGroup(*entries)
    row.arrange(RIGHT, buff=0.38)
    return row


def _cell(word, fill, txt_color=BG, w=CELL_W, h=CELL_H, fs=CELL_FS):
    box = RoundedRectangle(
        width=w,
        height=h,
        corner_radius=0.08,
        fill_color=fill,
        fill_opacity=1.0,
        stroke_color=BLOCK_STROKE,
        stroke_width=1.5,
    )
    lbl = text(str(word), font_size=fs, color=txt_color)
    if lbl.width > w * 0.86:
        lbl.scale_to_fit_width(w * 0.86)
    if lbl.height > h * 0.78:
        lbl.scale_to_fit_height(h * 0.78)
    lbl.move_to(box.get_center())
    return VGroup(box, lbl)


def _cells(words, fill, txt_color=BG):
    row = VGroup(*[_cell(w, fill, txt_color) for w in words])
    row.arrange(RIGHT, buff=CELL_BUFF)
    return row


def _alloc_bar(width, fill=WARN, opacity=0.92):
    return RoundedRectangle(
        width=width,
        height=CELL_H,
        corner_radius=0.08,
        fill_color=fill,
        fill_opacity=opacity,
        stroke_color=BLOCK_STROKE,
        stroke_width=1.5,
    )


def _ghost_slots(width, fill, n=8, opacity=0.7):
    buff = 0.045
    w = (width - (n - 1) * buff) / n
    w = max(0.14, min(w, 0.40))
    slots = VGroup(*[
        RoundedRectangle(
            width=w,
            height=CELL_H * 0.90,
            corner_radius=0.05,
            fill_color=fill,
            fill_opacity=opacity,
            stroke_color=BLOCK_STROKE,
            stroke_width=1.0,
        )
        for _ in range(n)
    ])
    slots.arrange(RIGHT, buff=buff)
    if slots.width > width:
        slots.scale_to_fit_width(width)
    return slots


def _count_block(label, width, fill, opacity=1.0, height=CELL_H, txt_color=FG):
    rect = RoundedRectangle(
        width=width,
        height=height,
        corner_radius=0.08,
        fill_color=fill,
        fill_opacity=opacity,
        stroke_color=BLOCK_STROKE,
        stroke_width=1.5,
    )
    txt = text(label, font_size=TINY_SIZE, color=txt_color)
    if txt.width > width * 0.9:
        txt.scale_to_fit_width(width * 0.9)
    txt.move_to(rect.get_center())
    block = VGroup(rect, txt)
    block.rect = rect
    return block


def _under(mob, lines, color=MUTED, buff=0.18):
    cap = text(lines, font_size=14, color=color, line_spacing=1.1)
    if cap.width > max(mob.width, 1.2):
        cap.scale_to_fit_width(max(mob.width, 1.2))
    cap.next_to(mob, DOWN, buff=buff)
    return cap


def _over(mob, lines, color=MUTED, buff=0.16):
    cap = text(lines, font_size=14, color=color, line_spacing=1.1)
    if cap.width > max(mob.width, 1.2):
        cap.scale_to_fit_width(max(mob.width, 1.2))
    cap.next_to(mob, UP, buff=buff)
    return cap


def _stack_bar(name, parts, h=3.15, w=1.35):
    """Vertical stacked bar. parts = [(frac, color, opacity, label), ...] bottom to top."""
    outline = Rectangle(width=w, height=h, stroke_color=BLOCK_STROKE, stroke_width=2)
    segs = VGroup()
    labels = VGroup()
    y = outline.get_bottom()[1]
    cx = outline.get_center()[0]
    for frac, color, opacity, lab in parts:
        sh = h * frac
        rect = Rectangle(
            width=w,
            height=max(sh, 1e-4),
            fill_color=color,
            fill_opacity=opacity,
            stroke_width=0,
        )
        rect.move_to([cx, y + sh / 2, 0])
        segs.add(rect)
        if lab and sh >= 0.20:
            t = text(lab, font_size=13 if sh >= 0.32 else 11, color=FG)
            if t.width > w * 0.88:
                t.scale_to_fit_width(w * 0.88)
            t.move_to(rect.get_center())
            labels.add(t)
        y += sh
    name_t = text(name, font_size=16, color=FG, line_spacing=1.05)
    if name_t.width > w + 0.6:
        name_t.scale_to_fit_width(w + 0.6)
    name_t.next_to(outline, DOWN, buff=0.20)
    g = VGroup(outline, segs, labels, name_t)
    g.outline = outline
    g.segs = segs
    g.seg_labels = labels
    g.name_t = name_t
    return g


class S4Problem(Slide):
    def construct(self):
        apply_theme(self)

        # -------------------------------------------------------------
        # Beat 1: the question
        # -------------------------------------------------------------
        heading = title("How do you allocate memory for\nsomething whose final size is unknown?")
        if heading.width > 13.2:
            heading.scale_to_fit_width(13.2)
            heading.to_edge(UP)
        sub = caption("a request's output length is unknown until it emits <eos>")
        sub.next_to(heading, DOWN, buff=0.45)

        leftover = RoundedRectangle(
            width=12.2,
            height=1.15,
            corner_radius=0.12,
            stroke_color=BLOCK_STROKE,
            stroke_width=2,
            fill_opacity=0.0,
        )
        leftover.move_to(DOWN * 0.15)
        leftover_lab = caption("leftover VRAM  ·  KV cache region")
        leftover_lab.next_to(leftover, DOWN, buff=0.28)

        self.play(FadeIn(heading), run_time=0.6)
        self.play(FadeIn(sub), FadeIn(leftover), FadeIn(leftover_lab), run_time=0.6)
        self._hold()

        # -------------------------------------------------------------
        # Beat 2: one contiguous tensor
        # -------------------------------------------------------------
        heading = self._retitle(heading, "Existing systems: one contiguous tensor")
        self.play(FadeOut(sub), run_time=0.35)
        self._dump(sub)

        claim = RoundedRectangle(
            width=12.2,
            height=1.15,
            corner_radius=0.12,
            fill_color=WARN,
            fill_opacity=0.95,
            stroke_width=0,
        )
        claim.move_to(leftover.get_center())
        claim_lab = text("one contiguous tensor per request", font_size=SMALL_SIZE, color=BG)
        claim_lab.move_to(claim.get_center())
        why = caption("deep learning frameworks require tensors in contiguous memory")
        why.next_to(leftover_lab, DOWN, buff=0.28)

        self.play(GrowFromEdge(claim, LEFT), run_time=1.1)
        self.play(FadeIn(claim_lab), FadeIn(why), run_time=0.5)
        self._hold()

        # -------------------------------------------------------------
        # Beat 3: unlike a traditional tensor
        # -------------------------------------------------------------
        heading = self._retitle(heading, "Unlike a traditional deep learning tensor")
        self.play(
            FadeOut(claim), FadeOut(claim_lab), FadeOut(leftover),
            FadeOut(leftover_lab), FadeOut(why),
            run_time=0.45,
        )
        self._dump(VGroup(claim, claim_lab, leftover, leftover_lab, why))

        left_title = small("Traditional DL tensor", color=MUTED)
        left_grid = VGroup(*[
            Square(side_length=0.38, fill_color=GOOD, fill_opacity=0.9, stroke_width=0.6, stroke_color=BG)
            for _ in range(16)
        ])
        left_grid.arrange_in_grid(rows=4, cols=4, buff=0.07)
        left_cap = caption("shape known before the run")
        left_note = caption("contiguous allocation is perfect")
        left_panel = VGroup(left_title, left_grid, left_cap, left_note)
        left_panel.arrange(DOWN, buff=0.28)
        left_panel.move_to(LEFT * 3.5 + DOWN * 0.15)

        right_title = small("KV cache", color=MUTED)
        right_outline = RoundedRectangle(
            width=2.6, height=1.8, corner_radius=0.1,
            stroke_color=BLOCK_STROKE, stroke_width=2,
        )
        right_fill = RoundedRectangle(
            width=2.6, height=0.40, corner_radius=0.08,
            fill_color=ACCENT, fill_opacity=0.95, stroke_width=0,
        )
        right_fill.move_to(right_outline.get_bottom(), aligned_edge=DOWN)
        right_q = text("?", font_size=36, color=BAD)
        right_q.next_to(right_outline, UP, buff=0.12)
        right_cap = caption("grows and shrinks over time")
        right_note = caption("lifetime and length not known a priori")
        right_col = VGroup(right_title, right_outline, right_cap, right_note)
        right_col.arrange(DOWN, buff=0.28)
        right_col.move_to(RIGHT * 3.5 + DOWN * 0.15)
        right_fill.move_to(right_outline.get_bottom(), aligned_edge=DOWN)
        right_q.move_to(right_outline.get_center() + UP * 0.28)

        divider = Rectangle(width=0.02, height=4.4, fill_color=MUTED, fill_opacity=0.45, stroke_width=0)
        divider.move_to(DOWN * 0.2)

        self.play(FadeIn(left_panel), FadeIn(divider), run_time=0.7)
        self.play(
            FadeIn(right_title), FadeIn(right_outline),
            FadeIn(right_cap), FadeIn(right_note),
            run_time=0.5,
        )
        self.play(GrowFromEdge(right_fill, DOWN), run_time=0.7)
        fill_tall = RoundedRectangle(
            width=2.6, height=1.35, corner_radius=0.08,
            fill_color=ACCENT, fill_opacity=0.95, stroke_width=0,
        )
        fill_tall.move_to(right_outline.get_bottom(), aligned_edge=DOWN)
        fill_short = RoundedRectangle(
            width=2.6, height=0.45, corner_radius=0.08,
            fill_color=ACCENT, fill_opacity=0.95, stroke_width=0,
        )
        fill_short.move_to(right_outline.get_bottom(), aligned_edge=DOWN)
        self.play(Transform(right_fill, fill_tall), run_time=0.55)
        self.play(Transform(right_fill, fill_short), run_time=0.45)
        self.play(FadeIn(right_q), run_time=0.4)
        self._hold()

        # -------------------------------------------------------------
        # Beat 4: pre-allocate A to 2048
        # -------------------------------------------------------------
        heading = self._retitle(heading, "Statically allocate to the maximum length")
        self.play(
            FadeOut(left_panel), FadeOut(divider),
            FadeOut(right_title), FadeOut(right_outline), FadeOut(right_fill),
            FadeOut(right_q), FadeOut(right_cap), FadeOut(right_note),
            run_time=0.45,
        )
        self._dump(VGroup(
            left_panel, divider, right_title, right_outline, right_fill,
            right_q, right_cap, right_note,
        ))

        sub = caption("irrespective of the actual input or eventual output length")
        sub.next_to(heading, DOWN, buff=0.28)

        a_alloc = _alloc_bar(12.0)
        a_alloc.next_to(sub, DOWN, buff=0.95)
        tag_a = caption("Request A  ·  max 2048")
        tag_a.next_to(a_alloc, UP, buff=0.16)
        tag_a.align_to(a_alloc, LEFT)
        systems = caption("FasterTransformer, Orca")
        systems.next_to(a_alloc, DOWN, buff=0.45)

        self.play(FadeIn(sub), FadeIn(tag_a), run_time=0.4)
        self.play(GrowFromEdge(a_alloc, LEFT), run_time=1.35)
        self.play(FadeIn(systems), run_time=0.4)
        self._hold()

        # -------------------------------------------------------------
        # Beat 5: Request B, max 512
        # -------------------------------------------------------------
        heading = self._retitle(heading, "Same rule, a smaller maximum")
        dy = 1.25 - a_alloc.get_y()
        self.play(
            FadeOut(sub), FadeOut(systems),
            a_alloc.animate.shift(UP * dy),
            tag_a.animate.shift(UP * dy),
            run_time=0.7,
        )
        self._dump(VGroup(sub, systems))

        b_alloc = _alloc_bar(7.0)
        b_alloc.align_to(a_alloc, LEFT)
        b_alloc.set_y(-0.55)
        tag_b = caption("Request B  ·  max 512")
        tag_b.next_to(b_alloc, RIGHT, buff=0.20)
        self.play(GrowFromEdge(b_alloc, LEFT), FadeIn(tag_b), run_time=1.1)
        self._hold()

        # -------------------------------------------------------------
        # Beat 6: Fig 3 prompt tokens paint into A's slab
        # -------------------------------------------------------------
        heading = self._retitle(heading, "Fig. 3  —  Request A, max 2048")
        self.play(
            b_alloc.animate.set_fill(WARN, 0.28).set_stroke(BLOCK_STROKE, 1.5, 0.35),
            tag_b.animate.set_opacity(0.35),
            run_time=0.4,
        )
        legend = _frag_legend()
        legend.scale(0.86)
        legend.next_to(heading, DOWN, buff=0.20)
        if legend.width > 13.2:
            legend.scale_to_fit_width(13.2)
            legend.next_to(heading, DOWN, buff=0.20)

        filled = _cells(FOUR_SCORE[:7], ACCENT)
        filled.align_to(a_alloc, LEFT)
        filled.shift(RIGHT * 0.06)
        filled.set_y(a_alloc.get_y())

        self.play(FadeIn(legend), run_time=0.4)
        self.play(
            LaggedStart(
                *[FadeIn(c, shift=RIGHT * 0.18) for c in filled],
                lag_ratio=0.14,
            ),
            run_time=1.6,
        )
        self._hold()

        # -------------------------------------------------------------
        # Beat 7: current iteration
        # -------------------------------------------------------------
        current = _cell("brought", ACCENT2)
        current.next_to(filled, RIGHT, buff=0.10)
        current.set_y(a_alloc.get_y())
        self.play(GrowFromCenter(current), run_time=0.7)
        self.play(Indicate(current, color=ACCENT2, scale_factor=1.08), run_time=0.55)
        self._hold()

        # -------------------------------------------------------------
        # Beat 8: reserved
        # -------------------------------------------------------------
        reserved = _cells(["forth", "<eos>"], WARN)
        reserved.next_to(current, RIGHT, buff=0.10)
        reserved.set_y(a_alloc.get_y())
        cap_rsv = caption("eventually used  ·  occupied for the whole lifetime")
        cap_rsv.to_edge(DOWN, buff=0.28)
        self.play(
            LaggedStart(*[GrowFromCenter(c) for c in reserved], lag_ratio=0.18),
            run_time=0.8,
        )
        self.play(Indicate(reserved, color=FG, scale_factor=1.06), run_time=0.6)
        self.play(FadeIn(cap_rsv), run_time=0.4)
        self._hold()

        # -------------------------------------------------------------
        # Beat 9: internal fragmentation — empty slots cascade
        # -------------------------------------------------------------
        remain_left = reserved.get_right()[0] + 0.08
        remain_right = a_alloc.get_right()[0] - 0.05
        remain_w = max(remain_right - remain_left, 1.6)
        ghosts = _ghost_slots(remain_w, BAD, n=10, opacity=0.55)
        ghosts.move_to([(remain_left + remain_right) / 2, a_alloc.get_y(), 0])
        if ghosts.width > remain_w:
            ghosts.scale_to_fit_width(remain_w)
            ghosts.move_to([(remain_left + remain_right) / 2, a_alloc.get_y(), 0])

        self.play(FadeOut(cap_rsv), run_time=0.3)
        self._dump(cap_rsv)
        self.play(
            LaggedStart(*[FadeIn(g, scale=0.45) for g in ghosts], lag_ratio=0.07),
            run_time=1.7,
        )
        internal = _count_block("2038 slots never used", remain_w, BAD)
        internal.move_to(ghosts.get_center())
        if internal.width > remain_w:
            internal.scale_to_fit_width(remain_w)
            internal.move_to(ghosts.get_center())
        cap_internal = _over(internal, "internal fragmentation", color=BAD)
        timing_int = caption("pure waste  ·  known only after sampling finishes")
        timing_int.to_edge(DOWN, buff=0.26)
        self.play(
            FadeOut(ghosts, lag_ratio=0.04),
            GrowFromEdge(internal, LEFT),
            run_time=0.95,
        )
        self._dump(ghosts)
        self.play(FadeIn(cap_internal), FadeIn(timing_int), run_time=0.45)
        self._hold()

        # -------------------------------------------------------------
        # Beat 10: Request B fills, then the same empty-slot cascade
        # -------------------------------------------------------------
        heading = self._retitle(heading, "Fig. 3  —  Request B, max 512")
        self.play(
            FadeOut(timing_int), FadeOut(cap_internal),
            b_alloc.animate.set_fill(WARN, 0.92).set_stroke(BLOCK_STROKE, 1.5, 1),
            tag_b.animate.set_opacity(1),
            run_time=0.5,
        )
        self._dump(VGroup(timing_int, cap_internal))

        b_filled = _cells(["You", "only", "live"], ACCENT)
        b_filled.align_to(b_alloc, LEFT)
        b_filled.shift(RIGHT * 0.06)
        b_filled.set_y(b_alloc.get_y())
        b_current = _cell("once", ACCENT2)
        b_current.next_to(b_filled, RIGHT, buff=0.10)
        b_current.set_y(b_alloc.get_y())
        b_reserved = _cell("<eos>", WARN)
        b_reserved.next_to(b_current, RIGHT, buff=0.10)
        b_reserved.set_y(b_alloc.get_y())

        self.play(
            LaggedStart(
                *[FadeIn(c, shift=RIGHT * 0.14) for c in b_filled],
                lag_ratio=0.14,
            ),
            run_time=1.0,
        )
        self.play(GrowFromCenter(b_current), GrowFromCenter(b_reserved), run_time=0.55)

        b_left = b_reserved.get_right()[0] + 0.08
        b_right = b_alloc.get_right()[0] - 0.05
        b_remain = max(b_right - b_left, 1.2)
        b_ghosts = _ghost_slots(b_remain, BAD, n=6, opacity=0.55)
        b_ghosts.move_to([(b_left + b_right) / 2, b_alloc.get_y(), 0])
        if b_ghosts.width > b_remain:
            b_ghosts.scale_to_fit_width(b_remain)
            b_ghosts.move_to([(b_left + b_right) / 2, b_alloc.get_y(), 0])
        b_internal = _count_block("507 slots never used", b_remain, BAD)
        b_internal.move_to(b_ghosts.get_center())
        if b_internal.width > b_remain:
            b_internal.scale_to_fit_width(b_remain)
            b_internal.move_to(b_ghosts.get_center())

        self.play(
            LaggedStart(*[FadeIn(g, scale=0.45) for g in b_ghosts], lag_ratio=0.08),
            run_time=1.15,
        )
        self.play(
            FadeOut(b_ghosts, lag_ratio=0.04),
            GrowFromEdge(b_internal, LEFT),
            run_time=0.8,
        )
        self._dump(b_ghosts)
        self._hold()

        # -------------------------------------------------------------
        # Beat 11: external fragmentation — a request that does not fit
        # -------------------------------------------------------------
        heading = self._retitle(heading, "Fig. 3  —  two requests, one address space")
        gap_w = 4.4
        gap_box = RoundedRectangle(
            width=gap_w, height=0.52, corner_radius=0.08,
            fill_opacity=0.0, stroke_color=MUTED, stroke_width=2,
        )
        mid_y = (a_alloc.get_bottom()[1] + b_alloc.get_top()[1]) / 2
        gap_box.move_to([a_alloc.get_left()[0] + gap_w / 2, mid_y, 0])
        gap_dash = DashedVMobject(gap_box.copy(), num_dashes=16)
        self.play(Create(gap_dash), run_time=0.65)

        try_g = _count_block("another request", gap_w + 1.4, ACCENT, txt_color=BG)
        try_g.move_to(gap_box.get_center() + UP * 1.65)
        self.play(FadeIn(try_g, shift=DOWN * 0.25), run_time=0.4)
        self.play(try_g.animate.move_to(gap_box.get_center()), run_time=0.55)
        self._nudge(try_g)

        gap_fill = _count_block("external fragmentation", gap_w, BAD, opacity=0.5)
        gap_fill.move_to(gap_box.get_center())
        known = caption("known before serving  ·  will never hold generated tokens")
        known.to_edge(DOWN, buff=0.42)
        fit_note = caption("three wastes prevent other requests from fitting into the memory")
        fit_note.to_edge(DOWN, buff=0.18)
        self.play(
            FadeOut(try_g, shift=UP * 0.8),
            FadeOut(gap_dash),
            FadeIn(gap_fill),
            run_time=0.75,
        )
        self._dump(try_g)
        self.play(FadeIn(known), FadeIn(fit_note), run_time=0.4)
        self._hold()

        # -------------------------------------------------------------
        # Beat 12: even if — unused pink still owned by A
        # -------------------------------------------------------------
        heading = self._retitle(heading, "Even if the actual length is known a priori")
        self.play(FadeOut(known), FadeOut(fit_note), run_time=0.3)
        self._dump(VGroup(known, fit_note))
        self.play(Indicate(internal, color=BAD, scale_factor=1.06), run_time=0.7)
        internal.rect.set_fill(BAD, 1.0)

        probe = _count_block("shorter request", 2.4, ACCENT2, txt_color=BG)
        probe.next_to(internal, UP, buff=0.85)
        self.play(FadeIn(probe, shift=DOWN * 0.2), run_time=0.4)
        self.play(probe.animate.move_to(internal.get_center()), run_time=0.5)
        self._nudge(probe)
        lock = caption("still reserved to A  ·  other requests cannot use it")
        lock.to_edge(DOWN, buff=0.26)
        self.play(
            FadeOut(probe, shift=UP * 0.45),
            FadeIn(lock),
            run_time=0.55,
        )
        self._dump(probe)
        self._hold()

        self.play(FadeOut(*self.mobjects), run_time=0.5)

        # -------------------------------------------------------------
        # Beat 13: Fig 2, Orca (Max)
        # -------------------------------------------------------------
        heading = title("Fig. 2  —  Orca (Max)")
        if heading.width > 13.2:
            heading.scale_to_fit_width(13.2)
            heading.to_edge(UP)

        legend2 = VGroup(
            _legend_swatch("token states", GOOD),
            _legend_swatch("reservation", WARN),
            _legend_swatch("internal frag.", BAD),
            _legend_swatch("external frag. & others", BAD, opacity=0.5),
        )
        legend2.arrange(RIGHT, buff=0.40)
        legend2.scale(0.86)
        legend2.next_to(heading, DOWN, buff=0.32)
        if legend2.width > 13.2:
            legend2.scale_to_fit_width(13.2)
            legend2.next_to(heading, DOWN, buff=0.32)

        bar_h, bar_w = 3.05, 1.32
        bar_max = _stack_bar(
            "Orca\n(Max)",
            [
                (0.204, GOOD, 1.0, "20.4"),
                (0.133, WARN, 1.0, "13.3"),
                (0.573, BAD, 1.0, "57.3"),
                (0.089, BAD, 0.5, "8.9"),
            ],
            h=bar_h, w=bar_w,
        )
        bar_pow2 = _stack_bar(
            "Orca\n(Pow2)",
            [
                (0.268, GOOD, 1.0, "26.8"),
                (0.179, WARN, 1.0, "17.9"),
                (0.136, BAD, 1.0, "13.6"),
                (0.416, BAD, 0.5, "41.6"),
            ],
            h=bar_h, w=bar_w,
        )
        bar_oracle = _stack_bar(
            "Orca\n(Oracle)",
            [
                (0.382, GOOD, 1.0, "38.2"),
                (0.252, WARN, 1.0, "25.2"),
                (0.366, BAD, 0.5, "36.6"),
            ],
            h=bar_h, w=bar_w,
        )
        bar_vllm = _stack_bar(
            "vLLM",
            [(1.0, MUTED, 0.55, "")],
            h=bar_h, w=bar_w,
        )
        vllm_q = text("?", font_size=TITLE_SIZE, color=FG)
        vllm_q.move_to(bar_vllm.outline.get_center())

        bars = VGroup(bar_max, bar_pow2, bar_oracle, bar_vllm)
        bars.arrange(RIGHT, buff=0.55, aligned_edge=DOWN)

        axis = Line(ORIGIN, UP * bar_h, color=MUTED, stroke_width=1.5)
        ticks = VGroup()
        for pct in (0, 20, 40, 60, 80, 100):
            tick = Line(LEFT * 0.07, RIGHT * 0.07, color=MUTED, stroke_width=1.2)
            lab = text(str(pct), font_size=12, color=MUTED)
            ticks.add(VGroup(tick, lab))
        y_lab = text("KV cache usage (%)", font_size=13, color=MUTED)
        y_lab.rotate(PI / 2)

        axis_group = VGroup(y_lab, axis, ticks)
        chart = VGroup(axis_group, bars)
        chart.arrange(RIGHT, buff=0.35, aligned_edge=DOWN)
        chart.next_to(legend2, DOWN, buff=0.28)

        axis.put_start_and_end_on(
            [axis.get_center()[0], bar_max.outline.get_bottom()[1], 0],
            [axis.get_center()[0], bar_max.outline.get_top()[1], 0],
        )
        for i, pct in enumerate((0, 20, 40, 60, 80, 100)):
            y = bar_max.outline.get_bottom()[1] + bar_h * (pct / 100.0)
            ticks[i][0].move_to([axis.get_center()[0], y, 0])
            ticks[i][1].next_to(ticks[i][0], LEFT, buff=0.10)
        y_lab.next_to(axis, LEFT, buff=0.58)

        if chart.get_bottom()[1] < -3.15:
            chart.scale(0.90)
            chart.next_to(legend2, DOWN, buff=0.18)
            axis.put_start_and_end_on(
                [axis.get_center()[0], bar_max.outline.get_bottom()[1], 0],
                [axis.get_center()[0], bar_max.outline.get_top()[1], 0],
            )
            for i, pct in enumerate((0, 20, 40, 60, 80, 100)):
                y = bar_max.outline.get_bottom()[1] + bar_h * (pct / 100.0)
                ticks[i][0].move_to([axis.get_center()[0], y, 0])
                ticks[i][1].next_to(ticks[i][0], LEFT, buff=0.10)
            y_lab.next_to(axis, LEFT, buff=0.55)
        if chart.get_left()[0] < -6.9:
            chart.shift(RIGHT * (-6.7 - chart.get_left()[0]))
        vllm_q.move_to(bar_vllm.outline.get_center())

        landing = caption("only 20.4% – 38.2% of KV memory stores actual token states")
        landing.to_edge(DOWN, buff=0.22)

        def _grow_bar(bar):
            return LaggedStart(
                *[GrowFromEdge(seg, DOWN) for seg in bar.segs],
                FadeIn(bar.seg_labels),
                FadeIn(bar.outline),
                FadeIn(bar.name_t),
                lag_ratio=0.08,
            )

        self.play(FadeIn(heading), FadeIn(legend2), FadeIn(axis), FadeIn(ticks), FadeIn(y_lab), run_time=0.55)
        self.play(_grow_bar(bar_max), run_time=1.0)
        self.play(Indicate(bar_max.segs[2], color=BAD, scale_factor=1.05), run_time=0.55)
        self._hold()

        heading = self._retitle(heading, "Fig. 2  —  Orca (Pow2)")
        self.play(_grow_bar(bar_pow2), run_time=1.0)
        self.play(Indicate(bar_pow2.segs[3], color=BAD, scale_factor=1.05), run_time=0.55)
        self._hold()

        heading = self._retitle(heading, "Fig. 2  —  Orca (Oracle)")
        self.play(_grow_bar(bar_oracle), run_time=1.0)
        oracle_note = caption("no internal fragmentation  ·  still 36.6% wasted")
        oracle_note.to_edge(DOWN, buff=0.22)
        self.play(FadeIn(oracle_note), Indicate(bar_oracle.segs[2], color=BAD, scale_factor=1.05), run_time=0.55)
        self._hold()

        heading = self._retitle(heading, "Fig. 2  —  vLLM is a question mark")
        self.play(FadeOut(oracle_note), _grow_bar(bar_vllm), FadeIn(vllm_q), run_time=0.8)
        self._dump(oracle_note)
        self.play(FadeIn(landing), run_time=0.5)
        self._hold()

        self.play(FadeOut(*self.mobjects), run_time=0.5)

        # -------------------------------------------------------------
        # Beat 17: two sequences — copy the prompt
        # -------------------------------------------------------------
        heading = title("Second: contiguous chunks cannot share")
        if heading.width > 13.2:
            heading.scale_to_fit_width(13.2)
            heading.to_edge(UP)
        share_sub = caption("each sequence's KV cache is stored in separate contiguous spaces")
        share_sub.next_to(heading, DOWN, buff=0.28)

        prompt_words = FOUR_SCORE[:7]
        copy1 = _cells(prompt_words, ACCENT)
        copy2 = _cells(prompt_words, ACCENT)
        suf1 = _cells(["…A"], ACCENT2, txt_color=BG)
        suf2 = _cells(["…B"], ACCENT2, txt_color=BG)
        row1 = VGroup(copy1, suf1).arrange(RIGHT, buff=0.10)
        row2 = VGroup(copy2, suf2).arrange(RIGHT, buff=0.10)
        lab1 = caption("sequence 1")
        lab2 = caption("sequence 2")
        lab1.next_to(row1, LEFT, buff=0.22)
        lab2.next_to(row2, LEFT, buff=0.22)
        seq1 = VGroup(lab1, row1)
        seq2 = VGroup(lab2, row2)
        seqs = VGroup(seq1, seq2)
        seqs.arrange(DOWN, buff=0.70)
        seqs.next_to(share_sub, DOWN, buff=0.70)
        if seqs.get_left()[0] < -6.8:
            seqs.shift(RIGHT * (-6.6 - seqs.get_left()[0]))
        if seqs.width > 13.0:
            seqs.scale_to_fit_width(13.0)
            seqs.next_to(share_sub, DOWN, buff=0.70)

        self.play(FadeIn(heading), FadeIn(share_sub), run_time=0.5)
        self.play(FadeIn(lab1), FadeIn(row1), run_time=0.7)
        self.play(
            LaggedStart(
                *[TransformFromCopy(copy1[i], copy2[i]) for i in range(len(prompt_words))],
                lag_ratio=0.10,
            ),
            FadeIn(lab2),
            FadeIn(suf2),
            run_time=1.5,
        )
        self._hold()

        heading = self._retitle(heading, "Prompt KV stored twice")
        brace_box = SurroundingRectangle(VGroup(copy1, copy2), color=BAD, buff=0.14, stroke_width=2)
        dup = text("prompt KV  ·  12% of total  ·  stored twice", font_size=SMALL_SIZE, color=BAD)
        if dup.width > 13.0:
            dup.scale_to_fit_width(13.0)
        dup.next_to(seqs, DOWN, buff=0.40)
        beam = caption("beam search: up to 55% memory saving, if sequences could share")
        beam.next_to(dup, DOWN, buff=0.28)
        self.play(Circumscribe(VGroup(copy1, copy2), color=BAD, buff=0.14), run_time=0.8)
        self.play(FadeIn(brace_box), FadeIn(dup), run_time=0.55)
        self.play(FadeIn(beam), run_time=0.45)
        self._hold()

        self.play(FadeOut(*self.mobjects), run_time=0.5)

        # -------------------------------------------------------------
        # Beat 19: one cause
        # -------------------------------------------------------------
        heading = title("One design, two failures")
        cause = body("each request's KV cache is one contiguous, pre-allocated chunk", font_size=26)
        if cause.width > 13.2:
            cause.scale_to_fit_width(13.2)
        cause.next_to(heading, DOWN, buff=0.40)

        swatches = VGroup(
            _legend_swatch("reserved", WARN),
            _legend_swatch("internal frag.", BAD),
            _legend_swatch("external frag.", BAD, opacity=0.5),
            _legend_swatch("no sharing", ACCENT2),
        )
        swatches.arrange(RIGHT, buff=0.45)
        swatches.next_to(cause, DOWN, buff=0.55)
        if swatches.width > 13.2:
            swatches.scale_to_fit_width(13.2)
            swatches.next_to(cause, DOWN, buff=0.55)

        land = body("Memory, not compute, caps the batch.", font_size=32)
        land.next_to(swatches, DOWN, buff=0.70)

        self.play(FadeIn(heading), run_time=0.5)
        self.play(FadeIn(cause), run_time=0.5)
        self.play(LaggedStart(*[FadeIn(s, shift=UP * 0.12) for s in swatches], lag_ratio=0.18), run_time=1.1)
        self.play(FadeIn(land), run_time=0.5)
        self._hold()

        self.play(FadeOut(*self.mobjects), run_time=0.5)

        # -------------------------------------------------------------
        # Beat 20: seam to Act II
        # -------------------------------------------------------------
        act_checkpoint(
            self,
            2,
            "The idea: blocks, not slabs",
            done=["Act I — Why memory is the bottleneck"],
            current="Act II — Blocks, not slabs",
            upcoming=["Act III — The payoffs"],
        )

        self.wait(0.45)


    def _nudge(self, mob, dx=0.18):
        self.play(mob.animate.shift(RIGHT * dx), rate_func=there_and_back, run_time=0.55)

    def _hold(self):
        self.wait(0.45)
        self.next_slide()

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
