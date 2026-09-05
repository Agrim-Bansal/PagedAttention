"""S4 — The problem, derived by the audience (Act I climax).

NARRATION
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
"""

from manim import (
    DOWN,
    LEFT,
    ORIGIN,
    RIGHT,
    UP,
    FadeIn,
    FadeOut,
    GrowFromEdge,
    Rectangle,
    Square,
    Text,
    VGroup,
)
from manim_slides import Slide

from talk.theme import *
from talk.components import *


# ---------------------------------------------------------------------------
# Local helpers (scene-specific; not shared components)
# ---------------------------------------------------------------------------


def _legend_swatch(label, color, opacity=1.0):
    sq = Square(side_length=0.24, fill_color=color, fill_opacity=opacity, stroke_width=0)
    txt = caption(label)
    txt.next_to(sq, RIGHT, buff=0.12)
    return VGroup(sq, txt)


def _frag_legend():
    entries = [
        _legend_swatch("in use", ACCENT),
        _legend_swatch("current iteration", ACCENT2),
        _legend_swatch("reserved (idle)", WARN),
        _legend_swatch("internal frag.", BAD),
        _legend_swatch("external frag.", BAD, opacity=0.5),
    ]
    row = VGroup(*entries)
    row.arrange(RIGHT, buff=0.5)
    return row


def _kv_strip(filled_words, current_word, n_reserved, internal_width, internal_label, cell=0.62):
    """One request's KV strip: filled tokens, one current-iteration token,
    a few small reserved cells, then one abstracted internal-fragmentation
    block whose width encodes the real slot count. Returns a VGroup with
    .filled, .current, .reserved, .internal attributes."""
    filled = VGroup(*[TokenBox(w, color=BG, fill=ACCENT, height=cell) for w in filled_words])
    filled.arrange(RIGHT, buff=0.08)

    current = TokenBox(current_word, color=BG, fill=ACCENT2, height=cell)

    reserved = VGroup(*[
        Square(side_length=cell * 0.7, fill_color=WARN, fill_opacity=1.0, stroke_color=BLOCK_STROKE, stroke_width=1.5)
        for _ in range(n_reserved)
    ])
    reserved.arrange(RIGHT, buff=0.08)

    internal_rect = Rectangle(
        width=internal_width,
        height=cell,
        fill_color=BAD,
        fill_opacity=1.0,
        stroke_color=BLOCK_STROKE,
        stroke_width=1.5,
    )
    internal_txt = Text(internal_label, font_size=TINY_SIZE, color=FG)
    if internal_txt.width > internal_width * 0.9:
        internal_txt.scale_to_fit_width(internal_width * 0.9)
    internal_txt.move_to(internal_rect.get_center())
    internal = VGroup(internal_rect, internal_txt)

    row = VGroup(filled, current, reserved, internal)
    row.arrange(RIGHT, buff=0.12)

    row.filled = filled
    row.current = current
    row.reserved = reserved
    row.internal = internal
    return row


class S4Problem(Slide):
    def construct(self):
        apply_theme(self)

        # -------------------------------------------------------------
        # Beat 1: the question
        # -------------------------------------------------------------
        question = Text(
            "How do you allocate memory for\nsomething whose final size is unknown?",
            font_size=36,
            color=FG,
            line_spacing=1.3,
        )
        question.move_to(UP * 0.8)
        subline = caption(
            "a request's output length is unknown until it emits <eos>"
        )
        subline.next_to(question, DOWN, buff=0.6)

        self.play(FadeIn(question))
        self.play(FadeIn(subline))
        self.next_slide()

        self.play(FadeOut(question), FadeOut(subline))

        # -------------------------------------------------------------
        # Beat 2: reserve the maximum
        # -------------------------------------------------------------
        heading = title("Existing systems: reserve the maximum")
        sub = caption("FasterTransformer, Orca — contiguous allocation, sized to the model's max length")
        sub.next_to(heading, DOWN, buff=0.25)

        strip_outline = Rectangle(width=12.0, height=1.0, stroke_color=BLOCK_STROKE, stroke_width=2)
        strip_outline.move_to(ORIGIN)
        claim = Rectangle(width=12.0, height=1.0, fill_color=WARN, fill_opacity=1.0, stroke_width=0)
        claim.move_to(strip_outline.get_left(), aligned_edge=LEFT)
        claim_label = Text("Request A — 2048 slots reserved", font_size=SMALL_SIZE, color=BG)
        claim_label.move_to(strip_outline.get_center())

        used_caption = caption("actual need: unknown until <eos>")
        used_caption.next_to(strip_outline, DOWN, buff=0.4)

        self.play(FadeIn(heading), FadeIn(sub))
        self.play(FadeIn(strip_outline))
        self.play(GrowFromEdge(claim, LEFT), run_time=1.2)
        self.play(FadeIn(claim_label))
        self.play(FadeIn(used_caption))
        self.next_slide()

        self.play(
            FadeOut(heading), FadeOut(sub), FadeOut(strip_outline),
            FadeOut(claim), FadeOut(claim_label), FadeOut(used_caption),
        )

        # -------------------------------------------------------------
        # Beat 3: Fig 3, zoomed on Request A
        # -------------------------------------------------------------
        heading = title('Fig. 3 — "Four score and seven years..."')
        legend = _frag_legend()
        legend.scale(0.85)
        legend.next_to(heading, DOWN, buff=0.3)

        row_a = _kv_strip(
            filled_words=FOUR_SCORE[:7],
            current_word="brought",
            n_reserved=2,
            internal_width=4.0,
            internal_label="2038 slots never used",
            cell=0.6,
        )
        row_a.scale_to_fit_width(min(row_a.width, 13.0))
        row_a.move_to(ORIGIN + DOWN * 0.3)

        cap_filled = caption("7 KV states\n(prompt)")
        cap_filled.next_to(row_a.filled, DOWN, buff=0.35)
        cap_current = caption("current\niteration")
        cap_current.next_to(row_a.current, DOWN, buff=0.35)
        cap_reserved = caption("reserved\n(idle)")
        cap_reserved.next_to(row_a.reserved, DOWN, buff=0.35)
        cap_internal = Text("INTERNAL FRAGMENTATION", font_size=TINY_SIZE, color=BAD)
        cap_internal.next_to(row_a.internal, UP, buff=0.35)

        self.play(FadeIn(heading), FadeIn(legend))
        self.play(FadeIn(row_a.filled), FadeIn(cap_filled))
        self.play(FadeIn(row_a.current), FadeIn(cap_current))
        self.play(FadeIn(row_a.reserved), FadeIn(cap_reserved))
        self.play(GrowFromEdge(row_a.internal, LEFT), FadeIn(cap_internal))
        self.next_slide()

        # -------------------------------------------------------------
        # Beat 4: Request B + external fragmentation
        # -------------------------------------------------------------
        row_a_group = VGroup(row_a, cap_filled, cap_current, cap_reserved, cap_internal)
        self.play(row_a_group.animate.shift(UP * 1.5))

        bottom_caps = VGroup(cap_filled, cap_current, cap_reserved)
        gap = Rectangle(width=row_a.width * 0.5, height=0.35, fill_color=BAD, fill_opacity=0.5, stroke_width=0)
        gap.next_to(bottom_caps, DOWN, buff=0.4).align_to(row_a, LEFT).shift(RIGHT * 1.0)
        gap_label = Text("external fragmentation — too small for any request", font_size=TINY_SIZE, color=FG)
        gap_label.next_to(gap, DOWN, buff=0.15)

        row_b = _kv_strip(
            filled_words=["You", "only", "live"],
            current_word="once",
            n_reserved=1,
            internal_width=1.0,
            internal_label="507 never used",
            cell=0.6,
        )
        row_b.next_to(gap_label, DOWN, buff=0.5).align_to(row_a, LEFT)

        cap_b = caption('Request B: "You only live once"')
        cap_b.next_to(row_b, DOWN, buff=0.3)

        self.play(FadeIn(gap), FadeIn(gap_label))
        self.play(FadeIn(row_b), FadeIn(cap_b))
        self.next_slide()

        self.play(FadeOut(*self.mobjects))

        # -------------------------------------------------------------
        # Beat 5: Fig 2, zoom out to the whole KV region
        # -------------------------------------------------------------
        heading = title("Fig. 2 — memory waste, whole KV region")
        legend2 = VGroup(
            _legend_swatch("token states", GOOD),
            _legend_swatch("reservation", WARN),
            _legend_swatch("internal frag.", BAD),
            _legend_swatch("external frag. & other", BAD, opacity=0.5),
        )
        legend2.arrange(RIGHT, buff=0.45)
        legend2.scale(0.8)
        legend2.next_to(heading, DOWN, buff=0.3)

        def _bar(label, segs, width=8.5):
            b = MemoryBar(segs, width=width, height=0.55, show_pct=False)
            row = VGroup(small(label, font_size=SMALL_SIZE), b)
            row.arrange(RIGHT, buff=0.4)
            return row, b

        row_max, bar_max = _bar("Orca (Max)", [
            ("20%", 0.204, GOOD),
            ("13%", 0.133, WARN),
            ("57%", 0.573, BAD),
            ("", 0.089, BAD),
        ])
        row_pow2, bar_pow2 = _bar("Orca (Pow2)", [
            ("27%", 0.268, GOOD),
            ("18%", 0.179, WARN),
            ("14%", 0.136, BAD),
            ("42%", 0.416, BAD),
        ])
        row_oracle, bar_oracle = _bar("Orca (Oracle)", [
            ("38%", 0.382, GOOD),
            ("25%", 0.252, WARN),
            ("37%", 0.366, BAD),
        ])
        row_vllm, bar_vllm = _bar("vLLM", [
            ("", 1.0, MUTED),
        ])

        # fix external-fragmentation opacity to match the 50% convention
        bar_max.segs[3].set_opacity(0.5)
        bar_pow2.segs[3].set_opacity(0.5)
        bar_oracle.segs[2].set_opacity(0.5)

        rows = VGroup(row_max, row_pow2, row_oracle, row_vllm)
        rows.arrange(DOWN, buff=0.45, aligned_edge=LEFT)
        for r in rows:
            r[1].align_to(rows[0][1], LEFT)
        rows.next_to(legend2, DOWN, buff=0.6)

        vllm_q = Text("?", font_size=TITLE_SIZE, color=MUTED)
        vllm_q.move_to(bar_vllm.get_center())

        landing = body("Only 20-40% of the KV memory holds actual token state.")
        landing.next_to(rows, DOWN, buff=0.5)

        self.play(FadeIn(heading), FadeIn(legend2))
        self.play(FadeIn(row_max[0]), bar_max.animate_in())
        self.play(FadeIn(row_pow2[0]), bar_pow2.animate_in())
        self.play(FadeIn(row_oracle[0]), bar_oracle.animate_in())
        self.play(FadeIn(row_vllm[0]), bar_vllm.animate_in(), FadeIn(vllm_q))
        self.play(FadeIn(landing))
        self.next_slide()

        self.play(FadeOut(*self.mobjects))

        # -------------------------------------------------------------
        # Beat 6: novelty — paging never needed before LLMs
        # -------------------------------------------------------------
        heading = title("Paging was never needed before LLMs")

        left_title = small("Before: a DNN tensor", color=MUTED)
        left_grid = VGroup(*[
            Square(side_length=0.35, fill_color=GOOD, fill_opacity=0.9, stroke_width=0.5, stroke_color=BG)
            for _ in range(16)
        ])
        left_grid.arrange_in_grid(rows=4, cols=4, buff=0.06)
        left_shape = caption("shape known before running:\n[32, 3, 224, 224]")
        left_note = caption("contiguous allocation is perfect —\nnothing to fragment")
        left_panel = VGroup(left_title, left_grid, left_shape, left_note)
        left_panel.arrange(DOWN, buff=0.3)
        left_panel.move_to(LEFT * 3.6)

        right_title = small("Now: the KV cache tensor", color=MUTED)
        right_bar_outline = Rectangle(width=2.2, height=1.6, stroke_color=BLOCK_STROKE, stroke_width=2)
        right_bar = Rectangle(width=2.2, height=0.5, fill_color=BAD, fill_opacity=1.0, stroke_width=0)
        right_bar.move_to(right_bar_outline.get_bottom(), aligned_edge=DOWN)
        right_q = Text("?", font_size=32, color=BAD)
        right_q.next_to(right_bar_outline, UP, buff=0.15)
        right_visual = VGroup(right_bar_outline, right_bar, right_q)
        right_shape = caption("length grows one token at a time,\nper request")
        right_note = caption("nobody knows where it stops\nuntil it stops")
        right_panel = VGroup(right_title, right_visual, right_shape, right_note)
        right_panel.arrange(DOWN, buff=0.3)
        right_panel.move_to(RIGHT * 3.6)

        divider = Rectangle(width=0.02, height=4.5, fill_color=MUTED, fill_opacity=0.5, stroke_width=0)
        divider.move_to(ORIGIN).shift(DOWN * 0.3)

        panels = VGroup(left_panel, right_panel, divider)
        panels.next_to(heading, DOWN, buff=0.6)

        self.play(FadeIn(heading))
        self.play(FadeIn(left_panel), FadeIn(divider))
        self.play(FadeIn(right_panel))
        self.next_slide()

        self.play(FadeOut(*self.mobjects))

        # -------------------------------------------------------------
        # Beat 7: no sharing
        # -------------------------------------------------------------
        heading = title("The second gap: no sharing")
        sub = caption("two requests, identical prompt — contiguous layout forces two full copies")
        sub.next_to(heading, DOWN, buff=0.25)

        shared_words = FOUR_SCORE[:5]
        copy1 = token_sequence(shared_words, fill=ACCENT, color=BG, height=0.55)
        copy2 = token_sequence(shared_words, fill=ACCENT, color=BG, height=0.55)
        label1 = caption("Request 1's copy")
        label2 = caption("Request 2's copy")
        copy1_group = VGroup(label1, copy1)
        copy1_group.arrange(DOWN, buff=0.2)
        copy2_group = VGroup(label2, copy2)
        copy2_group.arrange(DOWN, buff=0.2)
        copies = VGroup(copy1_group, copy2_group)
        copies.arrange(DOWN, buff=0.6)
        copies.next_to(sub, DOWN, buff=0.6)

        highlight = Rectangle(
            width=copies.width + 0.6,
            height=copies.height + 0.6,
            stroke_color=BAD,
            stroke_width=3,
        )
        highlight.move_to(copies.get_center())
        dup_label = Text("same content, stored twice", font_size=SMALL_SIZE, color=BAD)
        dup_label.next_to(highlight, DOWN, buff=0.3)

        self.play(FadeIn(heading), FadeIn(sub))
        self.play(FadeIn(copy1_group))
        self.play(FadeIn(copy2_group))
        self.play(FadeIn(highlight), FadeIn(dup_label))
        self.next_slide()

        self.play(FadeOut(*self.mobjects))

        # -------------------------------------------------------------
        # Beat 8: summary
        # -------------------------------------------------------------
        heading = title("Four wastes, one cause")
        items = VGroup(
            body("Reservation — held for tokens not yet generated", color=WARN, font_size=SMALL_SIZE),
            body("Internal fragmentation — guessed max length wrong", color=BAD, font_size=SMALL_SIZE),
            body("External fragmentation — gaps too small to reuse", color=BAD, font_size=SMALL_SIZE),
            body("No sharing — identical content stored twice", color=ACCENT2, font_size=SMALL_SIZE),
        )
        items.arrange(DOWN, buff=0.35, aligned_edge=LEFT)
        items.next_to(heading, DOWN, buff=0.7)

        cause = caption("one design choice: each request's KV cache is one contiguous block")
        cause.next_to(items, DOWN, buff=0.5)

        landing2 = body("Memory, not compute, caps throughput.", font_size=34)
        landing2.next_to(cause, DOWN, buff=0.5)

        self.play(FadeIn(heading))
        self.play(FadeIn(items))
        self.play(FadeIn(cause))
        self.play(FadeIn(landing2))
        self.next_slide()

        self.play(FadeOut(*self.mobjects))

        # -------------------------------------------------------------
        # Beat 9: seam to Act II
        # -------------------------------------------------------------
        act_checkpoint(
            self,
            2,
            "The idea: blocks, not slabs",
            done=["Act I — Why memory is the bottleneck"],
            current="Act II — Blocks, not slabs",
            upcoming=["Act III — The payoffs"],
        )

        self.wait(0.3)
