"""S3 — The KV cache (Act I).

NARRATION
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
"""

from manim import (
    DOWN,
    LEFT,
    RIGHT,
    UP,
    ORIGIN,
    AnimationGroup,
    Arrow,
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


class S3KVCache(Slide):
    def construct(self):
        apply_theme(self)

        # -------------------------------------------------------------
        # Beat 1: naive recomputation — a growing triangle of redundant work
        # -------------------------------------------------------------
        heading = title("Recomputing K/V every step is wasteful")
        sub = caption("Naive: rebuild every previous token's K,V, every step")
        sub.next_to(heading, DOWN, buff=0.25)
        self.play(FadeIn(heading), FadeIn(sub))

        n_steps = 6
        rows = VGroup()
        for step in range(1, n_steps + 1):
            row = VGroup(*[
                Square(side_length=0.32, fill_color=K_COLOR, fill_opacity=0.85, stroke_width=0)
                for _ in range(step)
            ])
            row.arrange(RIGHT, buff=0.08)
            label = Text(f"step {step}", font_size=TINY_SIZE, color=MUTED)
            label.next_to(row, LEFT, buff=0.3)
            full_row = VGroup(label, row)
            rows.add(full_row)
        rows.arrange(DOWN, buff=0.18, aligned_edge=LEFT)
        rows.next_to(sub, DOWN, buff=0.5)

        redo_caption = caption("Every step rebuilds work already done in earlier steps")
        redo_caption.next_to(rows, DOWN, buff=0.35)

        self.play(AnimationGroup(*[FadeIn(r) for r in rows], lag_ratio=0.25))
        self.play(FadeIn(redo_caption))
        self.wait(0.3)
        self.next_slide()
        self.play(FadeOut(heading), FadeOut(sub), FadeOut(rows), FadeOut(redo_caption))

        # -------------------------------------------------------------
        # Beat 2: cache them instead — the KV cache on "Four score..."
        # -------------------------------------------------------------
        heading2 = title('The KV cache: "Four score and seven years ago our fathers"')
        heading2.scale(0.8)

        words = FOUR_SCORE[:8]
        seq = token_sequence(words, height=0.55, font_size=SMALL_SIZE)
        seq.scale_to_fit_width(min(seq.width, 12.5))
        seq.next_to(heading2, DOWN, buff=0.7)

        k_row = VGroup()
        v_row = VGroup()
        for tb in seq:
            k_sq = Square(side_length=0.28, fill_color=K_COLOR, fill_opacity=0.9, stroke_width=0)
            v_sq = Square(side_length=0.28, fill_color=V_COLOR, fill_opacity=0.9, stroke_width=0)
            k_sq.next_to(tb, DOWN, buff=0.3)
            v_sq.next_to(k_sq, DOWN, buff=0.1)
            k_row.add(k_sq)
            v_row.add(v_sq)

        legend = VGroup(
            VGroup(Square(side_length=0.22, fill_color=K_COLOR, fill_opacity=0.9, stroke_width=0),
                   small("K", font_size=SMALL_SIZE, color=MUTED)).arrange(RIGHT, buff=0.12),
            VGroup(Square(side_length=0.22, fill_color=V_COLOR, fill_opacity=0.9, stroke_width=0),
                   small("V", font_size=SMALL_SIZE, color=MUTED)).arrange(RIGHT, buff=0.12),
        )
        legend.arrange(RIGHT, buff=0.6)
        legend.next_to(VGroup(seq, k_row, v_row), DOWN, buff=0.45)

        row_caption = caption("One row per request, growing by one K,V pair per token")
        row_caption.next_to(legend, DOWN, buff=0.3)

        self.play(FadeIn(heading2))
        self.play(FadeIn(seq))
        self.play(AnimationGroup(*[FadeIn(VGroup(k, v)) for k, v in zip(k_row, v_row)], lag_ratio=0.15))
        self.play(FadeIn(legend))
        self.play(FadeIn(row_caption))
        self.wait(0.3)
        self.next_slide()
        self.play(
            FadeOut(heading2), FadeOut(seq), FadeOut(k_row), FadeOut(v_row),
            FadeOut(legend), FadeOut(row_caption),
        )

        # -------------------------------------------------------------
        # Beat 3: per-request, two caches, lives in VRAM
        # -------------------------------------------------------------
        heading3 = title("Per request, and it lives in VRAM")

        req_a_words = FOUR_SCORE[:5]
        req_b_words = ["it", "was", "the", "best"]

        req_a_seq = token_sequence(req_a_words, height=0.5, font_size=SMALL_SIZE)
        req_b_seq = token_sequence(req_b_words, height=0.5, font_size=SMALL_SIZE)

        def kv_strip(seq, k_color, v_color):
            k_r = VGroup()
            v_r = VGroup()
            for tb in seq:
                k_sq = Square(side_length=0.24, fill_color=k_color, fill_opacity=0.9, stroke_width=0)
                v_sq = Square(side_length=0.24, fill_color=v_color, fill_opacity=0.9, stroke_width=0)
                k_sq.next_to(tb, DOWN, buff=0.22)
                v_sq.next_to(k_sq, DOWN, buff=0.08)
                k_r.add(k_sq)
                v_r.add(v_sq)
            return k_r, v_r

        # Position the two rows first (so the K/V strips can be placed
        # under each token below), then attach the "Request X" labels to
        # the left without disturbing the already-aligned K/V squares.
        req_a_seq.move_to(UP * 0.9)
        req_b_seq.move_to(DOWN * 1.5)
        req_a_seq.align_to(req_b_seq, LEFT).shift(RIGHT * 1.6)
        req_b_seq.shift(RIGHT * 1.6)

        req_a_k, req_a_v = kv_strip(req_a_seq, K_COLOR, V_COLOR)
        req_b_k, req_b_v = kv_strip(req_b_seq, K_COLOR, V_COLOR)

        label_a = small("Request A", font_size=SMALL_SIZE, color=ACCENT)
        label_a.next_to(req_a_seq, LEFT, buff=0.4)
        label_b = small("Request B", font_size=SMALL_SIZE, color=ACCENT2)
        label_b.next_to(req_b_seq, LEFT, buff=0.4)

        group_a = VGroup(label_a, req_a_seq, req_a_k, req_a_v)
        group_b = VGroup(label_b, req_b_seq, req_b_k, req_b_v)
        both = VGroup(group_a, group_b)
        both.scale_to_fit_width(min(both.width, 13.0))
        both.next_to(heading3, DOWN, buff=0.6)

        vram_caption = caption("Two requests, two independent caches — both live in VRAM")
        vram_caption.next_to(both, DOWN, buff=0.5)

        self.play(FadeIn(heading3))
        self.play(FadeIn(group_a))
        self.play(FadeIn(group_b))
        self.play(FadeIn(vram_caption))
        self.wait(0.3)
        self.next_slide()
        self.play(FadeOut(heading3), FadeOut(both), FadeOut(vram_caption))

        # -------------------------------------------------------------
        # Beat 4: it's big — the arithmetic
        # -------------------------------------------------------------
        heading4 = title("It's big: ~800 KB per token (OPT-13B)")
        heading4.scale(0.85)

        arithmetic = small("2 (K and V)  x  5120 (hidden)  x  40 (layers)  x  2 bytes (FP16)", font_size=SMALL_SIZE)
        arithmetic.next_to(heading4, DOWN, buff=0.7)
        equals = body("= 800 KB / token", font_size=BODY_SIZE, color=ACCENT)
        equals.next_to(arithmetic, DOWN, buff=0.4)

        request_line = small("A 2048-token request  x  800 KB", font_size=SMALL_SIZE)
        request_line.next_to(equals, DOWN, buff=0.7)
        request_equals = body("~= 1.6 GB for one request", font_size=BODY_SIZE, color=ACCENT)
        request_equals.next_to(request_line, DOWN, buff=0.4)

        self.play(FadeIn(heading4))
        self.play(FadeIn(arithmetic))
        self.play(FadeIn(equals))
        self.play(FadeIn(request_line))
        self.play(FadeIn(request_equals))
        self.wait(0.3)
        self.next_slide()
        self.play(
            FadeOut(heading4), FadeOut(arithmetic), FadeOut(equals),
            FadeOut(request_line), FadeOut(request_equals),
        )

        # -------------------------------------------------------------
        # Beat 5: the memory budget — recreate Fig 1 (left)
        # -------------------------------------------------------------
        heading5 = title("Memory budget: 13B model on an A100 40GB")
        heading5.scale(0.85)

        segments = [
            ("weights", 0.65, ACCENT),
            ("KV cache", 0.30, ACCENT2),
            ("other", 0.05, MUTED),
        ]
        pie = MemoryPie(segments, radius=1.9)
        pie.next_to(heading5, DOWN, buff=0.8)

        fixed_caption = caption("Weights: fixed the moment the model loads")
        flex_caption = caption("KV cache: grows and shrinks while serving")
        fixed_caption.next_to(pie, DOWN, buff=0.5)
        flex_caption.next_to(fixed_caption, DOWN, buff=0.25)

        self.play(FadeIn(heading5))
        self.play(FadeIn(pie))
        self.play(FadeIn(fixed_caption))
        self.play(FadeIn(flex_caption))
        self.wait(0.3)
        self.next_slide()
        self.play(FadeOut(heading5), FadeOut(pie), FadeOut(fixed_caption), FadeOut(flex_caption))

        # -------------------------------------------------------------
        # Beat 6: landing + seam to S4
        # -------------------------------------------------------------
        landing = title("KV cache fitted = requests batched = throughput")
        landing.scale(0.8)
        landing.move_to(UP * 0.6)
        seam = body("So how do we lay these caches out in memory?", font_size=BODY_SIZE, color=MUTED)
        seam.next_to(landing, DOWN, buff=0.9)

        self.play(FadeIn(landing))
        self.play(FadeIn(seam))
        self.wait(0.3)
        self.next_slide()
        self.play(FadeOut(landing), FadeOut(seam))
        self.wait(0.3)
