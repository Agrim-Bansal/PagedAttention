"""S10 — Ablations: kernel overhead and block size (Act III, Fig 18).

NARRATION
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
"""

from manim import (
    DOWN,
    LEFT,
    RIGHT,
    UP,
    ORIGIN,
    FadeIn,
    FadeOut,
    VGroup,
)
from manim_slides import Slide

from talk.theme import *
from talk.components import *


class S10Ablations(Slide):
    def construct(self):
        apply_theme(self)

        # -------------------------------------------------------------
        # Beat 1: Fig 18(a) — attention-kernel micro-benchmark
        # -------------------------------------------------------------
        heading = title("Ablation 1: the kernel is slower on its own")
        self.play(FadeIn(heading))

        categories = ["Context 64", "Context 128", "Context 256"]
        series = {
            "FT (bs 8)": [32, 57, 90],
            "FT (bs 32)": [52, 97, 150],
            "vLLM (bs 8)": [40, 70, 110],
            "vLLM (bs 32)": [65, 120, 185],
        }
        colors = {
            "FT (bs 8)": MUTED,
            "FT (bs 32)": ACCENT2,
            "vLLM (bs 8)": WARN,
            "vLLM (bs 32)": ACCENT,
        }
        chart = bar_chart(
            categories, series, y_label="Kernel latency (us)",
            colors=colors, width=9.0, height=3.6,
        )
        chart.move_to(ORIGIN).shift(UP * 0.15)

        self.play(chart.animate_in())

        callout = small("~20-26% slower attention kernel — but attention is only\npart of one step's total latency", font_size=SMALL_SIZE, color=ACCENT)
        callout.to_edge(DOWN, buff=0.3)
        self.play(FadeIn(callout))
        self.wait(0.3)
        self.next_slide()
        self.play(FadeOut(heading), FadeOut(chart), FadeOut(callout))

        # -------------------------------------------------------------
        # Beat 2: Fig 18(b) — block-size sweep
        # -------------------------------------------------------------
        heading2 = title("Ablation 2: picking the block size")
        self.play(FadeIn(heading2))

        block_sizes = [1, 2, 4, 8, 16, 32, 64, 128, 256]
        x_positions = list(range(len(block_sizes)))  # log2(block_size): evenly spaced
        line_series = {
            "ShareGPT": [5.0, 3.0, 2.0, 1.3, 1.0, 1.0, 1.1, 1.3, 4.5],
            "Alpaca": [4.0, 2.5, 1.6, 1.1, 1.0, 1.2, 2.5, 6.0, 15.0],
        }
        line = line_chart(
            x_positions, line_series,
            y_label="Normalized latency (s/token)",
            x_range=[0, 8, 1], y_range=[0, 17.5, 2.5],
            colors={"ShareGPT": ACCENT2, "Alpaca": ACCENT},
            width=9.0, height=4.2,
        )
        # Component limitation: line_chart labels ticks with the raw x
        # values it was given, and has no log-x mode. We plot at log2(block
        # size) positions (evenly spaced, since sizes are powers of two) and
        # swap in the true block-size numbers as the tick text afterward.
        for lbl, bs in zip(line.x_labels, block_sizes):
            real_lbl = text(str(bs), font_size=TINY_SIZE, color=MUTED)
            real_lbl.move_to(lbl)
            lbl.become(real_lbl)

        line.move_to(ORIGIN).shift(DOWN * 0.2)
        x_axis_caption2 = caption("Block size (tokens per block), log scale")
        x_axis_caption2.next_to(line.axes, DOWN, buff=0.9)

        self.play(line.animate_in(), FadeIn(x_axis_caption2))

        default_arrow = arrow(
            line.axes.c2p(4, 12.5), line.axes.c2p(4, 2.5),
            buff=0.1, color=WARN,
        )
        default_label = small("vLLM default: 16", font_size=SMALL_SIZE, color=WARN)
        default_label.next_to(default_arrow, UP, buff=0.15)
        self.play(shoot(default_arrow), FadeIn(default_label))
        self.wait(0.3)
        self.next_slide()
        self.play(
            FadeOut(heading2), FadeOut(line), FadeOut(x_axis_caption2),
            FadeOut(default_arrow), FadeOut(default_label),
        )

        # -------------------------------------------------------------
        # Beat 3: landing
        # -------------------------------------------------------------
        landing = title("Trade a slightly slower kernel for a far")
        landing2 = title("better memory story: net 2-4x")
        landing2.next_to(landing, DOWN, buff=0.3)
        group = VGroup(landing, landing2)
        group.scale(0.85)
        group.move_to(ORIGIN)
        self.play(FadeIn(group))
        self.wait(0.3)
        self.next_slide()
        self.play(FadeOut(group))
        self.wait(0.3)
