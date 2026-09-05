"""S0 — Title / cost hook + roadmap.

NARRATION

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
"""

from manim import (
    DOWN,
    LEFT,
    ORIGIN,
    RIGHT,
    UP,
    Circle,
    FadeIn,
    FadeOut,
    GrowFromEdge,
    RoundedRectangle,
    Text,
    VGroup,
)
from manim_slides import Slide

from talk.theme import (
    ACCENT,
    BAD,
    BODY_SIZE,
    FG,
    MUTED,
    SMALL_SIZE,
    TINY_SIZE,
    TITLE_SIZE,
    WARN,
    apply_theme,
    body,
    caption,
    small,
)
from talk.components import MemoryBar


def _coin_stack(n, color, cell=0.22):
    """A small stack of `n` squares ("coins"/cost units), bottom-aligned."""
    from manim import Square

    stack = VGroup()
    for i in range(n):
        sq = Square(side_length=cell, fill_color=color, fill_opacity=0.9, stroke_color=FG, stroke_width=1)
        stack.add(sq)
    stack.arrange(UP, buff=0.03)
    return stack


class S0Title(Slide):
    def construct(self):
        apply_theme(self)

        # ------------------------------------------------------------------
        # Beat 1 — Title card
        # ------------------------------------------------------------------
        main_title = Text(
            "Efficient Memory Management for\nLLM Serving with PagedAttention",
            font_size=TITLE_SIZE,
            color=FG,
            line_spacing=1.2,
        )
        main_title.move_to(UP * 1.3)

        subtitle = Text(
            "Kwon et al., SOSP '23 — the vLLM paper",
            font_size=BODY_SIZE,
            color=ACCENT,
        )
        subtitle.next_to(main_title, DOWN, buff=0.6)

        presenter = Text(
            "Presented by ___",
            font_size=SMALL_SIZE,
            color=MUTED,
        )
        presenter.next_to(subtitle, DOWN, buff=1.0)

        self.play(FadeIn(main_title, shift=UP * 0.2))
        self.play(FadeIn(subtitle))
        self.play(FadeIn(presenter))
        self.next_slide()

        self.play(FadeOut(main_title), FadeOut(subtitle), FadeOut(presenter))

        # ------------------------------------------------------------------
        # Beat 2 — Cost hook: small stack vs large stack of coins/bars
        # ------------------------------------------------------------------
        hook_title = Text("The cost of one request", font_size=TITLE_SIZE, color=FG)
        hook_title.to_edge(UP)

        search_stack = _coin_stack(2, MUTED)
        llm_stack = _coin_stack(20, ACCENT)

        search_label = small("Keyword\nsearch", color=MUTED, line_spacing=0.9)
        llm_label = small("LLM request\n(~10x)", color=ACCENT, line_spacing=0.9)

        stacks = VGroup(search_stack, llm_stack).arrange(RIGHT, buff=2.2)
        stacks.move_to(ORIGIN + DOWN * 0.3)

        # align bottoms
        search_stack.align_to(llm_stack, DOWN)
        search_label.next_to(search_stack, DOWN, buff=0.3)
        llm_label.next_to(llm_stack, DOWN, buff=0.3)

        self.play(FadeIn(hook_title))
        self.play(GrowFromEdge(search_stack, DOWN), FadeIn(search_label))
        self.play(GrowFromEdge(llm_stack, DOWN), FadeIn(llm_label))
        self.wait(0.3)
        self.next_slide()

        self.play(FadeOut(hook_title), FadeOut(stacks), FadeOut(search_label), FadeOut(llm_label))

        # ------------------------------------------------------------------
        # Beat 3 — Memory hook: MemoryBar with wasted KV region foreshadowed
        # ------------------------------------------------------------------
        mem_title = Text("Where that cost lives: GPU memory", font_size=TITLE_SIZE, color=FG)
        mem_title.to_edge(UP)

        bar = MemoryBar(
            segments=[
                ("Weights (65%)", 0.65, MUTED),
                ("KV cache (30%)", 0.30, ACCENT),
                ("", 0.05, MUTED),
            ],
            width=10.0,
            height=1.0,
            show_pct=False,
        )
        bar.move_to(UP * 0.3)

        other_caption = caption("Other ~5%")
        other_caption.next_to(bar.segs[2], UP, buff=0.15)

        kv_caption = caption(">30% of GPU memory is the KV cache")
        kv_caption.next_to(bar, DOWN, buff=0.5)

        self.play(FadeIn(mem_title))
        self.play(bar.animate_in())
        self.play(FadeIn(kv_caption), FadeIn(other_caption))

        # Zoom in on the KV segment to foreshadow waste inside it
        waste_bar = MemoryBar(
            segments=[
                ("used", 0.30, ACCENT),
                ("wasted", 0.70, BAD),
            ],
            width=6.0,
            height=0.8,
            show_pct=False,
        )
        waste_bar.next_to(bar, DOWN, buff=1.4)
        waste_caption = caption("of the KV region: ~20-40% used (details later)")
        waste_caption.next_to(waste_bar, DOWN, buff=0.35)

        self.play(FadeIn(waste_bar.outline))
        self.play(*[GrowFromEdge(s, LEFT) for s in waste_bar.segs])
        self.play(FadeIn(waste_bar.labels), FadeIn(waste_caption))
        self.wait(0.3)
        self.next_slide()

        self.play(
            FadeOut(mem_title),
            FadeOut(bar),
            FadeOut(kv_caption),
            FadeOut(other_caption),
            FadeOut(waste_bar),
            FadeOut(waste_caption),
        )

        # ------------------------------------------------------------------
        # Beat 4 — Roadmap: three acts as pills
        # ------------------------------------------------------------------
        road_title = Text("The plan: three acts", font_size=TITLE_SIZE, color=FG)
        road_title.to_edge(UP)

        act_specs = [
            ("Act I", "Why memory is\nthe bottleneck", MUTED),
            ("Act II", "Blocks,\nnot slabs", ACCENT),
            ("Act III", "The payoffs", WARN),
        ]
        pills = VGroup()
        for name, desc, color in act_specs:
            circle = RoundedRectangle(
                width=1.9, height=0.9, corner_radius=0.45,
                color=color, fill_opacity=0.12, stroke_width=3,
            )
            label = Text(name, font_size=BODY_SIZE, color=color, weight="BOLD")
            if label.width > circle.width * 0.82:
                label.scale_to_fit_width(circle.width * 0.82)
            label.move_to(circle.get_center())
            desc_txt = Text(desc, font_size=SMALL_SIZE, color=FG, line_spacing=1.0)
            desc_txt.next_to(circle, DOWN, buff=0.35)
            pill = VGroup(circle, label, desc_txt)
            pills.add(pill)
        pills.arrange(RIGHT, buff=1.4)
        pills.move_to(ORIGIN + DOWN * 0.2)

        self.play(FadeIn(road_title))
        self.play(
            FadeIn(pills[0], shift=UP * 0.2),
            FadeIn(pills[1], shift=UP * 0.2),
            FadeIn(pills[2], shift=UP * 0.2),
            lag_ratio=0.3,
        )
        self.wait(0.3)
        self.next_slide()
