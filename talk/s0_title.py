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
    RoundedRectangle,
    Square,
    VGroup,
)
from manim_slides import Slide

from talk.theme import (
    ACCENT,
    BAD,
    BLOCK_FILL,
    BLOCK_STROKE,
    BODY_SIZE,
    FG,
    GOOD,
    MUTED,
    SMALL_SIZE,
    TINY_SIZE,
    TITLE_SIZE,
    WARN,
    apply_theme,
    body,
    caption,
    small,
    text,
)


def _coin_stack(n, color, cell=0.22):
    """A small stack of `n` squares ("coins"/cost units), bottom-aligned."""
    stack = VGroup()
    for i in range(n):
        sq = Square(side_length=cell, fill_color=color, fill_opacity=0.9, stroke_color=FG, stroke_width=1)
        stack.add(sq)
    stack.arrange(UP, buff=0.03)
    return stack


def _core_grid(cols=6, rows=5, cell=0.32):
    """Dim GPU-core grid. Returns (grid, list of cores to light)."""
    cores = VGroup()
    for _ in range(rows * cols):
        sq = Square(
            side_length=cell,
            fill_color=MUTED,
            fill_opacity=0.22,
            stroke_color=BLOCK_STROKE,
            stroke_width=1,
        )
        cores.add(sq)
    cores.arrange_in_grid(rows=rows, cols=cols, buff=cell * 0.28)
    n_lit = max(1, (rows * cols) // 4)
    lit = [cores[i] for i in range(n_lit)]
    return cores, lit


def _vram_tank(width=1.35, height=2.55):
    """Empty VRAM tank plus a fill rect sized to the interior. Position fill after layout."""
    outline = RoundedRectangle(
        width=width,
        height=height,
        corner_radius=0.1,
        stroke_color=BLOCK_STROKE,
        stroke_width=3,
        fill_opacity=0.0,
    )
    pad = 0.12
    fill = RoundedRectangle(
        width=width - pad,
        height=height - pad,
        corner_radius=0.08,
        fill_color=ACCENT,
        fill_opacity=0.92,
        stroke_width=0,
    )
    return outline, fill


def _request_chip(label="req"):
    box = RoundedRectangle(
        width=0.85,
        height=0.42,
        corner_radius=0.08,
        fill_color=BLOCK_FILL,
        fill_opacity=1.0,
        stroke_color=BAD,
        stroke_width=2,
    )
    txt = text(label, font_size=TINY_SIZE, color=FG)
    txt.move_to(box.get_center())
    return VGroup(box, txt)


class S0Title(Slide):
    def construct(self):
        apply_theme(self)

        # ------------------------------------------------------------------
        # Beat 1 — Title card
        # ------------------------------------------------------------------
        main_title = text(
            "Efficient Memory Management for\nLLM Serving with PagedAttention",
            font_size=TITLE_SIZE,
            color=FG,
            line_spacing=1.2,
        )
        main_title.move_to(UP * 1.3)

        subtitle = text(
            "Kwon et al., SOSP '23 — the vLLM paper",
            font_size=BODY_SIZE,
            color=ACCENT,
        )
        subtitle.next_to(main_title, DOWN, buff=0.6)

        presenter = text(
            "Presented by Agrim Bansal, Harit Mangal",
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
        hook_title = text("The cost of one request", font_size=TITLE_SIZE, color=FG)
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
        # Beat 3 — Memory vs compute: idle cores, packed VRAM, blocked requests
        # ------------------------------------------------------------------
        mem_title = text("Where that cost lives", font_size=TITLE_SIZE, color=FG)
        mem_title.to_edge(UP)

        compute_hdr = small("COMPUTE", color=MUTED)
        cores, lit_cores = _core_grid()
        compute_badge = small("spare capacity", color=GOOD)
        compute_col = VGroup(compute_hdr, cores, compute_badge).arrange(DOWN, buff=0.25)

        memory_hdr = small("MEMORY", color=MUTED)
        tank_outline, tank_fill = _vram_tank()
        memory_badge = small("packed — no room", color=BAD)
        memory_col = VGroup(memory_hdr, tank_outline, memory_badge).arrange(DOWN, buff=0.25)

        cols = VGroup(compute_col, memory_col).arrange(RIGHT, buff=2.2)
        cols.next_to(mem_title, DOWN, buff=0.4)
        cols.shift(LEFT * 0.85)

        tank_fill.move_to(tank_outline.get_center())
        tank_fill.set_z_index(2)
        tank_outline.set_z_index(3)

        chips = VGroup(*[_request_chip("req") for _ in range(3)])
        chips.arrange(DOWN, buff=0.12)
        chips.next_to(tank_outline, RIGHT, buff=0.35)
        chip_hit = chips.get_center()
        chips.shift(RIGHT * 2.0)

        punchline = body("Can't batch more — they don't fit", color=ACCENT)
        punchline.to_edge(DOWN, buff=1.15)

        kv_line = caption("KV cache: >30% of VRAM, mostly wasted (later)")
        kv_line.next_to(punchline, DOWN, buff=0.2)

        self.play(FadeIn(mem_title))
        self.play(FadeIn(compute_hdr), FadeIn(cores), FadeIn(memory_hdr), FadeIn(tank_outline))
        self.play(*[c.animate.set_fill(GOOD, opacity=0.9) for c in lit_cores])
        self.play(FadeIn(compute_badge))
        self.play(GrowFromEdge(tank_fill, DOWN))
        self.play(FadeIn(memory_badge))
        self.play(chips.animate.move_to(chip_hit), run_time=0.45)
        self.play(chips.animate.shift(RIGHT * 0.4), run_time=0.18)
        x_mark = text("X", font_size=TITLE_SIZE, color=BAD, weight="BOLD")
        x_mark.next_to(chips, RIGHT, buff=0.18)
        self.play(FadeIn(x_mark))
        self.play(FadeIn(punchline), FadeIn(kv_line))
        self.wait(0.3)
        self.next_slide()

        self.play(
            FadeOut(mem_title),
            FadeOut(compute_col),
            FadeOut(memory_col),
            FadeOut(tank_fill),
            FadeOut(chips),
            FadeOut(x_mark),
            FadeOut(punchline),
            FadeOut(kv_line),
        )

        # ------------------------------------------------------------------
        # Beat 4 — Roadmap: three acts as pills
        # ------------------------------------------------------------------
        road_title = text("The plan: three acts", font_size=TITLE_SIZE, color=FG)
        road_title.to_edge(UP)

        act_specs = [
            ("Act I", "Prerequisite\nUnderstanding the problem", MUTED),
            ("Act II", "Blocks,\nnot slabs", ACCENT),
            ("Act III", "The payoffs", GOOD),
        ]
        pills = VGroup()
        for name, desc, color in act_specs:
            circle = RoundedRectangle(
                width=1.9, height=0.9, corner_radius=0.45,
                color=color, fill_opacity=0.12, stroke_width=3,
            )
            label = text(name, font_size=BODY_SIZE, color=color, weight="BOLD")
            if label.width > circle.width * 0.82:
                label.scale_to_fit_width(circle.width * 0.82)
            label.move_to(circle.get_center())
            desc_txt = text(desc, font_size=SMALL_SIZE, color=FG, line_spacing=1.0)
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
