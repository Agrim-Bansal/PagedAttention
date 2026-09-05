"""Scratch scene exercising every component and theme helper.

NARRATION
---------
Beat 1 — Tokens: the recurring "Four score..." example as token boxes.
Beat 2 — KV block states: one block cycling through empty/filled/reserved/
internal/external.
Beat 3 — Physical memory + block table: a grid of blocks, a logical->physical
table, and an arrow mapping one to the other.
Beat 4 — Ref counts: a badge counting shared references.
Beat 5 — Memory bar + pie: a 65/30/5 split shown two ways.
Beat 6 — GPU schematic: cores plus a VRAM bar filling up.
Beat 7 — Attention diagram: Q attends over K/V for a short token sequence.
Beat 8 — Bar chart: three grouped series across categories.
Beat 9 — Line chart: log-scale multi-series line chart.
Beat 10 — Act checkpoint: the "where we are" seam beat.
"""

from manim import DOWN, LEFT, RIGHT, UP, ORIGIN, FadeIn, FadeOut
from manim_slides import Slide

from talk.theme import *
from talk.components import *


class ScratchComponents(Slide):
    def construct(self):
        apply_theme(self)

        # Beat 1: tokens
        heading = title("Tokens")
        seq = token_sequence(FOUR_SCORE)
        seq.scale_to_fit_width(min(seq.width, 13.5))
        seq.next_to(heading, DOWN, buff=0.8)
        self.play(FadeIn(heading))
        self.play(FadeIn(seq))
        self.next_slide()
        self.play(FadeOut(heading), FadeOut(seq))

        # Beat 2: KV block states
        heading = title("KV block states")
        block = KVBlock(slots=5, index=3)
        block.move_to(ORIGIN)
        self.play(FadeIn(heading), FadeIn(block))
        self.next_slide()
        for i, state in enumerate(["filled", "reserved", "internal", "external", "empty"]):
            self.play(block.set_state(i, state))
        self.next_slide()
        self.play(FadeOut(heading), FadeOut(block))

        # Beat 3: physical mem grid + block table + arrow map
        heading = title("Physical memory + block table")
        grid = PhysicalMemGrid(n_blocks=6, slots=4, cols=3, cell=0.5)
        grid.scale(0.9)
        grid.to_edge(LEFT, buff=0.8).shift(DOWN * 0.3)
        table = BlockTable(n_rows=3)
        table.scale(0.9)
        table.to_edge(RIGHT, buff=1.2).shift(DOWN * 0.3)
        self.play(FadeIn(heading))
        self.play(FadeIn(grid), FadeIn(table))
        self.next_slide()
        self.play(table.set_row(0, 2, 4))
        self.play(table.highlight_row(0))
        arrow = arrow_map(table.rows[0]["mobject"], grid.block(2))
        self.play(FadeIn(arrow))
        self.wait(0.4)
        self.play(table.add_row(4, 1))
        self.wait(0.4)
        self.next_slide()
        self.play(FadeOut(heading), FadeOut(grid), FadeOut(table), FadeOut(arrow))

        # Beat 4: ref count badge
        heading = title("Reference count")
        block2 = KVBlock(slots=4, index=2)
        block2.move_to(ORIGIN)
        badge = RefCountBadge(value=1)
        badge.next_to(block2.cells.get_corner(UP + RIGHT), UP + RIGHT, buff=-0.15)
        self.play(FadeIn(heading), FadeIn(block2), FadeIn(badge))
        self.next_slide()
        for v in [2, 3, 1]:
            self.play(badge.set_value(v))
        self.next_slide()
        self.play(FadeOut(heading), FadeOut(block2), FadeOut(badge))

        # Beat 5: memory bar + pie
        heading = title("Memory budget: 65/30/5")
        segments = [("weights", 0.65, ACCENT), ("KV cache", 0.30, ACCENT2), ("other", 0.05, MUTED)]
        bar = MemoryBar(segments)
        bar.next_to(heading, DOWN, buff=1.0)
        pie = MemoryPie(segments)
        pie.next_to(bar, DOWN, buff=1.0)
        self.play(FadeIn(heading))
        self.play(bar.animate_in())
        self.play(FadeIn(pie))
        self.next_slide()
        self.play(FadeOut(heading), FadeOut(bar), FadeOut(pie))

        # Beat 6: GPU schematic
        heading = title("GPU schematic")
        gpu = GPUSchematic(cores=(8, 6), width=8)
        gpu.move_to(ORIGIN)
        self.play(FadeIn(heading), FadeIn(gpu))
        self.next_slide()
        self.play(gpu.fill_vram(0.85, color=GOOD))
        self.next_slide()
        self.play(FadeOut(heading), FadeOut(gpu))

        # Beat 7: attention diagram
        heading = title("Attention: Q attends over K/V")
        tokens = FOUR_SCORE[:6]
        diagram = attention_diagram(tokens, query_index=5)
        diagram.scale(0.85)
        diagram.move_to(ORIGIN).shift(DOWN * 0.3)
        self.play(FadeIn(heading))
        self.play(FadeIn(diagram.tokens))
        self.next_slide()
        self.play(FadeIn(diagram.keys), FadeIn(diagram.values), FadeIn(diagram.query))
        self.next_slide()
        self.play(*[shoot(a) for a in diagram.arrows])
        self.next_slide()
        self.play(FadeIn(diagram.weights))
        self.play(FadeIn(diagram.output))
        self.next_slide()
        self.play(FadeOut(heading), FadeOut(diagram))

        # Beat 8: bar chart, grouped, 3 series
        heading = title("Throughput comparison")
        chart = bar_chart(
            categories=["OPT-13B", "OPT-66B", "OPT-175B"],
            series={
                "vLLM": [30, 22, 18],
                "Orca (Oracle)": [18, 14, 11],
                "FasterTransformer": [7, 6, 5],
            },
            y_label="req/s",
            value_labels=True,
        )
        chart.scale(0.85)
        chart.next_to(heading, DOWN, buff=0.6)
        self.play(FadeIn(heading))
        self.play(FadeIn(chart.axes), FadeIn(chart.x_labels), FadeIn(chart.legend))
        self.play(chart.animate_in())
        self.next_slide()
        self.play(FadeOut(heading), FadeOut(chart))

        # Beat 9: line chart, log_y
        heading = title("Latency vs request rate (log scale)")
        lchart = line_chart(
            x=[1, 2, 4, 8, 16],
            series={
                "vLLM": [1.0, 1.1, 1.3, 2.0, 5.0],
                "Orca": [1.2, 1.6, 3.0, 12.0, 60.0],
            },
            x_label="requests/s",
            y_label="normalized latency",
            log_y=True,
        )
        lchart.scale(0.85)
        lchart.next_to(heading, DOWN, buff=0.6)
        self.play(FadeIn(heading))
        self.play(FadeIn(lchart.axes), FadeIn(lchart.x_labels), FadeIn(lchart.y_ticks), FadeIn(lchart.legend))
        self.play(lchart.animate_in())
        self.next_slide()
        self.play(FadeOut(heading), FadeOut(lchart))

        # Beat 10: act checkpoint
        act_checkpoint(
            self,
            act_no=2,
            act_title="The idea: page the KV cache",
            done=("Why memory is the bottleneck",),
            current="PagedAttention mechanics",
            upcoming=("Sharing", "Scheduling", "Results"),
        )

        self.wait(0.3)
