"""S9 — Results (Act III).

NARRATION
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
"""

from manim import (
    DOWN,
    LEFT,
    RIGHT,
    UP,
    ORIGIN,
    FadeIn,
    FadeOut,
    Square,
    VGroup,
)
from manim_slides import Slide

from talk.theme import *
from talk.components import *

# Colors per SCENE_BRIEF: vLLM=ACCENT, FasterTransformer=MUTED,
# Orca(Oracle)=ACCENT2, Orca(Pow2)=V_COLOR, Orca(Max)=WARN
COLORS = {
    "vLLM": ACCENT,
    "FasterTransformer": MUTED,
    "Orca (Oracle)": ACCENT2,
    "Orca (Pow2)": V_COLOR,
    "Orca (Max)": WARN,
}


class S9Results(Slide):
    def construct(self):
        apply_theme(self)

        # -------------------------------------------------------------
        # Beat 1: setup — models/GPUs table + baselines + metric
        # -------------------------------------------------------------
        heading = title("Putting vLLM to the test")
        self.play(FadeIn(heading))

        model_rows = [
            ("OPT-13B", "1x A100"),
            ("OPT-66B", "4x A100"),
            ("OPT-175B", "8x A100 80GB"),
            ("LLaMA-13B", "1x A100"),
        ]
        model_group = VGroup()
        for name, gpu in model_rows:
            row = VGroup(
                small(name, font_size=SMALL_SIZE, color=FG),
                small(gpu, font_size=SMALL_SIZE, color=MUTED),
            )
            row.arrange(RIGHT, buff=0.5)
            model_group.add(row)
        model_group.arrange(DOWN, buff=0.22, aligned_edge=LEFT)
        model_group.move_to(LEFT * 3.4 + UP * 0.3)
        model_title = caption("Models & hardware")
        model_title.next_to(model_group, UP, buff=0.3).align_to(model_group, LEFT)

        baseline_group = VGroup()
        for name, desc in [
            ("FasterTransformer", "latency-optimized engine"),
            ("Orca (Oracle)", "knows true output length"),
            ("Orca (Pow2)", "reserves next power of 2"),
            ("Orca (Max)", "reserves full 2048 max"),
        ]:
            row = VGroup(
                small(name, font_size=SMALL_SIZE, color=COLORS.get(name, FG)),
                caption(desc),
            )
            row.arrange(RIGHT, buff=0.35)
            baseline_group.add(row)
        baseline_group.arrange(DOWN, buff=0.22, aligned_edge=LEFT)
        baseline_group.move_to(RIGHT * 2.0 + UP * 0.3)
        baseline_title = caption("Baselines")
        baseline_title.next_to(baseline_group, UP, buff=0.3).align_to(baseline_group, LEFT)

        self.play(FadeIn(model_title), FadeIn(model_group))
        self.play(FadeIn(baseline_title), FadeIn(baseline_group))

        metric = small(
            "Metric: normalized latency (s/token) vs request rate",
            font_size=SMALL_SIZE, color=ACCENT,
        )
        metric.next_to(VGroup(model_group, baseline_group), DOWN, buff=0.7)
        self.play(FadeIn(metric))
        self.wait(0.3)
        self.next_slide()
        self.play(FadeOut(VGroup(
            heading, model_title, model_group, baseline_title, baseline_group, metric,
        )))

        # -------------------------------------------------------------
        # Beat 2: Fig 11 — ShareGPT vs Alpaca length distributions
        # -------------------------------------------------------------
        heading2 = title("Two very different workloads")
        self.play(FadeIn(heading2))

        chart2 = bar_chart(
            ["ShareGPT", "Alpaca"],
            {"Input (mean tokens)": [161, 19], "Output (mean tokens)": [338, 58]},
            y_label="",
            colors={"Input (mean tokens)": ACCENT2, "Output (mean tokens)": ACCENT},
            width=7.5, height=4.0, value_labels=True,
        )
        chart2.move_to(ORIGIN + DOWN * 0.2)
        self.play(chart2.animate_in())
        note2 = caption("ShareGPT: 8.4x longer prompts, 5.8x longer outputs -> more KV pressure")
        note2.next_to(chart2, DOWN, buff=0.5)
        self.play(FadeIn(note2))
        self.wait(0.3)
        self.next_slide()
        self.play(FadeOut(VGroup(heading2, chart2, note2)))

        # -------------------------------------------------------------
        # Beat 3: Fig 12 — normalized latency vs request rate, OPT-13B
        # -------------------------------------------------------------
        heading3 = title("Latency vs request rate: OPT-13B")
        self.play(FadeIn(heading3))

        # ShareGPT panel (approx knees from PAPER_REFERENCE Fig 12a)
        def latency_curve(knee, xmax, steepness=2.2):
            xs = [round(xmax * i / 12, 2) for i in range(13)]
            ys = []
            for xv in xs:
                if xv <= 0:
                    ys.append(0.05)
                else:
                    ratio = xv / knee
                    ys.append(min(0.05 + 0.9 * (ratio ** steepness), 1.05))
            return xs, ys

        share_x, _ = latency_curve(1.9, 2.0)
        share_series = {}
        knees_share = {"FasterTransformer": 0.3, "Orca (Max)": 0.5, "Orca (Pow2)": 0.75,
                        "Orca (Oracle)": 1.1, "vLLM": 1.9}
        for name, knee in knees_share.items():
            _, ys = latency_curve(knee, 2.0)
            share_series[name] = ys

        chart3a = line_chart(
            share_x, share_series,
            x_label="Request rate (req/s)", y_label="Norm. latency (s/tok)",
            y_range=[0, 1.1, 0.25], colors=COLORS, width=6.0, height=3.6, markers=False,
        )
        chart3a.scale(0.85)
        chart3a.move_to(LEFT * 3.6 + DOWN * 0.4)
        title3a = small("ShareGPT", font_size=SMALL_SIZE, color=FG)
        title3a.next_to(chart3a, UP, buff=0.15)

        alpaca_x, _ = latency_curve(28, 30)
        knees_alpaca = {"FasterTransformer": 5, "Orca (Max)": 7, "Orca (Pow2)": 13,
                         "Orca (Oracle)": 20, "vLLM": 28}
        alpaca_series = {}
        for name, knee in knees_alpaca.items():
            _, ys = latency_curve(knee, 30)
            alpaca_series[name] = ys

        chart3b = line_chart(
            alpaca_x, alpaca_series,
            x_label="Request rate (req/s)", y_label="Norm. latency (s/tok)",
            y_range=[0, 1.1, 0.25], colors=COLORS, width=6.0, height=3.6, markers=False,
        )
        chart3b.scale(0.85)
        chart3b.move_to(RIGHT * 3.6 + DOWN * 0.4)
        title3b = small("Alpaca", font_size=SMALL_SIZE, color=FG)
        title3b.next_to(chart3b, UP, buff=0.15)

        # shared legend below, to avoid duplicate legends crowding the frame
        chart3a.legend.set_opacity(0)
        chart3b.legend.set_opacity(0)
        shared_legend = VGroup()
        for name, color in COLORS.items():
            swatch = Square(side_length=0.16, fill_color=color, fill_opacity=1.0, stroke_width=0)
            txt = caption(name)
            txt.next_to(swatch, RIGHT, buff=0.1)
            shared_legend.add(VGroup(swatch, txt))
        shared_legend.arrange(RIGHT, buff=0.35)
        shared_legend.next_to(VGroup(chart3a, chart3b), DOWN, buff=0.5)

        self.play(FadeIn(title3a), FadeIn(title3b))
        self.play(FadeIn(chart3a), FadeIn(chart3b))
        self.play(FadeIn(shared_legend))
        note3 = caption("vLLM: 1.7-2.7x Oracle, 2.7-8x Orca (Max), up to 22x FasterTransformer")
        note3.next_to(shared_legend, DOWN, buff=0.3)
        self.play(FadeIn(note3))
        self.wait(0.3)
        self.next_slide()
        self.play(FadeOut(VGroup(
            heading3, chart3a, chart3b, title3a, title3b, shared_legend, note3,
        )))

        # -------------------------------------------------------------
        # Beat 4: Fig 13 — average batched requests
        # -------------------------------------------------------------
        heading4 = title("Why: more requests batched at once")
        self.play(FadeIn(heading4))

        chart4 = bar_chart(
            ["ShareGPT (2 req/s)", "Alpaca (30 req/s)"],
            {
                "Orca (Max)": [7.0, 7.0],
                "Orca (Pow2)": [9.81, 43.24],
                "Orca (Oracle)": [13.62, 72.75],
                "vLLM": [30.42, 132.44],
            },
            y_label="",
            colors=COLORS, width=8.5, height=4.2, value_labels=True,
        )
        chart4.scale(0.92)
        chart4.move_to(ORIGIN + DOWN * 0.2)
        self.play(chart4.animate_in())
        note4 = caption("vLLM batches 2.2x more than Orca (Oracle) on ShareGPT")
        note4.next_to(chart4, DOWN, buff=0.35)
        self.play(FadeIn(note4))
        self.wait(0.3)
        self.next_slide()
        self.play(FadeOut(VGroup(heading4, chart4, note4)))

        # -------------------------------------------------------------
        # Beat 5: callback to Fig 2 — waste bars
        # -------------------------------------------------------------
        heading5 = title("Callback: where did the memory go?")
        self.play(FadeIn(heading5))

        waste_data = [
            ("Orca (Max)", [("Token states", 20.4, GOOD), ("Reservation", 13.3, WARN),
                             ("Internal frag.", 57.3, BAD), ("External frag.", 8.9, BAD)]),
            ("Orca (Pow2)", [("Token states", 26.8, GOOD), ("Reservation", 17.9, WARN),
                              ("Internal frag.", 13.6, BAD), ("External frag.", 41.6, BAD)]),
            ("Orca (Oracle)", [("Token states", 38.2, GOOD), ("Reservation", 25.2, WARN),
                                ("External frag.", 36.6, BAD)]),
            ("vLLM", [("Token states", 96.3, GOOD), ("Other", 3.7, MUTED)]),
        ]
        labels5 = VGroup(*[small(label, font_size=SMALL_SIZE, color=FG) for label, _ in waste_data])
        labels5.arrange(DOWN, buff=0.42, aligned_edge=RIGHT)
        bars5 = VGroup()
        for label, segs in waste_data:
            fracs = [(name, val / 100.0, col) for name, val, col in segs]
            mb = MemoryBar(fracs, width=7.5, height=0.55, show_pct=False)
            bars5.add(mb)
        bars5.arrange(DOWN, buff=0.42)
        labels5.next_to(bars5, LEFT, buff=0.4)
        for lbl, mb in zip(labels5, bars5):
            lbl.match_y(mb)
        rows5 = VGroup(labels5, bars5)
        rows5.move_to(ORIGIN + DOWN * 0.1)

        self.play(FadeIn(labels5))
        self.play(*[mb.animate_in() for mb in bars5])
        note5 = caption("20.4% -> 38.2% actual use in Orca variants; vLLM: 96.3%")
        note5.next_to(rows5, DOWN, buff=0.5)
        self.play(FadeIn(note5))
        self.wait(0.3)
        self.next_slide()
        self.play(FadeOut(VGroup(heading5, rows5, note5)))

        # -------------------------------------------------------------
        # Beat 6: Fig 14 — parallel sampling & beam search
        # -------------------------------------------------------------
        heading6 = title("Harder decoding: sampling & beam search")
        self.play(FadeIn(heading6))

        knee_bar = bar_chart(
            ["parallel=2", "parallel=4", "parallel=6"],
            {
                "Orca (Max)": [2, 1.5, 1],
                "Orca (Pow2)": [7, 4, 2.5],
                "Orca (Oracle)": [10, 6, 4],
                "vLLM": [15, 10, 6.5],
            },
            y_label="",
            colors=COLORS, width=6.0, height=3.4, value_labels=False,
        )
        knee_bar.scale(0.85)
        knee_bar.move_to(LEFT * 3.6 + DOWN * 0.4)
        title6a = small("Parallel sampling", font_size=SMALL_SIZE, color=FG)
        title6a.next_to(knee_bar, UP, buff=0.15)

        beam_bar = bar_chart(
            ["beam=2", "beam=4", "beam=6"],
            {
                "Orca (Max)": [2, 1, 1],
                "Orca (Pow2)": [7, 4, 2.5],
                "Orca (Oracle)": [10, 5.5, 4],
                "vLLM": [16, 9.5, 7],
            },
            y_label="",
            colors=COLORS, width=6.0, height=3.4, value_labels=False,
        )
        beam_bar.scale(0.85)
        beam_bar.move_to(RIGHT * 3.6 + DOWN * 0.4)
        title6b = small("Beam search", font_size=SMALL_SIZE, color=FG)
        title6b.next_to(beam_bar, UP, buff=0.15)

        knee_bar.legend.set_opacity(0)
        beam_bar.legend.set_opacity(0)

        self.play(FadeIn(title6a), FadeIn(title6b))
        self.play(knee_bar.animate_in(), beam_bar.animate_in())
        note6 = caption("vLLM's edge over Orca (Oracle) widens: 1.3x (sampling) -> 2.3x (beam-6)")
        note6.next_to(VGroup(knee_bar, beam_bar), DOWN, buff=0.5)
        self.play(FadeIn(note6))
        self.wait(0.3)
        self.next_slide()
        self.play(FadeOut(VGroup(heading6, knee_bar, beam_bar, title6a, title6b, note6)))

        # -------------------------------------------------------------
        # Beat 7: Fig 15 — memory savings from sharing
        # -------------------------------------------------------------
        heading7 = title("Memory saved by sharing KV blocks")
        self.play(FadeIn(heading7))

        chart7 = bar_chart(
            ["2 / width 2", "4 / width 4", "6 / width 6"],
            {
                "Parallel sampling (Alpaca)": [6.09, 8.53, 9.79],
                "Beam search (Alpaca)": [37.56, 53.13, 55.16],
            },
            y_label="",
            colors={"Parallel sampling (Alpaca)": ACCENT2, "Beam search (Alpaca)": ACCENT},
            width=7.5, height=4.0, value_labels=True,
        )
        chart7.move_to(ORIGIN + DOWN * 0.2)
        self.play(chart7.animate_in())
        note7 = caption("ShareGPT (longer sequences): 16.2-30.5% / 44.3-66.3%")
        note7.next_to(chart7, DOWN, buff=0.45)
        self.play(FadeIn(note7))
        self.wait(0.3)
        self.next_slide()
        self.play(FadeOut(VGroup(heading7, chart7, note7)))

        # -------------------------------------------------------------
        # Beat 8: Fig 16 — shared prefix translation
        # -------------------------------------------------------------
        heading8 = title("Shared prefixes: translation (LLaMA-13B)")
        self.play(FadeIn(heading8))

        chart8 = bar_chart(
            ["1-shot prefix\n(80 tok)", "5-shot prefix\n(341 tok)"],
            {"Orca (Oracle)": [1.0, 1.0], "vLLM": [1.67, 3.58]},
            y_label="",
            colors={"Orca (Oracle)": ACCENT2, "vLLM": ACCENT},
            width=6.5, height=4.0, value_labels=True,
        )
        chart8.move_to(ORIGIN + DOWN * 0.2)
        self.play(chart8.animate_in())
        note8 = caption("More shared prefix -> bigger vLLM advantage: 1.67x -> 3.58x")
        note8.next_to(chart8, DOWN, buff=0.45)
        self.play(FadeIn(note8))
        self.wait(0.3)
        self.next_slide()
        self.play(FadeOut(VGroup(heading8, chart8, note8)))

        # -------------------------------------------------------------
        # Beat 9: Fig 17 — chatbot workload
        # -------------------------------------------------------------
        heading9 = title("Chatbot workload: long prompts")
        self.play(FadeIn(heading9))

        def chat_curve(knee, xmax=0.9, steepness=2.5):
            xs = [round(xmax * i / 12, 3) for i in range(13)]
            ys = []
            for xv in xs:
                if xv <= 0:
                    ys.append(0.05)
                else:
                    ratio = xv / knee
                    ys.append(min(0.05 + 0.9 * (ratio ** steepness), 1.05))
            return xs, ys

        chat_x, _ = chat_curve(0.78)
        chat_knees = {"Orca (Max)": 0.55, "Orca (Pow2)": 0.58, "Orca (Oracle)": 0.6, "vLLM": 0.78}
        chat_series = {}
        for name, knee in chat_knees.items():
            _, ys = chat_curve(knee)
            chat_series[name] = ys

        chart9 = line_chart(
            chat_x, chat_series,
            x_label="Request rate (req/s)", y_label="Norm. latency (s/tok)",
            y_range=[0, 1.1, 0.25], colors=COLORS, width=8.0, height=4.0, markers=False,
        )
        chart9.move_to(ORIGIN + DOWN * 0.2)
        self.play(FadeIn(chart9))
        note9 = caption("vLLM sustains ~2x the request rate; Orca variants converge")
        note9.next_to(chart9, DOWN, buff=0.35)
        self.play(FadeIn(note9))
        self.wait(0.3)
        self.next_slide()
        self.play(FadeOut(VGroup(heading9, chart9, note9)))

        # -------------------------------------------------------------
        # Beat 10: summary
        # -------------------------------------------------------------
        heading10 = title("The bottom line")
        self.play(FadeIn(heading10))

        lines10 = VGroup(
            body("2-4x throughput at the same latency vs Orca", color=ACCENT),
            body("Up to 22x vs FasterTransformer", color=FG),
            body("No change to the model", color=MUTED),
        )
        lines10.arrange(DOWN, buff=0.45)
        lines10.move_to(ORIGIN)
        self.play(FadeIn(lines10, lag_ratio=0.2))
        self.wait(0.3)
        self.next_slide()
        self.play(FadeOut(VGroup(heading10, lines10)))
        self.wait(0.3)
