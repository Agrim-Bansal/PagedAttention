"""S9 — Results (Act III). Paper §6, figures as drawn.

Walks the evaluation in paper order: §6.1 setup, Fig 1-right (the page-1
claim), Fig 11, §6.2 Fig 12 then Fig 13, §6.3 Fig 14 then Fig 15, §6.4
Fig 16, §6.5 Fig 17. Latency charts keep the paper's axes and series;
vLLM is drawn last so the gap is visible. Curve knees are schematic
stand-ins for the paper's plots; printed bar values and quoted multipliers
are exact.

NARRATION
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
"""

from __future__ import annotations

import math

from manim import (
    DOWN,
    LEFT,
    ORIGIN,
    RIGHT,
    UP,
    DashedLine,
    FadeIn,
    FadeOut,
    GrowFromEdge,
    Square,
    VGroup,
)
from manim_slides import Slide

from talk.theme import *
from talk.components import *

COLORS = {
    "vLLM": ACCENT,
    "FasterTransformer": MUTED,
    "Orca (Oracle)": ACCENT2,
    "Orca (Pow2)": V_COLOR,
    "Orca (Max)": WARN,
    "Existing systems": MUTED,
}
MARKERS = {
    "vLLM": "circle",
    "FasterTransformer": "x",
    "Orca (Max)": "x",
    "Orca (Pow2)": "triangle",
    "Orca (Oracle)": "square",
    "Existing systems": "x",
}
STROKES = {
    "vLLM": 4.5,
    "FasterTransformer": 2.5,
    "Orca (Oracle)": 2.8,
    "Orca (Pow2)": 2.8,
    "Orca (Max)": 2.8,
    "Existing systems": 2.8,
}
SERIES_FT = [
    "FasterTransformer",
    "Orca (Max)",
    "Orca (Pow2)",
    "Orca (Oracle)",
    "vLLM",
]
SERIES_ORCA = ["Orca (Max)", "Orca (Pow2)", "Orca (Oracle)", "vLLM"]


def _xs(xmax, n=20):
    return [xmax * i / n for i in range(n + 1)]


def _latency(x, knee, ymin=0.045, ymax=1.0, power=5.5):
    if knee <= 0:
        return ymax
    r = max(x, 0.0) / knee
    return min(ymin + (ymax - ymin) * (r ** power), ymax)


def _x_step(xmax):
    if xmax <= 1.0:
        return 0.25
    if xmax <= 2.5:
        return 0.5
    if xmax <= 8:
        return 2
    if xmax <= 25:
        return 5
    return 10


def latency_chart(xmax, knees, names, width, height, show_legend=False, y_label="", x_label="", x_range=None):
    xs = _xs(xmax)
    series = {n: [_latency(x, knees[n]) for x in xs] for n in names}
    if x_range is None:
        x_range = [0, xmax, _x_step(xmax)]
    return line_chart(
        xs,
        series,
        x_label=x_label,
        y_label=y_label,
        x_range=x_range,
        y_range=[0, 1.0, 0.25],
        colors=COLORS,
        marker_shapes=MARKERS,
        stroke_widths=STROKES,
        width=width,
        height=height,
        show_legend=show_legend,
        markers=True,
    )


def shared_legend(names):
    items = VGroup()
    for name in names:
        color = COLORS[name]
        swatch = Square(
            side_length=0.16,
            fill_color=color,
            fill_opacity=1.0,
            stroke_width=0,
        )
        txt = caption(name)
        txt.next_to(swatch, RIGHT, buff=0.1)
        items.add(VGroup(swatch, txt))
    items.arrange(RIGHT, buff=0.32)
    return items


def density_xy(mean, sigma, peak, xmax=2000, n=80):
    xs = [xmax * i / n for i in range(n + 1)]
    raw = []
    for x in xs:
        if x < 1:
            raw.append(0.0)
            continue
        mu = math.log(mean) - 0.5 * sigma * sigma
        p = math.exp(-((math.log(x) - mu) ** 2) / (2 * sigma * sigma)) / (
            x * sigma * math.sqrt(2 * math.pi)
        )
        raw.append(p)
    m = max(raw) or 1.0
    return xs, [peak * v / m for v in raw]


def make_panel(xmax, knees, names, heading, w=3.95, h=2.45):
    chart = latency_chart(
        xmax, knees, names, width=w, height=h, show_legend=False,
    )
    hdr = small(heading, font_size=TINY_SIZE, color=FG)
    grp = VGroup(hdr, chart)
    grp.arrange(DOWN, buff=0.08)
    return grp, chart


class S9Results(Slide):
    def construct(self):
        apply_theme(self)

        # -------------------------------------------------------------
        # Beat 1: §6.1 setup
        # -------------------------------------------------------------
        heading = title("Section 6: putting vLLM to the test")
        self.play(FadeIn(heading))

        model_rows = [
            ("OPT-13B", "1x A100", "12 GB KV"),
            ("OPT-66B", "4x A100", "21 GB KV"),
            ("OPT-175B", "8x A100 80GB", "264 GB KV"),
            ("LLaMA-13B", "1x A100", "translation"),
        ]
        model_group = VGroup()
        for name, gpu, kv in model_rows:
            row = VGroup(
                small(name, font_size=SMALL_SIZE, color=FG),
                small(gpu, font_size=TINY_SIZE, color=MUTED),
                caption(kv),
            )
            row.arrange(RIGHT, buff=0.35)
            model_group.add(row)
        model_group.arrange(DOWN, buff=0.18, aligned_edge=LEFT)
        model_title = caption("Table 1 — models and hardware")
        model_title.next_to(model_group, UP, buff=0.22).align_to(model_group, LEFT)
        left = VGroup(model_title, model_group)
        left.move_to(LEFT * 3.5 + UP * 0.15)

        baseline_group = VGroup()
        for name, desc in [
            ("FasterTransformer", "latency-optimized; dynamic batching"),
            ("Orca (Oracle)", "knows true output length (infeasible)"),
            ("Orca (Pow2)", "over-reserves by up to 2x"),
            ("Orca (Max)", "always reserves 2048"),
        ]:
            row = VGroup(
                small(name, font_size=TINY_SIZE, color=COLORS.get(name, FG)),
                caption(desc),
            )
            row.arrange(RIGHT, buff=0.28)
            baseline_group.add(row)
        baseline_group.arrange(DOWN, buff=0.18, aligned_edge=LEFT)
        baseline_title = caption("Baselines")
        baseline_title.next_to(baseline_group, UP, buff=0.22).align_to(baseline_group, LEFT)
        right = VGroup(baseline_title, baseline_group)
        right.move_to(RIGHT * 3.15 + UP * 0.15)

        self.play(FadeIn(left), FadeIn(right))
        metric = small(
            "Normalized latency = mean(end-to-end latency / output length) vs request rate",
            font_size=TINY_SIZE,
            color=ACCENT,
        )
        metric.to_edge(DOWN, buff=0.55)
        note = caption("Better = keep latency low out to a higher request rate. Poisson arrivals.")
        note.next_to(metric, UP, buff=0.15)
        self.play(FadeIn(metric), FadeIn(note))
        self.wait(0.3)
        self.next_slide()
        self.play(FadeOut(VGroup(heading, left, right, metric, note)))

        # -------------------------------------------------------------
        # Beat 2: Fig 1-right
        # -------------------------------------------------------------
        heading2 = title("Fig 1 (right): the claim")
        self.play(FadeIn(heading2))

        batches = list(range(0, 41, 2))
        existing_mem = [min(26.0 + 1.55 * b, 40.0) for b in batches]
        vllm_mem = [min(26.0 + 0.32 * b, 39.2) for b in batches]
        existing_tput = [min(0.14 * b, 1.12) for b in batches]
        vllm_tput = [min(0.075 * b, 2.15) for b in batches]

        mem = line_chart(
            batches,
            {"Existing systems": existing_mem, "vLLM": vllm_mem},
            y_label="Memory usage (GB)",
            x_range=[0, 40, 10],
            y_range=[0, 40, 10],
            colors=COLORS,
            marker_shapes=MARKERS,
            stroke_widths=STROKES,
            width=9.2,
            height=2.15,
            show_legend=True,
            markers=True,
        )
        mem.next_to(heading2, DOWN, buff=0.18)

        tput = line_chart(
            batches,
            {"Existing systems": existing_tput, "vLLM": vllm_tput},
            x_label="Batch size (# requests)",
            y_label="Throughput (tok/s)",
            x_range=[0, 40, 10],
            y_range=[0, 2.5, 0.5],
            colors=COLORS,
            marker_shapes=MARKERS,
            stroke_widths=STROKES,
            width=9.2,
            height=2.0,
            show_legend=False,
            markers=True,
        )
        tput.next_to(mem, DOWN, buff=0.28)
        tput.align_to(mem, LEFT)

        param = DashedLine(
            mem.axes.c2p(0, 26),
            mem.axes.c2p(40, 26),
            color=WARN,
            stroke_width=1.5,
            dash_length=0.12,
        )
        param_lbl = caption("Parameter size 26 GB")
        pp = mem.axes.c2p(18, 29)
        param_lbl.move_to([pp[0], pp[1], 0])

        self.play(mem.fade_frame())
        self.play(mem.draw_series("Existing systems"), run_time=0.7)
        self.play(FadeIn(param), FadeIn(param_lbl))
        self.play(mem.draw_series("vLLM"), run_time=0.9)
        self.play(tput.fade_frame())
        self.play(tput.draw_series("Existing systems"), run_time=0.7)
        self.play(tput.draw_series("vLLM"), run_time=0.9)
        self.wait(0.3)
        self.next_slide()
        self.play(FadeOut(VGroup(heading2, mem, tput, param, param_lbl)))

        # -------------------------------------------------------------
        # Beat 3: Fig 11 histograms
        # -------------------------------------------------------------
        heading3 = title("Fig 11: ShareGPT vs Alpaca")
        self.play(FadeIn(heading3))

        sx, s_in = density_xy(161.31, 1.12, 1.85)
        _, s_out = density_xy(337.99, 1.02, 1.25)
        ax, a_in = density_xy(19.31, 0.72, 7.4)
        _, a_out = density_xy(58.45, 0.88, 4.3)

        dens_colors = {
            "Input (mean: 161.31)": ACCENT2,
            "Output (mean: 337.99)": ACCENT,
            "Input (mean: 19.31)": ACCENT2,
            "Output (mean: 58.45)": ACCENT,
        }
        chart11a = line_chart(
            sx,
            {"Input (mean: 161.31)": s_in, "Output (mean: 337.99)": s_out},
            x_label="# Tokens",
            y_label="Density (x 1e-2)",
            x_range=[0, 2000, 500],
            y_range=[0, 2.0, 0.5],
            colors=dens_colors,
            width=5.6,
            height=3.4,
            markers=False,
            show_legend=True,
        )
        chart11b = line_chart(
            ax,
            {"Input (mean: 19.31)": a_in, "Output (mean: 58.45)": a_out},
            x_label="# Tokens",
            y_label="",
            x_range=[0, 2000, 500],
            y_range=[0, 8.0, 2],
            colors=dens_colors,
            width=5.6,
            height=3.4,
            markers=False,
            show_legend=True,
        )
        t11a = small("(a) ShareGPT", font_size=SMALL_SIZE)
        t11b = small("(b) Alpaca", font_size=SMALL_SIZE)
        col_a = VGroup(t11a, chart11a).arrange(DOWN, buff=0.1)
        col_b = VGroup(t11b, chart11b).arrange(DOWN, buff=0.1)
        row11 = VGroup(col_a, col_b).arrange(RIGHT, buff=0.45)
        row11.next_to(heading3, DOWN, buff=0.2)

        self.play(FadeIn(t11a), FadeIn(t11b))
        self.play(chart11a.fade_frame(), chart11b.fade_frame())
        self.play(
            chart11a.draw_series("Input (mean: 161.31)"),
            chart11b.draw_series("Input (mean: 19.31)"),
            run_time=0.8,
        )
        self.play(
            chart11a.draw_series("Output (mean: 337.99)"),
            chart11b.draw_series("Output (mean: 58.45)"),
            run_time=0.8,
        )
        note3 = caption("ShareGPT: 8.4x longer prompts, 5.8x longer outputs, higher variance")
        note3.to_edge(DOWN, buff=0.28)
        self.play(FadeIn(note3))
        self.wait(0.3)
        self.next_slide()
        self.play(FadeOut(VGroup(heading3, row11, note3)))

        # -------------------------------------------------------------
        # Beat 4: Fig 12(a) OPT-13B ShareGPT, full frame
        # -------------------------------------------------------------
        heading4 = title("Fig 12: single-sequence generation")
        self.play(FadeIn(heading4))
        legend_ft = shared_legend(SERIES_FT)
        legend_ft.next_to(heading4, DOWN, buff=0.12)
        self.play(FadeIn(legend_ft))

        knees_12a = {
            "FasterTransformer": 0.3,
            "Orca (Max)": 0.5,
            "Orca (Pow2)": 0.75,
            "Orca (Oracle)": 1.1,
            "vLLM": 1.9,
        }
        t12a = small("(a) OPT-13B, 1 GPU, ShareGPT", font_size=SMALL_SIZE)
        chart12a = latency_chart(
            2.0, knees_12a, SERIES_FT, width=9.4, height=3.7,
            y_label="Normalized latency (s/token)",
            x_label="Request rate (req/s)",
        )
        col12a = VGroup(t12a, chart12a).arrange(DOWN, buff=0.1)
        col12a.next_to(legend_ft, DOWN, buff=0.12)

        self.play(FadeIn(t12a), chart12a.fade_frame())
        for name in SERIES_FT:
            self.play(chart12a.draw_series(name), run_time=0.65)
        note4 = caption("ShareGPT: 1.7-2.7x Oracle, 2.7-8x Max, up to 22x FasterTransformer")
        note4.to_edge(DOWN, buff=0.22)
        self.play(FadeIn(note4))
        self.wait(0.3)
        self.next_slide()

        # -------------------------------------------------------------
        # Beat 5: Fig 12 ShareGPT row (a)(b)(c)
        # -------------------------------------------------------------
        self.play(FadeOut(VGroup(col12a, note4)))

        share_specs = [
            ("(a) OPT-13B, 1 GPU", 2.0, knees_12a),
            (
                "(b) OPT-66B, 4 GPU",
                1.0,
                {
                    "FasterTransformer": 0.1,
                    "Orca (Max)": 0.2,
                    "Orca (Pow2)": 0.4,
                    "Orca (Oracle)": 0.65,
                    "vLLM": 0.95,
                },
            ),
            (
                "(c) OPT-175B, 8 GPU",
                2.5,
                {
                    "FasterTransformer": 0.5,
                    "Orca (Max)": 1.0,
                    "Orca (Pow2)": 1.5,
                    "Orca (Oracle)": 1.8,
                    "vLLM": 2.4,
                },
            ),
        ]
        share_panels = []
        share_charts = []
        for hdr, xmax, knees in share_specs:
            grp, ch = make_panel(xmax, knees, SERIES_FT, hdr)
            share_panels.append(grp)
            share_charts.append(ch)
        share_row = VGroup(*share_panels).arrange(RIGHT, buff=0.22, aligned_edge=UP)
        share_row.next_to(legend_ft, DOWN, buff=0.15)
        xlab5 = caption("Request rate (req/s)  —  ShareGPT")
        xlab5.next_to(share_row, DOWN, buff=0.12)

        self.play(FadeIn(xlab5), *[FadeIn(p[0]) for p in share_panels])
        self.play(*[ch.fade_frame() for ch in share_charts])
        for name in SERIES_FT:
            self.play(*[ch.draw_series(name) for ch in share_charts], run_time=0.55)
        self.wait(0.3)
        self.next_slide()

        # -------------------------------------------------------------
        # Beat 6: Fig 12 Alpaca row (d)(e)(f)
        # -------------------------------------------------------------
        self.play(FadeOut(VGroup(share_row, xlab5)))

        alpaca_specs = [
            (
                "(d) OPT-13B, Alpaca",
                30,
                {
                    "FasterTransformer": 5,
                    "Orca (Max)": 7,
                    "Orca (Pow2)": 13,
                    "Orca (Oracle)": 20,
                    "vLLM": 28,
                },
            ),
            (
                "(e) OPT-66B, Alpaca",
                20,
                {
                    "FasterTransformer": 3,
                    "Orca (Max)": 5,
                    "Orca (Pow2)": 9,
                    "Orca (Oracle)": 13,
                    "vLLM": 18,
                },
            ),
            (
                "(f) OPT-175B, Alpaca",
                22,
                {
                    "FasterTransformer": 4,
                    "Orca (Max)": 6,
                    "Orca (Pow2)": 13,
                    "Orca (Oracle)": 16,
                    "vLLM": 20,
                },
            ),
        ]
        alp_panels = []
        alp_charts = []
        for hdr, xmax, knees in alpaca_specs:
            grp, ch = make_panel(xmax, knees, SERIES_FT, hdr)
            alp_panels.append(grp)
            alp_charts.append(ch)
        alp_row = VGroup(*alp_panels).arrange(RIGHT, buff=0.22, aligned_edge=UP)
        alp_row.next_to(legend_ft, DOWN, buff=0.15)
        xlab6 = caption("Request rate (req/s)  —  Alpaca")
        xlab6.next_to(alp_row, DOWN, buff=0.1)

        self.play(FadeIn(xlab6), *[FadeIn(p[0]) for p in alp_panels])
        self.play(*[ch.fade_frame() for ch in alp_charts])
        for name in SERIES_FT:
            self.play(*[ch.draw_series(name) for ch in alp_charts], run_time=0.55)
        note6 = caption("(f): 175B + short Alpaca is less memory-bound, so the gap shrinks")
        note6.to_edge(DOWN, buff=0.22)
        self.play(FadeIn(note6))
        self.wait(0.3)
        self.next_slide()
        self.play(FadeOut(VGroup(heading4, legend_ft, alp_row, xlab6, note6)))

        # -------------------------------------------------------------
        # Beat 7: Fig 13 batched requests
        # -------------------------------------------------------------
        heading7 = title("Fig 13: average batched requests, OPT-13B")
        self.play(FadeIn(heading7))

        cats13 = ["Orca (Max)", "Orca (Pow2)", "Orca (Oracle)", "vLLM"]
        fill13 = [WARN, V_COLOR, ACCENT2, ACCENT]

        def bars_one(values, y_max, panel_title):
            ch = bar_chart(
                cats13,
                {"n": values},
                y_label="# Batched requests",
                y_max=y_max,
                colors={"n": ACCENT},
                width=5.6,
                height=3.5,
                value_labels=True,
                show_legend=False,
            )
            for bar, col in zip(ch.bars["n"], fill13):
                bar.set_fill(col)
            hdr = small(panel_title, font_size=SMALL_SIZE)
            grp = VGroup(hdr, ch).arrange(DOWN, buff=0.12)
            return grp, ch

        g13a, c13a = bars_one(
            [7.00, 9.81, 13.62, 30.42], 35, "(a) ShareGPT, 2 req/s",
        )
        g13b, c13b = bars_one(
            [7.00, 43.24, 72.75, 132.44], 150, "(b) Alpaca, 30 req/s",
        )
        row13 = VGroup(g13a, g13b).arrange(RIGHT, buff=0.5, aligned_edge=UP)
        row13.next_to(heading7, DOWN, buff=0.25)

        self.play(FadeIn(g13a[0]), FadeIn(g13b[0]))
        self.play(c13a.fade_frame(), c13b.fade_frame())
        for i in range(4):
            anims = [
                GrowFromEdge(c13a.bars["n"][i], DOWN),
                GrowFromEdge(c13b.bars["n"][i], DOWN),
                FadeIn(c13a.value_labels["n"][i]),
                FadeIn(c13b.value_labels["n"][i]),
            ]
            self.play(*anims, run_time=0.55)
        note7 = caption("ShareGPT: 2.2x Oracle, 4.3x Max  —  more leftover VRAM holds real tokens")
        note7.to_edge(DOWN, buff=0.28)
        self.play(FadeIn(note7))
        self.wait(0.3)
        self.next_slide()
        self.play(FadeOut(VGroup(heading7, row13, note7)))

        # -------------------------------------------------------------
        # Beat 8: Fig 14 parallel sampling
        # -------------------------------------------------------------
        heading8 = title("Fig 14: parallel generation, OPT-13B Alpaca")
        self.play(FadeIn(heading8))
        legend_o = shared_legend(SERIES_ORCA)
        legend_o.next_to(heading8, DOWN, buff=0.12)
        self.play(FadeIn(legend_o))

        par_specs = [
            ("(a) parallel size = 2", 17, {"Orca (Max)": 2, "Orca (Pow2)": 7, "Orca (Oracle)": 10, "vLLM": 15}),
            ("(b) parallel size = 4", 10, {"Orca (Max)": 1.5, "Orca (Pow2)": 4, "Orca (Oracle)": 6, "vLLM": 10}),
            ("(c) parallel size = 6", 7, {"Orca (Max)": 1, "Orca (Pow2)": 2.5, "Orca (Oracle)": 4, "vLLM": 6.5}),
        ]
        par_panels, par_charts = [], []
        for hdr, xmax, knees in par_specs:
            grp, ch = make_panel(xmax, knees, SERIES_ORCA, hdr)
            par_panels.append(grp)
            par_charts.append(ch)
        par_row = VGroup(*par_panels).arrange(RIGHT, buff=0.22, aligned_edge=UP)
        par_row.next_to(legend_o, DOWN, buff=0.12)
        xlab8 = caption("Request rate (req/s)")
        xlab8.next_to(par_row, DOWN, buff=0.1)

        self.play(FadeIn(xlab8), *[FadeIn(p[0]) for p in par_panels])
        self.play(*[ch.fade_frame() for ch in par_charts])
        for name in SERIES_ORCA:
            self.play(*[ch.draw_series(name) for ch in par_charts], run_time=0.55)
        self.wait(0.3)
        self.next_slide()

        # -------------------------------------------------------------
        # Beat 9: Fig 14 beam search
        # -------------------------------------------------------------
        self.play(FadeOut(VGroup(heading8, par_row, xlab8)))
        heading9 = title("Fig 14: beam search, OPT-13B Alpaca")
        heading9.move_to(heading8)
        self.play(FadeIn(heading9))

        beam_specs = [
            ("(d) beam width = 2", 17, {"Orca (Max)": 2, "Orca (Pow2)": 7, "Orca (Oracle)": 10, "vLLM": 16}),
            ("(e) beam width = 4", 10, {"Orca (Max)": 1, "Orca (Pow2)": 4, "Orca (Oracle)": 5.5, "vLLM": 9.5}),
            ("(f) beam width = 6", 7, {"Orca (Max)": 1, "Orca (Pow2)": 2.5, "Orca (Oracle)": 4, "vLLM": 7}),
        ]
        beam_panels, beam_charts = [], []
        for hdr, xmax, knees in beam_specs:
            grp, ch = make_panel(xmax, knees, SERIES_ORCA, hdr)
            beam_panels.append(grp)
            beam_charts.append(ch)
        beam_row = VGroup(*beam_panels).arrange(RIGHT, buff=0.22, aligned_edge=UP)
        beam_row.next_to(legend_o, DOWN, buff=0.12)
        xlab9 = caption("Request rate (req/s)")
        xlab9.next_to(beam_row, DOWN, buff=0.1)

        self.play(FadeIn(xlab9), *[FadeIn(p[0]) for p in beam_panels])
        self.play(*[ch.fade_frame() for ch in beam_charts])
        for name in SERIES_ORCA:
            self.play(*[ch.draw_series(name) for ch in beam_charts], run_time=0.55)
        note9 = caption("vs Orca (Oracle) on Alpaca: 1.3x in sampling -> 2.3x at beam width 6")
        note9.to_edge(DOWN, buff=0.22)
        self.play(FadeIn(note9))
        self.wait(0.3)
        self.next_slide()
        self.play(FadeOut(VGroup(heading9, legend_o, beam_row, xlab9, note9)))

        # -------------------------------------------------------------
        # Beat 10: Fig 15 memory saving
        # -------------------------------------------------------------
        heading10 = title("Fig 15: memory saving from sharing, Alpaca")
        self.play(FadeIn(heading10))

        c15a = bar_chart(
            ["2", "4", "6"],
            {"saving": [6.09, 8.53, 9.79]},
            y_label="Memory saving (%)",
            y_max=10,
            colors={"saving": ACCENT2},
            width=5.4,
            height=3.6,
            value_labels=True,
            show_legend=False,
            x_label="# Output sequences",
        )
        c15b = bar_chart(
            ["2", "4", "6"],
            {"saving": [37.56, 53.13, 55.16]},
            y_label="Memory saving (%)",
            y_max=60,
            colors={"saving": ACCENT},
            width=5.4,
            height=3.6,
            value_labels=True,
            show_legend=False,
            x_label="Beam width",
        )
        t15a = small("(a) Parallel sampling", font_size=SMALL_SIZE)
        t15b = small("(b) Beam search", font_size=SMALL_SIZE)
        g15a = VGroup(t15a, c15a).arrange(DOWN, buff=0.12)
        g15b = VGroup(t15b, c15b).arrange(DOWN, buff=0.12)
        row15 = VGroup(g15a, g15b).arrange(RIGHT, buff=0.55)
        row15.next_to(heading10, DOWN, buff=0.2)

        self.play(FadeIn(t15a), FadeIn(t15b))
        self.play(c15a.fade_frame())
        self.play(c15a.grow_series("saving"), run_time=0.8)
        self.play(c15b.fade_frame())
        self.play(c15b.grow_series("saving"), run_time=1.0)
        note10 = caption("ShareGPT (text): 16.2-30.5% parallel sampling, 44.3-66.3% beam search")
        note10.to_edge(DOWN, buff=0.28)
        self.play(FadeIn(note10))
        self.wait(0.3)
        self.next_slide()
        self.play(FadeOut(VGroup(heading10, row15, note10)))

        # -------------------------------------------------------------
        # Beat 11: Fig 16 shared prefix curves
        # -------------------------------------------------------------
        heading11 = title("Fig 16: shared prefix, LLaMA-13B En-De")
        self.play(FadeIn(heading11))
        legend16 = shared_legend(["Orca (Oracle)", "vLLM"])
        legend16.next_to(heading11, DOWN, buff=0.12)
        self.play(FadeIn(legend16))

        names16 = ["Orca (Oracle)", "vLLM"]
        g16a, c16a = make_panel(
            50,
            {"Orca (Oracle)": 24, "vLLM": 42},
            names16,
            "(a) 1-shot prefix (80 tok)",
            w=5.5,
            h=3.2,
        )
        g16b, c16b = make_panel(
            50,
            {"Orca (Oracle)": 12, "vLLM": 43},
            names16,
            "(b) 5-shot prefix (341 tok)",
            w=5.5,
            h=3.2,
        )
        row16 = VGroup(g16a, g16b).arrange(RIGHT, buff=0.4, aligned_edge=UP)
        row16.next_to(legend16, DOWN, buff=0.12)
        xlab16 = caption("Request rate (req/s)")
        xlab16.next_to(row16, DOWN, buff=0.08)

        self.play(FadeIn(xlab16), FadeIn(g16a[0]), FadeIn(g16b[0]))
        self.play(c16a.fade_frame(), c16b.fade_frame())
        self.play(c16a.draw_series("Orca (Oracle)"), c16b.draw_series("Orca (Oracle)"), run_time=0.7)
        self.play(c16a.draw_series("vLLM"), c16b.draw_series("vLLM"), run_time=0.9)
        note11 = caption("Throughput vs Orca (Oracle): 1.67x (1-shot) -> 3.58x (5-shot)")
        note11.to_edge(DOWN, buff=0.22)
        self.play(FadeIn(note11))
        self.wait(0.3)
        self.next_slide()
        self.play(FadeOut(VGroup(heading11, legend16, row16, xlab16, note11)))

        # -------------------------------------------------------------
        # Beat 12: Fig 17 chatbot
        # -------------------------------------------------------------
        heading12 = title("Fig 17: chatbot workload, OPT-13B")
        self.play(FadeIn(heading12))

        chart17 = latency_chart(
            0.9,
            {"Orca (Max)": 0.55, "Orca (Pow2)": 0.58, "Orca (Oracle)": 0.6, "vLLM": 0.78},
            SERIES_ORCA,
            width=9.2,
            height=3.8,
            show_legend=True,
            y_label="Normalized latency (s/token)",
            x_label="Request rate (req/s)",
            x_range=[0, 0.9, 0.3],
        )
        chart17.next_to(heading12, DOWN, buff=0.25)

        self.play(chart17.fade_frame())
        for name in SERIES_ORCA:
            self.play(chart17.draw_series(name), run_time=0.6)
        note12 = caption("Orca variants cluster (all reserve ~1024). vLLM still ~2x the request rate.")
        note12.to_edge(DOWN, buff=0.28)
        self.play(FadeIn(note12))
        self.wait(0.3)
        self.next_slide()
        self.play(FadeOut(VGroup(heading12, chart17, note12)))

        # -------------------------------------------------------------
        # Beat 13: landing
        # -------------------------------------------------------------
        heading13 = title("The evaluation, in one line")
        self.play(FadeIn(heading13))
        lines = VGroup(
            body("2-4x throughput vs Orca at the same latency", color=ACCENT),
            body("Up to 22x vs FasterTransformer", color=FG),
            body("Larger gains: longer sequences, larger models, harder decoding", color=MUTED),
            body("No change to the model", color=MUTED),
        )
        lines.arrange(DOWN, buff=0.38)
        lines.move_to(ORIGIN + DOWN * 0.15)
        self.play(FadeIn(lines, lag_ratio=0.2))
        handoff = caption("Next: the attention kernel itself is 20-26% slower  —  section 7")
        handoff.to_edge(DOWN, buff=0.45)
        self.play(FadeIn(handoff))
        self.wait(0.3)
        self.next_slide()
        self.play(FadeOut(VGroup(heading13, lines, handoff)))
        self.wait(0.3)
