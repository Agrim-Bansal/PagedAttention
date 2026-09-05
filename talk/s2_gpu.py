"""S2 — Why GPUs (Act I). Standalone primer; written to sit before transformers.

NARRATION
---------
Beat 1 — Same math, over and over.
Neural-net serving is not fancy one-off logic. Under the hood it is the same
matrix multiply, again and again, against a huge shared weight matrix W.
One small input, one giant W, one small output — and that pattern repeats
across the whole model. [PAUSE] That "same operation, many times" shape is
exactly what a GPU is built for.

Beat 2 — CPU vs GPU.
A CPU has a handful of big, fast cores. Great at one complicated thing at a
time. Watch them light up in sequence. A GPU flips the trade: thousands of
small, simple cores. Give it a pile of identical multiplies and it runs them
all at once, one piece per core. [PAUSE] Our workload — the same multiply,
many times — maps almost perfectly onto that grid.

Beat 3 — Two memory pools.
Here is the catch that matters for serving. The GPU does not borrow the
computer's regular RAM. It has its own memory, called VRAM, sitting next to
the cores. CPU DRAM and GPU VRAM are two separate pools, joined by a PCIe
link that is slow compared to on-device memory. Cores can only multiply data
that is already in VRAM. If it is still on the CPU side, the GPU is waiting.

Beat 4 — Weights persist in VRAM.
So the model's weights have to live in VRAM for the whole time we are
serving. For OPT-13B on an A100, that is about 26 gigabytes of a 40-gigabyte
card. Sixty-five percent of the GPU's memory is gone the moment the model
loads — before we have served a single request. [PAUSE] Those weights stay
there. They do not come and go per request.

Beat 5 — One request vs 26 GB.
Now send in one request. The cores have to stream that whole 26 GB of
weights to produce one small result. Most of the time they are waiting on
memory, not multiplying. A single request is a tiny amount of math against a
huge W, so the machine looks idle even though VRAM is already packed.
Serving one-at-a-time wastes the GPU.

Beat 6 — Batching: one hub, many requests.
The fix is batching. The weights are identical for every request, so we load
W once and send many requests through the same multiply. Watch: every
request arrow ends at one point on W, and every result arrow starts from
that same point. One pass over the weights, N results. [PAUSE] Throughput
becomes a question of how many requests we can pack into that one pass.

Beat 7 — Leftover VRAM is the budget.
Almost for free — except leftover VRAM is finite. Weights already took
26 GB. The empty slice at the top is all we have for live request state.
Some requests fit; the rest bounce off. We cannot batch more than that
leftover space can hold. [PAUSE]

Beat 8 — Landing.
So: leftover VRAM decides the maximum batch, and the maximum batch decides
throughput. That is the resource this talk is about. Everything that follows
is about how we spend that leftover slice.
"""

from manim import (
    DOWN,
    LEFT,
    ORIGIN,
    RIGHT,
    UP,
    AnimationGroup,
    Circle,
    DashedLine,
    FadeIn,
    FadeOut,
    GrowFromEdge,
    Rectangle,
    ReplacementTransform,
    RoundedRectangle,
    Square,
    VGroup,
)

from manim_slides import Slide

from talk.theme import (
    ACCENT,
    ACCENT2,
    BAD,
    BG,
    BLOCK_FILL,
    BLOCK_STROKE,
    FG,
    GOOD,
    MUTED,
    SMALL_SIZE,
    TINY_SIZE,
    apply_theme,
    body,
    caption,
    small,
    text,
    title,
)
from talk.components import arrow, shoot


WEIGHTS_FRAC = 26.0 / 40.0


def _chip(label, color=FG, fill=BLOCK_FILL, width=1.15, height=0.48, font_size=TINY_SIZE):
    box = RoundedRectangle(
        corner_radius=0.08,
        width=width,
        height=height,
        fill_color=fill,
        fill_opacity=1.0,
        stroke_color=BLOCK_STROKE,
        stroke_width=2,
    )
    txt = text(label, font_size=font_size, color=color)
    if txt.width > width * 0.82:
        txt.scale_to_fit_width(width * 0.82)
    txt.move_to(box.get_center())
    return VGroup(box, txt)


def _core_grid(cols=8, rows=6, cell=0.26, fill=ACCENT2, opacity=0.28):
    cores = VGroup()
    for _ in range(rows * cols):
        sq = Square(
            side_length=cell,
            fill_color=fill,
            fill_opacity=opacity,
            stroke_width=0.5,
            stroke_color=BG,
        )
        cores.add(sq)
    cores.arrange_in_grid(rows=rows, cols=cols, buff=cell * 0.22)
    return cores


def _gpu_chip(cores):
    chip = RoundedRectangle(
        corner_radius=0.1,
        width=cores.width + 0.38,
        height=cores.height + 0.38,
        fill_color=BLOCK_FILL,
        fill_opacity=1.0,
        stroke_color=BLOCK_STROKE,
        stroke_width=2,
    )
    chip.move_to(cores.get_center())
    cores.set_z_index(1)
    return VGroup(chip, cores)


def _vram_tank(width=1.55, height=3.35):
    outline = RoundedRectangle(
        corner_radius=0.1,
        width=width,
        height=height,
        stroke_color=BLOCK_STROKE,
        stroke_width=3,
        fill_opacity=0.0,
    )
    pad = 0.14
    inner_w = width - pad
    inner_h = height - pad
    fill = RoundedRectangle(
        corner_radius=0.08,
        width=inner_w,
        height=max(inner_h * WEIGHTS_FRAC, 0.05),
        fill_color=ACCENT,
        fill_opacity=0.92,
        stroke_width=0,
    )
    label = text("VRAM", font_size=TINY_SIZE, color=MUTED)
    group = VGroup(outline, fill, label)
    group.outline = outline
    group.fill = fill
    group.caption = label
    group.inner_w = inner_w
    group.inner_h = inner_h
    group.pad = pad
    return group


def _place_tank_fill(tank):
    """Sit the pre-sized weights fill on the bottom of the tank interior."""
    outline = tank.outline
    fill = tank.fill
    bottom = outline.get_bottom()[1] + tank.pad / 2
    fill.move_to([outline.get_center()[0], bottom + fill.height / 2, 0])
    tank.caption.next_to(outline, DOWN, buff=0.12)


def _matmul_pipeline():
    """One x → W → y row. x, W box, and y share a y so both arrows are horizontal."""
    x_in = _chip("x", width=0.95, height=0.7, font_size=SMALL_SIZE)
    w_box = RoundedRectangle(
        corner_radius=0.1,
        width=2.6,
        height=1.55,
        fill_color=ACCENT,
        fill_opacity=0.88,
        stroke_color=BLOCK_STROKE,
        stroke_width=2,
    )
    w_lab = small("W", font_size=SMALL_SIZE, color=FG)
    y_out = _chip("y", width=0.95, height=0.7, font_size=SMALL_SIZE, fill=GOOD, color=BG)

    x_in.move_to(LEFT * 2.2)
    w_box.move_to(ORIGIN)
    y_out.move_to(RIGHT * 2.2)
    w_lab.next_to(w_box, UP, buff=0.14)

    a_in = arrow(x_in.get_right(), w_box.get_left(), buff=0.06)
    a_out = arrow(w_box.get_right(), y_out.get_left(), buff=0.06)
    row = VGroup(x_in, a_in, w_box, w_lab, a_out, y_out)
    row.x_in = x_in
    row.w_box = w_box
    row.w_lab = w_lab
    row.y_out = y_out
    row.a_in = a_in
    row.a_out = a_out
    return row


def _weights_block(width=2.5, height=1.9):
    box = RoundedRectangle(
        corner_radius=0.12,
        width=width,
        height=height,
        fill_color=ACCENT,
        fill_opacity=0.88,
        stroke_color=BLOCK_STROKE,
        stroke_width=2,
    )
    label = small("model weights  (same W)", font_size=SMALL_SIZE, color=FG)
    label.next_to(box, UP, buff=0.18)
    hub = Circle(
        radius=0.14,
        color=FG,
        fill_color=BG,
        fill_opacity=1.0,
        stroke_width=2.5,
    )
    hub.move_to(box.get_center())
    hub.set_z_index(5)
    group = VGroup(box, label, hub)
    group.box = box
    group.label = label
    group.hub = hub
    return group


class S2GPU(Slide):
    def construct(self):
        apply_theme(self)

        # -------------------------------------------------------------
        # Beat 1 — the unit of work is a giant matmul
        # -------------------------------------------------------------
        heading = title("The same multiply, over and over")
        self.play(FadeIn(heading))

        pipeline = _matmul_pipeline()
        pipeline.shift(DOWN * 0.1)

        self.play(FadeIn(pipeline.x_in))
        self.play(FadeIn(pipeline.w_box), FadeIn(pipeline.w_lab))
        self.play(shoot(pipeline.a_in))
        self.play(shoot(pipeline.a_out), FadeIn(pipeline.y_out))

        copies = VGroup(*[_matmul_pipeline() for _ in range(6)])
        copies.scale(0.42)
        copies.arrange_in_grid(rows=2, cols=3, buff=0.38)
        copies.move_to(DOWN * 0.5)

        many_cap = caption("Same op, many times — the shape a GPU is built for")
        many_cap.next_to(copies, DOWN, buff=0.32)

        others = VGroup(*[copies[i] for i in range(6) if i != 1])
        self.play(
            ReplacementTransform(pipeline, copies[1]),
            FadeIn(others, lag_ratio=0.08),
            run_time=0.9,
        )
        self.play(FadeIn(many_cap))
        self.wait(0.3)
        self.next_slide()

        self.play(FadeOut(heading), FadeOut(copies), FadeOut(many_cap))

        # -------------------------------------------------------------
        # Beat 2 — CPU sequential vs GPU parallel
        # -------------------------------------------------------------
        heading2 = title("CPU vs GPU")
        self.play(FadeIn(heading2))

        cpu_lab = small("CPU: a few big cores", color=MUTED)
        cpu_cores = VGroup(*[
            RoundedRectangle(
                corner_radius=0.08,
                width=1.05,
                height=1.05,
                fill_color=BLOCK_FILL,
                fill_opacity=1.0,
                stroke_color=BLOCK_STROKE,
                stroke_width=2,
            )
            for _ in range(4)
        ])
        cpu_cores.arrange_in_grid(rows=2, cols=2, buff=0.22)
        cpu_lab.next_to(cpu_cores, UP, buff=0.28)
        cpu_full = VGroup(cpu_lab, cpu_cores)
        cpu_full.to_edge(LEFT, buff=1.15).shift(DOWN * 0.15)

        gpu_lab = small("GPU: thousands of tiny cores", color=MUTED)
        gpu_cores = _core_grid()
        gpu = _gpu_chip(gpu_cores)
        gpu_lab.next_to(gpu, UP, buff=0.28)
        gpu_full = VGroup(gpu_lab, gpu)
        gpu_full.to_edge(RIGHT, buff=0.7).shift(DOWN * 0.15)

        map_cap = caption("Identical work: a few at a time  vs  all at once")
        map_cap.to_edge(DOWN, buff=0.45)

        self.play(FadeIn(cpu_full), FadeIn(gpu_full))
        self.play(FadeIn(map_cap))
        for core in cpu_cores:
            self.play(core.animate.set_fill(ACCENT, opacity=0.9), run_time=0.28)
        self.play(
            AnimationGroup(
                *[c.animate.set_fill(ACCENT, opacity=0.9) for c in gpu_cores],
                lag_ratio=0.0,
            ),
            run_time=0.45,
        )
        self.wait(0.3)
        self.next_slide()

        self.play(FadeOut(heading2), FadeOut(cpu_full), FadeOut(map_cap))

        # GPU chip stays. Center it in a CPU | PCIe | GPU | VRAM row.
        gpu_keep = gpu
        gpu_keep_lab = small("GPU cores", color=MUTED)

        # -------------------------------------------------------------
        # Beat 3 — two memory pools
        # -------------------------------------------------------------
        heading3 = title("Two memory pools — VRAM is not RAM")

        band_h = gpu_keep.height
        cpu_chip = RoundedRectangle(
            corner_radius=0.1,
            width=2.15,
            height=band_h,
            fill_color=BLOCK_FILL,
            fill_opacity=1.0,
            stroke_color=BLOCK_STROKE,
            stroke_width=2,
        )
        cpu_chip_lab = small("CPU", font_size=SMALL_SIZE, color=FG)
        cpu_chip_lab.move_to(cpu_chip.get_center())
        dram = Rectangle(
            width=0.72,
            height=band_h,
            stroke_color=BLOCK_STROKE,
            stroke_width=2,
            fill_color=MUTED,
            fill_opacity=0.35,
        )
        dram_lab = text("DRAM", font_size=TINY_SIZE, color=MUTED)
        dram.next_to(cpu_chip, RIGHT, buff=0.18)
        dram.match_y(cpu_chip)
        dram_lab.next_to(dram, DOWN, buff=0.12)
        host = VGroup(cpu_chip, cpu_chip_lab, dram, dram_lab)

        tank = _vram_tank()

        mid_y = -0.2
        pcie_gap = 1.85
        gpu_tank_buff = 0.5
        total_w = host.width + pcie_gap + gpu_keep.width + gpu_tank_buff + tank.outline.width
        left_x = -total_w / 2

        host.move_to([left_x + host.width / 2, 0, 0])
        host.shift(UP * (mid_y - cpu_chip.get_center()[1]))

        gpu_x = left_x + host.width + pcie_gap + gpu_keep.width / 2
        gpu_target = [gpu_x, mid_y, 0]

        tank_x = gpu_x + gpu_keep.width / 2 + gpu_tank_buff + tank.outline.width / 2
        tank.move_to([tank_x, 0, 0])
        tank.shift(UP * (mid_y - tank.outline.get_center()[1]))
        _place_tank_fill(tank)

        pcie_start = [dram.get_right()[0] + 0.12, mid_y, 0]
        pcie_end = [gpu_x - gpu_keep.width / 2 - 0.12, mid_y, 0]
        pcie = DashedLine(
            pcie_start,
            pcie_end,
            color=MUTED,
            dash_length=0.12,
            stroke_width=2,
        )
        pcie_lab = caption("PCIe  (slow)")
        pcie_lab.move_to([
            (pcie_start[0] + pcie_end[0]) / 2,
            mid_y + 0.32,
            0,
        ])

        gpu_keep_lab.move_to([
            gpu_x,
            mid_y + gpu_keep.height / 2 + 0.22 + gpu_keep_lab.height / 2,
            0,
        ])

        pool_cap = caption("Cores can only multiply data already in VRAM")
        pool_cap.to_edge(DOWN, buff=0.4)

        self.play(FadeOut(gpu_lab), FadeIn(heading3))
        self.play(gpu_keep.animate.move_to(gpu_target), FadeIn(gpu_keep_lab))
        self.play(FadeIn(host))
        self.play(FadeIn(pcie), FadeIn(pcie_lab))
        self.play(FadeIn(tank.outline), FadeIn(tank.caption))
        self.play(FadeIn(pool_cap))
        self.wait(0.3)
        self.next_slide()

        # -------------------------------------------------------------
        # Beat 4 — weights persist in VRAM
        # -------------------------------------------------------------
        heading4 = title("Weights sit in VRAM for the whole serve")
        self.play(FadeOut(heading3), FadeIn(heading4))
        self.play(FadeOut(pool_cap))

        self.play(GrowFromEdge(tank.fill, DOWN))

        w_on_fill = text("26 GB", font_size=SMALL_SIZE, color=BG)
        if w_on_fill.width > tank.fill.width * 0.85:
            w_on_fill.scale_to_fit_width(tank.fill.width * 0.85)
        w_on_fill.move_to(tank.fill.get_center() + UP * 0.12)
        w_sub = text("weights", font_size=TINY_SIZE, color=BG)
        w_sub.next_to(w_on_fill, DOWN, buff=0.05)
        tank.add(w_on_fill, w_sub)

        fact = small("OPT-13B on an A100 40 GB  ·  ~65% gone before any request", color=ACCENT)
        fact.to_edge(DOWN, buff=0.42)

        self.play(FadeIn(w_on_fill), FadeIn(w_sub))
        self.play(FadeIn(fact))
        self.wait(0.3)
        self.next_slide()

        # -------------------------------------------------------------
        # Beat 5 — one request vs 26 GB
        # -------------------------------------------------------------
        heading5 = title("One request vs 26 GB of weights")
        self.play(FadeOut(heading4), FadeIn(heading5))
        self.play(FadeOut(fact))

        # Host / PCIe have made their point; keep GPU + VRAM.
        self.play(
            FadeOut(host),
            FadeOut(pcie),
            FadeOut(pcie_lab),
            FadeOut(gpu_keep_lab),
        )
        self.play(
            gpu_keep.animate.move_to(LEFT * 1.35 + DOWN * 0.05),
            tank.animate.move_to(RIGHT * 3.55 + DOWN * 0.05),
        )

        req1 = _chip("req 1", width=1.2, height=0.5)
        req1.to_edge(LEFT, buff=0.55).shift(DOWN * 0.05)
        out1 = _chip("result", width=1.2, height=0.5, fill=GOOD, color=BG)
        out1.next_to(gpu_keep, DOWN, buff=0.42)

        a_req = arrow(req1.get_right(), gpu_keep.get_left(), buff=0.12)
        a_res = arrow(gpu_keep.get_bottom(), out1.get_top(), buff=0.1)

        idle_cap = caption("Cores wait on memory — one request is too little math")
        idle_cap.to_edge(DOWN, buff=0.42)

        self.play(FadeIn(req1))
        self.play(shoot(a_req))
        self.play(
            AnimationGroup(
                *[c.animate.set_fill(MUTED, opacity=0.22) for c in gpu_cores],
                lag_ratio=0.0,
            ),
            run_time=0.5,
        )
        self.play(shoot(a_res), FadeIn(out1))
        self.play(FadeIn(idle_cap))
        self.wait(0.3)
        self.next_slide()

        # -------------------------------------------------------------
        # Beat 6 — batching through one hub
        # -------------------------------------------------------------
        heading6 = title("Batching: many requests, one pass")
        self.play(FadeOut(heading5), FadeIn(heading6))
        self.play(
            FadeOut(idle_cap),
            FadeOut(req1),
            FadeOut(out1),
            FadeOut(a_req),
            FadeOut(a_res),
        )

        # Park GPU + VRAM on the far right as a reminder.
        self.play(
            gpu_keep.animate.scale(0.52).move_to(RIGHT * 5.9 + UP * 1.15),
            tank.animate.scale(0.52).move_to(RIGHT * 5.9 + DOWN * 1.2),
        )

        w_block = _weights_block()
        w_block.move_to(LEFT * 0.15 + DOWN * 0.25)
        hub_pt = w_block.hub.get_center()

        n = 5
        reqs = VGroup(*[_chip(f"req {i + 1}", width=1.2, height=0.46) for i in range(n)])
        reqs.arrange(DOWN, buff=0.16)
        reqs.to_edge(LEFT, buff=0.55).shift(DOWN * 0.2)

        outs = VGroup(*[
            _chip("result", width=1.15, height=0.46, fill=GOOD, color=BG)
            for _ in range(n)
        ])
        outs.arrange(DOWN, buff=0.16)
        outs.move_to(RIGHT * 3.05 + DOWN * 0.2)

        self.play(FadeIn(w_block.label), FadeIn(w_block.box), FadeIn(w_block.hub))
        self.play(FadeIn(reqs, lag_ratio=0.06))

        arrows_in = VGroup(*[
            arrow(reqs[i].get_right(), hub_pt, buff=0.18, color=MUTED)
            for i in range(n)
        ])
        for a in arrows_in:
            a.set_z_index(4)
        self.play(AnimationGroup(*[shoot(a) for a in arrows_in], lag_ratio=0.05))

        arrows_out = VGroup(*[
            arrow(hub_pt, outs[i].get_left(), buff=0.18, color=MUTED)
            for i in range(n)
        ])
        for a in arrows_out:
            a.set_z_index(4)
        self.play(
            FadeIn(outs, lag_ratio=0.05),
            AnimationGroup(*[shoot(a) for a in arrows_out], lag_ratio=0.05),
        )
        self.play(
            AnimationGroup(
                *[c.animate.set_fill(ACCENT, opacity=0.9) for c in gpu_cores],
                lag_ratio=0.0,
            ),
            run_time=0.4,
        )

        batch_cap = caption("N requests through one point on W  ·  one load")
        batch_cap.to_edge(DOWN, buff=0.4)
        self.play(FadeIn(batch_cap))
        self.wait(0.3)
        self.next_slide()

        # -------------------------------------------------------------
        # Beat 7 — leftover VRAM is the serving budget
        # -------------------------------------------------------------
        heading7 = title("Leftover VRAM is the serving budget")
        batch_bits = VGroup(
            w_block, reqs, outs, arrows_in, arrows_out, batch_cap,
        )
        self.play(FadeOut(heading6), FadeIn(heading7), FadeOut(batch_bits))

        # Bring GPU + tank back to a readable size, tank as the focus.
        self.play(
            gpu_keep.animate.scale(1.85).move_to(LEFT * 2.7 + DOWN * 0.1),
            tank.animate.scale(2.05).move_to(RIGHT * 1.45 + DOWN * 0.05),
        )

        leftover_h = tank.fill.height * (1.0 - WEIGHTS_FRAC) / WEIGHTS_FRAC
        leftover = RoundedRectangle(
            corner_radius=0.08,
            width=tank.fill.width,
            height=max(leftover_h, 0.08),
            fill_color=GOOD,
            fill_opacity=0.18,
            stroke_color=GOOD,
            stroke_width=1.5,
        )
        leftover.next_to(tank.fill, UP, buff=0.04)
        leftover_lab = text("leftover", font_size=TINY_SIZE, color=GOOD)
        leftover_lab.move_to(leftover.get_center())
        if leftover_lab.height > leftover.height * 0.7:
            leftover_lab.scale_to_fit_height(leftover.height * 0.65)

        n_fit, n_bounce = 3, 2
        fit_chips = VGroup(*[
            _chip(f"req {i + 1}", width=1.05, height=0.42)
            for i in range(n_fit)
        ])
        fit_chips.arrange(DOWN, buff=0.1)
        bounce_chips = VGroup(*[
            _chip(f"req {i + 1}", width=1.05, height=0.42)
            for i in range(n_fit, n_fit + n_bounce)
        ])
        bounce_chips.arrange(DOWN, buff=0.1)
        all_try = VGroup(fit_chips, bounce_chips).arrange(DOWN, buff=0.1)
        all_try.next_to(tank.outline, RIGHT, buff=1.35)
        parked = all_try.get_center().copy()
        all_try.shift(RIGHT * 1.6)

        budget_cap = caption("Weights already took 26 GB  ·  leftover decides the batch")
        budget_cap.to_edge(DOWN, buff=0.4)

        self.play(FadeIn(leftover), FadeIn(leftover_lab))
        self.play(all_try.animate.move_to(parked), run_time=0.5)
        self.play(fit_chips.animate.next_to(leftover, RIGHT, buff=0.28), run_time=0.4)
        self.play(bounce_chips.animate.shift(RIGHT * 0.45), run_time=0.2)
        x_marks = VGroup()
        for ch in bounce_chips:
            x = text("X", font_size=SMALL_SIZE, color=BAD)
            x.next_to(ch, RIGHT, buff=0.12)
            x_marks.add(x)
        self.play(FadeIn(x_marks))
        for ch in fit_chips:
            ch[0].set_stroke(GOOD, width=2)
        self.play(FadeIn(budget_cap))
        self.wait(0.3)
        self.next_slide()

        # -------------------------------------------------------------
        # Beat 8 — landing
        # -------------------------------------------------------------
        heading8 = title("Leftover VRAM  →  max batch  →  throughput")
        self.play(FadeOut(heading7), FadeIn(heading8))
        self.play(
            FadeOut(bounce_chips),
            FadeOut(x_marks),
            FadeOut(budget_cap),
        )

        land_cap = caption("That leftover slice is the resource this talk is about")
        land_cap.to_edge(DOWN, buff=0.55)
        self.play(FadeIn(land_cap))
        self.wait(0.3)
        self.next_slide()
