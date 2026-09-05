"""S2 — Why GPUs (Act I).

NARRATION
---------
Beat 1 — Same math, everywhere.
Every decoding step we just saw is, under the hood, the same handful of
matrix multiplies applied over and over: one token in, one token's worth of
math against every weight matrix in the model. Now imagine that happening
for every layer, and — once we start batching requests — for many tokens at
once. It's not complicated math. It's just an enormous amount of *identical*
math, fanned out over and over. [PAUSE] That "same operation, many times"
shape is exactly what a GPU is built for.

Beat 2 — CPU vs GPU.
A CPU has a handful of big, fast cores — great at doing one complicated
thing quickly, one after another. A GPU flips that trade: thousands of
small, simple cores. Give it one huge matrix multiply and it slices the
work into thousands of tiny pieces and runs them all at the same time, one
piece per core. Our "do the same multiply-and-add for every token" workload
maps almost perfectly onto that grid of cores. [PAUSE]

Beat 3 — Batching amortizes the weights.
Here's the trick that makes serving efficient: the model's weights are
identical for every request — request 1, request 2, request N all multiply
against the exact same matrices. So instead of loading those weights once
per request, we load them once and run many requests' tokens through them
in the same pass. Throughput becomes a question of how many requests we can
batch together into one pass over the same weights. More batching, more
throughput — for free, almost.

Beat 4 — The catch: VRAM is limited.
Almost for free. Here's the catch: the GPU doesn't borrow the computer's
regular memory — it has its own memory, called VRAM, and it's finite. An
A100 GPU, for example, has 40GB of it. Anything the model touches while it
runs — the weights, the intermediate activations, and any per-request state
we want to keep around — has to fit inside that 40GB. The weights alone for
a 13-billion-parameter model like OPT-13B are about 26GB. That's already
almost two-thirds of an A100's memory, gone, before we've served a single
request. [PAUSE]

Beat 5 — Landing.
So: weights take a big, fixed bite out of VRAM the moment the model loads.
Whatever is left over is what we have to work with for everything else —
and that leftover space is what decides how many requests we can actually
batch together at once.
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
    RoundedRectangle,
    VGroup,
)
from manim_slides import Slide

from talk.theme import *
from talk.components import *


class S2GPU(Slide):
    def construct(self):
        apply_theme(self)

        # -------------------------------------------------------------
        # Beat 1: recap — many identical small operations fanning out
        # -------------------------------------------------------------
        heading = title("One step, the same math — many times")
        sub = caption("Every generated token repeats the same matrix multiplies")
        sub.next_to(heading, DOWN, buff=0.25)

        seed_box = RoundedRectangle(
            corner_radius=0.08, width=1.6, height=0.8,
            fill_color=BLOCK_FILL, fill_opacity=1.0, stroke_color=BLOCK_STROKE, stroke_width=2,
        )
        seed_label = small("A x W", font_size=SMALL_SIZE)
        seed_label.move_to(seed_box.get_center())
        seed_group = VGroup(seed_box, seed_label)
        seed_group.move_to(UP * 1.6)

        n_cols, n_rows = 8, 3
        copies = VGroup()
        for r in range(n_rows):
            for c in range(n_cols):
                box = seed_group.copy()
                box.scale(0.42)
                copies.add(box)
        copies.arrange_in_grid(rows=n_rows, cols=n_cols, buff=0.28)
        copies.move_to(DOWN * 0.7)

        fan_caption = caption("Same multiply-and-add, once per token, per layer")
        fan_caption.next_to(copies, DOWN, buff=0.4)

        self.play(FadeIn(heading), FadeIn(sub))
        self.play(FadeIn(seed_group))
        self.play(FadeOut(seed_group), FadeIn(copies, lag_ratio=0.02))
        self.play(FadeIn(fan_caption))
        self.wait(0.3)
        self.next_slide()
        self.play(FadeOut(heading), FadeOut(sub), FadeOut(copies), FadeOut(fan_caption))

        # -------------------------------------------------------------
        # Beat 2: CPU vs GPU — few big cores vs thousands of tiny cores,
        # the fanned-out work mapping onto the GPU's cores.
        # -------------------------------------------------------------
        heading2 = title("CPU vs GPU")

        cpu_label = small("CPU: a few big cores", font_size=SMALL_SIZE, color=MUTED)
        cpu_cores = VGroup(*[
            RoundedRectangle(corner_radius=0.08, width=1.1, height=1.1,
                              fill_color=BLOCK_FILL, fill_opacity=1.0,
                              stroke_color=BLOCK_STROKE, stroke_width=2)
            for _ in range(4)
        ])
        cpu_cores.arrange_in_grid(rows=2, cols=2, buff=0.25)
        cpu_label.next_to(cpu_cores, UP, buff=0.3)
        cpu_full = VGroup(cpu_label, cpu_cores)
        cpu_full.to_edge(LEFT, buff=1.3).shift(DOWN * 0.2)

        gpu_label = small("GPU: thousands of tiny cores", font_size=SMALL_SIZE, color=MUTED)
        gpu = GPUSchematic(cores=(10, 8), width=6.0)
        gpu_label.next_to(gpu, UP, buff=0.3)
        gpu_full = VGroup(gpu_label, gpu)
        gpu_full.to_edge(RIGHT, buff=0.6).shift(DOWN * 0.2)

        map_caption = caption("The fanned-out work lands one piece per core")
        map_caption.next_to(VGroup(cpu_full, gpu_full), DOWN, buff=0.6)

        self.play(FadeIn(heading2))
        self.play(FadeIn(cpu_full), FadeIn(gpu_full))
        self.play(FadeIn(map_caption))
        self.play(gpu.cores.animate.set_fill(ACCENT, opacity=0.9))
        self.wait(0.3)
        self.next_slide()
        self.play(FadeOut(heading2), FadeOut(cpu_full), FadeOut(gpu_full), FadeOut(map_caption))

        # -------------------------------------------------------------
        # Beat 3: batching amortizes the weights
        # -------------------------------------------------------------
        heading3 = title("Batching: many requests, one pass")

        weights_box = RoundedRectangle(
            corner_radius=0.1, width=3.4, height=1.6,
            fill_color=ACCENT, fill_opacity=0.85, stroke_color=BLOCK_STROKE, stroke_width=2,
        )
        weights_label = small("model weights\n(same for everyone)", font_size=SMALL_SIZE, color=BG)
        weights_label.move_to(weights_box.get_center())
        weights_group = VGroup(weights_box, weights_label)

        n_requests = 5
        req_boxes = VGroup(*[
            TokenBox(f"req {i + 1}", color=FG, fill=BLOCK_FILL, height=0.55, font_size=SMALL_SIZE)
            for i in range(n_requests)
        ])
        req_boxes.arrange(DOWN, buff=0.22)
        req_boxes.to_edge(LEFT, buff=1.2)

        out_boxes = VGroup(*[
            TokenBox("token", color=BG, fill=GOOD, height=0.55, font_size=SMALL_SIZE)
            for _ in range(n_requests)
        ])
        out_boxes.arrange(DOWN, buff=0.22)
        out_boxes.to_edge(RIGHT, buff=1.2)

        weights_group.next_to(req_boxes, RIGHT, buff=1.4)
        out_boxes.next_to(weights_group, RIGHT, buff=1.4)

        batch_caption = caption("N requests through the same weights = N tokens for ~1 load")
        batch_caption.next_to(VGroup(req_boxes, weights_group, out_boxes), DOWN, buff=0.6)

        # Arrows land on the weights box's left/right edges (clipped to its
        # height), not its center, so they never cross the label text.
        half_h = weights_box.height / 2 * 0.85

        def edge_point(box, side, y):
            y_clipped = max(min(y, box.get_center()[1] + half_h), box.get_center()[1] - half_h)
            x = box.get_left()[0] if side == "left" else box.get_right()[0]
            return [x, y_clipped, 0]

        self.play(FadeIn(heading3))
        self.play(FadeIn(req_boxes))
        self.play(FadeIn(weights_group))
        arrows_in = VGroup(*[
            Arrow(
                rb.get_right(), edge_point(weights_box, "left", rb.get_y()),
                buff=0.08, stroke_width=2.5, color=MUTED, max_tip_length_to_length_ratio=0.15,
            )
            for rb in req_boxes
        ])
        self.play(AnimationGroup(*[GrowFromEdge(a, LEFT) for a in arrows_in], lag_ratio=0.05))
        self.play(FadeIn(out_boxes))
        arrows_out = VGroup(*[
            Arrow(
                edge_point(weights_box, "right", ob.get_y()), ob.get_left(),
                buff=0.08, stroke_width=2.5, color=MUTED, max_tip_length_to_length_ratio=0.15,
            )
            for ob in out_boxes
        ])
        self.play(AnimationGroup(*[GrowFromEdge(a, LEFT) for a in arrows_out], lag_ratio=0.05))
        self.play(FadeIn(batch_caption))
        self.wait(0.3)
        self.next_slide()
        self.play(
            FadeOut(heading3), FadeOut(req_boxes), FadeOut(weights_group), FadeOut(out_boxes),
            FadeOut(arrows_in), FadeOut(arrows_out), FadeOut(batch_caption),
        )

        # -------------------------------------------------------------
        # Beat 4: the catch — VRAM is limited, weights already fill most of it
        # -------------------------------------------------------------
        heading4 = title("The catch: the GPU has its own memory")

        gpu2 = GPUSchematic(cores=(10, 8), width=7.0)
        gpu2.move_to(ORIGIN).shift(DOWN * 0.3)
        vram_caption = caption("VRAM: finite, and everything the model touches must live here")
        vram_caption.next_to(gpu2, UP, buff=0.35)
        fact = small("OPT-13B weights ≈ 26 GB on an A100 40 GB", font_size=SMALL_SIZE, color=ACCENT)
        fact.next_to(gpu2, DOWN, buff=0.5)

        self.play(FadeIn(heading4))
        self.play(FadeIn(gpu2), FadeIn(vram_caption))
        self.play(FadeIn(fact))
        self.play(gpu2.fill_vram(26 / 40, color=ACCENT))
        self.wait(0.3)
        self.next_slide()
        self.play(FadeOut(heading4), FadeOut(gpu2), FadeOut(vram_caption), FadeOut(fact))

        # -------------------------------------------------------------
        # Beat 5: landing
        # -------------------------------------------------------------
        landing = title("What's left over decides how many requests we can batch")
        landing.scale(0.85)
        landing.move_to(ORIGIN)
        self.play(FadeIn(landing))
        self.wait(0.3)
        self.next_slide()
        self.play(FadeOut(landing))
        self.wait(0.3)
