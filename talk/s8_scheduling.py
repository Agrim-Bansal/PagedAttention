"""S8 — Scheduling under memory pressure (Act III).

NARRATION
---------
Beat 1 — The vLLM system overview (Fig 4).
Zoom out to how vLLM is actually organized. A central Scheduler decides which
requests run in each step. It talks to a KV Cache Manager, which is the one
place that owns every block table in the system — logical block to physical
block, for every request. Underneath, a CPU Block Allocator and a GPU Block
Allocator hand out physical blocks on each side. And on the right, a row of
Workers — Worker 0 through Worker N-1 — each on its own GPU, each running a
shard of the model plus a Cache Engine that moves blocks around exactly when
the scheduler tells it to. [PAUSE] One brain, many hands.

Beat 2 — Running out of room.
In normal operation this is boring: requests arrive, the scheduler admits
them, blocks get allocated on demand as each one generates tokens, and the
batch grows. But GPU memory is finite, so eventually the free block pool runs
dry — some request's next token needs a new block, and there isn't one.
[PAUSE] Which request gives up its memory, and what do we do with it?

Beat 3 — Policy: first-come, first-served.
vLLM's answer is deliberately simple: first-come, first-served, with
preemption. Requests are served oldest-first; when the scheduler must free
space, it preempts the most recently arrived request first. That guarantees
no request starves waiting behind an endless stream of newer arrivals — the
one that's been running longest is the last one ever kicked out.

Beat 4 — All-or-nothing eviction.
Here's a detail that only makes sense once you remember how attention works:
every generated token reads every block of a sequence, every single step. So
evicting half a sequence's blocks buys you nothing — the sequence still can't
run without the other half. vLLM evicts a sequence's blocks all at once, and
if several sequences share blocks — say, all the beams of one request — the
whole gang goes together. An OS, by contrast, happily evicts individual
pages one at a time, because a process only touches a handful of them per
instruction.

Beat 5 — Recovery option 1: swapping.
Once a sequence is preempted, vLLM needs to get its blocks out of the way
without losing them, and there are two ways to do that. The first is
swapping: the CPU block allocator copies the evicted blocks into ordinary CPU
RAM, over PCIe, and swaps them back in later when the request is
rescheduled. The cost is PCIe bandwidth. And notice the swap space can never
grow unbounded — it's capped by exactly the GPU's total KV block capacity,
since that's the most that could ever be evicted at once.

Beat 6 — Recovery option 2: recomputation.
The second option is more radical: just drop the evicted blocks entirely.
When the request is rescheduled, concatenate its original prompt with all
the tokens it had already generated, and treat that whole thing as one new
prompt — run a single prefill pass over it. That rebuilds the entire KV
cache in one parallel pass instead of one slow token at a time, so it's
often cheaper than it sounds.

Beat 7 — So which one wins? [PAUSE]
It depends on block size. In the paper's microbenchmark, recomputation's
overhead is essentially flat no matter the block size — it never touches a
KV block. Swapping is expensive at small block sizes, because it means many
tiny PCIe transfers, but gets cheaper as blocks get larger and transfers get
bigger. They cross over somewhere in the medium range, roughly block size 16
to 64, where end-to-end performance is comparable either way. These numbers
are approximate, read off the paper's Figure 19.

Beat 8 — Landing.
Put together, this is the payoff of paging: preemption, all-or-nothing
eviction, swap, recompute — none of that exists in a system where memory is
one fixed contiguous slab per request. Paging gives you a knob the
contiguous systems never had: you can take memory back, and give it back
later.
"""

from manim import (
    DOWN,
    LEFT,
    ORIGIN,
    RIGHT,
    UP,
    AnimationGroup,
    Cross,
    FadeIn,
    FadeOut,
    Rectangle,
    RoundedRectangle,
    Square,
    VGroup,
)
from manim_slides import Slide

from talk.theme import *
from talk.components import *


def _box(label, width=2.4, height=0.9, fill=BLOCK_FILL, stroke=BLOCK_STROKE, text_color=FG, font_size=SMALL_SIZE):
    """A labeled RoundedRectangle. Returns VGroup with .box, .label."""
    b = RoundedRectangle(
        corner_radius=0.1, width=width, height=height,
        fill_color=fill, fill_opacity=1.0, stroke_color=stroke, stroke_width=2,
    )
    t = text(label, font_size=font_size, color=text_color)
    if t.width > width * 0.9:
        t.scale_to_fit_width(width * 0.9)
    t.move_to(b.get_center())
    g = VGroup(b, t)
    g.box = b
    g.label = t
    return g


class S8Scheduling(Slide):
    def construct(self):
        apply_theme(self)

        # ==========================================================
        # Beat 1: Fig 4 — vLLM system overview
        # ==========================================================
        heading = title("The vLLM system overview")
        self.play(FadeIn(heading))

        scheduler = _box("Scheduler", width=2.6, height=0.9, fill=ACCENT, text_color=BG)
        scheduler.move_to(UP * 2.0 + LEFT * 4.8)
        self.play(FadeIn(scheduler))

        kv_mgr = _box("KV Cache Manager\n(block tables)", width=3.2, height=1.2, fill=BLOCK_FILL)
        kv_mgr.move_to(LEFT * 4.8 + UP * 0.2)
        sched_kv_arrow = arrow_map(scheduler, kv_mgr, color=MUTED)

        cpu_alloc = _box("CPU Block\nAllocator", width=2.1, height=0.85, font_size=TINY_SIZE)
        gpu_alloc = _box("GPU Block\nAllocator", width=2.1, height=0.85, font_size=TINY_SIZE)
        cpu_alloc.next_to(kv_mgr, DOWN, buff=0.55).shift(LEFT * 1.15)
        gpu_alloc.next_to(kv_mgr, DOWN, buff=0.55).shift(RIGHT * 1.15)
        kv_cpu_arrow = arrow_map(kv_mgr, cpu_alloc, color=MUTED)
        kv_gpu_arrow = arrow_map(kv_mgr, gpu_alloc, color=MUTED)

        self.play(
            FadeIn(kv_mgr), shoot(sched_kv_arrow),
        )
        self.play(
            FadeIn(cpu_alloc), FadeIn(gpu_alloc),
            shoot(kv_cpu_arrow), shoot(kv_gpu_arrow),
        )

        worker_names = ["Worker 0", "Worker 1", "...", "Worker N-1"]
        worker_xs = [0.9, 2.5, 4.0, 5.9]
        workers = VGroup()
        worker_arrows = VGroup()
        for name, x in zip(worker_names, worker_xs):
            if name == "...":
                dots = text("...", font_size=BODY_SIZE, color=MUTED)
                dots.move_to(RIGHT * x + UP * 0.3)
                workers.add(dots)
                continue
            outer = RoundedRectangle(
                corner_radius=0.12, width=1.5, height=2.4,
                fill_color=BLOCK_FILL, fill_opacity=1.0, stroke_color=ACCENT2, stroke_width=2,
            )
            outer.move_to(RIGHT * x + UP * 0.3)
            wname = text(name, font_size=TINY_SIZE, color=ACCENT2)
            wname.next_to(outer, UP, buff=0.15)
            cache_engine = _box("Cache\nEngine", width=1.25, height=0.85, font_size=TINY_SIZE)
            model_shard = _box("Model\nShard", width=1.25, height=0.85, font_size=TINY_SIZE)
            cache_engine.move_to(outer.get_center() + UP * 0.55)
            model_shard.move_to(outer.get_center() + DOWN * 0.55)
            wgroup = VGroup(outer, wname, cache_engine, model_shard)
            workers.add(wgroup)
            arrow = arrow_map(scheduler, outer, color=MUTED)
            worker_arrows.add(arrow)

        self.play(FadeIn(workers, lag_ratio=0.15))
        self.play(*[shoot(a) for a in worker_arrows])
        cap = caption("Scheduler decides who runs; KV manager owns every block table; "
                       "workers move blocks on their own GPU")
        cap.to_edge(DOWN, buff=0.35)
        self.play(FadeIn(cap))
        self.wait(0.3)
        self.next_slide()

        fig4_group = VGroup(
            scheduler, kv_mgr, cpu_alloc, gpu_alloc, sched_kv_arrow, kv_cpu_arrow, kv_gpu_arrow,
            workers, worker_arrows, cap,
        )
        self.play(FadeOut(heading), FadeOut(fig4_group))

        # ==========================================================
        # Beat 2: normal operation -> pool exhausted
        # ==========================================================
        heading2 = title("Running out of room")
        self.play(FadeIn(heading2))

        grid = PhysicalMemGrid(n_blocks=6, slots=1, cols=6, cell=0.9, title="Free block pool")
        grid.move_to(UP * 0.6)
        self.play(FadeIn(grid))

        req_colors = [ACCENT, ACCENT2, V_COLOR]
        req_names = ["Req A", "Req B", "Req C"]
        batch_row = VGroup()
        block_idx = 0
        for name, color in zip(req_names, req_colors):
            tb = TokenBox(name, color=BG, fill=color, height=0.55, font_size=SMALL_SIZE)
            batch_row.add(tb)
            anims = []
            for _ in range(2):
                anims.append(grid.block(block_idx).fill_slot(0, "", color=color))
                block_idx += 1
            self.play(*anims)
        batch_row.arrange(RIGHT, buff=0.3)
        batch_row.next_to(grid, DOWN, buff=0.7)
        self.play(FadeIn(batch_row))

        new_token = TokenBox("new token", color=BG, fill=BAD, height=0.55, font_size=SMALL_SIZE)
        new_token.next_to(grid, UP, buff=0.4)
        cross = Cross(scale_factor=0.35, stroke_color=BAD, stroke_width=6)
        cross.move_to(grid.get_center())
        self.play(FadeIn(new_token))
        self.play(FadeIn(cross))
        question = body("Which request gives up its memory,\nand what do we do with it?")
        question.scale(0.85)
        question.next_to(batch_row, DOWN, buff=0.5)
        self.play(FadeIn(question))
        self.wait(0.3)
        self.next_slide()
        self.play(FadeOut(heading2), FadeOut(grid), FadeOut(batch_row), FadeOut(new_token),
                   FadeOut(cross), FadeOut(question))

        # ==========================================================
        # Beat 3: FCFS + preemption
        # ==========================================================
        heading3 = title("Policy: first-come, first-served")
        self.play(FadeIn(heading3))

        arrivals = [("Req A", "t = 0"), ("Req B", "t = 1"), ("Req C", "t = 2")]
        rows = VGroup()
        for name, t in arrivals:
            tb = TokenBox(name, color=FG, fill=BLOCK_FILL, width=2.2, height=0.7, font_size=SMALL_SIZE)
            tlabel = caption(t)
            tlabel.next_to(tb, RIGHT, buff=0.4)
            row = VGroup(tb, tlabel)
            rows.add(row)
        rows.arrange(DOWN, buff=0.45)
        rows.move_to(LEFT * 3.4)
        self.play(FadeIn(rows, lag_ratio=0.2))

        oldest_tag = small("oldest — never starves", font_size=SMALL_SIZE, color=GOOD)
        oldest_tag.next_to(rows[0], RIGHT, buff=0.6)
        newest_tag = small("newest — preempted first", font_size=SMALL_SIZE, color=BAD)
        newest_tag.next_to(rows[-1], RIGHT, buff=0.6)
        self.play(rows[0][0].box.animate.set_stroke(GOOD, width=4), FadeIn(oldest_tag))
        self.play(rows[-1][0].box.animate.set_stroke(BAD, width=4), FadeIn(newest_tag))
        self.wait(0.3)
        self.next_slide()
        self.play(FadeOut(heading3), FadeOut(rows), FadeOut(oldest_tag), FadeOut(newest_tag))

        # ==========================================================
        # Beat 4: all-or-nothing eviction
        # ==========================================================
        heading4 = title("All-or-nothing eviction")
        self.play(FadeIn(heading4))

        left_label = small("vLLM: whole sequence", font_size=SMALL_SIZE, color=ACCENT)
        left_label.move_to(LEFT * 3.6 + UP * 2.2)
        seq_blocks = VGroup(*[
            Square(side_length=0.65, fill_color=V_COLOR, fill_opacity=1.0, stroke_color=BLOCK_STROKE, stroke_width=2)
            for _ in range(4)
        ])
        seq_blocks.arrange(RIGHT, buff=0.15)
        seq_blocks.move_to(LEFT * 3.6 + UP * 1.0)
        req_c_label = caption("Req C's blocks")
        req_c_label.next_to(seq_blocks, DOWN, buff=0.25)

        right_label = small("An OS: one page at a time", font_size=SMALL_SIZE, color=MUTED)
        right_label.move_to(RIGHT * 3.6 + UP * 2.2)
        os_pages = VGroup(*[
            Square(side_length=0.65, fill_color=BLOCK_FILL, fill_opacity=1.0, stroke_color=BLOCK_STROKE, stroke_width=2)
            for _ in range(4)
        ])
        os_pages.arrange(RIGHT, buff=0.15)
        os_pages.move_to(RIGHT * 3.6 + UP * 1.0)
        os_label = caption("only page 2 evicted")
        os_label.next_to(os_pages, DOWN, buff=0.25)

        self.play(
            FadeIn(left_label), FadeIn(seq_blocks), FadeIn(req_c_label),
            FadeIn(right_label), FadeIn(os_pages), FadeIn(os_label),
        )

        useless = caption("useless: every token reads every block")
        useless.next_to(seq_blocks, DOWN, buff=0.9)
        self.play(
            seq_blocks.animate.shift(DOWN * 2.2).set_opacity(0.25),
            FadeIn(useless),
        )
        self.play(os_pages[1].animate.set_fill(BAD, opacity=0.9))
        cross2 = Cross(scale_factor=0.32, stroke_color=BAD, stroke_width=5)
        cross2.move_to(os_pages[1].get_center())
        self.play(FadeIn(cross2))
        self.wait(0.3)
        self.next_slide()
        self.play(FadeOut(heading4), FadeOut(left_label), FadeOut(seq_blocks), FadeOut(req_c_label),
                   FadeOut(useless), FadeOut(right_label), FadeOut(os_pages), FadeOut(os_label),
                   FadeOut(cross2))

        # ==========================================================
        # Beat 5: swapping
        # ==========================================================
        heading5 = title("Recovery option 1: swapping")
        self.play(FadeIn(heading5))

        gpu_box = _box("GPU Block\nAllocator", width=2.6, height=1.1, fill=BLOCK_FILL)
        gpu_box.move_to(LEFT * 3.6 + UP * 0.9)
        cpu_box = _box("CPU Block\nAllocator", width=2.6, height=1.1, fill=BLOCK_FILL)
        cpu_box.move_to(RIGHT * 3.6 + UP * 0.9)
        pcie_arrow = arrow_map(gpu_box, cpu_box, color=WARN)
        pcie_label = caption("over PCIe")
        pcie_label.next_to(pcie_arrow, UP, buff=0.15)

        self.play(FadeIn(gpu_box), FadeIn(cpu_box), shoot(pcie_arrow), FadeIn(pcie_label))

        swap_blocks = VGroup(*[
            Square(side_length=0.5, fill_color=V_COLOR, fill_opacity=1.0, stroke_color=BLOCK_STROKE, stroke_width=2)
            for _ in range(3)
        ])
        swap_blocks.arrange(RIGHT, buff=0.12)
        swap_blocks.move_to(gpu_box.get_center() + DOWN * 1.1)
        self.play(FadeIn(swap_blocks))
        target = swap_blocks.copy().move_to(cpu_box.get_center() + DOWN * 1.1)
        self.play(swap_blocks.animate.move_to(target.get_center()))
        self.wait(0.2)

        bound_cap = caption("swap space bounded by the GPU's total KV block capacity")
        bar_gpu = MemoryBar([("GPU KV space", 1.0, ACCENT2)], width=4.5, height=0.5, show_pct=False)
        bar_cpu = MemoryBar([("CPU swap space", 1.0, WARN)], width=4.5, height=0.5, show_pct=False)
        bars = VGroup(bar_gpu, bar_cpu)
        bars.arrange(DOWN, buff=0.35)
        bars.next_to(VGroup(gpu_box, cpu_box, swap_blocks), DOWN, buff=0.6)
        bound_cap.next_to(bars, UP, buff=0.2)
        self.play(FadeIn(bound_cap), FadeIn(bars))
        self.wait(0.3)
        self.next_slide()
        self.play(FadeOut(heading5), FadeOut(gpu_box), FadeOut(cpu_box), FadeOut(pcie_arrow),
                   FadeOut(pcie_label), FadeOut(swap_blocks), FadeOut(bound_cap), FadeOut(bars))

        # ==========================================================
        # Beat 6: recomputation
        # ==========================================================
        heading6 = title("Recovery option 2: recomputation")
        self.play(FadeIn(heading6))

        prompt_tokens = token_sequence(["Four", "score", "and", "seven"], height=0.5, font_size=SMALL_SIZE)
        gen_tokens = token_sequence(["our", "fathers"], height=0.5, font_size=SMALL_SIZE, fill=ACCENT)
        concat = VGroup(prompt_tokens, gen_tokens)
        concat.arrange(RIGHT, buff=0.25)
        concat.move_to(UP * 1.6)
        concat_label = caption("prompt + already-generated tokens, as one new prompt")
        concat_label.next_to(concat, DOWN, buff=0.25)

        self.play(FadeIn(concat), FadeIn(concat_label))

        prefill_arrow = arrow_map(concat, VGroup(concat).copy().shift(DOWN * 1.6), color=ACCENT)
        prefill_arrow.move_to(concat.get_center() + DOWN * 0.9)
        prefill_label = small("one prefill pass", font_size=SMALL_SIZE, color=ACCENT)
        prefill_label.next_to(prefill_arrow, RIGHT, buff=0.3)

        rebuilt = VGroup(*[
            Square(side_length=0.5, fill_color=ACCENT, fill_opacity=1.0, stroke_color=BLOCK_STROKE, stroke_width=2)
            for _ in range(6)
        ])
        rebuilt.arrange(RIGHT, buff=0.12)
        rebuilt.move_to(DOWN * 1.6)
        rebuilt_label = caption("whole KV cache rebuilt at once")
        rebuilt_label.next_to(rebuilt, DOWN, buff=0.25)

        self.play(shoot(prefill_arrow), FadeIn(prefill_label))
        self.play(FadeIn(rebuilt, lag_ratio=0.1), FadeIn(rebuilt_label))
        compare = caption("faster than decoding the whole thing token by token")
        compare.next_to(rebuilt_label, DOWN, buff=0.35)
        self.play(FadeIn(compare))
        self.wait(0.3)
        self.next_slide()
        self.play(FadeOut(heading6), FadeOut(concat), FadeOut(concat_label), FadeOut(prefill_arrow),
                   FadeOut(prefill_label), FadeOut(rebuilt), FadeOut(rebuilt_label), FadeOut(compare))

        # ==========================================================
        # Beat 7: trade-off (Fig 19)
        # ==========================================================
        heading7 = title("Swap vs recompute: it depends on block size")
        self.play(FadeIn(heading7))

        block_sizes = [1, 16, 32, 64, 128, 256]
        recompute_ms = [25, 24, 25, 23, 24, 25]
        swap_ms = [135, 55, 35, 25, 18, 15]
        chart = line_chart(
            block_sizes,
            {"Recompute": recompute_ms, "Swap": swap_ms},
            x_label="Block size",
            y_label="Time (ms)",
            colors={"Recompute": BAD, "Swap": ACCENT2},
            width=8.5, height=4.2,
        )
        chart.move_to(DOWN * 0.3)
        self.play(chart.animate_in())

        note = caption("approx values from Fig 19; crossover around block size 16-64, "
                        "where both give similar end-to-end performance")
        note.next_to(chart, DOWN, buff=0.5)
        self.play(FadeIn(note))
        self.wait(0.3)
        self.next_slide()
        self.play(FadeOut(heading7), FadeOut(chart), FadeOut(note))

        # ==========================================================
        # Beat 8: landing
        # ==========================================================
        landing = title("Paging gives you a knob contiguous systems never had")
        landing.scale(0.85)
        landing.move_to(UP * 0.4)
        sub_landing = body("Take memory back. Give it back later.")
        sub_landing.next_to(landing, DOWN, buff=0.6)
        self.play(FadeIn(landing))
        self.play(FadeIn(sub_landing))
        self.wait(0.3)
        self.next_slide()
        self.play(FadeOut(landing), FadeOut(sub_landing))
        self.wait(0.3)
