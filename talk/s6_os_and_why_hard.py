"""S6 — This is what an OS does, and why it's hard (Act II).

NARRATION
---------
Beat 1 — The reveal.
Look at what we just built: logical blocks for a request, a block table that
maps them to physical slots, and a pool of physical blocks on the GPU. [PAUSE]
Take a step back and squint at this picture. Logical blocks that get mapped,
on demand, onto scattered physical blocks through a table... this is exactly
what an operating system does when it manages a process's memory. We just
reinvented virtual memory paging — for the KV cache.

Beat 2 — The same picture, OS vocabulary.
Same picture, new labels. What we called a block, the OS calls a page. What
we called a token, the OS calls a byte. What we called a request, the OS
calls a process. And our block table is just a page table, mapping a
process's virtual pages onto physical RAM. It's the same idea, one layer
lower in the stack.

Beat 3 — Paging, for anyone who skipped the OS class.
Here's the whole idea in one breath: every process gets to believe it has
one big, contiguous chunk of memory. Underneath, the page table quietly
scatters that memory across whatever physical frames happen to be free.
Frames get handed out only when the process actually touches that page — not
up front — so there's no need to reserve a giant contiguous region, and no
external fragmentation between processes.

Beat 4 — So was this just copying the OS?
So, fair question: did we just copy fifty-year-old operating systems ideas
onto a GPU and call it a paper? [PAUSE] No. The idea transfers, but making it
work on a GPU, for attention, required real systems work that a textbook page
table never has to do.

Beat 5 — Reason one and two: no MMU, and the kernel itself changes.
First: your CPU has dedicated hardware for this — a memory management unit
that walks page tables and a TLB that caches recent translations, all in
silicon, off the critical path. The GPU has none of that for our purposes;
vLLM does every logical-to-physical translation in software, inside the
kernel, on every access. Second, and bigger: an OS page fault handler never
touches your program's computation — it just finds the page and hands control
back. Here, the computation *is* the memory access. The attention kernel
itself had to be rewritten to gather scattered KV blocks fast: a fused
reshape-and-write kernel, a fused block-read-and-attention kernel, and a
fused block-copy kernel for copy-on-write.

Beat 6 — Reason three and four: this isn't a rare fault, and OS policy doesn't fit.
Third: a page fault is a rare event — maybe once every few thousand
instructions. Our "fault" happens on *every token, every step*: attention
reads every block, every time. That's why the PagedAttention kernel itself
runs about 20 to 26 percent slower than FasterTransformer's kernel on
contiguous memory — Figure 18(a) in the paper. And yet the end-to-end system
is 2 to 4 times faster, because the memory efficiency gain swamps that
per-kernel cost. [PAUSE] Fourth: the OS doesn't know our workload. We needed
domain-specific policies it never had — all-or-nothing eviction of a whole
sequence's blocks at once, gang-scheduling sequences that share blocks, and
tuning the block size itself. More on those shortly.

Beat 7 — Landing.
So: an OS idea, re-engineered for a workload the OS was never designed for.

Beat 8 — Checkpoint.
"""

from manim import (
    DOWN,
    LEFT,
    ORIGIN,
    RIGHT,
    UP,
    AnimationGroup,
    FadeIn,
    FadeOut,
    Rectangle,
    Transform,
    VGroup,
)
from manim_slides import Slide

from talk.theme import *
from talk.components import *


class S6OSAndWhyHard(Slide):
    def construct(self):
        apply_theme(self)

        # -------------------------------------------------------------
        # Beat 1: the reveal — recreate the S5 picture
        # -------------------------------------------------------------
        heading = title("...this is what an operating system does.")
        heading.scale(0.85)

        logical_title = small("Logical blocks", font_size=SMALL_SIZE, color=MUTED)
        logical_blocks = VGroup(
            KVBlock(slots=4, cell=0.6, words=["Four", "score", "and", "seven"], index=0),
            KVBlock(slots=4, cell=0.6, words=["years", "ago", "our", "fathers"], index=1),
        )
        for kb in logical_blocks:
            for cell in kb.cells:
                cell.set_fill(ACCENT, opacity=1.0)
        logical_blocks.arrange(DOWN, buff=0.5)
        logical_group = VGroup(logical_title, logical_blocks)
        logical_title.next_to(logical_blocks, UP, buff=0.3)

        table = BlockTable(n_rows=2, title="Block table")

        # Build the physical grid manually (rather than via fill_slot after the
        # fact) so the KV word labels are correct from the start; this avoids
        # PhysicalMemGrid/KVBlock.fill_slot leaving orphaned label mobjects
        # behind when the whole picture is later scaled/moved as a group.
        phys_b0 = KVBlock(slots=4, cell=0.6, words=["years", "ago", "our", "fathers"], index=0)
        phys_b1 = KVBlock(slots=4, cell=0.6, index=1)
        phys_b2 = KVBlock(slots=4, cell=0.6, words=["Four", "score", "and", "seven"], index=2)
        phys_b3 = KVBlock(slots=4, cell=0.6, index=3)
        for kb in (phys_b0, phys_b2):
            for cell in kb.cells:
                cell.set_fill(ACCENT, opacity=1.0)
        phys_grid = VGroup(phys_b0, phys_b1, phys_b2, phys_b3)
        phys_grid.arrange_in_grid(rows=2, cols=2, buff=0.35)
        physical_title_txt = small("Physical GPU memory", font_size=SMALL_SIZE, color=FG)
        physical_title_txt.next_to(phys_grid, UP, buff=0.3)
        physical = VGroup(physical_title_txt, phys_grid)
        physical.title = physical_title_txt

        logical_group.move_to(LEFT * 5.3 + DOWN * 0.3)
        table.move_to(LEFT * 0.8 + DOWN * 0.3)
        physical.move_to(RIGHT * 4.3 + DOWN * 0.3)

        self.play(FadeIn(heading))
        self.play(FadeIn(logical_group))
        self.play(FadeIn(table))
        self.play(FadeIn(physical))
        self.play(table.set_row(0, 2, 4), table.set_row(1, 0, 4))

        arrow1 = arrow_map(logical_group, table, color=MUTED)
        arrow2 = arrow_map(table, physical, color=MUTED)
        self.play(shoot(arrow1), shoot(arrow2))
        self.wait(0.3)
        self.next_slide()

        # -------------------------------------------------------------
        # Beat 2: morph labels into OS vocabulary
        # -------------------------------------------------------------
        self.play(FadeOut(heading))
        heading2 = title("Same picture, OS vocabulary")
        heading2.scale(0.85)
        self.play(FadeIn(heading2))

        new_logical_title = small("Pages (per process)", font_size=SMALL_SIZE, color=MUTED)
        new_logical_title.move_to(logical_title.get_center())
        new_table_title = small("Page table", font_size=SMALL_SIZE, color=FG)
        new_table_title.move_to(table.title.get_center())
        new_physical_title = small("Physical RAM", font_size=SMALL_SIZE, color=FG)
        new_physical_title.move_to(physical.title.get_center())

        self.play(
            Transform(logical_title, new_logical_title),
            Transform(table.title, new_table_title),
            Transform(physical.title, new_physical_title),
        )

        mapping_pairs = [
            ("blocks", "pages"),
            ("tokens", "bytes"),
            ("requests", "processes"),
            ("block table", "page table"),
            ("GPU memory", "physical RAM"),
        ]
        map_rows = VGroup()
        for our_term, os_term in mapping_pairs:
            left_cell = small(our_term, font_size=SMALL_SIZE, color=ACCENT)
            arrow_cell = small("->", font_size=SMALL_SIZE, color=MUTED)
            right_cell = small(os_term, font_size=SMALL_SIZE, color=ACCENT2)
            row = VGroup(left_cell, arrow_cell, right_cell)
            row.arrange(RIGHT, buff=0.35)
            map_rows.add(row)
        map_rows.arrange(DOWN, buff=0.28, aligned_edge=LEFT)
        map_header = VGroup(small("vLLM term", font_size=TINY_SIZE, color=MUTED),
                             small("", font_size=TINY_SIZE),
                             small("OS term", font_size=TINY_SIZE, color=MUTED))
        map_full = VGroup(map_rows)
        map_full.scale(0.85)
        map_full.to_edge(DOWN, buff=0.5)

        picture_group = VGroup(logical_group, table, physical, arrow1, arrow2)
        self.play(picture_group.animate.scale(0.72).to_edge(UP, buff=1.3))
        map_full.next_to(picture_group, DOWN, buff=0.5)

        for row in map_rows:
            self.play(FadeIn(row), run_time=0.4)
        self.wait(0.3)
        self.next_slide()

        self.play(FadeOut(heading2), FadeOut(picture_group), FadeOut(map_full))

        # -------------------------------------------------------------
        # Beat 3: OS paging intuition
        # -------------------------------------------------------------
        heading3 = title("Paging, in one breath")
        self.play(FadeIn(heading3))

        intuition_lines = VGroup(
            small("Each process believes it has one contiguous address space", font_size=SMALL_SIZE),
            small("The page table scatters it across free physical frames", font_size=SMALL_SIZE),
            small("Frames are handed out on demand, not reserved up front", font_size=SMALL_SIZE),
            small("Result: no external fragmentation", font_size=SMALL_SIZE, color=GOOD),
        )
        intuition_lines.arrange(DOWN, buff=0.35, aligned_edge=LEFT)
        intuition_lines.move_to(LEFT * 4.0)

        physical2 = PhysicalMemGrid(n_blocks=6, slots=4, cols=3, cell=0.45, title="Physical frames")
        physical2.move_to(RIGHT * 3.6)
        fill_anims = []
        for i in (1, 4):
            for s in range(4):
                fill_anims.append(physical2.block(i).set_state(s, "filled"))

        self.play(FadeIn(intuition_lines))
        self.play(FadeIn(physical2))
        self.play(*fill_anims)
        self.wait(0.3)
        self.next_slide()
        self.play(FadeOut(heading3), FadeOut(intuition_lines), FadeOut(physical2))

        # -------------------------------------------------------------
        # Beat 4: "So was this just copying the OS?" -> "No."
        # -------------------------------------------------------------
        question = title("So was this just... copying the OS?")
        question.scale(0.85)
        question.move_to(UP * 0.5)
        self.play(FadeIn(question))

        no_text = title("No.")
        no_text.set_color(ACCENT)
        no_text.move_to(DOWN * 1.0)
        self.play(FadeIn(no_text))
        self.wait(0.3)
        self.next_slide()
        self.play(FadeOut(question), FadeOut(no_text))

        # -------------------------------------------------------------
        # Beat 5: reasons (a) no HW MMU  (b) kernel rewritten
        # -------------------------------------------------------------
        heading5 = title("Reason 1 & 2: no hardware help, and the kernel changes")
        heading5.scale(0.75)
        self.play(FadeIn(heading5))

        mmu_box = Rectangle(width=4.6, height=1.6, fill_color=BLOCK_FILL, fill_opacity=1.0,
                             stroke_color=BLOCK_STROKE, stroke_width=2)
        mmu_box.move_to(LEFT * 4.8 + UP * 0.9)
        mmu_label = small("No GPU MMU / TLB", font_size=SMALL_SIZE, color=BAD)
        mmu_sub = caption("Translation done in software,\ninside the kernel, every access")
        mmu_sub.next_to(mmu_label, DOWN, buff=0.2)
        mmu_group = VGroup(mmu_box, VGroup(mmu_label, mmu_sub).move_to(mmu_box.get_center()))

        self.play(FadeIn(mmu_group))
        self.next_slide()

        kernel_caption = small("The attention kernel itself is rewritten:", font_size=SMALL_SIZE, color=MUTED)
        kernel_caption.next_to(mmu_group, DOWN, buff=0.7).align_to(mmu_group, LEFT)

        stage_names = [
            "fused reshape\n+ block write",
            "fused block read\n+ attention",
            "fused block copy\n(copy-on-write)",
        ]
        stages = VGroup()
        for name in stage_names:
            box = Rectangle(width=2.6, height=1.3, fill_color=ACCENT2, fill_opacity=0.15,
                             stroke_color=ACCENT2, stroke_width=2)
            lbl = small(name, font_size=TINY_SIZE, color=FG)
            lbl.move_to(box.get_center())
            stages.add(VGroup(box, lbl))
        stages.arrange(RIGHT, buff=0.55)
        stages.next_to(kernel_caption, DOWN, buff=0.4)
        stages.move_to(RIGHT * 1.0 + DOWN * 1.6)
        kernel_caption.next_to(stages, UP, buff=0.35)

        stage_arrows = VGroup(*[
            arrow(
                stages[i].get_right(), stages[i + 1].get_left(),
                buff=0.1,
            )
            for i in range(len(stages) - 1)
        ])

        self.play(FadeIn(kernel_caption))
        self.play(FadeIn(stages[0]))
        self.play(shoot(stage_arrows[0]), FadeIn(stages[1]))
        self.play(shoot(stage_arrows[1]), FadeIn(stages[2]))
        self.wait(0.3)
        self.next_slide()
        self.play(
            FadeOut(heading5), FadeOut(mmu_group), FadeOut(kernel_caption),
            FadeOut(stages), FadeOut(stage_arrows),
        )

        # -------------------------------------------------------------
        # Beat 6: reasons (c) access pattern / Fig 18a  (d) domain policies
        # -------------------------------------------------------------
        heading6 = title("Reason 3 & 4: not a rare fault, and no OS policy fits")
        heading6.scale(0.7)
        self.play(FadeIn(heading6))

        access_caption = small("Every token, every step touches every block", font_size=SMALL_SIZE, color=WARN)
        access_caption.move_to(LEFT * 3.6 + UP * 1.6)
        self.play(FadeIn(access_caption))

        overhead_chart = bar_chart(
            categories=["Kernel latency", "End-to-end throughput"],
            series={
                "FasterTransformer": [1.0, 1.0],
                "vLLM": [1.23, 3.0],
            },
            y_label="relative",
            width=6.0, height=3.0,
            value_labels=True,
        )
        overhead_chart.scale(0.72)
        overhead_chart.move_to(LEFT * 3.6 + DOWN * 1.0)
        self.play(FadeIn(overhead_chart))
        overhead_note = caption("Schematic only — exact Fig. 18(a) benchmark appears later")
        overhead_note.next_to(overhead_chart, DOWN, buff=0.3)
        self.play(FadeIn(overhead_note))
        self.next_slide()

        policy_title = small("Domain-specific policies the OS never needed:", font_size=SMALL_SIZE, color=MUTED)
        policy_title.move_to(RIGHT * 3.6 + UP * 1.6)
        policies = VGroup(
            small("- all-or-nothing eviction of a sequence's blocks", font_size=TINY_SIZE),
            small("- gang-scheduling sequences that share blocks", font_size=TINY_SIZE),
            small("- tuned block size (parallelism vs fragmentation)", font_size=TINY_SIZE),
        )
        policies.arrange(DOWN, buff=0.3, aligned_edge=LEFT)
        policies.next_to(policy_title, DOWN, buff=0.35).align_to(policy_title, LEFT)
        self.play(FadeIn(policy_title))
        self.play(FadeIn(policies), run_time=0.6)
        self.wait(0.3)
        self.next_slide()
        self.play(
            FadeOut(heading6), FadeOut(access_caption), FadeOut(overhead_chart),
            FadeOut(overhead_note), FadeOut(policy_title), FadeOut(policies),
        )

        # -------------------------------------------------------------
        # Beat 7: landing
        # -------------------------------------------------------------
        landing = title("An OS idea, re-engineered for a workload the OS never had.")
        landing.scale(0.72)
        landing.move_to(ORIGIN)
        self.play(FadeIn(landing))
        self.wait(0.3)
        self.next_slide()
        self.play(FadeOut(landing))

        # -------------------------------------------------------------
        # Beat 8: act checkpoint -> seam to Act III
        # -------------------------------------------------------------
        act_checkpoint(
            self, 3, "The payoffs",
            done=["Act I — Why memory is the bottleneck", "Act II — The idea: page the KV cache"],
            current="Act III — The payoffs",
            upcoming=[],
        )
        self.wait(0.3)
