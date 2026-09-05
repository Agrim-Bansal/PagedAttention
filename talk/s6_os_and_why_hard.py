"""S6 — This is what an OS does, and why it's hard (Act II).

NARRATION
---------
Beat 1 — Pickup.
Look at what we just built. Same Lincoln sentence, same mapping as the last
scene. Logical 0 lives in physical 7. Logical 1 in physical 1. Logical 2 in
physical 3. A request, a block table, a pool of GPU blocks. Sit with this
picture — we are going to look at it again. [PAUSE]

Beat 2 — The reveal.
Take a step back and squint. Logical blocks mapped, on demand, onto
scattered physical blocks through a table... this is exactly what an
operating system does when it manages a process's memory. We just
reinvented virtual memory paging — for the KV cache. [PAUSE]

Beat 3 — OS vocabulary.
Same picture, new labels. A block is a page. A token is a byte. A request
is a process. And the block table is a page table: virtual pages onto
physical frames. Same idea, one layer lower in the stack. [PAUSE]

Beat 4 — Demand paging, shown.
The process believes it has one contiguous chunk — pages 0, 1, 2 in order,
on the left. Underneath, those pages sit in whatever frames were free: 7,
1, and 3, not next to each other. Frames are handed out when the process
actually touches them, not reserved up front. No giant slab. No external
fragmentation. [PAUSE]

Beat 5 — The question.
So, fair question. Did we just copy fifty-year-old operating systems ideas
onto a GPU and call it a paper? Sit with that. [PAUSE]

Beat 6 — The answer.
The idea transfers. The engineering does not. Making paging work on a GPU,
for attention, required real systems work that a textbook page table never
has to do. Two reasons. We will take them one at a time.

Beat 7 — No GPU MMU.
First: your CPU has dedicated hardware for this — a memory management unit
and a TLB, in silicon, off the critical path. The GPU has none of that for
our purposes. Every logical-to-physical translation happens in software,
inside the kernel, on every access. This table is walked by the kernel
itself. [PAUSE]

Beat 8 — The kernel is the pager.
Second: an OS page-fault handler never touches your program's computation.
It finds the page and hands control back. Here the computation is the
memory access. The attention kernel had to be rewritten to gather those
scattered KV blocks and attend in one fused pass. An OS pager never
rewrites your program. [PAUSE]

Beat 9 — The kernel is slower.
And that rewrite is not free. A page fault is rare. Ours happens on every
token, every step: attention reads every block, every time. That's why the
PagedAttention kernel itself runs about 20 to 26 percent slower than
FasterTransformer's kernel on contiguous memory. [PAUSE]

Beat 10 — And yet.
And yet the end-to-end system is 2 to 4 times faster. Leftover VRAM now
holds a bigger batch. The memory win swamps the per-kernel cost. We will
measure that kernel overhead after the results. [PAUSE]

Beat 11 — Landing.
So: an OS idea, re-engineered for a workload the OS was never designed for.
A new kernel. New policies still to come — because every block of a
sequence is always touched together, and OS-style per-page eviction does
not apply.
"""

from manim import (
    DOWN,
    LEFT,
    ORIGIN,
    RIGHT,
    UP,
    FadeIn,
    FadeOut,
    SurroundingRectangle,
    Transform,
    VGroup,
)
from manim_slides import Slide

from talk.components import (
    KVBlock,
    arrow_map,
    shoot,
)
from talk.theme import (
    ACCENT,
    ACCENT2,
    FG,
    GOOD,
    MUTED,
    TINY_SIZE,
    WARN,
    apply_theme,
    act_checkpoint,
    caption,
    small,
    text,
    title,
)

CELL = 0.50
FRAME_W = 13.2
PICTURE_BOTTOM = -3.20


def _fit_title(content):
    t = title(content)
    if t.width > FRAME_W:
        t.scale_to_fit_width(FRAME_W)
        t.to_edge(UP)
    return t


def _kv(index, words=None, cell=CELL):
    words = words or ["", "", "", ""]
    block = KVBlock(slots=4, cell=cell, words=words, index=None)
    block.index_label.set_opacity(0)
    block.index_label.scale(0.01)
    block.index_label.move_to(block.cells.get_center())
    idx = text(str(index), font_size=TINY_SIZE, color=MUTED)
    idx.next_to(block.cells, LEFT, buff=0.10)
    for i, w in enumerate(words):
        if w:
            block.cells[i].set_fill(ACCENT, opacity=1.0)
    row = VGroup(idx, block)
    row.cells = block.cells
    row.kv = block
    return row


def _stack(title_str, rows):
    body = VGroup(*rows)
    body.arrange(DOWN, buff=0.16, aligned_edge=LEFT)
    head = small(title_str, color=FG)
    if head.width > body.width:
        head.scale_to_fit_width(body.width)
    head.next_to(body, UP, buff=0.16)
    group = VGroup(head, body)
    group.head = head
    group.body = body
    group.blocks = rows
    return group


def _mini_table(title_str, rows):
    head = small(title_str, color=FG)
    headers = ["logical", "physical", "filled"]
    hdr = VGroup(*[caption(h) for h in headers])
    hdr.arrange(RIGHT, buff=0.40)
    body = VGroup()
    for log, phys, filled in rows:
        row = VGroup(
            text(str(log), font_size=18, color=FG),
            text(str(phys), font_size=18, color=ACCENT),
            text(str(filled), font_size=18, color=FG),
        )
        row.arrange(RIGHT, buff=0.40)
        body.add(row)
    body.arrange(DOWN, buff=0.14, aligned_edge=LEFT)
    for row in body:
        for cell, hcell in zip(row, hdr):
            cell.align_to(hcell, ORIGIN)
            cell.set_x(hcell.get_x())
    table = VGroup(head, hdr, body)
    table.arrange(DOWN, buff=0.16)
    table.head = head
    return table


def _place_under_title(mob, heading, bottom=PICTURE_BOTTOM):
    mob.next_to(heading, DOWN, buff=0.32)
    room = heading.get_bottom()[1] - 0.32 - bottom
    if mob.height > room:
        mob.scale_to_fit_height(room)
        mob.next_to(heading, DOWN, buff=0.32)
    if mob.width > FRAME_W:
        mob.scale_to_fit_width(FRAME_W)
        mob.next_to(heading, DOWN, buff=0.32)
    return mob


def _stat(value, label, color):
    v = text(value, font_size=56, color=color)
    lab = small(label, color=MUTED)
    if lab.width > 5.4:
        lab.scale_to_fit_width(5.4)
    g = VGroup(v, lab)
    g.arrange(DOWN, buff=0.22)
    return g


class S6OSAndWhyHard(Slide):
    def construct(self):
        apply_theme(self)

        # -------------------------------------------------------------
        # Beat 1: pickup — S5 Fig 6 end state, in frame
        # -------------------------------------------------------------
        heading = _fit_title("The mapping we just built")

        logical = _stack(
            "Logical blocks",
            [
                _kv(0, ["Four", "score", "and", "seven"]),
                _kv(1, ["years", "ago", "our", "fathers"]),
                _kv(2, ["brought", "", "", ""]),
            ],
        )
        table = _mini_table(
            "Block table",
            [(0, 7, "4/4"), (1, 1, "4/4"), (2, 3, "1/4")],
        )
        physical = _stack(
            "Physical GPU memory",
            [
                _kv(0),
                _kv(1, ["years", "ago", "our", "fathers"]),
                _kv(3, ["brought", "", "", ""]),
                _kv(7, ["Four", "score", "and", "seven"]),
            ],
        )

        picture = VGroup(logical, table, physical)
        picture.arrange(RIGHT, buff=0.55, aligned_edge=UP)
        _place_under_title(picture, heading)

        self.play(FadeIn(heading), run_time=0.6)
        self.play(FadeIn(logical), run_time=0.7)
        self.play(FadeIn(table), run_time=0.6)
        self.play(FadeIn(physical), run_time=0.8)
        a_lt = arrow_map(logical, table, color=MUTED)
        a_tp = arrow_map(table, physical, color=MUTED)
        self.play(shoot(a_lt), shoot(a_tp), run_time=0.8)
        self.wait(0.5)
        self.next_slide()

        # -------------------------------------------------------------
        # Beat 2: reveal
        # -------------------------------------------------------------
        heading = self._retitle(heading, "...this is what an operating system does.")
        self.wait(0.5)
        self.next_slide()

        # -------------------------------------------------------------
        # Beat 3: relabel in place — one title at a time
        # -------------------------------------------------------------
        heading = self._retitle(heading, "Same picture, OS vocabulary")

        new_logical = small("Virtual pages", color=FG)
        if new_logical.width > logical.body.width + 0.4:
            new_logical.scale_to_fit_width(logical.body.width + 0.4)
        new_logical.move_to(logical.head.get_center())

        new_table = small("Page table", color=FG)
        new_table.move_to(table.head.get_center())

        new_phys = small("Physical frames", color=FG)
        if new_phys.width > physical.body.width:
            new_phys.scale_to_fit_width(physical.body.width)
        new_phys.move_to(physical.head.get_center())

        self.play(Transform(logical.head, new_logical), run_time=0.7)
        self.play(Transform(table.head, new_table), run_time=0.7)
        self.play(Transform(physical.head, new_phys), run_time=0.7)
        gloss = caption("tokens → bytes     request → process")
        gloss.to_edge(DOWN, buff=0.28)
        self.play(FadeIn(gloss), run_time=0.5)
        self.wait(0.5)
        self.next_slide()

        # -------------------------------------------------------------
        # Beat 4: demand paging shown on the same picture
        # -------------------------------------------------------------
        heading = self._retitle(heading, "Demand paging", FadeOut(gloss))
        self._dump(gloss)

        virt_ring = SurroundingRectangle(
            logical.body, buff=0.10, color=ACCENT2, stroke_width=2,
        )
        phys_ring = SurroundingRectangle(
            physical.body, buff=0.10, color=ACCENT, stroke_width=2,
        )
        foot = caption(
            "contiguous virtual view  ·  scattered frames  ·  on demand, no external holes"
        )
        foot.to_edge(DOWN, buff=0.26)
        if foot.width > FRAME_W:
            foot.scale_to_fit_width(FRAME_W)
            foot.to_edge(DOWN, buff=0.26)

        self.play(FadeIn(virt_ring), run_time=0.6)
        self.play(FadeIn(phys_ring), run_time=0.6)
        self.play(FadeIn(foot), run_time=0.5)
        self.wait(0.5)
        self.next_slide()

        # -------------------------------------------------------------
        # Beat 5: the question only — sit with it
        # -------------------------------------------------------------
        heading = self._retitle(
            heading,
            "So was this just copying the OS?",
            FadeOut(virt_ring),
            FadeOut(phys_ring),
            FadeOut(foot),
        )
        self._dump(VGroup(virt_ring, phys_ring, foot))
        self.wait(0.5)
        self.next_slide()

        # -------------------------------------------------------------
        # Beat 6: the answer
        # -------------------------------------------------------------
        answer = small("The idea transfers. The engineering does not.", color=ACCENT)
        if answer.width > FRAME_W:
            answer.scale_to_fit_width(FRAME_W)
        answer.to_edge(DOWN, buff=0.28)
        self.play(FadeIn(answer), run_time=0.7)
        self.wait(0.5)
        self.next_slide()

        # -------------------------------------------------------------
        # Beat 7: no GPU MMU — stay on the picture, highlight the table
        # -------------------------------------------------------------
        heading = self._retitle(
            heading,
            "The GPU has no MMU",
            FadeOut(answer),
        )
        self._dump(answer)

        table_ring = SurroundingRectangle(
            table, buff=0.12, color=WARN, stroke_width=2,
        )
        mmu_foot = caption("The kernel walks this table itself, every access.")
        mmu_foot.to_edge(DOWN, buff=0.28)
        if mmu_foot.width > FRAME_W:
            mmu_foot.scale_to_fit_width(FRAME_W)
            mmu_foot.to_edge(DOWN, buff=0.28)
        self.play(FadeIn(table_ring), run_time=0.6)
        self.play(FadeIn(mmu_foot), run_time=0.5)
        self.wait(0.5)
        self.next_slide()

        # -------------------------------------------------------------
        # Beat 8: the kernel is the pager — still on the picture
        # -------------------------------------------------------------
        heading = self._retitle(
            heading,
            "The kernel is the pager",
            FadeOut(table_ring),
            FadeOut(mmu_foot),
        )
        self._dump(VGroup(table_ring, mmu_foot))

        phys_ring2 = SurroundingRectangle(
            physical.body, buff=0.10, color=ACCENT2, stroke_width=2,
        )
        pager_foot = caption(
            "Gather scattered KV and attend in one fused pass.  An OS pager never rewrites your program."
        )
        pager_foot.to_edge(DOWN, buff=0.26)
        if pager_foot.width > FRAME_W:
            pager_foot.scale_to_fit_width(FRAME_W)
            pager_foot.to_edge(DOWN, buff=0.26)
        self.play(FadeIn(phys_ring2), run_time=0.6)
        self.play(FadeIn(pager_foot), run_time=0.5)
        self.wait(0.5)
        self.next_slide()

        # -------------------------------------------------------------
        # Beat 9: the kernel is slower — one number
        # -------------------------------------------------------------
        self.play(
            FadeOut(heading),
            FadeOut(picture),
            FadeOut(a_lt),
            FadeOut(a_tp),
            FadeOut(phys_ring2),
            FadeOut(pager_foot),
            run_time=0.6,
        )
        self._dump(VGroup(heading, picture, a_lt, a_tp, phys_ring2, pager_foot))

        heading = _fit_title("And that kernel is slower")
        self.play(FadeIn(heading), run_time=0.6)

        left = _stat("+20–26%", "slower attention kernel", WARN)
        left.move_to(ORIGIN + UP * 0.2)
        vs = caption("vs FasterTransformer, contiguous KV")
        vs.next_to(left, DOWN, buff=0.22)
        hot = small("Every token, every block — not a rare page fault.", color=FG)
        if hot.width > FRAME_W:
            hot.scale_to_fit_width(FRAME_W)
        hot.to_edge(DOWN, buff=0.4)

        self.play(FadeIn(left), run_time=0.7)
        self.play(FadeIn(vs), run_time=0.45)
        self.play(FadeIn(hot), run_time=0.5)
        self.wait(0.5)
        self.next_slide()

        # -------------------------------------------------------------
        # Beat 10: and yet, 2–4×
        # -------------------------------------------------------------
        self.play(left.animate.shift(LEFT * 2.6), vs.animate.shift(LEFT * 2.6), run_time=0.7)
        right = _stat("2–4×", "faster end to end", GOOD)
        right.move_to(ORIGIN + RIGHT * 2.6 + UP * 0.2)
        why = caption("leftover VRAM holds a bigger batch")
        why.next_to(right, DOWN, buff=0.22)
        self.play(FadeIn(right), FadeIn(why), run_time=0.7)
        self.wait(0.5)
        self.next_slide()

        # -------------------------------------------------------------
        # Beat 11: landing + Act III checkpoint
        # -------------------------------------------------------------
        land = small("An OS idea. A new kernel. New policies still to come.", color=ACCENT)
        if land.width > FRAME_W:
            land.scale_to_fit_width(FRAME_W)
        land.move_to(hot.get_center())
        self.play(FadeOut(hot), FadeIn(land), run_time=0.7)
        self._dump(hot)
        self.wait(0.5)
        self.next_slide()

        self.play(
            FadeOut(heading),
            FadeOut(left),
            FadeOut(right),
            FadeOut(vs),
            FadeOut(why),
            FadeOut(land),
            run_time=0.5,
        )
        act_checkpoint(
            self,
            3,
            "The payoffs",
            done=[
                "Act I — Why memory is the bottleneck",
                "Act II — The idea: page the KV cache",
            ],
            current="Act III — The payoffs",
            upcoming=[],
        )

    def _retitle(self, old, new_str, *anims, run_time=0.7):
        new = _fit_title(new_str)
        extra = list(anims)
        if old is not None:
            extra = [FadeOut(old, shift=UP * 0.15)] + extra
            self.play(FadeIn(new, shift=DOWN * 0.08), *extra, run_time=run_time)
            self._dump(old)
        else:
            self.play(FadeIn(new), *extra, run_time=run_time)
        return new

    def _dump(self, mob):
        self.remove(mob)
