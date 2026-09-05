"""S7 — Sharing across sequences (Act III).

Pickup S4's unpaid bill on the S5 picture, then run Figures 8, 9, and 10.
Measured savings (Fig 15/16) stay in S9.

NARRATION
---------
Beat 1 — Pickup.
Look at the mapping we just built. Same Lincoln sentence: logical 0 lives in
physical 7, logical 1 in physical 1, logical 2 in physical 3. Figure 7 put a
second request into that same pool — interleaved, not shared. Two sequences
still owned two copies of everything. [PAUSE] Act I already named the bill:
a contiguous KV cache cannot point two sequences at the same physical
memory. Prompt copies were 12 percent of the cache. Beam search could save
up to 55 percent. We now have a block table. We can pay that bill.

Beat 2 — Parallel sampling, the product.
A common trick for better outputs: one prompt, several tries. "Give me two
samples." Call them A1 and A2. Each is its own sequence with its own KV
cache, but they start from the exact same prompt. [PAUSE] In a contiguous
system that means copying the prompt's entire KV cache, once per sample.
With blocks, we rewind to just after prefill — and we do not copy.

Beat 3 — Same physical blocks, two logical views.
Figure 8. A1's block table and A2's block table both map logical 0 to
physical 7 and logical 1 to physical 1. No copy has happened — both samples
are pointers into one shared region. Each shared physical block carries a
reference count: here, 2, because two logical blocks point at it.

Beat 4 — Only the last block can move.
Physical 7 is packed. Four out of four prompt tokens. Nobody ever writes
there again, so it can stay shared forever. Physical 1 is different: one
open slot, reserved for the next token. That is the only place a write can
happen. [PAUSE]

Beat 5 — Copy-on-write.
A1 samples a different continuation and tries to write "mothers" into that
open slot. vLLM checks the reference count, sees 2, and refuses to write in
place. Allocate a fresh physical block — physical 3 — copy the three shared
tokens into it, write "mothers" into the copy, retarget A1's table, and
decrement the original block's count to 1. [PAUSE] A2 still points at the
original, untouched. The cost: only that last, not-yet-full block is copied.
Every packed block just stays shared.

Beat 6 — Write in place.
Now A2 writes "fathers" into the original block. The reference count is
already 1, so the write happens in place. No copy. [PAUSE] This is the same
trick an operating system uses when you fork a process: share the pages,
copy only the one that someone actually writes. Two samples, one prompt's
worth of KV, plus a single last-block copy.

Beat 7 — Beam search shares a tree.
Beam search uses the same mechanism more aggressively. A beam of width 4
keeps four candidate sequences at every step. They do not just share the
prompt — they share prefixes of each other's generated tokens, because
every candidate is an extension of a shared history. Picture a tree: one
shared trunk, a private branch for the candidate that diverged early, and
four live heads. Each head is only the new block that candidate needed.

Beat 8 — One prune step.
Watch the next iteration. Candidates 0 and 3 fall out of the top 4. Every
physical block that only they were using has its reference count drop to
zero — and those blocks go straight back to the free pool, with no bulk
copy. New blocks are allocated for the surviving heads. [PAUSE] The sharing
pattern is being recomputed, cheaply, at every single decoding step. This
is the case that can save up to 55 percent of KV memory. We will measure
that in the results.

Beat 9 — Shared prefix, the paper's example.
Same trick, now across different requests. A production translation service
ships the same few-shot instruction with every call: "Translate English to
French," plus three examples — sea otter, peppermint, plush giraffe.
Sequence A asks for "cheese?" and gets "fromage." Sequence B asks "I love
you?" and gets "Je t'aime." Look at how much of each request is identical.
[PAUSE]

Beat 10 — Cached blocks, two tables.
vLLM computes that shared prefix's KV blocks exactly once, caches them, and
every new request's block table just points at those cached blocks. The
last shared block is marked copy-on-write. Only the task-specific suffix —
the actual question — needs new computation and new blocks.

Beat 11 — Three primitives.
None of this is three special features. The engine exposes three operations.
Fork: create a new sequence from an existing one, share its blocks, bump
the reference counts. Append: write a new token; copy-on-write if the block
is still shared. Free: drop a sequence, decrement, reclaim at zero.
Parallel sampling is fork then append. Beam search is fork, append, and
free, every step. A shared prefix is a fork onto a cache that was computed
once. One API.

Beat 12 — Landing.
Reference counts and copy-on-write pay Act I's bill: sequences can now
point at the same physical memory. Next: what happens when that free pool
is empty.
"""

from manim import (
    DOWN,
    LEFT,
    ORIGIN,
    RIGHT,
    UP,
    UR,
    AnimationGroup,
    Create,
    Cross,
    FadeIn,
    FadeOut,
    GrowFromCenter,
    Indicate,
    LaggedStart,
    Line,
    Rectangle,
    ReplacementTransform,
    RoundedRectangle,
    SurroundingRectangle,
    VGroup,
)
from manim_slides import Slide

from talk.theme import *
from talk.components import *


SLOT_W, SLOT_H = 0.96, 0.46
IDX_W = 0.50
FRAME_W = 13.2
PICTURE_BOTTOM = -3.18
CHIP_W, CHIP_H = 0.90, 0.50
_LONGEST = "brought"
_FS_CACHE = {}
_TOK_H = 0.48
_TOK_PAD = 0.46


def _fit(mob, max_w):
    if mob.width > max_w:
        mob.scale_to_fit_width(max_w)
    return mob


def _fit_title(content):
    t = title(content)
    if t.width > FRAME_W:
        t.scale_to_fit_width(FRAME_W)
        t.to_edge(UP)
    return t


def _foot(s, color=MUTED, size=SMALL_SIZE):
    t = text(s, font_size=size, color=color)
    _fit(t, FRAME_W)
    t.to_edge(DOWN, buff=0.26)
    return t


def _tx(content, **kw):
    """Text with a recorded baseline offset so short words sit on the same line."""
    t = text(content, **kw)
    t.dy = 0.0
    if content:
        probe = text("Ág" + content + "Ág", **kw)
        n = len(probe.submobjects)
        if n == len(t.submobjects) + 4:
            inner = VGroup(*probe.submobjects[2:n - 2])
            t.dy = inner.get_center()[1] - probe.get_center()[1]
    return t


def _center(mob, point):
    mob.move_to(point)
    mob.shift(UP * getattr(mob, "dy", 0.0))
    return mob


def _word_fs(avail_w, base=16):
    """One font size for every token: sized so the longest word still fits."""
    key = (round(avail_w, 3), base)
    if key not in _FS_CACHE:
        probe = text(_LONGEST, font_size=base)
        _FS_CACHE[key] = base * min(1.0, avail_w / max(probe.width, 1e-6))
    return _FS_CACHE[key]


_SLOT_FS = _word_fs(0.84 * SLOT_W, base=16)
_TOK_FS = 16


def _label_at(word, point, txt_color=BG):
    lbl = _tx(str(word) if word else "", font_size=_SLOT_FS, color=txt_color if word else FG)
    if word:
        _center(lbl, point)
    else:
        lbl.move_to(point)
    return lbl


def _fill_slot(block, i, word, color=ACCENT, txt_color=BG):
    new_label = _label_at(word, block.cells[i].get_center(), txt_color)
    old_label = block.labels[i]
    block.labels.submobjects[i] = new_label
    fill_color = color if word else BLOCK_FILL
    return AnimationGroup(
        block.cells[i].animate.set_fill(fill_color, opacity=1.0),
        FadeOut(old_label),
        FadeIn(new_label),
        group=block,
    )


class _SlotRow(VGroup):
    """Four equal rounded slots. One type size; boxes never scale per word."""

    def __init__(self, words=None, color=ACCENT):
        super().__init__()
        words = list(words or ["", "", "", ""])
        while len(words) < 4:
            words.append("")
        self.cell = SLOT_W
        self.cells = VGroup()
        self.labels = VGroup()
        for w in words:
            cell = Rectangle(
                width=SLOT_W,
                height=SLOT_H,
                fill_color=color if w else BLOCK_FILL,
                fill_opacity=1.0,
                stroke_width=0,
            )
            self.cells.add(cell)
        self.cells.arrange(RIGHT, buff=0)
        frame = RoundedRectangle(
            width=self.cells.width,
            height=SLOT_H,
            corner_radius=0.10,
            fill_opacity=0,
            stroke_color=BLOCK_STROKE,
            stroke_width=1.8,
        )
        frame.move_to(self.cells.get_center())
        dividers = VGroup()
        for i in range(1, 4):
            x = self.cells[i].get_left()[0]
            dividers.add(
                Line(
                    [x, self.cells.get_top()[1], 0],
                    [x, self.cells.get_bottom()[1], 0],
                    color=BLOCK_STROKE,
                    stroke_width=1.4,
                )
            )
        for i, w in enumerate(words):
            self.labels.add(_label_at(w, self.cells[i].get_center(), BG))
        self.frame = frame
        self.add(self.cells, frame, dividers, self.labels)


class _PRow(VGroup):
    """Index in a fixed-width column + four equal token slots."""

    def __init__(self, index, words=None, color=ACCENT, ghost=False):
        super().__init__()
        self.kv = _SlotRow(words=words, color=color)
        self.idx = _tx(str(index), font_size=TINY_SIZE, color=MUTED)
        _center(self.idx, self.kv.cells.get_left() + LEFT * (IDX_W / 2 + 0.04))
        self.add(self.idx, self.kv)
        self.cells = self.kv.cells
        self.index = index
        self._ghost = ghost
        if ghost:
            self.set_opacity(0.28)

    def left_port(self):
        """Arrow endpoint just left of the index, so the tip never covers the number."""
        p = self.idx.get_left().copy()
        p += LEFT * 0.18
        return p

    def right_port(self):
        """Arrow endpoint at the block's right edge, below the ref badge."""
        return self.cells.get_right()

    def reserve(self, i):
        return self.kv.cells[i].animate.set_fill(WARN, opacity=0.60)

    def unghost(self):
        self._ghost = False
        return self.animate.set_opacity(1.0)


class _Stack(VGroup):
    def __init__(self, head_str, rows, head_color=FG):
        super().__init__()
        self.body = VGroup(*rows)
        self.body.arrange(DOWN, buff=0.20, aligned_edge=LEFT)
        self.head = small(head_str, color=head_color)
        _fit(self.head, max(self.body.width, 2.6))
        self.head.next_to(self.body, UP, buff=0.26)
        self.add(self.head, self.body)
        self.rows = rows


class _Table(VGroup):
    """Logical | physical | filled, as a quiet card."""

    def __init__(self, n, head_str, head_color=FG):
        super().__init__()
        self.head = small(head_str, color=head_color)
        cols = VGroup(caption("logical"), caption("physical"), caption("filled"))
        cols.arrange(RIGHT, buff=0.42)
        self.cols = cols
        self.phys = []
        self.fill = []
        self.rows = VGroup()
        for i in range(n):
            lg = _tx(str(i), font_size=18, color=MUTED)
            ph = _tx("–", font_size=18, color=MUTED)
            fl = _tx("–", font_size=18, color=MUTED)
            row = VGroup(lg, ph, fl)
            self.phys.append(ph)
            self.fill.append(fl)
            self.rows.add(row)
        self.rows.arrange(DOWN, buff=0.49)
        self.rows.next_to(cols, DOWN, buff=0.14)
        for row in self.rows:
            for c, h in zip(row, cols):
                c.set_x(h.get_center()[0])
        inner = VGroup(cols, self.rows)
        self.frame = RoundedRectangle(
            width=inner.width + 0.62,
            height=inner.height + 0.36,
            corner_radius=0.12,
            stroke_color=BLOCK_STROKE,
            stroke_width=1.5,
            fill_color=BLOCK_FILL,
            fill_opacity=0.55,
        )
        self.frame.move_to(inner.get_center())
        self.head.next_to(self.frame, UP, buff=0.22)
        self.add(self.frame, self.head, cols, self.rows)

    def set_mapping(self, i, physical, filled, color=FG):
        new_p = _tx(str(physical), font_size=18, color=color)
        new_f = _tx(str(filled), font_size=18, color=color)
        new_p.move_to(self.phys[i].get_center())
        new_f.move_to(self.fill[i].get_center())
        old_p, old_f = self.phys[i], self.fill[i]
        self.phys[i], self.fill[i] = new_p, new_f
        self.rows[i].submobjects[1] = new_p
        self.rows[i].submobjects[2] = new_f
        return AnimationGroup(
            FadeOut(old_p), FadeIn(new_p), FadeOut(old_f), FadeIn(new_f),
            group=self,
        )

    def row_right(self, i):
        p = self.frame.get_right().copy()
        p[1] = self.rows[i].get_center()[1]
        return p

    def row_left(self, i):
        p = self.frame.get_left().copy()
        p[1] = self.rows[i].get_center()[1]
        return p


def _place_under_title(mob, heading, bottom=PICTURE_BOTTOM):
    mob.next_to(heading, DOWN, buff=0.28)
    room = heading.get_bottom()[1] - 0.28 - bottom
    if mob.height > room:
        mob.scale_to_fit_height(room)
        mob.next_to(heading, DOWN, buff=0.28)
    if mob.width > FRAME_W:
        mob.scale_to_fit_width(FRAME_W)
        mob.next_to(heading, DOWN, buff=0.28)
    return mob


def _badge(row, value, color=ACCENT2):
    badge = RefCountBadge(value=value, color=color)
    badge.scale(0.72)
    badge.next_to(row.cells, RIGHT, buff=0.28)
    return badge


def _map_arrow(start, end, color, buff=0.06):
    """Straight if the row lines up; otherwise an elbow in the open gutter."""
    if abs(start[1] - end[1]) < 0.22:
        return arrow(start, end, color=color, buff=buff)
    gx = 0.5 * (start[0] + end[0])
    c1 = start.copy()
    c1[0] = gx
    c2 = end.copy()
    c2[0] = gx
    return VGroup(
        Line(start, c1, color=color, stroke_width=1.6),
        Line(c1, c2, color=color, stroke_width=1.6),
        arrow(c2, end, color=color, buff=0.04),
    )


def _shoot(mob):
    if isinstance(mob, VGroup) and getattr(mob, "tip", None) is None:
        return Create(mob)
    return shoot(mob)


def _link(src, dst):
    return Line(src.box.get_right(), dst.box.get_left(), color=MUTED, stroke_width=2)


def _chip(label, fill=BLOCK_FILL, stroke=BLOCK_STROKE, fg=FG):
    box = RoundedRectangle(
        width=CHIP_W, height=CHIP_H, corner_radius=0.10,
        fill_color=fill, fill_opacity=1.0,
        stroke_color=stroke, stroke_width=2,
    )
    lab = _tx(label, font_size=TINY_SIZE, color=fg)
    _fit(lab, CHIP_W * 0.86)
    _center(lab, box.get_center())
    g = VGroup(box, lab)
    g.box = box
    g.lab = lab
    return g


def _tok(word, fill, fg=BG, h=_TOK_H):
    """Hug the word. Same type size for every pill — never scale per phrase."""
    lab = _tx(word, font_size=_TOK_FS, color=fg)
    box = RoundedRectangle(
        width=max(lab.width + _TOK_PAD, 1.70),
        height=h, corner_radius=0.10,
        fill_color=fill, fill_opacity=1.0,
        stroke_color=fill, stroke_width=1.5,
    )
    _center(lab, box.get_center())
    g = VGroup(box, lab)
    g.box, g.label = box, lab
    return g


class S7Sharing(Slide):
    def construct(self):
        apply_theme(self)

        # -------------------------------------------------------------
        # Beat 1: pickup — S5 mapping, unpaid sharing bill
        # -------------------------------------------------------------
        heading = _fit_title("Two requests in one pool still did not share")

        logical = _Stack(
            "Logical blocks",
            [
                _PRow(0, ["Four", "score", "and", "seven"]),
                _PRow(1, ["years", "ago", "our", "fathers"]),
                _PRow(2, ["brought", "", "", ""]),
            ],
        )
        table = _Table(3, "Block table")
        physical = _Stack(
            "Physical KV blocks",
            [
                _PRow(7, ["Four", "score", "and", "seven"]),
                _PRow(1, ["years", "ago", "our", "fathers"]),
                _PRow(3, ["brought", "", "", ""]),
            ],
        )
        p7, p1, p3 = physical.rows
        picture = VGroup(logical, table, physical)
        picture.arrange(RIGHT, buff=0.70, aligned_edge=UP)
        _place_under_title(picture, heading)

        self.play(FadeIn(heading, shift=DOWN * 0.08), FadeIn(picture), run_time=0.75)
        self.play(
            LaggedStart(
                table.set_mapping(0, 7, "4/4", ACCENT),
                table.set_mapping(1, 1, "4/4", ACCENT),
                table.set_mapping(2, 3, "1/4", ACCENT),
                lag_ratio=0.18,
            ),
            run_time=0.8,
        )
        a07 = arrow(table.row_right(0), p7.left_port(), color=ACCENT, buff=0.06)
        a11 = arrow(table.row_right(1), p1.left_port(), color=ACCENT, buff=0.06)
        a23 = arrow(table.row_right(2), p3.left_port(), color=ACCENT, buff=0.06)
        self.play(LaggedStart(shoot(a07), shoot(a11), shoot(a23), lag_ratio=0.28), run_time=0.85)
        foot = _foot("Fig. 7 interleaved them. Contiguous KV still cannot share.", color=FG)
        self.play(FadeIn(foot), run_time=0.4)
        self.wait(0.3)
        self.next_slide()

        # -------------------------------------------------------------
        # Beat 2: parallel sampling — rewind to the prompt, two samples
        # -------------------------------------------------------------
        heading = self._retitle(
            heading,
            "Parallel sampling: one prompt, two tries",
            FadeOut(logical),
            FadeOut(table),
            FadeOut(a07), FadeOut(a11), FadeOut(a23),
            FadeOut(foot),
        )
        self._dump(VGroup(logical, table, a07, a11, a23, foot))

        a1_table = _Table(2, "Sample A1", head_color=ACCENT)
        a2_table = _Table(2, "Sample A2", head_color=ACCENT2)
        phys_start = physical.get_center()
        trio = VGroup(a1_table, physical, a2_table)
        trio.arrange(RIGHT, buff=1.00, aligned_edge=UP)
        if trio.width > FRAME_W:
            trio.arrange(RIGHT, buff=0.70, aligned_edge=UP)
        trio.next_to(heading, DOWN, buff=0.40)
        physical.body.shift(UP * (a1_table.rows[0].get_center()[1] - p7.cells.get_center()[1]))
        phys_end = physical.get_center()
        a1_end, a2_end = a1_table.get_center(), a2_table.get_center()
        physical.move_to(phys_start)
        a1_table.move_to(a1_end)
        a2_table.move_to(a2_end)
        self.play(physical.animate.move_to(phys_end), run_time=0.55)
        self.play(
            FadeIn(a1_table, shift=RIGHT * 0.18),
            FadeIn(a2_table, shift=LEFT * 0.18),
            run_time=0.5,
        )
        self.play(
            _fill_slot(p1.kv, 3, "", BLOCK_FILL, FG),
            _fill_slot(p3.kv, 0, "", BLOCK_FILL, FG),
            p1.reserve(3),
            run_time=0.65,
        )
        foot = _foot("Contiguous systems copy the whole prompt KV, once per sample.", color=FG)
        self.play(FadeIn(foot), run_time=0.4)
        self.wait(0.3)
        self.next_slide()

        # -------------------------------------------------------------
        # Beat 3: Fig 8 shared mapping, ref count 2
        # -------------------------------------------------------------
        heading = self._retitle(
            heading,
            "Both samples point at the same physical blocks",
            FadeOut(foot),
        )
        self._dump(foot)

        self.play(
            LaggedStart(
                a1_table.set_mapping(0, 7, "4/4", ACCENT),
                a1_table.set_mapping(1, 1, "3/4", ACCENT),
                a2_table.set_mapping(0, 7, "4/4", ACCENT2),
                a2_table.set_mapping(1, 1, "3/4", ACCENT2),
                lag_ratio=0.12,
            ),
            run_time=0.85,
        )
        badge7 = _badge(p7, 2, ACCENT2)
        badge1 = _badge(p1, 2, ACCENT2)
        arr_a1_0 = _map_arrow(a1_table.row_right(0), p7.left_port(), ACCENT)
        arr_a1_1 = _map_arrow(a1_table.row_right(1), p1.left_port(), ACCENT)
        arr_a2_0 = _map_arrow(a2_table.row_left(0), badge7.get_left(), ACCENT2, buff=0.08)
        arr_a2_1 = _map_arrow(a2_table.row_left(1), badge1.get_left(), ACCENT2, buff=0.08)
        arr = VGroup(arr_a1_0, arr_a1_1, arr_a2_0, arr_a2_1)
        self.play(GrowFromCenter(badge7), GrowFromCenter(badge1), run_time=0.35)
        self.play(_shoot(arr_a1_0), _shoot(arr_a2_0), run_time=0.45)
        self.play(_shoot(arr_a1_1), _shoot(arr_a2_1), run_time=0.45)
        foot = _foot("No copy. Two pointers. Reference count 2.", color=FG)
        self.play(FadeIn(foot), run_time=0.4)
        self.wait(0.3)
        self.next_slide()

        # -------------------------------------------------------------
        # Beat 4: packed block stays; only the last block can be written
        # -------------------------------------------------------------
        heading = self._retitle(heading, "Only the last block can ever be written", FadeOut(foot))
        self._dump(foot)

        ring7 = SurroundingRectangle(p7.cells, buff=0.08, color=GOOD, stroke_width=2.4, corner_radius=0.08)
        ring1 = SurroundingRectangle(p1.cells, buff=0.08, color=WARN, stroke_width=2.4, corner_radius=0.08)
        self.play(FadeIn(ring7), Indicate(p7.kv, color=GOOD, scale_factor=1.02), run_time=0.55)
        self.play(FadeIn(ring1), Indicate(p1.kv, color=WARN, scale_factor=1.02), run_time=0.55)
        foot = _foot("7 is packed and stays shared. 1 has the only open slot.", color=FG)
        self.play(FadeIn(foot), run_time=0.4)
        self.wait(0.3)
        self.next_slide()

        # -------------------------------------------------------------
        # Beat 5: copy-on-write — A1 writes "mothers" into phys 3
        # -------------------------------------------------------------
        heading = self._retitle(
            heading,
            "Copy-on-write: A1 writes \"mothers\"",
            FadeOut(ring7), FadeOut(ring1), FadeOut(foot),
        )
        self._dump(VGroup(ring7, ring1, foot))

        flash = SurroundingRectangle(p1.cells, buff=0.08, color=WARN, stroke_width=2.5, corner_radius=0.06)
        self.play(FadeIn(flash), run_time=0.3)
        self.play(flash.animate.move_to(p3.cells.get_center()), run_time=0.6)
        self.play(
            LaggedStart(
                _fill_slot(p3.kv, 0, "years", ACCENT),
                _fill_slot(p3.kv, 1, "ago", ACCENT),
                _fill_slot(p3.kv, 2, "our", ACCENT),
                lag_ratio=0.18,
            ),
            run_time=0.7,
        )
        self.play(_fill_slot(p3.kv, 3, "mothers", ACCENT), FadeOut(flash), run_time=0.5)
        badge3 = _badge(p3, 1, ACCENT)
        new_b1 = _badge(p1, 1, ACCENT2)
        self.play(
            FadeOut(badge1),
            FadeIn(new_b1),
            GrowFromCenter(badge3),
            a1_table.set_mapping(1, 3, "4/4", ACCENT),
            run_time=0.6,
        )
        badge1 = new_b1
        new_a1b = _map_arrow(a1_table.row_right(1), p3.left_port(), ACCENT)
        self.play(FadeOut(arr_a1_1), FadeIn(new_a1b), run_time=0.5)
        arr.submobjects[1] = new_a1b
        foot = _foot("Only the last, not-yet-full block is copied.", color=FG)
        self.play(FadeIn(foot), run_time=0.4)
        self.wait(0.3)
        self.next_slide()

        # -------------------------------------------------------------
        # Beat 6: A2 writes in place; OS fork
        # -------------------------------------------------------------
        heading = self._retitle(
            heading,
            "A2 writes in place — same trick as OS fork",
            FadeOut(foot),
        )
        self._dump(foot)

        self.play(
            Indicate(p1.kv.cells[3], color=ACCENT2, scale_factor=1.08),
            _fill_slot(p1.kv, 3, "fathers", ACCENT2),
            a2_table.set_mapping(1, 1, "4/4", ACCENT2),
            run_time=0.8,
        )
        in_place = caption("ref = 1  →  write in place, no copy")
        in_place.next_to(physical.body, DOWN, buff=0.28)
        _fit(in_place, 8.0)
        self.play(FadeIn(in_place), run_time=0.35)
        foot = _foot("Share the pages. Copy only the one that someone writes.", color=GOOD)
        self.play(FadeIn(foot), run_time=0.4)
        self.wait(0.3)
        self.next_slide()

        # -------------------------------------------------------------
        # Beat 7: Fig 9 beam-search tree
        # -------------------------------------------------------------
        fig8 = VGroup(
            heading, physical, a1_table, a2_table, arr,
            badge7, badge1, badge3, in_place, foot, new_a1b,
        )
        heading = self._retitle(
            heading,
            "Beam search: sharing a whole tree",
            *[FadeOut(m) for m in (
                physical, a1_table, a2_table, arr,
                badge7, badge1, badge3, in_place, foot, new_a1b,
            )],
        )
        self._dump(fig8)

        tree, tree_parts = self._beam_tree()
        _place_under_title(tree, heading, bottom=-3.08)
        c_labels, b0, b1, b3, b2, b4, heads, lines, trunk_note = tree_parts

        self.play(GrowFromCenter(b0), run_time=0.4)
        self.play(
            GrowFromCenter(b1), GrowFromCenter(b3),
            Create(lines["trunk"]),
            run_time=0.5,
        )
        self.play(
            GrowFromCenter(b2), GrowFromCenter(b4),
            Create(lines["side"]),
            run_time=0.45,
        )
        self.play(
            LaggedStart(
                *[AnimationGroup(GrowFromCenter(h), FadeIn(c)) for h, c in zip(heads, c_labels)],
                lag_ratio=0.12,
            ),
            Create(lines["heads"]),
            FadeIn(trunk_note),
            run_time=0.75,
        )
        foot = _foot("They share generated prefixes, not just the prompt.", color=FG)
        self.play(FadeIn(foot), run_time=0.4)
        self.wait(0.3)
        self.next_slide()

        # -------------------------------------------------------------
        # Beat 8: prune — candidates 0 and 3 drop
        # -------------------------------------------------------------
        heading = self._retitle(heading, "One step later: two candidates drop out", FadeOut(foot))
        self._dump(foot)

        c0, c1, c2, c3 = c_labels
        h0, h1, h2, h3 = heads
        crosses = VGroup(
            Cross(c0, color=BAD, stroke_width=3.5),
            Cross(c3, color=BAD, stroke_width=3.5),
            Cross(h0, color=BAD, stroke_width=3),
            Cross(h3, color=BAD, stroke_width=3),
            Cross(b2, color=BAD, stroke_width=3),
            Cross(b4, color=BAD, stroke_width=3),
        )
        self.play(FadeIn(crosses), run_time=0.45)

        tray = RoundedRectangle(
            width=6.2, height=0.80, corner_radius=0.12,
            fill_color=GOOD, fill_opacity=0.08,
            stroke_color=GOOD, stroke_width=1.6,
        )
        pool_lab = small("Free pool", color=GOOD)
        pool = VGroup(pool_lab, tray).arrange(DOWN, buff=0.10)
        pool.to_edge(DOWN, buff=0.78)

        doomed = [b2, b4, h0, h3]
        copies = VGroup()
        for m in doomed:
            c = m.copy()
            if m.width > 1e-6:
                c.scale_to_fit_width(m.width)
            c.move_to(m.get_center())
            c.box, c.lab = c[0], c[1]
            copies.add(c)
        cx, cy = tray.get_center()[:2]
        dests = [
            [cx - 1.95, cy, 0],
            [cx - 0.65, cy, 0],
            [cx + 0.65, cy, 0],
            [cx + 1.95, cy, 0],
        ]

        self.add(copies)
        gone = VGroup(crosses, c_labels, heads, b2, b4, lines["side"], lines["heads"])
        self.play(FadeOut(gone), FadeIn(pool), run_time=0.45)
        self._dump(gone)
        self.play(
            *[c.animate.move_to(d) for c, d in zip(copies, dests)],
            run_time=0.75,
        )
        self.play(
            *[c.box.animate.set_stroke(GOOD).set_fill(BLOCK_FILL, 1.0) for c in copies],
            *[c.lab.animate.set_color(GOOD) for c in copies],
            run_time=0.35,
        )
        new_heads = VGroup(
            _chip("N0", fill=ACCENT, stroke=ACCENT, fg=BG),
            _chip("N1", fill=ACCENT, stroke=ACCENT, fg=BG),
            _chip("N2", fill=ACCENT2, stroke=ACCENT2, fg=BG),
            _chip("N3", fill=ACCENT2, stroke=ACCENT2, fg=BG),
        )
        new_heads.arrange(DOWN, buff=0.14)
        new_heads.next_to(b3, RIGHT, buff=0.52)
        if new_heads.get_right()[0] > 6.4:
            new_heads.shift(LEFT * (new_heads.get_right()[0] - 6.4))
        new_lines = VGroup(
            *[_link(b3, nh) for nh in new_heads]
        )
        self.play(
            LaggedStart(*[GrowFromCenter(h) for h in new_heads], lag_ratio=0.12),
            Create(new_lines),
            run_time=0.65,
        )
        foot = _foot("Ref → 0, reclaimed instantly. Up to 55% saved — measured in the results.", color=FG)
        self.play(FadeIn(foot), FadeOut(trunk_note), run_time=0.45)
        self.wait(0.3)
        self.next_slide()

        # -------------------------------------------------------------
        # Beat 9: Fig 10 shared prefix as sequences
        # -------------------------------------------------------------
        beam = VGroup(
            heading, b0, b1, b3, heads, lines["trunk"],
            copies, pool, new_heads, new_lines, foot, c_labels,
        )
        heading = self._retitle(
            heading,
            "Shared prefix: one instruction, many requests",
            *[FadeOut(m) for m in (
                b0, b1, b3, lines["trunk"],
                copies, pool, new_heads, new_lines, foot,
            )],
        )
        self._dump(beam)

        prefix_card, seq_cards = self._prefix_sequences()
        seqs = VGroup(prefix_card, seq_cards)
        _place_under_title(seqs, heading, bottom=-3.05)
        self.play(FadeIn(prefix_card, shift=DOWN * 0.1), run_time=0.6)
        self.play(FadeIn(seq_cards, shift=UP * 0.08), run_time=0.55)
        foot = _foot("Most of each request is the same instruction.", color=FG)
        self.play(FadeIn(foot), run_time=0.4)
        self.wait(0.3)
        self.next_slide()

        # -------------------------------------------------------------
        # Beat 10: the card is the cache; tables point at it
        # -------------------------------------------------------------
        heading = self._retitle(
            heading,
            "Both tables point at the cached prefix",
            FadeOut(foot), FadeOut(seq_cards),
        )
        self._dump(VGroup(foot, seq_cards))

        ta = _Table(2, "Request A", head_color=ACCENT)
        tb = _Table(2, "Request B", head_color=ACCENT2)
        a_q = _tok("cheese?", ACCENT)
        b_q = _tok("I love you?", ACCENT2)
        a_col = VGroup(ta, a_q).arrange(DOWN, buff=0.22)
        b_col = VGroup(tb, b_q).arrange(DOWN, buff=0.22)

        self.play(prefix_card.animate.scale(0.82).move_to(UP * 1.35), run_time=0.55)
        pair = VGroup(a_col, b_col).arrange(RIGHT, buff=1.6)
        pair.next_to(prefix_card, DOWN, buff=0.38)
        if pair.get_bottom()[1] < -3.10:
            pair.shift(UP * (-3.10 - pair.get_bottom()[1]))
        if pair.width > FRAME_W:
            pair.scale_to_fit_width(FRAME_W)
            pair.next_to(prefix_card, DOWN, buff=0.32)
        self.play(FadeIn(a_col, shift=UP * 0.12), FadeIn(b_col, shift=UP * 0.12), run_time=0.6)
        self.play(
            ta.set_mapping(0, 0, "4/4", WARN),
            ta.set_mapping(1, 1, "4/4", WARN),
            tb.set_mapping(0, 0, "4/4", WARN),
            tb.set_mapping(1, 1, "4/4", WARN),
            run_time=0.55,
        )
        pfx_arrows = VGroup(
            arrow(ta.head.get_top(), prefix_card.get_bottom() + LEFT * 1.8, color=WARN, buff=0.14),
            arrow(tb.head.get_top(), prefix_card.get_bottom() + RIGHT * 1.8, color=WARN, buff=0.14),
        )
        bdg = RefCountBadge(value=2, color=ACCENT2)
        bdg.scale(0.9)
        bdg.move_to(prefix_card.get_corner(UR) + LEFT * 0.35 + DOWN * 0.38)
        self.play(shoot(pfx_arrows[0]), shoot(pfx_arrows[1]), GrowFromCenter(bdg), run_time=0.6)
        foot = _foot("Prefix computed once. Last shared block is copy-on-write.", color=FG)
        self.play(FadeIn(foot), run_time=0.4)
        self.wait(0.3)
        self.next_slide()

        # -------------------------------------------------------------
        # Beat 11: fork / append / free
        # -------------------------------------------------------------
        pfx_scene = VGroup(heading, prefix_card, a_col, b_col, pfx_arrows, bdg, foot)
        heading = self._retitle(
            heading,
            "Three primitives, not three features",
            FadeOut(prefix_card), FadeOut(a_col), FadeOut(b_col),
            FadeOut(pfx_arrows), FadeOut(bdg), FadeOut(foot),
        )
        self._dump(pfx_scene)

        cards = VGroup(
            self._prim("fork", "New sequence from an old one.\nShare its blocks. Bump the refs.", ACCENT),
            self._prim("append", "Write a new token.\nCopy-on-write if ref > 1.", ACCENT2),
            self._prim("free", "Drop a sequence.\nDecrement. Reclaim at 0.", GOOD),
        )
        cards.arrange(RIGHT, buff=0.35)
        _place_under_title(cards, heading, bottom=-2.35)
        self.play(LaggedStart(*[FadeIn(c, shift=UP * 0.12) for c in cards], lag_ratio=0.16), run_time=0.9)

        compose = VGroup(
            caption("parallel sampling  =  fork, then append"),
            caption("beam search  =  fork / append / free, every step"),
            caption("shared prefix  =  fork onto a cache computed once"),
        )
        compose.arrange(DOWN, buff=0.12)
        compose.next_to(cards, DOWN, buff=0.32)
        _fit(compose, FRAME_W)
        compose.next_to(cards, DOWN, buff=0.32)
        self.play(FadeIn(compose), run_time=0.5)
        self.wait(0.3)
        self.next_slide()

        # -------------------------------------------------------------
        # Beat 12: landing
        # -------------------------------------------------------------
        heading = self._retitle(heading, "Sequences can now point at the same physical memory")
        land = _foot("Next: the free pool runs dry.", color=ACCENT)
        self.play(FadeIn(land), run_time=0.45)
        self.wait(0.3)
        self.next_slide()

    # ------------------------------------------------------------------
    def _beam_tree(self):
        c0 = small("C0", color=ACCENT)
        c1 = small("C1", color=ACCENT)
        c2 = small("C2", color=ACCENT2)
        c3 = small("C3", color=ACCENT2)
        c_labels = VGroup(c0, c1, c2, c3)

        b0 = _chip("B0", fill=WARN, stroke=WARN, fg=BG)
        b1 = _chip("B1", fill=ACCENT, stroke=ACCENT, fg=BG)
        b3 = _chip("B3", fill=ACCENT, stroke=ACCENT, fg=BG)
        b2 = _chip("B2", fill=ACCENT2, stroke=ACCENT2, fg=BG)
        b4 = _chip("B4", fill=ACCENT2, stroke=ACCENT2, fg=BG)
        h0 = _chip("H0", fill=ACCENT, stroke=ACCENT, fg=BG)
        h1 = _chip("H1", fill=ACCENT, stroke=ACCENT, fg=BG)
        h2 = _chip("H2", fill=ACCENT2, stroke=ACCENT2, fg=BG)
        h3 = _chip("H3", fill=ACCENT2, stroke=ACCENT2, fg=BG)
        heads = VGroup(h0, h1, h2, h3)

        b0.move_to(ORIGIN + LEFT * 0.35)
        b1.move_to(b0.get_center() + RIGHT * 1.55 + UP * 0.72)
        b3.move_to(b1.get_center() + RIGHT * 1.55)
        b2.move_to(b0.get_center() + RIGHT * 1.55 + DOWN * 0.92)
        b4.move_to(b2.get_center() + RIGHT * 1.55)
        h0.move_to(b3.get_center() + RIGHT * 1.55 + UP * 0.66)
        h1.move_to(b3.get_center() + RIGHT * 1.55 + UP * 0.00)
        h2.move_to(b3.get_center() + RIGHT * 1.55 + DOWN * 0.66)
        h3.move_to(b4.get_center() + RIGHT * 1.55)

        c0.next_to(h0, RIGHT, buff=0.16)
        c1.next_to(h1, RIGHT, buff=0.16)
        c2.next_to(h2, RIGHT, buff=0.16)
        c3.next_to(h3, RIGHT, buff=0.16)

        ln_trunk = VGroup(_link(b0, b1), _link(b1, b3))
        ln_side = VGroup(_link(b0, b2), _link(b2, b4))
        ln_heads = VGroup(_link(b3, h0), _link(b3, h1), _link(b3, h2), _link(b4, h3))
        lines = {"trunk": ln_trunk, "side": ln_side, "heads": ln_heads}

        b0_badge = RefCountBadge(value=4, color=ACCENT2)
        b0_badge.scale(0.70)
        b0_badge.move_to(b0.box.get_corner(UR) + RIGHT * 0.04 + UP * 0.04)
        b0.add(b0_badge)

        trunk_note = caption("shared trunk  ·  C3 diverged early  ·  four live heads")
        tree_body = VGroup(c_labels, b0, b1, b3, b2, b4, heads, ln_trunk, ln_side, ln_heads)
        trunk_note.next_to(tree_body, DOWN, buff=0.28)
        _fit(trunk_note, FRAME_W)
        trunk_note.next_to(tree_body, DOWN, buff=0.28)

        tree = VGroup(tree_body, trunk_note)
        parts = (c_labels, b0, b1, b3, b2, b4, heads, lines, trunk_note)
        return tree, parts

    def _prefix_sequences(self):
        lines = [
            "Translate English to French:",
            "sea otter  =>  loutre de mer",
            "peppermint  =>  menthe poivrée",
            "plush giraffe  =>  girafe en peluche",
        ]
        body_lines = VGroup(*[small(s, color=BG) for s in lines])
        body_lines.arrange(DOWN, buff=0.10, aligned_edge=LEFT)
        box = RoundedRectangle(
            width=min(body_lines.width + 0.80, 12.2),
            height=body_lines.height + 0.58,
            corner_radius=0.14,
            fill_color=WARN, fill_opacity=1.0,
            stroke_color=WARN, stroke_width=1.5,
        )
        lab = caption("Shared prefix  (few-shot instruction)")
        body_lines.move_to(box.get_center())
        card = VGroup(box, body_lines, lab)
        lab.next_to(box, UP, buff=0.12)

        def _seq(name, question, answer, color):
            head = small(name, color=color)
            q = _tok(question, color)
            arrow_t = caption("→")
            a = _tok(answer, MUTED, fg=FG)
            row = VGroup(q, arrow_t, a).arrange(RIGHT, buff=0.18)
            return VGroup(head, row).arrange(DOWN, buff=0.14, aligned_edge=LEFT)

        sa = _seq("Request A", "cheese?", "fromage", ACCENT)
        sb = _seq("Request B", "I love you?", "Je t'aime", ACCENT2)
        seqs = VGroup(sa, sb).arrange(RIGHT, buff=1.8, aligned_edge=UP)
        VGroup(card, seqs).arrange(DOWN, buff=0.42)
        return card, seqs

    def _prim(self, name, body_str, color):
        box = RoundedRectangle(
            width=4.05, height=2.15, corner_radius=0.14,
            fill_color=BLOCK_FILL, fill_opacity=1.0,
            stroke_color=color, stroke_width=2.5,
        )
        h = body(name, font_size=28, color=color)
        lines = VGroup(*[small(s, color=FG) for s in body_str.split("\n")])
        lines.arrange(DOWN, buff=0.10, aligned_edge=LEFT)
        for ln in lines:
            _fit(ln, 3.55)
        inner = VGroup(h, lines).arrange(DOWN, buff=0.22)
        inner.move_to(box.get_center())
        return VGroup(box, inner)

    def _retitle(self, old, new_str, *anims, run_time=0.55):
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
        if mob is None:
            return
        family = list(mob.get_family())
        self.remove(mob, *family)
        mob.set_opacity(0)
        mob.shift(LEFT * 40)
