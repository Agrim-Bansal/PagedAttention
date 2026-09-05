"""S7 — Sharing across sequences (Act III).

NARRATION
---------
Beat 1 — Parallel sampling, the setup.
A common trick for better outputs: take one prompt and sample several
completions from it — "give me 2 tries at an answer." Each sample, call them
A1 and A2, is really its own sequence with its own KV cache. But they all
start from the exact same prompt. [PAUSE] In a contiguous-memory system,
that means copying the prompt's entire KV cache once per sample. With
blocks, we don't have to.

Beat 2 — Same physical blocks, two logical views.
Here's Figure 8 from the paper. A1's block table and A2's block table both
map their logical blocks 0 and 1 to the *same* physical blocks. No copy has
happened — both samples are just pointers into one shared region. Each
shared physical block carries a reference count: here, 2, because two
logical blocks point at it.

Beat 3 — Copy-on-write.
Block 0 is completely full — four out of four prompt tokens — so nobody
ever needs to write to it again; it can stay shared forever. Block 1 is
different: it still has one open slot. When A1 generates its next token and
tries to write into that slot, vLLM checks the reference count, sees it's 2,
and refuses to write in place. Instead: allocate a fresh physical block,
copy the old block's contents into it, write A1's new token into the copy,
and decrement the original block's count to 1. [PAUSE] A2 keeps using the
original block, untouched. Notice the cost: only that *last*, not-yet-full
block is ever copied. Every earlier, fully-packed block just stays shared.

Beat 4 — Beam search shares a tree.
Beam search shares even more aggressively. Multiple candidate beams don't
just share the prompt — they share prefixes of each other's generated
tokens too, because at every step they're extensions of a shared history.
Picture it as a tree: one shared trunk, branching into candidates, each
candidate touching only the new blocks it needed to extend the sequence.

Beat 5 — And it changes every step.
Now watch what happens when a candidate falls out of the top-k and gets
dropped. Every physical block that only it was using has its reference
count drop to zero — and those blocks go straight back to the free pool,
instantly, without any bulk copy. [PAUSE] The sharing pattern is being
recomputed, cheaply, at every single decoding step.

Beat 6 — Shared prefix / system prompt.
This same trick shows up in production systems today: a fixed instruction
or a handful of few-shot examples that every request in a workload shares.
vLLM computes that shared prefix's KV blocks exactly once, caches them, and
every new request's block table just points at those cached blocks. Only
the request-specific suffix — the actual question — needs new computation
and new blocks.

Beat 7 — What it's worth.
On the paper's numbers: parallel sampling saves 6.1 to 9.8 percent of memory
on Alpaca, and 16.2 to 30.5 percent on ShareGPT. Beam search saves far more
— 37.6 to 55.2 percent on Alpaca, 44.3 to 66.3 percent on ShareGPT. [PAUSE]
None of this is possible in a contiguous-memory system — there, "sharing" a
block would mean the copies have to physically exist somewhere, defeating
the point.

Beat 8 — Landing.
Two small mechanisms — reference counts and copy-on-write — turn a memory
layout trick into free, automatic sharing across requests.
"""

from manim import (
    DOWN,
    LEFT,
    ORIGIN,
    RIGHT,
    UP,
    UR,
    Cross,
    FadeIn,
    FadeOut,
    Line,
    Rectangle,
    Transform,
    VGroup,
)
from manim_slides import Slide

from talk.theme import *
from talk.components import *


def _badge_at(block, color=ACCENT2, value=1):
    badge = RefCountBadge(value=value, color=color)
    badge.move_to(block.cells.get_corner(UR))
    return badge


class S7Sharing(Slide):
    def construct(self):
        apply_theme(self)

        # -------------------------------------------------------------
        # Beat 1: parallel sampling setup
        # -------------------------------------------------------------
        heading = title("Parallel sampling: one prompt, many samples")
        heading.scale(0.85)
        self.play(FadeIn(heading))

        prompt = token_sequence(FOUR_SCORE[:7], height=0.55, font_size=SMALL_SIZE)
        prompt.scale_to_fit_width(min(prompt.width, 10.5))
        prompt_label = caption("Prompt")
        prompt_label.next_to(prompt, UP, buff=0.2)
        prompt_group = VGroup(prompt_label, prompt)
        prompt_group.move_to(UP * 1.6)

        a1_box = TokenBox("Sample A1", color=BG, fill=ACCENT, height=0.6, font_size=SMALL_SIZE)
        a2_box = TokenBox("Sample A2", color=BG, fill=ACCENT2, height=0.6, font_size=SMALL_SIZE)
        a1_box.move_to(LEFT * 3 + DOWN * 1.4)
        a2_box.move_to(RIGHT * 3 + DOWN * 1.4)

        self.play(FadeIn(prompt_group))
        self.next_slide()

        arrow_a1 = arrow_map(prompt_group, a1_box, color=ACCENT)
        arrow_a2 = arrow_map(prompt_group, a2_box, color=ACCENT2)
        self.play(FadeIn(a1_box), FadeIn(a2_box))
        self.play(shoot(arrow_a1), shoot(arrow_a2))
        note1 = caption("Same prompt, different sampled continuations")
        note1.next_to(VGroup(a1_box, a2_box), DOWN, buff=0.6)
        self.play(FadeIn(note1))
        self.wait(0.3)
        self.next_slide()
        self.play(
            FadeOut(heading), FadeOut(prompt_group), FadeOut(a1_box), FadeOut(a2_box),
            FadeOut(arrow_a1), FadeOut(arrow_a2), FadeOut(note1),
        )

        # -------------------------------------------------------------
        # Beat 2: Fig 8 — shared physical blocks, ref count 2
        # -------------------------------------------------------------
        heading2 = title("Both samples point at the same physical blocks")
        heading2.scale(0.75)
        self.play(FadeIn(heading2))

        a1_title = small("Sample A1", font_size=SMALL_SIZE, color=ACCENT)
        a1_table = BlockTable(n_rows=2, title="A1 block table")
        a1_group = VGroup(a1_title, a1_table)
        a1_title.next_to(a1_table, UP, buff=0.25)
        a1_group.move_to(LEFT * 5.0 + DOWN * 0.2)

        a2_title = small("Sample A2", font_size=SMALL_SIZE, color=ACCENT2)
        a2_table = BlockTable(n_rows=2, title="A2 block table")
        a2_group = VGroup(a2_title, a2_table)
        a2_title.next_to(a2_table, UP, buff=0.25)
        a2_group.move_to(RIGHT * 5.0 + DOWN * 0.2)

        phys0 = KVBlock(slots=4, cell=0.55, words=["Four", "score", "and", "seven"], index=0)
        phys1 = KVBlock(slots=4, cell=0.55, words=["years", "ago", "our", ""], index=1)
        for cell in phys0.cells:
            cell.set_fill(ACCENT, opacity=1.0)
        for i, cell in enumerate(phys1.cells):
            cell.set_fill(ACCENT if i < 3 else BLOCK_FILL, opacity=1.0)
        phys_group = VGroup(phys0, phys1)
        phys_group.arrange(DOWN, buff=0.6)
        phys_title = small("Physical KV blocks", font_size=SMALL_SIZE, color=FG)
        phys_title.next_to(phys_group, UP, buff=0.3)
        physical = VGroup(phys_title, phys_group)
        physical.move_to(DOWN * 0.2)

        self.play(FadeIn(a1_group), FadeIn(a2_group), FadeIn(physical))
        self.play(a1_table.set_row(0, 0, 4), a1_table.set_row(1, 1, 3))
        self.play(a2_table.set_row(0, 0, 4), a2_table.set_row(1, 1, 3))

        arrow_1a = arrow(a1_table.get_right(), phys0.cells.get_left(), color=ACCENT, buff=0.15)
        arrow_1b = arrow(a1_table.get_right(), phys1.cells.get_left(), color=ACCENT, buff=0.15)
        arrow_2a = arrow(a2_table.get_left(), phys0.cells.get_right(), color=ACCENT2, buff=0.15)
        arrow_2b = arrow(a2_table.get_left(), phys1.cells.get_right(), color=ACCENT2, buff=0.15)
        self.play(*[shoot(a) for a in (arrow_1a, arrow_1b)])
        self.play(*[shoot(a) for a in (arrow_2a, arrow_2b)])

        badge0 = _badge_at(phys0, color=ACCENT2, value=2)
        badge1 = _badge_at(phys1, color=ACCENT2, value=2)
        self.play(FadeIn(badge0), FadeIn(badge1))
        self.wait(0.3)
        self.next_slide()

        # -------------------------------------------------------------
        # Beat 3: copy-on-write
        # -------------------------------------------------------------
        self.play(FadeOut(heading2))
        heading3 = title("Copy-on-write: only the last block moves")
        heading3.scale(0.8)
        self.play(FadeIn(heading3))

        cow_label = small("A1 writes 'mothers' into block 1", font_size=SMALL_SIZE, color=ACCENT)
        cow_label.move_to(UP * 1.6)
        self.play(FadeIn(cow_label))
        self.next_slide()

        new_block = KVBlock(slots=4, cell=0.55, words=["years", "ago", "our", "mothers"], index=2)
        for cell in new_block.cells:
            cell.set_fill(ACCENT, opacity=1.0)
        new_block.move_to(phys1.get_center() + RIGHT * 3.4)

        cow_arrow = arrow(
            phys1.cells.get_right(), new_block.cells.get_left(),
            color=WARN, buff=0.15,
        )
        cow_arrow_label = caption("copy")
        cow_arrow_label.next_to(cow_arrow, UP, buff=0.1)

        self.play(FadeIn(new_block), shoot(cow_arrow), FadeIn(cow_arrow_label))
        self.play(badge1.set_value(1))
        self.play(a1_table.set_row(1, 2, 4))
        cost_note = caption("Block 0 (full) stays shared forever; only block 1 (the open one) is copied")
        cost_note.next_to(VGroup(a1_group, a2_group, physical, new_block), DOWN, buff=0.5)
        self.play(FadeIn(cost_note))
        self.wait(0.3)
        self.next_slide()
        self.play(
            FadeOut(heading3), FadeOut(cow_label), FadeOut(a1_group), FadeOut(a2_group),
            FadeOut(physical), FadeOut(new_block), FadeOut(cow_arrow), FadeOut(cow_arrow_label),
            FadeOut(badge0), FadeOut(badge1), FadeOut(cost_note),
            FadeOut(arrow_1a), FadeOut(arrow_1b), FadeOut(arrow_2a), FadeOut(arrow_2b),
        )

        # -------------------------------------------------------------
        # Beat 4: beam search — sharing a tree
        # -------------------------------------------------------------
        heading4 = title("Beam search: sharing a whole tree")
        heading4.scale(0.85)
        self.play(FadeIn(heading4))

        root = TokenBox("prompt", color=BG, fill=MUTED, height=0.55, font_size=SMALL_SIZE)
        root.move_to(UP * 1.7)
        root_badge = RefCountBadge(value=2, color=ACCENT2)
        root_badge.move_to(root.get_corner(UR))

        beam1 = TokenBox("beam 1", color=BG, fill=ACCENT, height=0.55, font_size=SMALL_SIZE)
        beam2 = TokenBox("beam 2", color=BG, fill=ACCENT2, height=0.55, font_size=SMALL_SIZE)
        beam1.move_to(LEFT * 2.8 + DOWN * 0.4)
        beam2.move_to(RIGHT * 2.8 + DOWN * 0.4)
        beam1_badge = RefCountBadge(value=1, color=ACCENT)
        beam1_badge.move_to(beam1.get_corner(UR))
        beam2_badge = RefCountBadge(value=1, color=ACCENT2)
        beam2_badge.move_to(beam2.get_corner(UR))

        branch1 = Line(root.get_bottom(), beam1.get_top(), color=MUTED, stroke_width=2.5)
        branch2 = Line(root.get_bottom(), beam2.get_top(), color=MUTED, stroke_width=2.5)

        beam1b = TokenBox("beam 1a", color=BG, fill=ACCENT, height=0.5, font_size=TINY_SIZE)
        beam1c = TokenBox("beam 1b", color=BG, fill=ACCENT, height=0.5, font_size=TINY_SIZE)
        beam1b.move_to(LEFT * 4.2 + DOWN * 2.2)
        beam1c.move_to(LEFT * 1.6 + DOWN * 2.2)
        branch1b = Line(beam1.get_bottom(), beam1b.get_top(), color=MUTED, stroke_width=2)
        branch1c = Line(beam1.get_bottom(), beam1c.get_top(), color=MUTED, stroke_width=2)

        self.play(FadeIn(root), FadeIn(root_badge))
        self.play(shoot(branch1), shoot(branch2), FadeIn(beam1), FadeIn(beam2))
        self.play(FadeIn(beam1_badge), FadeIn(beam2_badge))
        self.next_slide()

        tree_note = caption("Beams share the prompt's blocks, then fork off their own")
        tree_note.next_to(VGroup(beam1b, beam1c, beam2), DOWN, buff=0.9)
        self.play(
            shoot(branch1b), shoot(branch1c),
            FadeIn(beam1b), FadeIn(beam1c),
        )
        self.play(FadeIn(tree_note))
        self.wait(0.3)
        self.next_slide()

        # -------------------------------------------------------------
        # Beat 5: a beam dies, blocks return to the free pool
        # -------------------------------------------------------------
        self.play(FadeOut(heading4), FadeOut(tree_note))
        heading5 = title("One step later: a beam drops out")
        heading5.scale(0.85)
        self.play(FadeIn(heading5))

        cross = Cross(beam2, color=BAD, stroke_width=4)
        self.play(FadeIn(cross))
        self.play(beam2_badge.set_value(0))
        free_pool_label = small("Free pool", font_size=SMALL_SIZE, color=GOOD)
        free_pool_label.to_edge(DOWN, buff=1.1)
        freed_block = Rectangle(
            width=beam2.width, height=beam2.height, fill_color=BLOCK_FILL, fill_opacity=1.0,
            stroke_color=BLOCK_STROKE, stroke_width=2,
        )
        freed_block.next_to(free_pool_label, UP, buff=0.25)
        # Transform beam2 itself (in place) into the freed-block look, rather
        # than transforming a throwaway .copy() into it — the latter leaves
        # an orphaned mobject in the scene that never gets faded out.
        self.play(
            FadeOut(cross), FadeOut(beam2_badge),
            Transform(beam2, freed_block),
            FadeOut(branch2),
        )
        self.play(FadeIn(free_pool_label))
        died_note = caption("Ref count -> 0: its blocks are reclaimed instantly, no bulk copy")
        died_note.next_to(free_pool_label, DOWN, buff=0.3)
        self.play(FadeIn(died_note))
        self.wait(0.3)
        self.next_slide()
        self.play(
            FadeOut(heading5), FadeOut(root), FadeOut(root_badge), FadeOut(branch1),
            FadeOut(beam1), FadeOut(beam1_badge), FadeOut(branch1b), FadeOut(branch1c),
            FadeOut(beam1b), FadeOut(beam1c), FadeOut(beam2), FadeOut(free_pool_label),
            FadeOut(died_note),
        )

        # -------------------------------------------------------------
        # Beat 6: shared prefix / system prompt (Fig 10)
        # -------------------------------------------------------------
        heading6 = title("Shared prefix: one system prompt, many requests")
        heading6.scale(0.7)
        self.play(FadeIn(heading6))

        prefix_block = KVBlock(slots=4, cell=0.5, words=["Translate", "English", "to", "French:"], index=0)
        for cell in prefix_block.cells:
            cell.set_fill(WARN, opacity=1.0)
        prefix_block.move_to(UP * 1.3)
        prefix_label = small("Shared prefix (few-shot examples)", font_size=SMALL_SIZE, color=WARN)
        prefix_label.next_to(prefix_block, UP, buff=0.25)
        prefix_badge = RefCountBadge(value=3, color=ACCENT2)
        prefix_badge.move_to(prefix_block.cells.get_corner(UR))

        self.play(FadeIn(prefix_label), FadeIn(prefix_block), FadeIn(prefix_badge))
        self.next_slide()

        suffixes = VGroup()
        suffix_words = [["cheese?", "", "", ""], ["I", "love", "you?", ""], ["good", "morning", "", ""]]
        req_labels = ["Request 1", "Request 2", "Request 3"]
        for i, (words, lbl) in enumerate(zip(suffix_words, req_labels)):
            blk = KVBlock(slots=4, cell=0.45, words=words, index=1)
            for j, cell in enumerate(blk.cells):
                cell.set_fill(ACCENT if words[j] else BLOCK_FILL, opacity=1.0)
            req_lbl = caption(lbl)
            grp = VGroup(req_lbl, blk)
            req_lbl.next_to(blk, UP, buff=0.15)
            suffixes.add(grp)
        suffixes.arrange(RIGHT, buff=0.9)
        suffixes.move_to(DOWN * 1.6)

        arrows6 = VGroup(*[arrow_map(prefix_block, s, color=MUTED) for s in suffixes])
        self.play(FadeIn(suffixes))
        self.play(*[shoot(a) for a in arrows6])
        suffix_note = caption("Prefix computed once and cached; only the suffix is new per request")
        suffix_note.next_to(suffixes, DOWN, buff=0.45)
        self.play(FadeIn(suffix_note))
        self.wait(0.3)
        self.next_slide()
        self.play(
            FadeOut(heading6), FadeOut(prefix_block), FadeOut(prefix_label), FadeOut(prefix_badge),
            FadeOut(suffixes), FadeOut(arrows6), FadeOut(suffix_note),
        )

        # -------------------------------------------------------------
        # Beat 7: numbers — memory saved by sharing
        # -------------------------------------------------------------
        heading7 = title("What sharing is worth")
        self.play(FadeIn(heading7))

        chart = bar_chart(
            categories=["PS-Alpaca", "PS-ShareGPT", "BS-Alpaca", "BS-ShareGPT"],
            series={"low": [6.1, 16.2, 37.6, 44.3], "high": [9.8, 30.5, 55.2, 66.3]},
            colors={"low": MUTED, "high": ACCENT},
            width=9.0, height=4.2,
            value_labels=True,
        )
        chart.scale(0.85)
        chart.move_to(DOWN * 0.3)
        legend_note = caption("PS = parallel sampling, BS = beam search  (low-high across k)")
        legend_note.next_to(chart, UP, buff=0.15)

        self.play(FadeIn(legend_note))
        self.play(chart.animate_in())
        self.next_slide()

        impossible_note = small("Impossible in contiguous systems: the copies would have to exist", font_size=SMALL_SIZE, color=BAD)
        impossible_note.next_to(chart, DOWN, buff=0.4)
        self.play(FadeIn(impossible_note))
        self.wait(0.3)
        self.next_slide()
        self.play(
            FadeOut(heading7), FadeOut(chart), FadeOut(legend_note),
            FadeOut(impossible_note),
        )

        # -------------------------------------------------------------
        # Beat 8: landing
        # -------------------------------------------------------------
        landing = title("Reference counts + copy-on-write: a memory trick becomes free sharing")
        landing.scale(0.68)
        landing.move_to(ORIGIN)
        self.play(FadeIn(landing))
        self.wait(0.3)
        self.next_slide()
        self.play(FadeOut(landing))
        self.wait(0.3)
