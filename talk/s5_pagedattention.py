"""S5 — PagedAttention: fixed-size blocks, block table, non-contiguous kernel.

Act II core / heaviest scene. Presents PagedAttention entirely on its own terms:
no mention of operating systems, virtual memory, or "pages" as an OS concept
(that reveal is S6). The paper's own word "block" is used throughout.

NARRATION

Beat 1 — The idea in one line
Here's the paper's fix, in one sentence: instead of one growing, contiguous slab
of memory per request, chop each request's KV cache into small fixed-size
chunks called blocks. The paper's default block size is 16 tokens; to keep the
pictures readable I'll draw blocks of 4. Watch our running example, "Four score
and seven years ago our fathers brought forth," get sliced into three blocks of
four tokens each — the last one only half full. [PAUSE] Note that block size is
fixed once and for all, chosen ahead of time — it does not depend on how long
any particular request turns out to be.

Beat 2 — Logical vs. physical blocks
Here's the key trick: blocks don't have to live next to each other in memory.
Each request has a logical view — its blocks in order, 0, 1, 2 — but those
logical blocks can be scattered anywhere in physical GPU memory. A block table
records, for each logical block, which physical block it actually lives in, and
how many of its slots are currently filled. And blocks are handed out lazily:
vLLM only grabs a new physical block once the current last block is completely
full. That laziness is what kills internal fragmentation.

Beat 3 — The kernel: attention over scattered blocks (Fig. 5)
So if the blocks are scattered, how does attention even work? This is the
paper's Figure 5. The query vector for the newest token, "forth," still has to
attend to every earlier token — but those tokens now live in three separate,
non-contiguous physical blocks. The PagedAttention kernel just fetches each
block on its own, computes a partial attention score against it, and combines
the partial results at the end — that's the paper's Equation 4, block-by-block
attention instead of one contiguous sweep. Non-contiguous memory stops being a
problem the moment your kernel is written to expect it.

Beat 4 — Decode walkthrough: prefill (Fig. 6)
Let's walk through this exactly the way the paper does, step by step. Prompt:
"Four score and seven years ago our fathers," eight tokens. Step ①, prefill, packs the first
four tokens, "Four score and seven," into logical block 0, which lands on
physical block 7; the remaining four tokens, "years ago our fathers," go into logical
block 1 on physical block 1. The block table now
has two rows: logical 0 to physical 7, four filled; logical 1 to physical 1,
four filled.

Beat 5 — Step ② — block table grows
First decode step generates "brought." Logical block 1 is completely
full, so vLLM allocates a brand-new logical block 2, mapped to a fresh physical
block 3, and adds a new row to the block table. This is the only moment a new
block ever gets allocated: exactly when the previous one is completely full.

Beat 6 — Step ③ — fill the new block
Second decode step generates "forth," landing in physical block 3 right next to
"brought" — filled count ticks from one to two.

Beat 7 — The punchline on waste
Notice the pattern: at any
moment, at most one block per request is partially empty. Everything else is
either completely full or not yet allocated. That's why the paper measures up
to 96.3% of KV cache memory actually holding real token state — remember that
question mark from the waste chart earlier? This is the answer.

Beat 8 — Two requests at once (Fig. 7)
None of this is special to one request. Here's Request B, "it was the best of
times," arriving while Request A is still running. Its logical blocks land on
physical blocks 2 and 4 — completely interleaved with Request A's blocks 7, 1,
and 3. Blocks 0, 5, and 6 are still sitting free, available to whichever
request needs them next. Physical layout has nothing to do with logical order
anymore.

Beat 9 — Freeing blocks
When a request finishes, every physical block it was using goes straight back
to the free pool — instantly, no matter where in memory those blocks happened
to be. There's no need to find a same-sized contiguous hole for the next
request, because blocks are always the same fixed size. That's external
fragmentation eliminated entirely; the only fragmentation left is at most one
partial block per live request.

Beat 10 — Recap
So: fixed-size blocks, a block table mapping logical to physical, allocation
only on demand, and a kernel built from the ground up to read scattered blocks
and combine the results. Four ideas, and together they take KV cache
utilization from roughly 20 to 40 percent up to 96.3%.
"""

from manim import (
    DOWN,
    LEFT,
    ORIGIN,
    RIGHT,
    UP,
    AnimationGroup,
    Circle,
    FadeIn,
    FadeOut,
    GrowArrow,
    Rectangle,
    Square,
    SurroundingRectangle,
    Text,
    VGroup,
)
from manim_slides import Slide

from talk.theme import (
    ACCENT,
    ACCENT2,
    BODY_SIZE,
    FG,
    GOOD,
    MUTED,
    Q_COLOR,
    SMALL_SIZE,
    TINY_SIZE,
    TITLE_SIZE,
    WARN,
    apply_theme,
    body,
    caption,
    small,
    title,
)
from talk.components import (
    FOUR_SCORE,
    BlockTable,
    KVBlock,
    MemoryBar,
    PhysicalMemGrid,
    TokenBox,
    arrow_map,
    token_sequence,
)


def _step_banner(text_str, color=ACCENT):
    """Small bottom-of-frame step label."""
    t = Text(text_str, font_size=SMALL_SIZE, color=color, weight="BOLD")
    t.to_edge(DOWN, buff=0.3)
    return t


def _query_marker(label="Q"):
    """A small query-vector marker (square + label), Fig-5 style."""
    sq = Square(side_length=0.4, fill_color=Q_COLOR, fill_opacity=0.9, stroke_width=0)
    lbl = Text(label, font_size=TINY_SIZE, color="#0f1117")
    lbl.move_to(sq.get_center())
    return VGroup(sq, lbl)


def _fill_slot_clean(block, i, word, color=ACCENT):
    """Like KVBlock.fill_slot, but crossfades the label (FadeOut old / FadeIn
    new) instead of Transform-morphing glyphs into each other. Transform on
    adjacent cells filling in quick succession made neighboring words
    (e.g. "years"/"ago"/"our") transiently overlap mid-morph; a crossfade
    keeps each label inside its own cell throughout. Mirrors the bookkeeping
    KVBlock.fill_slot does on .labels so later calls (fill/clear) stay
    consistent."""
    new_label = Text(str(word), font_size=TINY_SIZE, color=FG)
    if new_label.width > 0.9 * block.cell:
        new_label.scale_to_fit_width(0.9 * block.cell)
    new_label.move_to(block.cells[i].get_center())
    old_label = block.labels[i]
    block.labels.submobjects[i] = new_label
    return AnimationGroup(
        block.cells[i].animate.set_fill(color, opacity=1.0),
        FadeOut(old_label),
        FadeIn(new_label),
    )


class S5PagedAttention(Slide):
    def construct(self):
        apply_theme(self)

        # ==================================================================
        # Beat 1 — The idea in one line: slice into fixed-size blocks
        # ==================================================================
        t1 = title("The idea: chop the KV cache into blocks")
        sub1 = caption("Paper default: block size 16. Visuals here use 4.")
        sub1.next_to(t1, DOWN, buff=0.25)

        tokens = token_sequence(FOUR_SCORE, gap=0.16, height=0.6, font_size=SMALL_SIZE)
        if tokens.width > 11.4:
            tokens.scale_to_fit_width(11.4)
        tokens.move_to(UP * 0.7)

        self.play(FadeIn(t1), FadeIn(sub1))
        self.play(FadeIn(tokens, shift=UP * 0.1))

        rect0 = SurroundingRectangle(VGroup(tokens[0], tokens[1], tokens[2], tokens[3]), buff=0.12, color=ACCENT, stroke_width=3)
        rect1 = SurroundingRectangle(VGroup(tokens[4], tokens[5], tokens[6], tokens[7]), buff=0.12, color=ACCENT, stroke_width=3)
        x_left = tokens[8].get_left()[0] - 0.12
        y_c = tokens[8].get_center()[1]
        rect2 = Rectangle(width=rect0.width, height=rect0.height, stroke_color=WARN, stroke_width=3, fill_opacity=0)
        rect2.move_to([x_left + rect2.width / 2, y_c, 0])

        lbl0 = small("Block 0", color=ACCENT)
        lbl1 = small("Block 1", color=ACCENT)
        lbl2 = small("Block 2 (2/4)", color=WARN)
        lbl0.next_to(rect0, DOWN, buff=0.2)
        lbl1.next_to(rect1, DOWN, buff=0.2)
        lbl2.next_to(rect2, DOWN, buff=0.2)

        self.play(FadeIn(rect0), FadeIn(lbl0))
        self.play(FadeIn(rect1), FadeIn(lbl1))
        self.play(FadeIn(rect2), FadeIn(lbl2))
        line1 = body("Same fixed size for every block, chosen once, up front.", font_size=SMALL_SIZE, color=MUTED)
        line1.next_to(VGroup(rect0, rect1, rect2, lbl0, lbl1, lbl2), DOWN, buff=0.55)
        self.play(FadeIn(line1))
        self.wait(0.3)
        self.next_slide()

        self.play(FadeOut(*self.mobjects))

        # ==================================================================
        # Beat 2 — Logical vs physical blocks + block table + allocate on demand
        # ==================================================================
        t2 = title("Logical view vs. physical memory")
        self.play(FadeIn(t2))

        logical_row = VGroup(*[KVBlock(slots=4, cell=0.45, index=i, label_pos=DOWN) for i in range(3)])
        logical_row.arrange(RIGHT, buff=0.8)
        logical_row.next_to(t2, DOWN, buff=0.55)
        logical_caption = caption("Logical blocks: the request's own view, in order")
        logical_caption.next_to(logical_row, DOWN, buff=0.3)

        table2 = BlockTable(n_rows=0, title="Block table")

        phys_perm = [3, 0, 4]  # logical i -> physical index (non-contiguous, arbitrary)
        physical_grid = PhysicalMemGrid(n_blocks=5, slots=4, cols=5, cell=0.4, title="Physical GPU memory")

        lower_row = VGroup(table2, physical_grid)
        lower_row.arrange(RIGHT, buff=0.9, aligned_edge=UP)
        lower_row.next_to(logical_caption, DOWN, buff=0.55)

        self.play(FadeIn(logical_row), FadeIn(logical_caption))
        self.play(FadeIn(table2))
        self.play(FadeIn(physical_grid))

        row_anims = []
        arrows2 = VGroup()
        for i in range(3):
            row_anims.append(table2.add_row(phys_perm[i], 4 if i < 2 else 2))
        self.play(AnimationGroup(*row_anims, lag_ratio=0.3))
        for i in range(3):
            a1 = arrow_map(logical_row[i], table2.rows[i]["mobject"][0], color=MUTED)
            a2 = arrow_map(table2.rows[i]["mobject"][1], physical_grid.block(phys_perm[i]), color=ACCENT)
            arrows2.add(a1, a2)
        self.play(AnimationGroup(*[GrowArrow(a) for a in arrows2], lag_ratio=0.08))

        for i in range(2):
            self.play(*[physical_grid.block(phys_perm[i]).set_state(s, "filled") for s in range(4)])
        self.play(
            physical_grid.block(phys_perm[2]).set_state(0, "filled"),
            physical_grid.block(phys_perm[2]).set_state(1, "filled"),
        )
        demand_line = body('A new physical block is grabbed only when the\nlast one fills up completely.', font_size=SMALL_SIZE, color=FG, line_spacing=1.1)
        demand_line.to_edge(DOWN, buff=0.3)
        self.play(FadeIn(demand_line))
        self.wait(0.3)
        self.next_slide()

        self.play(FadeOut(*self.mobjects))

        # ==================================================================
        # Beat 3 — Fig 5: attention over scattered, non-contiguous blocks
        # ==================================================================
        t3 = title("Attention over scattered blocks (Fig. 5)")
        self.play(FadeIn(t3))

        block1 = KVBlock(slots=4, cell=0.55, index=1, words=["years", "ago", "our", "fathers"], label_pos=UP)
        for s in range(4):
            block1.set_state(s, "filled")
        block1.move_to(UP * 1.7 + LEFT * 3.6)

        block2 = KVBlock(slots=4, cell=0.55, index=2, words=["brought", "forth", "", ""], label_pos=UP)
        block2.set_state(0, "filled")
        block2.set_state(1, "filled")
        block2.move_to(UP * 1.7 + RIGHT * 3.6)

        block0 = KVBlock(slots=4, cell=0.55, index=0, words=["Four", "score", "and", "seven"], label_pos=DOWN)
        for s in range(4):
            block0.set_state(s, "filled")
        block0.move_to(DOWN * 2.0)

        query = _query_marker("Q")
        q_label = caption('query = "forth"')
        query.move_to(LEFT * 0.2 + UP * 0.0)
        q_label.next_to(query, DOWN, buff=0.15)

        self.play(FadeIn(query), FadeIn(q_label))

        a1 = arrow_map(query, block1, color=Q_COLOR)
        s1 = small("score block 1", color=ACCENT)
        s1.next_to(block1, UP, buff=0.35)
        self.play(GrowArrow(a1), FadeIn(s1))

        a2 = arrow_map(query, block2, color=Q_COLOR)
        s2 = small("score block 2", color=ACCENT)
        s2.next_to(block2, UP, buff=0.35)
        self.play(GrowArrow(a2), FadeIn(s2))

        a0 = arrow_map(query, block0, color=Q_COLOR)
        s0 = small("score block 0", color=ACCENT)
        s0.next_to(block0, RIGHT, buff=0.4)
        self.play(GrowArrow(a0), FadeIn(s0))

        combine_line = body("kernel fetches each block, scores it, then combines\n(the paper's Eq. 4)", font_size=SMALL_SIZE, color=GOOD, line_spacing=1.1)
        combine_line.to_edge(DOWN, buff=0.4)
        self.play(FadeIn(combine_line))
        self.wait(0.3)
        self.next_slide()

        self.play(FadeOut(*self.mobjects))

        # ==================================================================
        # Beats 4-9 — Decode walkthrough (Fig 6) then two requests (Fig 7)
        # then freeing. Persistent layout: Request A logical blocks (left),
        # block table (left-center), physical grid of 8 blocks (right).
        # Built with .arrange()/.next_to() so nothing overlaps regardless of
        # exact component sizes.
        # ==================================================================
        t4 = title("Decode walkthrough: prefill (Fig. 6)")
        self.play(FadeIn(t4))

        prompt_cap = caption('Step ① — prefill 8 tokens: "Four score and seven years ago our fathers"')
        prompt_cap.next_to(t4, DOWN, buff=0.3)
        self.play(FadeIn(prompt_cap))

        lblock0 = KVBlock(slots=4, cell=0.5, index=0, words=["Four", "score", "and", "seven"], label_pos=DOWN)
        for s in range(4):
            lblock0.set_state(s, "filled")
        lblock1 = KVBlock(slots=4, cell=0.5, index=1, words=["years", "ago", "our", "fathers"], label_pos=DOWN)
        for s in range(4):
            lblock1.set_state(s, "filled")
        # Block 2 is not allocated yet (beat 6+); build it now, invisible, so
        # its space is reserved from the start and nothing needs to move later.
        lblock2 = KVBlock(slots=4, cell=0.5, index=2, label_pos=DOWN)
        logical_row = VGroup(lblock0, lblock1, lblock2)
        logical_row.arrange(RIGHT, buff=0.45)
        lblock2.set_opacity(0)
        logical_title = small("Request A: logical blocks", color=ACCENT)
        logical_stack = VGroup(logical_title, logical_row)
        logical_stack.arrange(DOWN, buff=0.3)
        logical_stack.next_to(prompt_cap, DOWN, buff=0.4)

        table_a = BlockTable(n_rows=0, title="Block table (A)")

        grid = PhysicalMemGrid(n_blocks=8, slots=4, cols=4, cell=0.48, title="Physical KV blocks")

        lower_row = VGroup(table_a, grid)
        lower_row.arrange(RIGHT, buff=0.7, aligned_edge=UP)
        lower_row.next_to(logical_stack, DOWN, buff=0.45)

        self.play(FadeIn(logical_title), FadeIn(lblock0), FadeIn(lblock1))
        self.play(FadeIn(table_a))
        self.play(FadeIn(grid))

        # fill physical block 7 (logical 0) and physical block 1 (logical 1)
        fill7 = AnimationGroup(*[_fill_slot_clean(grid.block(7), i, w, ACCENT) for i, w in enumerate(["Four", "score", "and", "seven"])], lag_ratio=0.15)
        fill1 = AnimationGroup(*[_fill_slot_clean(grid.block(1), i, w, ACCENT) for i, w in enumerate(["years", "ago", "our", "fathers"])], lag_ratio=0.15)
        self.play(fill7)
        self.play(fill1)

        row0_anim = table_a.add_row(7, 4)
        row1_anim = table_a.add_row(1, 4)
        self.play(row0_anim)
        self.play(row1_anim)
        arrow_a0 = arrow_map(table_a.rows[0]["mobject"][1], grid.block(7), color=ACCENT)
        arrow_a1 = arrow_map(table_a.rows[1]["mobject"][1], grid.block(1), color=ACCENT)
        self.play(GrowArrow(arrow_a0), GrowArrow(arrow_a1))
        self.wait(0.3)
        self.next_slide()

        # -------------------- Beat 5: step (2) generates "brought" --------
        self.play(FadeOut(t4), FadeOut(prompt_cap))
        t5 = title('Step ② — generate "brought"')
        self.play(FadeIn(t5))

        q5 = _query_marker("Q")
        q5.next_to(grid, UP, buff=0.6)
        aq7 = arrow_map(q5, grid.block(7), color=Q_COLOR)
        aq1 = arrow_map(q5, grid.block(1), color=Q_COLOR)
        self.play(FadeIn(q5))
        self.play(GrowArrow(aq7), GrowArrow(aq1))
        step5_note = _step_banner("PagedAttention over physical blocks 7 and 1")
        self.play(FadeIn(step5_note))

        self.play(FadeOut(q5), FadeOut(aq7), FadeOut(aq1))
        self.play(lblock2.animate.set_opacity(1))
        self.play(_fill_slot_clean(lblock2, 0, "brought", ACCENT))
        self.play(_fill_slot_clean(grid.block(3), 0, "brought", ACCENT))
        row2 = table_a.add_row(3, 1)
        self.play(row2)
        arrow_a2 = arrow_map(table_a.rows[2]["mobject"][1], grid.block(3), color=ACCENT)
        self.play(GrowArrow(arrow_a2))
        note5b = _step_banner("last block full → allocate logical 2 / physical 3", color=GOOD)
        self.play(FadeOut(step5_note), FadeIn(note5b))
        self.wait(0.3)
        self.next_slide()

        # -------------------- Beat 6: step (3) generates "forth" -----------
        self.play(FadeOut(t5), FadeOut(note5b))
        t6 = title('Step ③ — generate "forth"')
        self.play(FadeIn(t6))

        note6 = _step_banner("reuse physical block 3; filled 1 → 2", color=GOOD)
        self.play(FadeIn(note6))

        self.play(_fill_slot_clean(lblock2, 1, "forth", ACCENT))
        self.play(_fill_slot_clean(grid.block(3), 1, "forth", ACCENT))
        self.play(table_a.set_row(2, 3, 2))
        self.wait(0.3)
        self.next_slide()

        # -------------------- Beat 7: utilization payoff --------------------
        self.play(FadeOut(t6), FadeOut(note6))
        t7 = title("The payoff: waste is bounded")
        self.play(FadeIn(t7))

        note7 = _step_banner("waste ≤ one partially-filled block per request", color=GOOD)
        self.play(FadeIn(note7))

        util_bar = MemoryBar(
            segments=[("token state", 0.963, GOOD), ("waste", 0.037, MUTED)],
            width=8.0,
            height=0.5,
            show_pct=False,
        )
        util_bar.next_to(lower_row, DOWN, buff=0.3)
        util_cap = caption("vLLM: up to 96.3% of KV memory is real token state (vs. ~20-40% before)")
        util_cap.next_to(util_bar, DOWN, buff=0.15)
        self.play(FadeOut(note7))
        self.play(util_bar.animate_in())
        self.play(FadeIn(util_cap))
        self.wait(0.3)
        self.next_slide()

        self.play(FadeOut(util_bar), FadeOut(util_cap), FadeOut(t7))

        # -------------------- Beat 8: Fig 7 — two requests at once --------
        t8 = title("Two requests at once (Fig. 7)")
        self.play(FadeIn(t8))

        b_block0 = KVBlock(slots=4, cell=0.42, index=0, words=["it", "was", "the", "best"], label_pos=DOWN)
        for s in range(4):
            b_block0.set_state(s, "filled")
        b_block0.cells.set_fill(ACCENT2, opacity=1.0)
        b_block1 = KVBlock(slots=4, cell=0.42, index=1, words=["of", "times", "", ""], label_pos=DOWN)
        b_block1.set_state(0, "filled")
        b_block1.set_state(1, "filled")
        b_block1.cells[0].set_fill(ACCENT2, opacity=1.0)
        b_block1.cells[1].set_fill(ACCENT2, opacity=1.0)
        b_col = VGroup(b_block0, b_block1)
        b_col.arrange(RIGHT, buff=0.35)
        b_title = small('Request B: "it was the best of times"', color=ACCENT2)
        b_stack = VGroup(b_title, b_col)
        b_stack.arrange(DOWN, buff=0.2)
        # All of b_stack's content is already final (no later fill_slot calls
        # on it), so it's safe to scale the whole thing down to fit the
        # remaining vertical space below the grid, above the frame edge.
        b_stack.scale(0.55)
        b_stack.next_to(lower_row, DOWN, buff=0.3)

        self.play(FadeIn(b_title), FadeIn(b_col))

        fillB2 = AnimationGroup(*[_fill_slot_clean(grid.block(2), i, w, ACCENT2) for i, w in enumerate(["it", "was", "the", "best"])], lag_ratio=0.15)
        fillB4 = AnimationGroup(*[_fill_slot_clean(grid.block(4), i, w, ACCENT2) for i, w in enumerate(["of", "times"])], lag_ratio=0.15)
        self.play(fillB2)
        self.play(fillB4)
        arrow_b0 = arrow_map(b_block0, grid.block(2), color=ACCENT2)
        arrow_b1 = arrow_map(b_block1, grid.block(4), color=ACCENT2)
        self.play(GrowArrow(arrow_b0), GrowArrow(arrow_b1))
        free_cap = caption("Blocks 0, 5, 6 still free for anyone — interleaved, not adjacent")
        free_cap.next_to(b_stack, DOWN, buff=0.15)
        self.play(FadeIn(free_cap))
        self.wait(0.3)
        self.next_slide()

        self.play(FadeOut(t8), FadeOut(free_cap))

        # -------------------- Beat 9: freeing --------
        t9 = title("Freeing: blocks return to the pool instantly")
        self.play(FadeIn(t9))

        done_mark = Circle(radius=0.22, color=GOOD, fill_opacity=0.2, stroke_width=3)
        done_mark.move_to(logical_stack.get_corner(UP + LEFT) + LEFT * 0.35)
        done_txt = Text("done", font_size=TINY_SIZE, color=GOOD)
        done_txt.move_to(done_mark.get_center())
        self.play(FadeIn(done_mark), FadeIn(done_txt))

        self.play(
            grid.block(7).clear_slot(0), grid.block(7).clear_slot(1), grid.block(7).clear_slot(2), grid.block(7).clear_slot(3),
            grid.block(1).clear_slot(0), grid.block(1).clear_slot(1), grid.block(1).clear_slot(2), grid.block(1).clear_slot(3),
            grid.block(3).clear_slot(0), grid.block(3).clear_slot(1),
        )
        self.play(
            grid.block(7).set_state(0, "empty"), grid.block(7).set_state(1, "empty"), grid.block(7).set_state(2, "empty"), grid.block(7).set_state(3, "empty"),
            grid.block(1).set_state(0, "empty"), grid.block(1).set_state(1, "empty"), grid.block(1).set_state(2, "empty"), grid.block(1).set_state(3, "empty"),
            grid.block(3).set_state(0, "empty"), grid.block(3).set_state(1, "empty"),
        )
        free_note = caption("Blocks 7, 1, 3 rejoin the free pool, from anywhere — no external fragmentation.")
        free_note.next_to(b_stack, DOWN, buff=0.15)
        self.play(FadeIn(free_note))
        self.wait(0.3)
        self.next_slide()

        self.play(FadeOut(*self.mobjects))

        # ==================================================================
        # Beat 10 — Summary
        # ==================================================================
        t10 = title("Recap: PagedAttention")
        self.play(FadeIn(t10))

        recap_lines = VGroup(
            small("• Fixed-size blocks", color=FG),
            small("• Block table: logical → physical", color=FG),
            small("• Allocate a new block only on demand", color=FG),
            small("• Kernel reads scattered blocks, then combines", color=FG),
        )
        recap_lines.arrange(DOWN, buff=0.35, aligned_edge=LEFT)

        stat = Text("96.3%", font_size=64, color=GOOD, weight="BOLD")
        stat_cap = caption("of KV cache memory is real token state")
        stat_group = VGroup(stat, stat_cap)
        stat_group.arrange(DOWN, buff=0.2)

        recap_full = VGroup(recap_lines, stat_group)
        recap_full.arrange(RIGHT, buff=1.0, aligned_edge=UP)
        if recap_full.width > 13.0:
            recap_full.scale_to_fit_width(13.0)
        recap_full.move_to(ORIGIN + DOWN * 0.2)

        self.play(AnimationGroup(*[FadeIn(l, shift=UP * 0.1) for l in recap_lines], lag_ratio=0.2))
        self.play(FadeIn(stat_group, shift=RIGHT * 0.2))
        self.wait(0.3)
        self.next_slide()
