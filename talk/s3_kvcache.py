"""S3 — The KV cache (Act I). Stitch: S2's growing K/V bundle × S1's leftover VRAM.

NARRATION
---------
Beat 1 — Pickup.
Last scene, the box needed the Key and Value of every previous token to write
the next word. The scene before that, leftover VRAM was the serving budget.
That bundle is still sitting here — watch the Query on "years" look across
every Key. We are going to name this bundle, size it, and put it in that
leftover slice. [PAUSE]

Beat 2 — Recompute is a triangle of real work.
Suppose we throw the Keys and Values away after each step. To write the next
word, we would rebuild them: k equals W_K x, v equals W_V x, for every
previous token, every time. Step 2 rebuilds token 1. Step 3 rebuilds 1 and 2.
Step 5 rebuilds four tokens just to add one new pair. [PAUSE] That triangle
only grows. The sentence gets longer; the wasted multiply gets worse.

Beat 3 — Keep them: the KV cache.
So don't throw them away. The first time we compute a token's Key and Value,
we write them down and keep them. The next step reads that store and computes
only the new pair. Query is different: it is made fresh for the newest token
and discarded — only K and V persist. This store is the KV cache.

Beat 4 — Prefill writes; decode appends.
Serving fills that cache in two phases. Prefill: the whole prompt enters at
once, in parallel, and the box writes a K,V pair for every prompt token in
one pass. Decode: the loop we have been watching — one new pair appended per
step. The cache is the state that survives between those decode steps.

Beat 5 — Read all, write one.
Look at one decode step closely. To write the next word, the cores must read
every cached pair — the whole row — and then write exactly one new pair on
the end. Read all, write one. That is why decode is memory-bound, and why
this object will eat leftover VRAM as the sentence grows.

Beat 6 — Every layer has its own copy.
And it is not one row. The model has many layers — forty, for OPT-13B — and
each layer keeps its own Keys and Values for every token. What looks like a
single strip is forty copies stacked. That is the first reason one token is
expensive.

Beat 7 — 800 KB, built in public.
Here is the arithmetic, one factor at a time. One vector is 5120 numbers in
FP16 — two bytes each — about 10 kilobytes. Times two, because each token
stores a Key and a Value: about 20 kilobytes. Times forty layers: 800
kilobytes per token. That is the cost of one word. [PAUSE] How many words
does this request need?

Beat 8 — Unknown length, so reserve the max.
We do not know. The cache grows one token at a time until the model emits
stop — there is no content-length. The only number we can bank on is the
model's maximum, 2048. So the whole strip gets reserved up front. The tokens
we have actually written sit on the left. Everything past them is reserved
and cut off from every other request, whether we ever fill it or not. [PAUSE]

Beat 9 — That reserved strip is 1.6 GB.
Now the last multiply means something. 800 kilobytes times 2048 reserved
slots is about 1.6 gigabytes — not "how big this request is," but how much
VRAM one request has spoken for. Used or not, that whole strip is gone from
the leftover budget. [PAUSE]

Beat 10 — Per request, in leftover VRAM.
Every request owns its own reserved strip. Request A is "Four score…";
request B is "it was the best…" — they grow independently, different lengths,
not shared. Both of them have to live in leftover VRAM, beside the 26
gigabytes of weights that never move.

Beat 11 — Fig 1 left, and how many fit.
That leftover slice is the paper's Figure 1: on a 13B model and an A100
40-gigabyte card, about 65 percent is weights, more than 30 percent is KV
cache — 12 gigabytes — and a sliver is other, short-lived activations.
Weights are fixed. KV is the only region that grows and shrinks. 12 gigabytes
divided by 1.6 gigabytes reserved is about seven requests at maximum length.
An eighth does not fit.

Beat 12 — Two hard properties.
Two facts make this object awkward to place. First: we reserved the max
because the length was unknown — most of that strip may never fill. Second:
the same word at a different position has a different Key and Value — this
is a timeline, not a dictionary of words. Leftover VRAM is spent on these
growing rows; how we lay them out is the batch size. [PAUSE] So: how do you
allocate memory for something whose final size is unknown?
"""

from manim import (
    DOWN,
    LEFT,
    ORIGIN,
    RIGHT,
    UP,
    AnimationGroup,
    DashedLine,
    FadeIn,
    FadeOut,
    GrowFromEdge,
    LaggedStart,
    RoundedRectangle,
    Square,
    SurroundingRectangle,
    Transform,
    VGroup,
)

from manim_slides import Slide

from talk.theme import *
from talk.components import *

WEIGHTS_FRAC = 26.0 / 40.0
KV_FRAC = 0.30
OTHER_FRAC = 0.05
REQ_B = ["it", "was", "the", "best"]


def _chip(label, color=FG, fill=BLOCK_FILL, width=1.05, height=0.38, font_size=TINY_SIZE):
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


def _vram_tank(width=1.7, height=3.35):
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
    label = text("VRAM  40 GB", font_size=TINY_SIZE, color=MUTED)
    group = VGroup(outline, fill, label)
    group.outline = outline
    group.fill = fill
    group.caption = label
    group.inner_w = inner_w
    group.inner_h = inner_h
    group.pad = pad
    return group


def _place_tank_fill(tank):
    outline = tank.outline
    fill = tank.fill
    bottom = outline.get_bottom()[1] + tank.pad / 2
    fill.move_to([outline.get_center()[0], bottom + fill.height / 2, 0])
    tank.caption.next_to(outline, DOWN, buff=0.12)


def _req_row(label, words, label_color, height=0.42, font_size=16):
    lab = small(label, font_size=SMALL_SIZE, color=label_color)
    toks = VGroup(*[
        TokenKV(w, show_kv=True, kv_opacity=1.0, height=height, font_size=font_size)
        for w in words
    ])
    toks.arrange(RIGHT, buff=0.08)
    row = VGroup(lab, toks).arrange(RIGHT, buff=0.28)
    row.lab = lab
    row.toks = toks
    return row


def _recompute_triangle(n_steps=5, cell=0.16):
    """Growing triangle of K/V pairs: step k rebuilds k previous tokens."""
    rows = VGroup()
    for step in range(1, n_steps + 1):
        pairs = VGroup()
        for _ in range(step):
            k = Square(side_length=cell, fill_color=K_COLOR, fill_opacity=0.9, stroke_width=0)
            v = Square(side_length=cell, fill_color=V_COLOR, fill_opacity=0.9, stroke_width=0)
            pairs.add(VGroup(k, v).arrange(RIGHT, buff=0.035))
        pairs.arrange(RIGHT, buff=0.07)
        lab = text(f"step {step}", font_size=14, color=MUTED)
        row = VGroup(lab, pairs).arrange(RIGHT, buff=0.18)
        rows.add(row)
    rows.arrange(DOWN, buff=0.07, aligned_edge=LEFT)
    return rows


def _factor_line(left, right, result, result_color=FG):
    lft = text(left, font_size=20, color=MUTED)
    rgt = text(right, font_size=20, color=FG)
    arrow = text("→", font_size=20, color=MUTED)
    res = text(result, font_size=22, color=result_color)
    line = VGroup(lft, rgt, arrow, res).arrange(RIGHT, buff=0.22)
    line.lft = lft
    line.res = res
    return line


def _size_bar(width, color, height=0.42):
    return RoundedRectangle(
        corner_radius=0.08,
        width=max(width, 0.08),
        height=height,
        fill_color=color,
        fill_opacity=0.92,
        stroke_width=0,
    )


def _reserved_strip(words, n_empty=8, cell_h=0.40):
    """Live tokens on the left, a cutoff, then reserved empty slots to the max."""
    used = VGroup(*[
        TokenKV(w, height=cell_h, font_size=14) for w in words
    ])
    used.arrange(RIGHT, buff=0.05)
    empties = VGroup()
    for _ in range(n_empty):
        cell = RoundedRectangle(
            width=0.34,
            height=cell_h,
            corner_radius=0.06,
            fill_color=WARN,
            fill_opacity=0.32,
            stroke_color=WARN,
            stroke_width=1.2,
        )
        empties.add(cell)
    empties.arrange(RIGHT, buff=0.05)
    cut = DashedLine(
        UP * (cell_h * 0.55 + 0.12),
        DOWN * (cell_h * 0.55 + 0.12),
        color=BAD,
        stroke_width=3,
        dash_length=0.07,
    )
    dots = text("…", font_size=20, color=WARN)
    max_lab = text("2048", font_size=16, color=WARN)
    tail = VGroup(empties, dots, max_lab).arrange(RIGHT, buff=0.12)
    body = VGroup(used, cut, tail).arrange(RIGHT, buff=0.16)
    used_cap = caption("used so far")
    used_cap.next_to(used, DOWN, buff=0.14)
    res_cap = caption("reserved — cut off from other requests")
    res_cap.set_color(WARN)
    res_cap.next_to(tail, DOWN, buff=0.14)
    if res_cap.get_right()[0] > 6.7:
        res_cap.scale_to_fit_width(tail.width)
        res_cap.next_to(tail, DOWN, buff=0.14)
    group = VGroup(body, used_cap, res_cap)
    group.used = used
    group.cut = cut
    group.tail = tail
    group.used_cap = used_cap
    group.res_cap = res_cap
    return group


class S3KVCache(Slide):
    def construct(self):
        apply_theme(self)

        prompt = FOUR_SCORE[:5]
        loop = TransformerLoop(prompt, show_kv=True, kv_opacity=1.0)
        heading = title("The bundle that sits in memory")
        self._math = None

        # ------------------------------------------------------------------
        # Beat 1 — Pickup: S2 bundle × S1 leftover VRAM
        # ------------------------------------------------------------------
        note = caption("Query of the newest token  ·  every previous Key and Value")
        note.to_edge(DOWN, buff=0.32)

        self.play(FadeIn(heading), FadeIn(loop.tbox), run_time=0.7)
        self.play(
            LaggedStart(*[FadeIn(tok) for tok in loop.input], lag_ratio=0.12),
            run_time=1.0,
        )
        self.play(FadeIn(loop.in_arrow), FadeIn(loop.out_arrow), FadeIn(note), run_time=0.5)
        self._math = note

        q_chip = loop.input[-1].show_query()
        q_chip.set_opacity(0)
        self.play(q_chip.animate.set_opacity(1.0), run_time=0.45)
        self.play(loop.input.animate.move_to(loop.input_centered_pos()), run_time=0.4)
        loop.update_arrows()

        land = caption("leftover VRAM is where this bundle has to live")
        self._set_footer(land, run_time=0.45)
        self.wait(0.3)
        self.next_slide()

        # ------------------------------------------------------------------
        # Beat 2 — Recompute is a triangle of real work
        # ------------------------------------------------------------------
        heading = self._retitle(heading, "Throw them away, rebuild them every step")
        hide_q = loop.input[-1].hide_query()
        fade_kv = loop.set_kv_opacity(0.0)
        anims = list(fade_kv)
        if hide_q is not None:
            anims.append(FadeOut(hide_q))
        self.play(*anims, run_time=0.55)
        loop.update_arrows()

        tri = _recompute_triangle(n_steps=5)
        self._fit_inner(tri, loop)
        gloss = formula(
            "each rebuild:  k = W<sub>K</sub> x    v = W<sub>V</sub> x",
            font_size=18,
            color=MUTED,
        )
        self._clear_footer(run_time=0.35)
        self.play(
            loop.tbox.label.animate.set_opacity(0),
            loop.park_label_top(),
            run_time=0.4,
        )
        self.play(LaggedStart(*[FadeIn(row) for row in tri], lag_ratio=0.22), run_time=1.4)
        n_tok = len(loop.input)
        for step in range(1, min(5, n_tok) + 1):
            pulse = []
            for tok in loop.input[:step]:
                pulse.append(tok.k_chip.animate.set_opacity(0.85))
                pulse.append(tok.v_chip.animate.set_opacity(0.85))
            self.play(*pulse, run_time=0.22)
            dim = []
            for tok in loop.input[:step]:
                dim.append(tok.k_chip.animate.set_opacity(0.0))
                dim.append(tok.v_chip.animate.set_opacity(0.0))
            self.play(*dim, run_time=0.16)
        self._set_footer(gloss, run_time=0.4)
        self.wait(0.3)
        self.next_slide()

        # ------------------------------------------------------------------
        # Beat 3 — Keep them: the KV cache
        # ------------------------------------------------------------------
        heading = self._retitle(heading, "Keep them: this is the KV cache")
        self.play(
            FadeOut(tri, shift=LEFT * 2.2),
            loop.restore_label(),
            loop.tbox.label.animate.set_opacity(1),
            run_time=0.55,
        )
        self._dump(tri)
        self.play(*loop.set_kv_opacity(1.0), run_time=0.6)

        q_chip = loop.input[-1].show_query()
        q_chip.set_opacity(0)
        self.play(q_chip.animate.set_opacity(1.0), run_time=0.4)
        self.play(loop.input.animate.move_to(loop.input_centered_pos()), run_time=0.3)
        loop.update_arrows()

        q_note = caption("Query is made fresh and discarded  ·  only K and V persist")
        self._set_footer(q_note, run_time=0.4)
        hide_q = loop.input[-1].hide_query()
        if hide_q is not None:
            self.play(FadeOut(hide_q), run_time=0.4)
        self.play(loop.input.animate.move_to(loop.input_centered_pos()), run_time=0.3)
        loop.update_arrows()

        name = caption("the KV cache: write once, read every later step")
        self._set_footer(name, run_time=0.4)
        self._decode_step(loop, FOUR_SCORE[5], kv_opacity=1.0)
        self.wait(0.3)
        self.next_slide()

        # ------------------------------------------------------------------
        # Beat 4 — Prefill writes; decode appends
        # ------------------------------------------------------------------
        heading = self._retitle(heading, "Prefill writes the cache; decode appends")
        pre_card = self._phase_card(
            "PREFILL",
            ACCENT,
            ["whole prompt at once", "writes K,V for every token", "one pass"],
        )
        dec_card = self._phase_card(
            "DECODE",
            ACCENT2,
            ["one new token per step", "appends one K,V pair", "cache survives"],
        )
        phases = VGroup(pre_card, dec_card).arrange(RIGHT, buff=0.5, aligned_edge=UP)
        self._fit_inner(phases, loop)
        self._clear_footer(run_time=0.3)
        self.play(loop.park_label_top(), run_time=0.35)
        self.play(*loop.set_kv_opacity(0.0), run_time=0.4)
        self.play(FadeIn(pre_card), run_time=0.45)
        self.play(
            LaggedStart(*self._chip_anims(loop, 1.0), lag_ratio=0.04),
            run_time=0.7,
        )
        self.play(FadeIn(dec_card), run_time=0.4)
        foot4 = caption("the cache is the state that survives between decode steps")
        self._set_footer(foot4, run_time=0.4)
        self.wait(0.3)
        self.next_slide()

        # ------------------------------------------------------------------
        # Beat 5 — Read all, write one
        # ------------------------------------------------------------------
        heading = self._retitle(heading, "Each decode step: read all, write one")
        self.play(FadeOut(phases, shift=LEFT * 2.2), loop.restore_label(), run_time=0.5)
        self._dump(phases)
        loop.update_arrows()
        reader = SurroundingRectangle(loop.input, color=ACCENT2, buff=0.10, stroke_width=2)
        self.play(FadeIn(reader), run_time=0.35)
        self.play(
            LaggedStart(*[
                AnimationGroup(
                    tok.k_chip.animate.set_opacity(0.35),
                    tok.v_chip.animate.set_opacity(0.35),
                )
                for tok in loop.input
            ], lag_ratio=0.08),
            run_time=0.7,
        )
        self.play(
            AnimationGroup(*[tok.k_chip.animate.set_opacity(1.0) for tok in loop.input]),
            AnimationGroup(*[tok.v_chip.animate.set_opacity(1.0) for tok in loop.input]),
            run_time=0.35,
        )
        self.play(FadeOut(reader), run_time=0.3)
        self._dump(reader)
        foot5 = caption("stream the whole cache past the cores  ·  append one pair")
        self._set_footer(foot5, run_time=0.4)
        self._decode_step(loop, FOUR_SCORE[6], kv_opacity=1.0)
        write_tag = caption("write one")
        write_tag.next_to(loop.input[-1], UP, buff=0.08)
        write_tag.set_color(ACCENT)
        self.play(FadeIn(write_tag), run_time=0.3)
        self.play(FadeOut(write_tag), run_time=0.35)
        self._dump(write_tag)
        self.wait(0.3)
        self.next_slide()

        # ------------------------------------------------------------------
        # Beat 6 — Every layer has its own copy
        # ------------------------------------------------------------------
        heading = self._retitle(heading, "Every layer keeps its own copy")
        self._clear_footer(run_time=0.3)
        self.play(
            FadeOut(loop.tbox),
            FadeOut(loop.in_arrow),
            FadeOut(loop.out_arrow),
            run_time=0.5,
        )
        self._dump(loop.tbox)
        self._dump(loop.in_arrow)
        self._dump(loop.out_arrow)

        self.play(loop.input.animate.scale(0.88).move_to(UP * 1.85), run_time=0.5)
        base = loop.input
        ghosts = VGroup()
        layer_labs = VGroup()
        lab1 = text("layer 1", font_size=14, color=MUTED)
        lab1.next_to(base, LEFT, buff=0.22)
        layer_labs.add(lab1)
        for i in range(1, 4):
            g = base.copy()
            g.set_opacity(max(0.18, 0.42 - i * 0.10))
            g.shift(DOWN * 0.42 * i + RIGHT * 0.10 * i)
            ghosts.add(g)
            lab = text("layer " + ("…" if i == 3 else str(i + 1)), font_size=14, color=MUTED)
            lab.next_to(g, LEFT, buff=0.22)
            layer_labs.add(lab)
        times = text("×  40 layers", font_size=28, color=ACCENT)
        times.next_to(ghosts, DOWN, buff=0.45)
        times.align_to(base, LEFT)

        self.play(FadeIn(lab1), run_time=0.3)
        self.play(
            LaggedStart(*[FadeIn(g) for g in ghosts], lag_ratio=0.2),
            LaggedStart(*[FadeIn(lab) for lab in layer_labs[1:]], lag_ratio=0.2),
            run_time=0.9,
        )
        self.play(FadeIn(times), run_time=0.4)
        foot6 = caption("not one row of vectors — forty  ·  OPT-13B")
        self._set_footer(foot6, run_time=0.4)
        self.wait(0.3)
        self.next_slide()

        # ------------------------------------------------------------------
        # Beat 7 — 800 KB, built in public
        # ------------------------------------------------------------------
        heading = self._retitle(heading, "One token: 800 KB  (OPT-13B)")
        unit = TokenKV("years", show_kv=True, kv_opacity=1.0, height=0.52, font_size=18)
        unit_lab = caption("one token")
        unit_block = VGroup(unit, unit_lab).arrange(DOWN, buff=0.12)
        self.play(
            FadeOut(base),
            FadeOut(ghosts),
            FadeOut(layer_labs),
            FadeOut(times),
            run_time=0.5,
        )
        self._dump(base)
        self._dump(ghosts)
        self._dump(layer_labs)
        self._dump(times)

        unit_block.move_to(LEFT * 5.15 + UP * 0.55)
        self.play(FadeIn(unit_block), run_time=0.4)

        lines = VGroup(
            _factor_line("5120 hidden", "×  2 bytes (FP16)", "≈  10 KB"),
            _factor_line("×  2", "Key and Value", "≈  20 KB"),
            _factor_line("×  40", "layers", "800 KB / token", result_color=ACCENT),
        )
        lines.arrange(DOWN, buff=0.32, aligned_edge=LEFT)
        lines.next_to(unit_block, RIGHT, buff=0.55)
        lines.shift(UP * 0.15)
        if lines.get_right()[0] > 6.85:
            lines.scale_to_fit_width(9.4)
            lines.next_to(unit_block, RIGHT, buff=0.45)

        bar_anchor = LEFT * 5.4 + DOWN * 2.15
        barspec = [
            (1.1, ACCENT2, "~10 KB"),
            (2.0, ACCENT2, "~20 KB"),
            (4.4, ACCENT, "800 KB"),
        ]
        bar = None
        bar_lab = None
        for i, line in enumerate(lines):
            self.play(FadeIn(line), run_time=0.4)
            w, col, blab = barspec[i]
            new_bar = _size_bar(w, col)
            new_bar.move_to(bar_anchor, aligned_edge=LEFT)
            new_lab = text(blab, font_size=18, color=col)
            new_lab.next_to(new_bar, RIGHT, buff=0.18)
            if bar is None:
                self.play(GrowFromEdge(new_bar, LEFT), FadeIn(new_lab), run_time=0.45)
                bar, bar_lab = new_bar, new_lab
            else:
                self.play(
                    Transform(bar, new_bar),
                    FadeOut(bar_lab),
                    FadeIn(new_lab),
                    run_time=0.45,
                )
                bar_lab = new_lab
        foot7 = caption("per token  ·  how many tokens does a request need?")
        self._set_footer(foot7, run_time=0.4)
        self.wait(0.3)
        self.next_slide()

        # ------------------------------------------------------------------
        # Beat 8 — Unknown length, so reserve the max
        # ------------------------------------------------------------------
        heading = self._retitle(heading, "We don't know the length, so we reserve the max")
        arith7 = VGroup(unit_block, lines)
        self.play(
            FadeOut(bar),
            FadeOut(bar_lab),
            arith7.animate.scale(0.78).move_to(UP * 1.72),
            run_time=0.5,
        )
        self._dump(bar)
        self._dump(bar_lab)

        strip = _reserved_strip(FOUR_SCORE[:6], n_empty=8)
        if strip.width > 13.0:
            strip.scale_to_fit_width(13.0)
        strip.move_to(DOWN * 0.55)
        self.play(FadeIn(strip.used), run_time=0.4)
        self.play(FadeIn(strip.used_cap), run_time=0.25)
        self.play(FadeIn(strip.cut), run_time=0.3)
        self.play(FadeIn(strip.tail), FadeIn(strip.res_cap), run_time=0.45)
        foot8 = caption("no content-length  ·  the only number we can bank on is 2048")
        self._set_footer(foot8, run_time=0.4)
        self.wait(0.3)
        self.next_slide()

        # ------------------------------------------------------------------
        # Beat 9 — That reserved strip is 1.6 GB
        # ------------------------------------------------------------------
        heading = self._retitle(heading, "That reserved strip is 1.6 GB")
        line4 = _factor_line("×  2048", "reserved slots", "≈  1.6 GB spoken for", result_color=ACCENT)
        line4.scale(0.78)
        line4.next_to(lines, DOWN, buff=0.18)
        line4.align_to(lines, LEFT)
        if line4.get_right()[0] > 6.8:
            line4.scale_to_fit_width(8.6)
            line4.next_to(lines, DOWN, buff=0.18)
            line4.align_to(lines, LEFT)
        cost = text("1.6 GB reserved", font_size=22, color=ACCENT)
        cost.next_to(strip, DOWN, buff=0.22)
        self.play(FadeIn(line4), run_time=0.45)
        self.play(FadeIn(cost), run_time=0.4)
        bar, bar_lab = cost, None
        foot9 = caption("used or not, that whole strip is gone from leftover VRAM")
        self._set_footer(foot9, run_time=0.4)
        self.wait(0.3)
        self.next_slide()

        # ------------------------------------------------------------------
        # Beat 8 — Per request, in leftover VRAM
        # ------------------------------------------------------------------
        heading = self._retitle(heading, "Per request, and it lives in leftover VRAM")
        arith = VGroup(unit_block, lines, line4, strip, bar)
        self.play(FadeOut(arith), run_time=0.45)
        self._dump(arith)

        tank = _vram_tank()
        tank.move_to(RIGHT * 4.35 + DOWN * 0.05)
        _place_tank_fill(tank)
        leftover_h = tank.inner_h * (1.0 - WEIGHTS_FRAC)
        leftover = RoundedRectangle(
            corner_radius=0.08,
            width=tank.inner_w,
            height=max(leftover_h, 0.08),
            fill_color=GOOD,
            fill_opacity=0.22,
            stroke_color=GOOD,
            stroke_width=1.5,
        )
        leftover.next_to(tank.fill, UP, buff=0.04)
        leftover.align_to(tank.fill, LEFT)
        left_lab = text("leftover", font_size=14, color=GOOD)
        if left_lab.height > leftover.height * 0.7:
            left_lab.scale_to_fit_height(leftover.height * 0.62)
        left_lab.move_to(leftover.get_center())
        w_lab = text("weights  26 GB", font_size=14, color=BG)
        if w_lab.width > tank.fill.width * 0.9:
            w_lab.scale_to_fit_width(tank.fill.width * 0.88)
        w_lab.move_to(tank.fill.get_center())

        row_a = _req_row("Request A", FOUR_SCORE[:6], ACCENT)
        row_b = _req_row("Request B", REQ_B, ACCENT2)
        rows = VGroup(row_a, row_b).arrange(DOWN, buff=0.55, aligned_edge=LEFT)
        rows.scale_to_fit_width(8.6)
        rows.move_to(LEFT * 2.35 + UP * 0.15)

        self.play(FadeIn(tank.outline), FadeIn(tank.fill), FadeIn(tank.caption), FadeIn(w_lab), run_time=0.5)
        self.play(FadeIn(leftover), FadeIn(left_lab), run_time=0.35)
        self.play(FadeIn(row_a), run_time=0.45)
        self.play(FadeIn(row_b), run_time=0.45)
        foot8 = caption("two requests, two independent caches  ·  both in leftover VRAM")
        self._set_footer(foot8, run_time=0.4)
        self.wait(0.3)
        self.next_slide()

        # ------------------------------------------------------------------
        # Beat 9 — Fig 1 left + how many fit
        # ------------------------------------------------------------------
        heading = self._retitle(heading, "13B on an A100 40 GB: the leftover is KV")
        kv_h = tank.inner_h * KV_FRAC
        other_h = tank.inner_h * OTHER_FRAC
        kv_fill = RoundedRectangle(
            corner_radius=0.08,
            width=tank.inner_w,
            height=max(kv_h, 0.08),
            fill_color=ACCENT2,
            fill_opacity=0.92,
            stroke_width=0,
        )
        kv_fill.next_to(tank.fill, UP, buff=0.03)
        kv_fill.align_to(tank.fill, LEFT)
        other_fill = RoundedRectangle(
            corner_radius=0.06,
            width=tank.inner_w,
            height=max(other_h, 0.06),
            fill_color=MUTED,
            fill_opacity=0.85,
            stroke_width=0,
        )
        other_fill.next_to(kv_fill, UP, buff=0.03)
        other_fill.align_to(tank.fill, LEFT)

        kv_lab = text("KV  12 GB", font_size=13, color=BG)
        if kv_lab.height > kv_fill.height * 0.75:
            kv_lab.scale_to_fit_height(kv_fill.height * 0.7)
        kv_lab.move_to(kv_fill.get_center())
        other_side = text("other", font_size=13, color=MUTED)
        other_side.next_to(other_fill, RIGHT, buff=0.12)

        legend = VGroup(
            text("weights   65%", font_size=16, color=ACCENT),
            text("KV cache  >30%", font_size=16, color=ACCENT2),
            text("other     ~5%", font_size=16, color=MUTED),
        ).arrange(DOWN, buff=0.12, aligned_edge=LEFT)

        self.play(
            FadeOut(row_a),
            FadeOut(row_b),
            FadeOut(leftover),
            FadeOut(left_lab),
            run_time=0.45,
        )
        self._dump(row_a)
        self._dump(row_b)
        self._dump(leftover)
        self._dump(left_lab)

        self.play(FadeIn(kv_fill), FadeIn(other_fill), run_time=0.5)
        self.play(FadeIn(kv_lab), FadeIn(other_side), run_time=0.3)
        legend.next_to(tank.outline, LEFT, buff=0.45)
        legend.align_to(tank.outline, UP)
        self.play(FadeIn(legend), run_time=0.4)

        math_line = text("12 GB  ÷  1.6 GB   ≈   7 requests at max length", font_size=22, color=FG)
        math_line.move_to(LEFT * 1.7 + DOWN * 2.05)
        if math_line.get_left()[0] < -6.9:
            math_line.scale_to_fit_width(8.8)
            math_line.move_to(LEFT * 1.5 + DOWN * 2.05)

        chips = VGroup(*[_chip(f"req {i + 1}", width=0.92, height=0.34) for i in range(7)])
        chips.arrange(RIGHT, buff=0.08)
        bounce = _chip("req 8", width=0.92, height=0.34)
        pack = VGroup(chips, bounce).arrange(RIGHT, buff=0.08)
        pack.next_to(math_line, DOWN, buff=0.42)
        pack.align_to(math_line, LEFT)
        if pack.get_right()[0] > 6.9:
            pack.scale_to_fit_width(9.6)
            pack.next_to(math_line, DOWN, buff=0.42)
            pack.align_to(math_line, LEFT)

        self.play(FadeIn(math_line), run_time=0.4)
        self.play(LaggedStart(*[FadeIn(c) for c in chips], lag_ratio=0.08), run_time=0.7)
        self.play(FadeIn(bounce, shift=RIGHT * 0.15), run_time=0.3)
        x_mark = text("X", font_size=22, color=BAD)
        x_mark.next_to(bounce, UP, buff=0.06)
        bounce[0].set_stroke(BAD, width=2)
        self.play(FadeIn(x_mark), run_time=0.3)
        for c in chips:
            c[0].set_stroke(GOOD, width=2)
        foot9 = caption("seven strips of 1.6 GB  ·  an eighth has no room")
        self._set_footer(foot9, run_time=0.4)
        self.wait(0.3)
        self.next_slide()

        # ------------------------------------------------------------------
        # Beat 10 — Two hard properties, then the S4 question
        # ------------------------------------------------------------------
        heading = self._retitle(heading, "Two reasons this object is hard to place")
        old9 = VGroup(
            tank.outline, tank.fill, tank.caption, w_lab,
            kv_fill, other_fill, kv_lab, other_side, legend,
            math_line, pack, x_mark,
        )
        self.play(FadeOut(old9), run_time=0.5)
        self._dump(old9)

        card_w, card_h = 5.6, 3.15
        card_a = RoundedRectangle(
            width=card_w, height=card_h, corner_radius=0.14,
            fill_color=BLOCK_FILL, fill_opacity=1.0,
            stroke_color=BLOCK_STROKE, stroke_width=2,
        )
        card_b = card_a.copy()
        cards = VGroup(card_a, card_b).arrange(RIGHT, buff=0.45)
        cards.move_to(UP * 0.15)

        t_a = text("Unknown final length", font_size=20, color=ACCENT)
        t_a.next_to(card_a.get_top(), DOWN, buff=0.22)
        grow_words = FOUR_SCORE[:6]
        grow = VGroup(*[
            TokenKV(w, height=0.38, font_size=14) for w in grow_words
        ])
        grow.arrange(RIGHT, buff=0.06)
        q1 = TokenBox("?", color=MUTED, fill=BLOCK_FILL, height=0.38, font_size=16)
        q2 = TokenBox("?", color=MUTED, fill=BLOCK_FILL, height=0.38, font_size=16)
        dots = text("…", font_size=22, color=MUTED)
        grow_row = VGroup(grow, q1, q2, dots).arrange(RIGHT, buff=0.08)
        if grow_row.width > card_w - 0.5:
            grow_row.scale_to_fit_width(card_w - 0.55)
        grow_row.move_to(card_a.get_center() + DOWN * 0.15)
        cap_a = caption("so the whole max strip gets reserved")
        cap_a.next_to(card_a.get_bottom(), UP, buff=0.22)

        t_b = text("Same word, different K/V", font_size=20, color=ACCENT2)
        t_b.next_to(card_b.get_top(), DOWN, buff=0.22)
        s_left = TokenKV("score", height=0.46, font_size=16)
        s_right = TokenKV("score", height=0.46, font_size=16)
        # Tint the right-hand chips so the vectors are visibly not the same.
        s_right.k_chip[0].set_fill("#7ad4f0", opacity=0.95)
        s_right.v_chip[0].set_fill("#d4a8ff", opacity=0.95)
        pos2 = caption("position 2")
        pos7 = caption("position 7")
        col_l = VGroup(s_left, pos2).arrange(DOWN, buff=0.12)
        col_r = VGroup(s_right, pos7).arrange(DOWN, buff=0.12)
        neq = text("≠", font_size=36, color=BAD)
        pair = VGroup(col_l, neq, col_r).arrange(RIGHT, buff=0.28)
        pair.move_to(card_b.get_center() + DOWN * 0.05)
        cap_b = caption("a timeline, not a dictionary of words")
        cap_b.next_to(card_b.get_bottom(), UP, buff=0.22)

        panel_a = VGroup(card_a, t_a, grow_row, cap_a)
        panel_b = VGroup(card_b, t_b, pair, cap_b)

        self.play(FadeIn(card_a), FadeIn(card_b), run_time=0.4)
        self.play(FadeIn(t_a), FadeIn(grow_row), FadeIn(cap_a), run_time=0.5)
        self.play(FadeIn(t_b), FadeIn(pair), FadeIn(cap_b), run_time=0.5)

        question = text(
            "How do you allocate memory for something whose final size is unknown?",
            font_size=22,
            color=ACCENT,
        )
        if question.width > 13.2:
            question.scale_to_fit_width(13.2)
        self._set_footer(question, run_time=0.5)
        self.wait(0.3)
        self.next_slide()

    def _phase_card(self, heading, color, lines):
        head = text(heading, font_size=20, color=color)
        body_lines = VGroup(*[text(ln, font_size=15, color=MUTED) for ln in lines])
        body_lines.arrange(DOWN, buff=0.08, aligned_edge=LEFT)
        return VGroup(head, body_lines).arrange(DOWN, buff=0.14, aligned_edge=LEFT)

    def _chip_anims(self, loop, alpha):
        anims = []
        for tok in loop.input:
            anims.append(tok.k_chip.animate.set_opacity(alpha))
            anims.append(tok.v_chip.animate.set_opacity(alpha))
        return anims

    def _dump(self, mob):
        if mob is None:
            return
        family = list(mob.get_family())
        self.remove(mob, *family)
        mob.set_opacity(0)
        mob.shift(LEFT * 40)

    def _set_footer(self, new, run_time=0.45):
        new.to_edge(DOWN, buff=0.30)
        old = self._math
        if old is None:
            self.play(FadeIn(new), run_time=run_time)
        else:
            self.play(FadeOut(old, shift=LEFT * 2.2), FadeIn(new), run_time=run_time)
            self._dump(old)
        self._math = new

    def _clear_footer(self, run_time=0.35):
        old = self._math
        if old is None:
            return
        self.play(FadeOut(old, shift=LEFT * 2.2), run_time=run_time)
        self._dump(old)
        self._math = None

    def _fit_inner(self, mob, loop):
        max_w = loop.tbox.box.width - 0.7
        max_h = loop.tbox.box.height - 0.85
        if mob.width > max_w:
            mob.scale_to_fit_width(max_w)
        if mob.height > max_h:
            mob.scale_to_fit_height(max_h)
        mob.move_to(loop.inner_point())

    def _retitle(self, old, new_str, *anims, run_time=0.6):
        new = title(new_str)
        if new.width > 13.4:
            new.scale_to_fit_width(13.4)
            new.to_edge(UP)
        self.play(
            FadeOut(old, shift=UP * 0.22),
            FadeIn(new, shift=DOWN * 0.10),
            *anims,
            run_time=run_time,
        )
        self._dump(old)
        return new

    def _decode_step(self, loop, word, kv_opacity=1.0):
        out = loop.spawn_output(word, kv_opacity=kv_opacity)
        delta = loop.join_delta()
        join = loop.join_point() + delta
        self.play(loop.set_active(True), FadeIn(out, shift=DOWN * 0.1), run_time=0.45)
        self.wait(0.2)
        self.play(out.animate.move_to(join), loop.input.animate.shift(delta), run_time=0.75)
        loop.adopt_output()
        loop.update_arrows()
        self.play(loop.set_active(False), run_time=0.25)
        return out
