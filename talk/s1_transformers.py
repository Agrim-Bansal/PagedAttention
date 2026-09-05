"""S1 — Transformers & attention (Act I): the box that emits one token at a time.

NARRATION

Beat 1 — One word at a time
You do not need the word "transformer" yet. A language model does not write a
sentence in one go. It takes the words so far, runs them through a box, and the
box emits one next word. That word is appended, the sequence recenters, and the
loop runs again. Watch "Four score and seven" produce "years". [PAUSE] In the
paper that loop is Equation 1: the joint probability of a sentence is just the
product of these next-word guesses.

Beat 2 — Key: a label for each token
Open the box. Each token is first a vector x_i — a list of numbers that stands
for that word in this layer. A learned matrix W_K multiplies it: k_i equals
W_K x_i. That is the Key. Think of it as a label on the token: what this word
contains, and how later steps will find it. Every token gets one. [PAUSE]

Beat 3 — Value: what the token will add
A second matrix produces the Value: v_i equals W_V x_i. If a later step decides
this token matters, it is the Value that actually gets mixed into the output —
the payload, not the label. Key is how you find it; Value is what you take.
Every token now carries both. [PAUSE]

Beat 4 — Query: the question this step asks
A third vector, only for the token we are writing from. q_i equals W_Q x_i —
the Query. It is the question this step is asking: what should I look up to
choose the next word? Only the newest token needs a fresh Query. Every earlier
token just sits there with its Key and Value. [PAUSE]

Beat 5 — Query against every Key
Equation 3 starts with a score. The Query is compared to every Key from position
1 through i — including its own, never a future token. s_j equals q_i transpose
k_j over square root of d. The numbers here are illustrative. [PAUSE] Which
token do you think scores highest?

Beat 6 — Softmax: scores become weights
Those scores become attention weights by softmax: a_ij equals exp of the score,
divided by the sum of those exps from t equals 1 to i. The weights are positive
and they sum to one. Bigger score, bigger slice of attention — a budget to spend
across the tokens so far.

Beat 7 — Mix the Values, write the next word
Spend that budget on Values: o_i equals the sum from j equals 1 to i of a_ij
v_j. That mixture is what the box turns into the next word. Here that's "ago",
which exits, joins the sequence, and recenters. Close the box — same machine as
beat 1, we just know what's inside.

Beat 8 — Nothing is thrown away
Run the loop once more, box closed. Every old Key and Value stays on the input;
the new token arrives with one new pair; nothing is thrown away. Next iteration
the box will consume the full sequence and every previous K and V again.

Beat 9 — Prefill vs decode
Serving splits this into two phases. Prefill: the whole prompt enters at once,
in parallel, and the box writes K and V for every prompt token plus the first
output token — compute-bound. Decode: the loop we have been watching — one new
token per pass, re-reading the growing K/V bundle every time — memory-bound, and
that is the phase that dominates latency.

Beat 10 — The landing point
So the load-bearing fact: to emit the next token, the box must be handed the Key
and the Value of every previous token, not just the most recent one. That growing
bundle sits in memory between iterations — at every layer, every decode step, for
the entire request. That is the state the rest of this talk is about.
"""

import math

from manim import (
    DOWN,
    LEFT,
    RIGHT,
    UP,
    FadeIn,
    FadeOut,
    LaggedStart,
    Rectangle,
    VGroup,
)
from manim_slides import Slide

from talk.components import *
from talk.theme import *

# Illustrative scores for the 5-token prefix after beat 1 (query = "years").
_SCORES = [0.5, 0.9, 0.3, 1.6, 2.2]


def _softmax(xs):
    m = max(xs)
    exps = [math.exp(x - m) for x in xs]
    total = sum(exps)
    return [e / total for e in exps]


def _open_card(eq_markup, color, rows):
    """Equation plus short gloss lines, used while the box is open.

    Each row is (label, explanation) or (label, explanation, label_color).
    Labels that contain Pango tags are rendered with formula().
    """
    eq = formula(eq_markup, font_size=26, color=color)
    lines = VGroup()
    for row in rows:
        label, expl = row[0], row[1]
        lab_color = row[2] if len(row) > 2 else color
        if "<sub>" in label or "<sup>" in label:
            lab = formula(label, font_size=18, color=lab_color)
        else:
            lab = text(label, font_size=18, color=lab_color)
        lines.add(VGroup(lab, text(expl, font_size=18, color=MUTED)).arrange(RIGHT, buff=0.22))
    lines.arrange(DOWN, buff=0.12, aligned_edge=LEFT)
    return VGroup(eq, lines).arrange(DOWN, buff=0.26)


def _phase_card(title, color, lines):
    """Prefill / decode card: a heading plus short body lines."""
    head = text(title, font_size=22, color=color)
    body = VGroup(*[text(ln, font_size=16, color=MUTED) for ln in lines])
    body.arrange(DOWN, buff=0.1, aligned_edge=LEFT)
    return VGroup(head, body).arrange(DOWN, buff=0.16, aligned_edge=LEFT)


def _metrics_in_box(values, loop, as_bars=False):
    """One number (or bar) per input token, x-aligned, vertically inside the box."""
    peak = max(values) if values else 1.0
    y_mid = loop.inner_point()[1]
    cells = VGroup()
    for v, tok in zip(values, loop.input):
        x = tok.get_center()[0]
        if as_bars:
            h = 0.16 + 0.58 * (v / peak)
            bar = Rectangle(
                width=0.38,
                height=h,
                fill_color=ACCENT,
                fill_opacity=0.92,
                stroke_width=0,
            )
            bar.move_to([x, y_mid - 0.32 + h / 2, 0])
            num = text(f"{v:.2f}", font_size=15, color=FG)
            num.next_to(bar, UP, buff=0.05)
            cells.add(VGroup(bar, num))
        else:
            n = text(f"{v:.1f}", font_size=28, color=ACCENT)
            n.move_to([x, y_mid, 0])
            cells.add(n)
    return cells


class S1Transformers(Slide):
    def construct(self):
        apply_theme(self)

        loop = TransformerLoop(FOUR_SCORE[:4], show_kv=True, kv_opacity=0.0)
        heading = title("One word at a time")
        self._math = None
        inner = None

        # ------------------------------------------------------------------
        # Beat 1 — the loop, then Eq. 1
        # ------------------------------------------------------------------
        note = caption("words so far go in   →   one next word comes out")
        note.to_edge(DOWN, buff=0.36)

        self.play(FadeIn(heading), FadeIn(loop.tbox), run_time=0.8)
        self.play(
            LaggedStart(*[FadeIn(tok) for tok in loop.input], lag_ratio=0.2),
            run_time=1.3,
        )
        self.play(FadeIn(loop.in_arrow), FadeIn(loop.out_arrow), FadeIn(note), run_time=0.6)
        self._math = note
        self._decode_step(loop, FOUR_SCORE[4], kv_opacity=0.0)

        eq1 = formula(
            "P(x)  =  P(x<sub>1</sub>) · P(x<sub>2</sub> | x<sub>1</sub>) ··· P(x<sub>n</sub> | x<sub>1</sub>, …, x<sub>n−1</sub>)",
            font_size=20,
            color=MUTED,
        )
        eq_gloss = caption("the next word depends on every word before it")
        foot1 = VGroup(eq_gloss, eq1).arrange(DOWN, buff=0.12)
        self._set_footer(foot1, run_time=0.55)
        self.wait(0.6)
        self.next_slide()

        # ------------------------------------------------------------------
        # Beat 2 — Key
        # ------------------------------------------------------------------
        key_card = _open_card(
            "k<sub>i</sub>  =  W<sub>K</sub> x<sub>i</sub>",
            K_COLOR,
            [
                ("x<sub>i</sub>", "this token as a vector"),
                ("W<sub>K</sub>", "a learned matrix"),
                ("Key", "a label — what this token contains"),
            ],
        )
        self._fit_inner(key_card, loop)
        heading = self._retitle(heading, "Key: a label for each token")
        self._clear_footer()
        self.play(loop.park_label_top(), run_time=0.5)
        self.play(FadeIn(key_card), run_time=0.6)
        self.play(
            LaggedStart(*self._chip_anims(loop, "k", 1.0), lag_ratio=0.16),
            run_time=1.2,
        )
        inner = key_card
        self.wait(0.6)
        self.next_slide()

        # ------------------------------------------------------------------
        # Beat 3 — Value
        # ------------------------------------------------------------------
        val_card = _open_card(
            "v<sub>i</sub>  =  W<sub>V</sub> x<sub>i</sub>",
            V_COLOR,
            [
                ("Key", "how you find this token", K_COLOR),
                ("Value", "what you mix in if you pick it", V_COLOR),
            ],
        )
        self._fit_inner(val_card, loop)
        heading = self._retitle(
            heading,
            "Value: what the token will add",
            FadeOut(inner, shift=LEFT * 2.4),
            FadeIn(val_card),
            run_time=0.7,
        )
        self._dump(inner)
        self.play(
            LaggedStart(*self._chip_anims(loop, "v", 1.0), lag_ratio=0.16),
            run_time=1.2,
        )
        inner = val_card
        self.wait(0.6)
        self.next_slide()

        # ------------------------------------------------------------------
        # Beat 4 — Query
        # ------------------------------------------------------------------
        q_card = _open_card(
            "q<sub>i</sub>  =  W<sub>Q</sub> x<sub>i</sub>",
            Q_COLOR,
            [
                ("Query", "the question this step is asking"),
                ("only the newest token", "needs a fresh Query"),
                ("every token", "keeps its Key and Value"),
            ],
        )
        self._fit_inner(q_card, loop)
        heading = self._retitle(
            heading,
            "Query: the question this step asks",
            FadeOut(inner, shift=LEFT * 2.4),
            FadeIn(q_card),
            run_time=0.7,
        )
        self._dump(inner)
        q_chip = loop.input[-1].show_query()
        q_chip.set_opacity(0)
        self.play(q_chip.animate.set_opacity(1.0), run_time=0.55)
        self.play(loop.input.animate.move_to(loop.input_centered_pos()), run_time=0.45)
        loop.update_arrows()
        inner = q_card
        self.wait(0.6)
        self.next_slide()

        # ------------------------------------------------------------------
        # Beat 5 — scores
        # ------------------------------------------------------------------
        score_eq = formula(
            "s<sub>j</sub>  =  q<sub>i</sub><sup>T</sup> k<sub>j</sub> / √d      j = 1, …, i",
            font_size=24,
        )
        score_row = _metrics_in_box(_SCORES, loop, as_bars=False)
        heading = self._retitle(
            heading,
            "Match the Query against every Key",
            FadeOut(inner, shift=LEFT * 2.4),
            FadeIn(score_row),
            run_time=0.7,
        )
        self._dump(inner)
        self._set_footer(score_eq)
        inner = score_row
        self.wait(0.6)
        self.next_slide()

        # ------------------------------------------------------------------
        # Beat 6 — softmax
        # ------------------------------------------------------------------
        weights = _softmax(_SCORES)
        # Round to 2 decimals that still sum to 1 so the "budget of 1" is visible.
        weights = [round(w, 2) for w in weights]
        weights[-1] = round(1.0 - sum(weights[:-1]), 2)
        soft_eq = formula(
            "a<sub>ij</sub>  =  exp(s<sub>j</sub>) / Σ exp(s<sub>t</sub>)      t = 1, …, i",
            font_size=24,
        )
        bar_row = _metrics_in_box(weights, loop, as_bars=True)
        heading = self._retitle(
            heading,
            "Softmax turns scores into a budget of 1",
            FadeOut(inner, shift=LEFT * 2.4),
            FadeIn(bar_row),
            run_time=0.7,
        )
        self._dump(inner)
        self._set_footer(soft_eq)
        inner = bar_row
        self.wait(0.6)
        self.next_slide()

        # ------------------------------------------------------------------
        # Beat 7 — weighted values → next token, then close
        # ------------------------------------------------------------------
        out_eq = formula(
            "o<sub>i</sub>  =  Σ a<sub>ij</sub> v<sub>j</sub>      j = 1, …, i",
            font_size=24,
            color=V_COLOR,
        )
        heading = self._retitle(heading, "Mix the Values, write the next word", run_time=0.55)
        self._set_footer(out_eq)
        self._decode_step(loop, FOUR_SCORE[5], kv_opacity=1.0)

        loop.input[-2].hide_query()
        self.play(FadeOut(inner, shift=LEFT * 2.4), loop.restore_label(), run_time=0.6)
        self._dump(inner)
        inner = None
        loop.update_arrows()
        self.wait(0.6)
        self.next_slide()

        # ------------------------------------------------------------------
        # Beat 8 — closed loop, KV grows
        # ------------------------------------------------------------------
        grow = caption("none discarded — the next step still needs all of them")
        heading = self._retitle(heading, "Nothing is thrown away", run_time=0.55)
        self._set_footer(grow)
        self._decode_step(loop, FOUR_SCORE[6], kv_opacity=1.0)
        self.wait(0.6)
        self.next_slide()

        # ------------------------------------------------------------------
        # Beat 9 — prefill vs decode (inside the box, not a footer aside)
        # ------------------------------------------------------------------
        pre_card = _phase_card(
            "PREFILL",
            ACCENT,
            [
                "whole prompt at once, in parallel",
                "writes K,V for every token + first output",
                "compute-bound",
            ],
        )
        dec_card = _phase_card(
            "DECODE",
            ACCENT2,
            [
                "one new token per pass — this loop",
                "re-reads every previous K and V",
                "memory-bound — dominates latency",
            ],
        )
        phases = VGroup(pre_card, dec_card).arrange(RIGHT, buff=0.55, aligned_edge=UP)
        self._fit_inner(phases, loop)
        heading = self._retitle(heading, "Two phases: prefill and decode")
        self._clear_footer()
        self.play(loop.park_label_top(), run_time=0.45)
        self.play(FadeIn(pre_card), run_time=0.55)
        self.play(FadeIn(dec_card), run_time=0.55)
        inner = phases
        self.wait(0.6)
        self.next_slide()

        # ------------------------------------------------------------------
        # Beat 10 — landing
        # ------------------------------------------------------------------
        land = VGroup(
            text("To emit the next token you need the K and V of every previous token.", font_size=20),
            caption("every layer  ·  every decode step  ·  whole request"),
        ).arrange(DOWN, buff=0.16)
        heading = self._retitle(heading, "The box needs every previous K and V", run_time=0.55)
        self.play(FadeOut(inner, shift=LEFT * 2.4), loop.restore_label(), run_time=0.55)
        self._dump(inner)
        inner = None
        self._set_footer(land)
        self.wait(0.6)
        self.next_slide()

    def _chip_anims(self, loop, which, alpha):
        attr = "k_chip" if which == "k" else "v_chip"
        return [getattr(tok, attr).animate.set_opacity(alpha) for tok in loop.input]

    def _dump(self, mob):
        """Take a faded mobject out of the frame so Cairo cannot ghost it."""
        if mob is None:
            return
        family = list(mob.get_family())
        self.remove(mob, *family)
        mob.set_opacity(0)
        mob.shift(LEFT * 40)

    def _set_footer(self, new, run_time=0.5):
        new.to_edge(DOWN, buff=0.30)
        old = self._math
        if old is None:
            self.play(FadeIn(new), run_time=run_time)
        else:
            self.play(FadeOut(old, shift=LEFT * 2.4), FadeIn(new), run_time=run_time)
            self._dump(old)
        self._math = new

    def _clear_footer(self, run_time=0.45):
        old = self._math
        if old is None:
            return
        self.play(FadeOut(old, shift=LEFT * 2.4), run_time=run_time)
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

    def _retitle(self, old, new_str, *anims, run_time=0.65):
        new = title(new_str)
        self.play(
            FadeOut(old, shift=UP * 0.25),
            FadeIn(new, shift=DOWN * 0.12),
            *anims,
            run_time=run_time,
        )
        self._dump(old)
        return new

    def _decode_step(self, loop, word, kv_opacity=1.0):
        out = loop.spawn_output(word, kv_opacity=kv_opacity)
        delta = loop.join_delta()
        join = loop.join_point() + delta
        self.play(loop.set_active(True), FadeIn(out, shift=DOWN * 0.1), run_time=0.5)
        self.wait(0.3)
        self.play(out.animate.move_to(join), loop.input.animate.shift(delta), run_time=0.85)
        loop.adopt_output()
        loop.update_arrows()
        self.play(loop.set_active(False), run_time=0.3)
        return out
