"""S1 — Transformers & attention (Act I): just enough to see the relevance.

NARRATION

Beat 1 — Autoregressive generation
Here's how an LLM actually produces text: it does not write a whole sentence at once.
It looks at everything so far, predicts a single next token, appends that token to the
sequence, and repeats. Prompt in, one token out, append, repeat — that loop is the
entire generation process. Watch it run twice: from "Four score and seven" it produces
"years", then from "Four score and seven years" it produces "ago". [PAUSE] Notice: every
step re-reads the whole sequence so far.

Beat 2 — Q, K, V intuition
To decide what comes next, the model turns every token into three vectors. A Query:
"what am I looking for right now?" A Key: "what do I contain, that others might look
for?" And a Value: "what I'll actually contribute if someone attends to me." Only the
newest token needs a fresh Query; every token — old and new — carries a Key and a
Value.

Beat 3 — Query meets every Key
The newest token's Query gets compared against the Key of every token in the sequence,
including its own. Each comparison produces one number, a raw "score" — query dot key —
that says roughly how relevant that earlier token is to what we're looking for right
now. [PAUSE] Which earlier token do you think gets the highest score here?

Beat 4 — Softmax turns scores into weights
Those raw scores get squashed by softmax into weights that are all positive and sum to
one — a probability distribution over "how much attention to pay to each earlier
token." Bigger score, bigger slice of attention.

Beat 5 — Weighted sum → output → next token
Now take every token's Value vector, scale it by its attention weight, and add them all
up. That weighted sum of values is the output of this step, and it's what the model
turns into the next predicted token. Here, that's "years" — which gets appended right
back onto the sequence.

Beat 6 — One step later: the stacks grow
Run the loop again to produce "ago": the sequence is now one token longer, so there's
one more Key and one more Value in play than last time. Every single generation step
adds exactly one new K and one new V — and none of the old ones ever get thrown away,
because the next step still needs to compare against them too.

Beat 7 — The landing point
So here's the load-bearing fact for the rest of this talk: to produce the very next
token, you need the Key and the Value of every previous token, not just the most recent
one. That K/V state has to sit somewhere and stick around for the entire request. Keep
an eye on those colored stacks — that's exactly what we'll come back to.
"""

from manim import (
    DOWN,
    LEFT,
    ORIGIN,
    RIGHT,
    UP,
    FadeIn,
    FadeOut,
    GrowArrow,
    GrowFromEdge,
    Square,
    Text,
    Transform,
    VGroup,
)
from manim_slides import Slide

from talk.theme import (
    ACCENT,
    BG,
    FG,
    GOOD,
    K_COLOR,
    MUTED,
    Q_COLOR,
    SMALL_SIZE,
    TINY_SIZE,
    TITLE_SIZE,
    V_COLOR,
    apply_theme,
    body,
    caption,
    small,
    title,
)
from talk.components import FOUR_SCORE, TokenBox, attention_diagram, token_sequence


class S1Transformers(Slide):
    def construct(self):
        apply_theme(self)

        # ------------------------------------------------------------------
        # Beat 1 — the generation loop: prompt in -> one token out -> append -> repeat
        # ------------------------------------------------------------------
        t1 = title("How an LLM writes: one token at a time")
        self.play(FadeIn(t1))

        seq_words = FOUR_SCORE[:4]  # "Four score and seven"
        seq = token_sequence(seq_words, height=0.55, font_size=SMALL_SIZE)
        seq.move_to(ORIGIN + UP * 0.3)
        loop_caption = caption("prompt in -> one token out -> append -> repeat")
        loop_caption.next_to(seq, DOWN, buff=1.0)

        self.play(FadeIn(seq, shift=UP * 0.2))
        self.play(FadeIn(loop_caption))

        # step 1: produce "years"
        next_word = FOUR_SCORE[4]
        new_box = TokenBox(next_word, color=BG, fill=GOOD, height=0.55, font_size=SMALL_SIZE)
        new_box.next_to(seq, RIGHT, buff=0.6)
        self.play(GrowFromEdge(new_box, LEFT))
        self.play(new_box.animate.next_to(seq, RIGHT, buff=0.12))
        seq.add(new_box)

        # step 2: produce "ago"
        next_word2 = FOUR_SCORE[5]
        new_box2 = TokenBox(next_word2, color=BG, fill=GOOD, height=0.55, font_size=SMALL_SIZE)
        new_box2.next_to(seq, RIGHT, buff=0.6)
        self.play(GrowFromEdge(new_box2, LEFT))
        self.play(new_box2.animate.next_to(seq, RIGHT, buff=0.12))
        seq.add(new_box2)

        self.wait(0.3)
        self.next_slide()

        self.play(FadeOut(VGroup(t1, seq, loop_caption)))

        # ------------------------------------------------------------------
        # Beat 2 — Q/K/V intuition
        # ------------------------------------------------------------------
        t2 = title("Every token becomes three vectors")
        self.play(FadeIn(t2))

        prefix = FOUR_SCORE[:4]
        diagram = attention_diagram(prefix, query_index=3)
        diagram.scale(1.1)
        diagram.move_to(ORIGIN + DOWN * 0.2)

        self.play(FadeIn(diagram.tokens, shift=UP * 0.2))
        self.play(
            FadeIn(diagram.keys, lag_ratio=0.1),
            FadeIn(diagram.values, lag_ratio=0.1),
        )
        self.play(FadeIn(diagram.query))

        legend = VGroup(
            _swatch_line("Q = what am I looking for", Q_COLOR),
            _swatch_line("K = what do I contain", K_COLOR),
            _swatch_line("V = what I'll contribute", V_COLOR),
        )
        legend.arrange(DOWN, buff=0.2, aligned_edge=LEFT)
        legend.to_edge(RIGHT, buff=0.6).shift(UP * 0.5)
        self.play(FadeIn(legend))

        note = caption("only the newest token needs a fresh Query")
        note.next_to(diagram.tokens, UP, buff=1.2).align_to(diagram.tokens, LEFT)
        self.play(FadeIn(note))
        self.wait(0.3)
        self.next_slide()

        self.play(FadeOut(t2), FadeOut(legend), FadeOut(note))

        # ------------------------------------------------------------------
        # Beat 3 — query compared to every key -> scores
        # ------------------------------------------------------------------
        t3 = title("Query . Key -> a score for every token")
        self.play(FadeIn(t3))
        self.play(*[GrowArrow(a) for a in diagram.arrows])
        score_label = caption("query . key -> score  (one number per earlier token)")
        score_label.next_to(diagram.keys, DOWN, buff=1.6)
        self.play(FadeIn(score_label))
        self.wait(0.3)
        self.next_slide()

        self.play(FadeOut(t3), FadeOut(score_label))

        # ------------------------------------------------------------------
        # Beat 4 — softmax -> weights
        # ------------------------------------------------------------------
        t4 = title("Softmax turns scores into weights")
        self.play(FadeIn(t4))
        self.play(FadeOut(diagram.arrows))
        self.play(FadeIn(diagram.weights, lag_ratio=0.1))
        weights_label = caption("softmax -> weights (positive, sum to one)")
        weights_label.next_to(diagram.weights, DOWN, buff=0.3)
        self.play(FadeIn(weights_label))
        self.wait(0.3)
        self.next_slide()

        self.play(FadeOut(t4), FadeOut(weights_label))

        # ------------------------------------------------------------------
        # Beat 5 — weighted sum of values -> output -> next token
        # ------------------------------------------------------------------
        t5 = title("Weighted sum of values -> next token")
        self.play(FadeIn(t5))
        self.play(FadeIn(diagram.output, shift=UP * 0.2))
        output_label = caption("weighted sum of values -> output -> next-token prediction")
        output_label.next_to(diagram.output, DOWN, buff=0.3)
        self.play(FadeIn(output_label))

        predicted = TokenBox(FOUR_SCORE[4], color=FG, fill=GOOD, height=0.55, font_size=SMALL_SIZE)
        predicted.next_to(diagram.tokens, RIGHT, buff=0.6)
        arrow_to_seq = _bend_arrow(diagram.output, predicted)
        self.play(GrowArrow(arrow_to_seq))
        self.play(FadeIn(predicted, shift=UP * 0.2))
        self.wait(0.3)
        self.next_slide()

        self.play(
            FadeOut(t5),
            FadeOut(output_label),
            FadeOut(arrow_to_seq),
            FadeOut(diagram.output),
            FadeOut(diagram.weights),
        )

        # ------------------------------------------------------------------
        # Beat 6 — one step later: the stacks grow
        # ------------------------------------------------------------------
        t6 = title("Next step: one more K, one more V")
        self.play(FadeIn(t6))

        # merge predicted token into the sequence, then build the extended diagram
        self.play(FadeOut(predicted))
        prefix2 = FOUR_SCORE[:5]
        diagram2 = attention_diagram(prefix2, query_index=4)
        diagram2.scale(1.1)
        diagram2.move_to(ORIGIN + DOWN * 0.2)

        self.play(
            FadeOut(diagram.tokens),
            FadeOut(diagram.keys),
            FadeOut(diagram.values),
            FadeOut(diagram.query),
        )
        self.play(FadeIn(diagram2.tokens, shift=UP * 0.2))
        self.play(
            FadeIn(diagram2.keys, lag_ratio=0.1),
            FadeIn(diagram2.values, lag_ratio=0.1),
        )
        self.play(FadeIn(diagram2.query))

        grow_note = caption("every step: +1 Key, +1 Value -- none are ever discarded")
        grow_note.next_to(diagram2.tokens, UP, buff=1.2).align_to(diagram2.tokens, LEFT)
        self.play(FadeIn(grow_note))
        self.wait(0.3)
        self.next_slide()

        self.play(FadeOut(t6), FadeOut(grow_note))

        # ------------------------------------------------------------------
        # Beat 7 — landing statement
        # ------------------------------------------------------------------
        self.play(FadeOut(diagram2.tokens), FadeOut(diagram2.keys), FadeOut(diagram2.values), FadeOut(diagram2.query))

        landing = body(
            "To produce the next token you need the\nK and V of EVERY previous token.",
            font_size=TITLE_SIZE,
        )
        landing.move_to(ORIGIN + UP * 0.5)

        k_stack = _tall_stack(6, K_COLOR)
        v_stack = _tall_stack(6, V_COLOR)
        k_label = small("Keys (K)", color=K_COLOR)
        v_label = small("Values (V)", color=V_COLOR)
        stacks = VGroup(k_stack, v_stack).arrange(RIGHT, buff=1.4)
        stacks.next_to(landing, DOWN, buff=0.8)
        k_label.next_to(k_stack, DOWN, buff=0.2)
        v_label.next_to(v_stack, DOWN, buff=0.2)

        self.play(FadeIn(landing))
        self.play(FadeIn(stacks), FadeIn(k_label), FadeIn(v_label))
        self.wait(0.3)
        self.next_slide()


def _swatch_line(text_str, color):
    sq = Square(side_length=0.22, fill_color=color, fill_opacity=0.9, stroke_width=0)
    txt = Text(text_str, font_size=TINY_SIZE, color=FG)
    txt.next_to(sq, RIGHT, buff=0.15)
    return VGroup(sq, txt)


def _bend_arrow(src, dst):
    from manim import Arrow

    return Arrow(
        src.get_center(),
        dst.get_center(),
        buff=0.3,
        stroke_width=2.5,
        color=MUTED,
        max_tip_length_to_length_ratio=0.08,
    )


def _tall_stack(n, color, cell_w=0.9, cell_h=0.28):
    from manim import Rectangle

    stack = VGroup()
    for i in range(n):
        r = Rectangle(width=cell_w, height=cell_h, fill_color=color, fill_opacity=0.85, stroke_color=FG, stroke_width=1)
        stack.add(r)
    stack.arrange(UP, buff=0.05)
    return stack
