"""S11 — Takeaways + discussion (Act III, close).

NARRATION
---------
Beat 1 — Three takeaways.
Three things worth carrying out of this talk. One: the bottleneck in LLM
serving was never the model's math — it was memory, specifically the KV
cache. Two: vLLM's fix is almost embarrassingly simple in hindsight — fixed
size blocks, a block table indirection, and on-demand allocation get you to
about 96% memory utilization, and sharing between requests basically falls
out for free. Three: this is an OS idea, but re-engineered for GPUs — a
software address-translation layer, a new fused attention kernel to pay for
that indirection, and new eviction and recovery policies suited to how LLMs
actually behave.

Beat 2 — Impact.
The practical result: vLLM became the de-facto open-source engine for
serving large language models, and PagedAttention's block-based KV cache
design has been adopted well beyond the original project, across the
industry.

Beat 3 — Discussion prompt.
So here's the question I want to leave you with: paging worked for the KV
cache. What else in an LLM serving system could you page? [PAUSE] A few
things people have actually tried since this paper: caching and reusing
shared prompt prefixes across different requests automatically; paging
LoRA adapters themselves, so many fine-tuned variants can be swapped in and
out like blocks; splitting prefill and decode onto different machines;
and offloading colder KV blocks from GPU memory to CPU RAM or disk.

Beat 4 — Thanks and questions.
Thanks for listening. Happy to take questions.
"""

from manim import (
    DOWN,
    LEFT,
    RIGHT,
    UP,
    ORIGIN,
    FadeIn,
    FadeOut,
    Line,
    VGroup,
)
from manim_slides import Slide

from talk.theme import *
from talk.components import *


class S11Takeaways(Slide):
    def construct(self):
        apply_theme(self)

        # -------------------------------------------------------------
        # Beat 1: three takeaways, one at a time
        # -------------------------------------------------------------
        heading = title("Three takeaways")
        self.play(FadeIn(heading))

        takeaways = [
            "1. The bottleneck was memory, not the model.",
            "2. Fixed-size blocks + block table + on-demand\nallocation -> ~96% utilization, free sharing.",
            "3. An OS idea re-engineered for GPUs: software\ntranslation, a new kernel, new policies.",
        ]
        lines = [body(txt, font_size=BODY_SIZE) for txt in takeaways]
        shown = VGroup(*lines)
        shown.arrange(DOWN, buff=0.55, aligned_edge=LEFT)
        shown.move_to(ORIGIN).shift(DOWN * 0.2)
        for line in lines:
            self.play(FadeIn(line))
            self.wait(0.2)
        self.next_slide()

        self.play(FadeOut(heading), FadeOut(shown))

        # -------------------------------------------------------------
        # Beat 2: impact
        # -------------------------------------------------------------
        heading2 = title("Impact")
        self.play(FadeIn(heading2))

        impact1 = body("vLLM became the de-facto open-source engine\nfor serving large language models.", font_size=BODY_SIZE)
        impact2 = body("PagedAttention's block-based KV cache design\nis now used across the industry.", font_size=BODY_SIZE, color=ACCENT)
        impact_group = VGroup(impact1, impact2)
        impact_group.arrange(DOWN, buff=0.6, aligned_edge=LEFT)
        impact_group.move_to(ORIGIN)

        self.play(FadeIn(impact1))
        self.play(FadeIn(impact2))
        self.wait(0.3)
        self.next_slide()
        self.play(FadeOut(heading2), FadeOut(impact_group))

        # -------------------------------------------------------------
        # Beat 3: discussion prompt, then reveal real vLLM features
        # -------------------------------------------------------------
        prompt = title("What else could you page?")
        prompt.scale(1.15)
        prompt.move_to(UP * 1.3)
        self.play(FadeIn(prompt))
        self.wait(0.3)
        self.next_slide()

        examples = [
            "Automatic prefix caching (shared prefixes across requests)",
            "Paged LoRA adapters (many fine-tuned adapters, swapped like blocks)",
            "Prefill / decode disaggregation",
            "KV offloading to CPU RAM / disk",
        ]
        ex_group = VGroup(*[small(e, font_size=SMALL_SIZE, color=MUTED) for e in examples])
        ex_group.arrange(DOWN, buff=0.4, aligned_edge=LEFT)
        ex_group.next_to(prompt, DOWN, buff=0.8)
        caption_note = caption("later became real vLLM features")
        caption_note.next_to(ex_group, DOWN, buff=0.4)

        self.play(FadeIn(ex_group, lag_ratio=0.15))
        self.play(FadeIn(caption_note))
        self.wait(0.3)
        self.next_slide()
        self.play(FadeOut(prompt), FadeOut(ex_group), FadeOut(caption_note))

        # -------------------------------------------------------------
        # Beat 4: thanks / questions card
        # -------------------------------------------------------------
        thanks = title("Thanks! Questions?")
        thanks.scale(1.1)
        thanks.move_to(UP * 1.6)

        divider = Line(LEFT * 3.2, RIGHT * 3.2, color=BLOCK_STROKE, stroke_width=2)
        divider.next_to(thanks, DOWN, buff=0.5)

        citation = small(
            "Kwon et al., Efficient Memory Management for Large\n"
            "Language Model Serving with PagedAttention, SOSP 2023",
            font_size=SMALL_SIZE, color=MUTED,
        )
        citation.next_to(divider, DOWN, buff=0.5)

        contact = caption("Agrim Bansal")
        contact.next_to(citation, DOWN, buff=0.6)

        self.play(FadeIn(thanks))
        self.play(FadeIn(divider))
        self.play(FadeIn(citation))
        self.play(FadeIn(contact))
        self.wait(0.3)
        self.next_slide()
        self.play(FadeOut(thanks), FadeOut(divider), FadeOut(citation), FadeOut(contact))
        self.wait(0.3)
