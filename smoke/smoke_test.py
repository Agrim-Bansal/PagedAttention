from manim import *
from manim_slides import Slide


class SmokeTest(Slide):
    def construct(self):
        title = Text("PagedAttention")
        box = RoundedRectangle(corner_radius=0.2, width=4, height=2, color=BLUE)
        box.next_to(title, DOWN, buff=0.5)

        self.play(Write(title))
        self.play(Create(box))
        self.next_slide()

        self.play(title.animate.set_color(YELLOW), box.animate.set_color(GREEN))
        self.wait(0.5)
