"""S8 — Scheduling under memory pressure (Act III).

Paging packed leftover VRAM and let requests share blocks. The GPU can now
admit more work than it can finish at once. This scene is that pressure:
who gives up memory, how much, and how the victim comes back.

One claim per pause. The pool stays on stage through the two recoveries.

Figures: 4 (system overview) as a late zoom-out of jobs already watched.
No Fig 19 — swap and recompute are shown as mechanisms, not compared.

NARRATION
---------
Beat 1 — Pickup: request A.
Look at leftover VRAM — the slice we have been spending all talk. Request A
is live. Four blocks, scattered, allocated on demand. Three still free.
Paging put those blocks wherever there was room. Sit with this picture.
[PAUSE]

Beat 2 — Request B joins.
Request B draws from the same pool. Its blocks sit interleaved with A's —
not a contiguous slab, just whatever was free. Sharing from last scene
packed this leftover even tighter. Three slots remain. The GPU is fuller
than it has ever been in this talk. It is still finite.

Beat 3 — C fills the leftovers.
A third request arrives. Any free block fits anyone, so C lands in those
three leftover slots, interleaved with A and B. The pool is full. We have
now admitted more work than we can finish if everyone keeps growing.
[PAUSE]

Beat 4 — This picture is new.
Contiguous systems never faced this. They reserved two thousand and
forty-eight slots the moment a request arrived — the model's maximum —
so they never ran out mid-decode. They bounced new work at the door.
Paging created the ability to overcommit. That is why we suddenly need
a policy for taking memory back.

Beat 5 — The pool runs dry.
A generates another token and needs one more block. There isn't one.
[PAUSE] Which request gives up its memory? And what do we do with it?

Beat 6 — Name the scheduler.
The answer starts with a job. Someone has to pick who runs this step,
and who gives up space when the pool is empty. That is the scheduler.
We will watch it work.

Beat 7 — Preempt the newest.
The policy is deliberately simple: first-come, first-served, with
preemption. Serve oldest first. When someone has to leave, evict the
newest arrival — here, C. A has been running longest, so A is the last
request we ever kick. A stream of new arrivals cannot starve the ones
already on the GPU. [PAUSE]

Beat 8 — Try taking half.
Suppose we only take two of C's three blocks, so A can continue. A
gets a block. One of C's blocks is free. One is still C's. Sit with
that leftover. [PAUSE] Every decode step reads every block of a
sequence — that is why the PagedAttention kernel walks the whole table,
every token. C cannot run on one leftover block. Those two freed blocks
bought A a step and stranded C.

Beat 9 — All or none.
Put them back. The rule is all of C's blocks, or none of them. An
operating system will happily evict a single page. We cannot. That is
the policy the OS scene promised: every block of a sequence is always
touched together. [PAUSE]

Beat 10 — Sequence groups.
And if C were two beam candidates sharing a trunk — the copy-on-write
picture from last scene — kicking only one of them would strand the
shared blocks. Reference count two: both still need that memory. So
the scheduler treats a sequence group as one unit. Beams of the same
request are gang-preempted, and gang-resumed, together. Sharing couples
their lifetimes.

Beat 11 — The other pool.
C is leaving. Before we move anything: remember the two memory pools
from the GPU scene. GPU VRAM, and ordinary CPU RAM, joined by PCIe —
the slow bridge. Swap space lives on that other side. Watch it appear.
[PAUSE]

Beat 12 — Swap.
Copy C's blocks across the bridge. GPU slots empty. A takes one and
continues. Swap space cannot grow forever. It is capped at the GPU's
total KV capacity, because that is the most we could ever evict at
once. The cost is PCIe bandwidth.

Beat 13 — Or drop, keep the tokens.
Or we do something an OS almost never does: throw the KV cache away.
Undo the copy. Drop C's blocks. Keep the tokens — the original prompt,
plus whatever C had already generated. The words are cheap. The cache
was the expensive object. [PAUSE]

Beat 14 — One prefill.
When C is rescheduled, concatenate those tokens and treat the whole
string as one new prompt. One prefill pass rebuilds the entire cache
in parallel, not one slow decode step at a time.

Beat 15 — Fig. 4, and the knob.
Zoom out. Same three jobs we just watched. The scheduler picks who
runs and who is preempted. It talks to a KV cache manager that owns
every block table — and under that, a GPU allocator for the leftover
pool, and a CPU allocator for the swap strip. Workers, one GPU each,
run the step they are told to run. Contiguous slabs had no way to
take memory back. Paging is the knob.
"""

from manim import (
    DOWN,
    LEFT,
    RIGHT,
    UP,
    AnimationGroup,
    Create,
    FadeIn,
    FadeOut,
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


FRAME_W = 13.2
A_COLOR = ACCENT
B_COLOR = ACCENT2
C_COLOR = V_COLOR

A_IDX = (0, 2, 5, 7)
B_IDX = (1, 4, 8)
C_IDX = (3, 6, 9)


def _fit_title(content):
    t = title(content)
    if t.width > FRAME_W:
        t.scale_to_fit_width(FRAME_W)
        t.to_edge(UP)
    return t


def _box(label, width=2.4, height=0.7, fill=BLOCK_FILL, stroke=BLOCK_STROKE,
         text_color=FG, font_size=SMALL_SIZE):
    b = RoundedRectangle(
        corner_radius=0.1, width=width, height=height,
        fill_color=fill, fill_opacity=1.0, stroke_color=stroke, stroke_width=2,
    )
    t = text(label, font_size=font_size, color=text_color)
    if t.width > width * 0.88:
        t.scale_to_fit_width(width * 0.88)
    t.move_to(b.get_center())
    g = VGroup(b, t)
    g.box = b
    g.label = t
    return g


def _mem_pool(n=10, cols=5, cell=0.84, title_str="GPU KV blocks", color=FG):
    cells = VGroup(*[
        Square(
            side_length=cell,
            fill_color=BLOCK_FILL,
            fill_opacity=1.0,
            stroke_color=BLOCK_STROKE,
            stroke_width=2,
        )
        for _ in range(n)
    ])
    rows = (n + cols - 1) // cols
    cells.arrange_in_grid(rows=rows, cols=cols, buff=0.20)
    ttl = small(title_str, font_size=SMALL_SIZE, color=color)
    ttl.next_to(cells, UP, buff=0.22)
    g = VGroup(ttl, cells)
    g.cells = cells
    g.title = ttl
    return g


def _req_chip(name, color, arrival):
    box = TokenBox(name, color=BG, fill=color, width=1.7, height=0.55, font_size=SMALL_SIZE)
    tag = caption(arrival)
    tag.next_to(box, DOWN, buff=0.10)
    g = VGroup(box, tag)
    g.box = box
    g.tag = tag
    return g


def _role_box(label, role, width=3.0, height=1.08, fill=BLOCK_FILL, stroke=BLOCK_STROKE,
              text_color=FG, role_color=None, font_size=SMALL_SIZE):
    b = RoundedRectangle(
        corner_radius=0.1, width=width, height=height,
        fill_color=fill, fill_opacity=1.0, stroke_color=stroke, stroke_width=2,
    )
    t = text(label, font_size=font_size, color=text_color)
    if t.width > width * 0.88:
        t.scale_to_fit_width(width * 0.88)
    rc = MUTED if role_color is None else role_color
    r = caption(role, color=rc)
    if r.width > width * 0.90:
        r.scale_to_fit_width(width * 0.90)
    t.move_to(b.get_center() + UP * 0.18)
    r.move_to(b.get_center() + DOWN * 0.22)
    g = VGroup(b, t, r)
    g.body = b
    g.label = t
    g.role = r
    return g


def _edge_arrow(src, dst, src_edge=DOWN, dst_edge=UP, color=MUTED):
    return arrow(
        src.get_edge_center(src_edge),
        dst.get_edge_center(dst_edge),
        color=color,
        buff=0.14,
        stroke_width=2.5,
    )


class S8Scheduling(Slide):
    def construct(self):
        apply_theme(self)

        # -------------------------------------------------------------
        # Beat 1 — Pickup: A in leftover VRAM
        # -------------------------------------------------------------
        heading = _fit_title("Leftover VRAM, still finite")
        self.play(FadeIn(heading), run_time=0.5)

        gpu = _mem_pool(title_str="Leftover VRAM  ·  KV blocks")
        gpu.move_to(UP * 0.28)
        free_lbl = small("10 free", font_size=SMALL_SIZE, color=GOOD)
        free_lbl.next_to(gpu.title, RIGHT, buff=0.35)

        chip_a = _req_chip("Req A", A_COLOR, "t = 0")
        chip_b = _req_chip("Req B", B_COLOR, "t = 1")
        chip_c = _req_chip("Req C", C_COLOR, "t = 2")
        chips = VGroup(chip_a, chip_b, chip_c)
        chips.arrange(RIGHT, buff=0.50)
        chips.move_to(DOWN * 2.12)
        chip_b.set_opacity(0)
        chip_c.set_opacity(0)
        self.add(chip_b, chip_c)

        cap = caption("Paging placed these blocks wherever there was room.")
        cap.to_edge(DOWN, buff=0.30)

        self.play(FadeIn(gpu), FadeIn(free_lbl), run_time=0.6)
        self.play(FadeIn(chip_a), run_time=0.45)
        self.play(
            LaggedStart(
                *[gpu.cells[i].animate.set_fill(A_COLOR, 1) for i in A_IDX],
                lag_ratio=0.22,
            ),
            run_time=1.1,
        )
        self._set_free(free_lbl, "6 free", GOOD)
        self.play(FadeIn(cap), run_time=0.4)
        self.wait(0.4)
        self.next_slide()

        # -------------------------------------------------------------
        # Beat 2 — B joins, interleaved
        # -------------------------------------------------------------
        heading = self._retitle(heading, "A second request, same pool")
        cap = self._recaption(cap, "Interleaved. Sharing packed it tighter. Still finite.")
        self.play(chip_b.animate.set_opacity(1), run_time=0.5)
        self.play(
            LaggedStart(
                *[gpu.cells[i].animate.set_fill(B_COLOR, 1) for i in B_IDX],
                lag_ratio=0.22,
            ),
            run_time=0.9,
        )
        self._set_free(free_lbl, "3 free", GOOD)
        self.wait(0.4)
        self.next_slide()

        # -------------------------------------------------------------
        # Beat 3 — C fills the leftovers
        # -------------------------------------------------------------
        heading = self._retitle(heading, "A third request fills what is left")
        cap = self._recaption(cap, "Any free block fits anyone. The pool is full.")
        self.play(chip_c.animate.set_opacity(1), run_time=0.5)
        self.play(
            LaggedStart(
                *[gpu.cells[i].animate.set_fill(C_COLOR, 1) for i in C_IDX],
                lag_ratio=0.22,
            ),
            run_time=0.9,
        )
        self._set_free(free_lbl, "0 free", BAD)
        self.wait(0.45)
        self.next_slide()

        # -------------------------------------------------------------
        # Beat 4 — This picture is new
        # -------------------------------------------------------------
        heading = self._retitle(heading, "Contiguous systems never reached this picture")
        cap = self._recaption(
            cap,
            "They reserved 2048 slots at arrival. They never ran out mid-decode.",
        )
        pulse = SurroundingRectangle(gpu.cells, buff=0.14, color=WARN, stroke_width=2.5, corner_radius=0.08)
        self.play(Create(pulse), run_time=0.7)
        self.wait(0.45)
        self.next_slide()

        # -------------------------------------------------------------
        # Beat 5 — Pool runs dry
        # -------------------------------------------------------------
        heading = self._retitle(heading, "A needs one more block")
        cap = self._recaption(cap, "Which request gives up its memory, and what do we do with it?")
        self.play(FadeOut(pulse), run_time=0.35)
        need = small("needs 1 more block", font_size=SMALL_SIZE, color=BAD)
        need.next_to(chip_a, UP, buff=0.18)
        ring = SurroundingRectangle(gpu.cells, buff=0.12, color=BAD, stroke_width=2.5, corner_radius=0.08)
        self.play(FadeIn(need), chip_a.box.box.animate.set_stroke(BAD, width=4), run_time=0.5)
        self.play(Create(ring), run_time=0.55)
        self.wait(0.5)
        self.next_slide()

        # -------------------------------------------------------------
        # Beat 6 — Name the scheduler
        # -------------------------------------------------------------
        heading = self._retitle(heading, "Someone has to decide")
        cap = self._recaption(cap, "Who runs this step. Who gives up space when the pool is empty.")
        sched = _box("Scheduler", width=2.4, height=0.60, fill=ACCENT, stroke=ACCENT,
                     text_color=BG, font_size=SMALL_SIZE)
        sched.move_to(UP * 2.52)
        self.play(FadeOut(need), FadeOut(ring), chip_a.box.box.animate.set_stroke(BLOCK_STROKE, width=2), run_time=0.4)
        self.play(FadeIn(sched, shift=DOWN * 0.12), run_time=0.55)
        self.wait(0.45)
        self.next_slide()

        # -------------------------------------------------------------
        # Beat 7 — Preempt the newest
        # -------------------------------------------------------------
        heading = self._retitle(heading, "Preempt the newest")
        cap = self._recaption(cap, "First-come, first-served. A stream of new arrivals cannot starve A.")
        oldest = small("oldest — last to kick", font_size=TINY_SIZE, color=GOOD)
        oldest.next_to(chip_a, DOWN, buff=0.42)
        newest = small("newest — preempted", font_size=TINY_SIZE, color=BAD)
        newest.next_to(chip_c, DOWN, buff=0.42)
        self.play(FadeIn(oldest), run_time=0.4)
        self.play(
            chip_c.animate.set_opacity(0.38),
            *[gpu.cells[i].animate.set_fill(C_COLOR, 0.38) for i in C_IDX],
            FadeIn(newest),
            run_time=0.8,
        )
        self.wait(0.45)
        self.next_slide()

        # -------------------------------------------------------------
        # Beat 8 — Partial eviction, sit stranded
        # -------------------------------------------------------------
        heading = self._retitle(heading, "Try taking only half of C")
        cap = self._recaption(cap, "A gets a block. C is left with one. Sit with that leftover.")
        self.play(
            gpu.cells[3].animate.set_fill(A_COLOR, 1),
            gpu.cells[6].animate.set_fill(BLOCK_FILL, 1),
            run_time=0.75,
        )
        stranded = SurroundingRectangle(gpu.cells[9], buff=0.10, color=BAD, stroke_width=3, corner_radius=0.06)
        still = caption("C still needs all 3")
        still.next_to(gpu.cells[9], DOWN, buff=0.20)
        self.play(Create(stranded), FadeIn(still), run_time=0.55)
        self.wait(0.5)
        self.next_slide()

        # -------------------------------------------------------------
        # Beat 9 — All or none
        # -------------------------------------------------------------
        heading = self._retitle(heading, "All of C's blocks, or none")
        cap = self._recaption(cap, "Every decode step reads every block. An OS can evict one page. We cannot.")
        self.play(
            FadeOut(stranded),
            FadeOut(still),
            gpu.cells[3].animate.set_fill(C_COLOR, 0.38),
            gpu.cells[6].animate.set_fill(C_COLOR, 0.38),
            gpu.cells[9].animate.set_fill(C_COLOR, 0.38),
            run_time=0.7,
        )
        self.wait(0.45)
        self.next_slide()

        # -------------------------------------------------------------
        # Beat 10 — Sequence groups
        # -------------------------------------------------------------
        heading = self._retitle(heading, "Shared blocks go together")
        cap = self._recaption(cap, "If C is two beams sharing a trunk, kicking one strands the other.")
        self.play(FadeOut(oldest), run_time=0.3)

        c1 = _req_chip("C1", C_COLOR, "beam")
        c2 = _req_chip("C2", C_COLOR, "beam")
        pair = VGroup(c1, c2)
        pair.arrange(RIGHT, buff=0.18)
        pair.move_to(chip_c.get_center())
        pair.set_opacity(0.55)
        badge = RefCountBadge(value=2, color=C_COLOR)
        badge.next_to(pair, RIGHT, buff=0.18)
        gang = caption("sequence group — gang-preempted")
        gang.next_to(pair, UP, buff=0.14)

        self.play(FadeOut(chip_c), FadeOut(newest), FadeIn(pair), FadeIn(gang), FadeIn(badge), run_time=0.7)
        self.wait(0.45)
        self.next_slide()

        # -------------------------------------------------------------
        # Beat 11 — CPU RAM appears (S1 callback)
        # -------------------------------------------------------------
        heading = self._retitle(heading, "The other pool")
        cap = self._recaption(cap, "CPU RAM, joined by PCIe — the slow bridge. Swap lives on that side.")
        self.play(FadeOut(badge), FadeOut(gang), run_time=0.3)

        chip_c2 = _req_chip("Req C", C_COLOR, "preempted")
        chip_c2.move_to(pair.get_center())
        chip_c2.set_opacity(0.55)
        self.play(FadeOut(pair), FadeIn(chip_c2), run_time=0.45)

        cpu = _mem_pool(cell=0.56, title_str="CPU RAM", color=WARN)
        cpu.scale(0.92)
        self.play(gpu.animate.shift(RIGHT * 2.60), free_lbl.animate.shift(RIGHT * 2.60), run_time=0.7)
        cpu.next_to(gpu, LEFT, buff=1.20)
        cpu.align_to(gpu, UP)
        pcie = arrow_map(gpu, cpu, color=WARN)
        pcie_lab = caption("PCIe")
        pcie_lab.next_to(pcie, UP, buff=0.10)
        self.play(FadeIn(cpu), shoot(pcie), FadeIn(pcie_lab), run_time=0.7)
        self.wait(0.45)
        self.next_slide()

        # -------------------------------------------------------------
        # Beat 12 — Swap
        # -------------------------------------------------------------
        heading = self._retitle(heading, "Swap: copy out over PCIe")
        cap = self._recaption(cap, "Swap space is capped at the GPU's KV capacity. Cost: the slow bridge.")

        ghosts = VGroup()
        ghost_anims = []
        for i in C_IDX:
            ghost = gpu.cells[i].copy()
            ghost.set_fill(C_COLOR, 0.95)
            ghosts.add(ghost)
            ghost_anims.append(ghost.animate.move_to(cpu.cells[i].get_center()).scale(0.56 / 0.84))
        self.add(ghosts)
        self.play(
            LaggedStart(*ghost_anims, lag_ratio=0.18),
            AnimationGroup(*[gpu.cells[i].animate.set_fill(BLOCK_FILL, 1) for i in C_IDX]),
            run_time=1.15,
        )
        self.play(
            *[cpu.cells[i].animate.set_fill(C_COLOR, 1) for i in C_IDX],
            FadeOut(ghosts),
            run_time=0.45,
        )
        self.play(gpu.cells[3].animate.set_fill(A_COLOR, 1), run_time=0.55)
        self._set_free(free_lbl, "2 free", GOOD)
        bound = caption("CPU swap  ≤  GPU KV blocks")
        bound.next_to(cpu, DOWN, buff=0.18)
        self.play(FadeIn(bound), run_time=0.4)
        self.wait(0.45)
        self.next_slide()

        # -------------------------------------------------------------
        # Beat 13 — Drop instead; keep the tokens
        # -------------------------------------------------------------
        heading = self._retitle(heading, "Or drop the cache. Keep the tokens.")
        cap = self._recaption(cap, "The words are cheap. The KV cache was the expensive object.")
        self.play(
            FadeOut(bound),
            *[cpu.cells[i].animate.set_fill(BLOCK_FILL, 1) for i in C_IDX],
            *[gpu.cells[i].animate.set_fill(C_COLOR, 0.38) for i in C_IDX],
            run_time=0.7,
        )
        self.play(
            FadeOut(cpu), FadeOut(pcie), FadeOut(pcie_lab),
            gpu.animate.shift(LEFT * 2.60),
            free_lbl.animate.shift(LEFT * 2.60),
            run_time=0.6,
        )
        self._set_free(free_lbl, "0 free", BAD)
        self.play(*[gpu.cells[i].animate.set_fill(BLOCK_FILL, 1) for i in C_IDX], run_time=0.65)
        self._set_free(free_lbl, "3 free", GOOD)

        prompt = token_sequence(["You", "only", "live"], height=0.42, font_size=TINY_SIZE)
        gen = token_sequence(["once"], height=0.42, font_size=TINY_SIZE, fill=C_COLOR, color=BG)
        tokens = VGroup(prompt, gen)
        tokens.arrange(RIGHT, buff=0.14)
        self.play(
            chip_a.animate.shift(DOWN * 0.38),
            chip_b.animate.shift(DOWN * 0.38),
            chip_c2.animate.shift(DOWN * 0.38),
            run_time=0.35,
        )
        tokens.next_to(gpu, DOWN, buff=0.22)
        tok_lab = caption("prompt + already generated")
        tok_lab.next_to(tokens, DOWN, buff=0.10)
        self.play(FadeIn(tokens), FadeIn(tok_lab), run_time=0.55)
        self.wait(0.5)
        self.next_slide()

        # -------------------------------------------------------------
        # Beat 14 — One prefill
        # -------------------------------------------------------------
        heading = self._retitle(heading, "One prefill rebuilds the whole cache")
        cap = self._recaption(cap, "A parallel pass — not one slow decode step at a time.")
        c_cells = VGroup(*[gpu.cells[i] for i in C_IDX])
        prefill = arrow(tokens.get_top(), c_cells.get_bottom(), color=ACCENT, buff=0.10)
        pre_lab = small("one prefill", font_size=SMALL_SIZE, color=ACCENT)
        pre_lab.next_to(tokens, RIGHT, buff=0.25)
        self.play(shoot(prefill), FadeIn(pre_lab), run_time=0.55)
        self.play(
            LaggedStart(
                *[gpu.cells[i].animate.set_fill(C_COLOR, 1) for i in C_IDX],
                lag_ratio=0.18,
            ),
            run_time=0.85,
        )
        self._set_free(free_lbl, "0 free", BAD)
        resumed = caption("resumed")
        resumed.move_to(chip_c2.tag.get_center())
        self.play(
            chip_c2.animate.set_opacity(1),
            Transform(chip_c2.tag, resumed),
            run_time=0.45,
        )
        cap = self._recaption(cap, "The cache is back, in one pass.")
        self.wait(0.45)
        self.next_slide()

        # -------------------------------------------------------------
        # Beat 15 — Fig 4 zoom-out + land
        # -------------------------------------------------------------
        heading = self._retitle(
            heading, "Fig. 4  ·  the pieces we just watched",
            FadeOut(tokens), FadeOut(tok_lab), FadeOut(prefill), FadeOut(pre_lab),
            FadeOut(gpu), FadeOut(free_lbl), FadeOut(chip_a), FadeOut(chip_b), FadeOut(chip_c2),
            FadeOut(sched),
        )
        cap = self._recaption(
            cap,
            "Contiguous slabs had no way to take memory back. Paging is the knob.",
        )

        fig4 = self._fig4()
        fig4["all"].next_to(heading, DOWN, buff=0.36)
        if fig4["all"].get_bottom()[1] < -3.08:
            fig4["all"].shift(UP * (-3.08 - fig4["all"].get_bottom()[1]))
        fig4["all"].set_x(0)

        self.play(FadeIn(fig4["scheduler"]), run_time=0.5)
        self.play(FadeIn(fig4["kv"]), shoot(fig4["sk"]), run_time=0.65)
        self.play(FadeIn(fig4["workers"]), shoot(fig4["sw"]), run_time=0.7)
        self.wait(0.4)
        self.next_slide()

    # -----------------------------------------------------------------
    # helpers
    # -----------------------------------------------------------------

    def _dump(self, mob):
        if mob is None:
            return
        mob.set_opacity(0)
        mob.shift(LEFT * 40)

    def _retitle(self, old, new_str, *anims, run_time=0.5):
        new = _fit_title(new_str)
        extra = list(anims)
        if old is not None:
            extra = [FadeOut(old, shift=UP * 0.16)] + extra
            self.play(FadeIn(new, shift=DOWN * 0.08), *extra, run_time=run_time)
            self._dump(old)
        else:
            self.play(FadeIn(new), *extra, run_time=run_time)
        return new

    def _recaption(self, old, new_str):
        new = caption(new_str)
        new.to_edge(DOWN, buff=0.30)
        if new.width > FRAME_W:
            new.scale_to_fit_width(FRAME_W)
            new.to_edge(DOWN, buff=0.30)
        if old is not None:
            self.play(FadeOut(old), FadeIn(new), run_time=0.35)
            self._dump(old)
        else:
            self.play(FadeIn(new), run_time=0.35)
        return new

    def _set_free(self, free_lbl, s, color):
        nxt = small(s, font_size=SMALL_SIZE, color=color)
        nxt.move_to(free_lbl.get_center())
        self.play(Transform(free_lbl, nxt), run_time=0.35)

    def _fig4(self):
        panel_w, panel_h = 5.45, 2.70
        mem_frame = RoundedRectangle(
            corner_radius=0.14, width=panel_w, height=panel_h,
            fill_color=BLOCK_FILL, fill_opacity=1.0,
            stroke_color=BLOCK_STROKE, stroke_width=2,
        )
        wrk_frame = RoundedRectangle(
            corner_radius=0.14, width=panel_w, height=panel_h,
            fill_color=BLOCK_FILL, fill_opacity=1.0,
            stroke_color=ACCENT2, stroke_width=2,
        )
        panels = VGroup(mem_frame, wrk_frame)
        panels.arrange(RIGHT, buff=0.50)

        kv_title = small("KV Cache Manager")
        kv_role = caption("owns every block table")
        cpu_al = _role_box(
            "CPU allocator", "swap",
            width=2.20, height=0.95, stroke=WARN,
        )
        gpu_al = _role_box(
            "GPU allocator", "leftover VRAM",
            width=2.20, height=0.95, stroke=ACCENT,
        )
        allocs = VGroup(cpu_al, gpu_al).arrange(RIGHT, buff=0.28)
        mem_content = VGroup(kv_title, kv_role, allocs).arrange(DOWN, buff=0.14)
        mem_content.move_to(mem_frame.get_center())

        wrk_title = small("Workers", color=ACCENT2)
        wrk_role = caption("one GPU each  ·  run the step")
        wrow = VGroup(
            small("Worker 0", color=ACCENT2),
            small("Worker 1", color=ACCENT2),
            text("···", font_size=SMALL_SIZE, color=MUTED),
            small("Worker N-1", color=ACCENT2),
        ).arrange(RIGHT, buff=0.32)
        wrk_content = VGroup(wrk_title, wrk_role, wrow).arrange(DOWN, buff=0.18)
        wrk_content.move_to(wrk_frame.get_center())

        mem = VGroup(mem_frame, mem_content)
        workers = VGroup(wrk_frame, wrk_content)

        sched = _role_box(
            "Scheduler", "who runs  ·  who is preempted",
            width=4.6, height=1.05, fill=ACCENT, stroke=ACCENT,
            text_color=BG, role_color=BG,
        )
        sched.next_to(panels, UP, buff=0.42)
        sched.set_x(panels.get_x())

        sk = _edge_arrow(sched.body, mem_frame)
        sw = _edge_arrow(sched.body, wrk_frame)

        return {
            "scheduler": sched,
            "kv": mem,
            "sk": sk,
            "workers": workers,
            "sw": sw,
            "all": VGroup(sched, mem, workers, sk, sw),
        }
