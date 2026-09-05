"""Reusable visual components for the PagedAttention talk deck.

Every component is a `VGroup` subclass (or a function returning one), built
from native mobjects only (no LaTeX). Sub-mobjects are exposed as attributes
so scene files can animate pieces individually.
"""

from manim import (
    DOWN,
    LEFT,
    ORIGIN,
    RIGHT,
    UP,
    OUT,
    Angle,
    AnimationGroup,
    Annulus,
    AnnularSector,
    Arrow,
    Circle,
    Create,
    Dot,
    FadeIn,
    FadeOut,
    GrowArrow,
    GrowFromEdge,
    Indicate,
    Line,
    Rectangle,
    RoundedRectangle,
    Square,
    Text,
    Transform,
    VGroup,
    Wait,
    Axes,
)
import math

from talk.theme import (
    ACCENT,
    ACCENT2,
    BAD,
    BG,
    BLOCK_FILL,
    BLOCK_STROKE,
    BODY_SIZE,
    FG,
    GOOD,
    K_COLOR,
    MUTED,
    Q_COLOR,
    SMALL_SIZE,
    TINY_SIZE,
    V_COLOR,
    WARN,
)

__all__ = [
    "TokenBox",
    "token_sequence",
    "FOUR_SCORE",
    "KVBlock",
    "PhysicalMemGrid",
    "BlockTable",
    "arrow_map",
    "RefCountBadge",
    "MemoryBar",
    "MemoryPie",
    "GPUSchematic",
    "attention_diagram",
    "bar_chart",
    "line_chart",
]

FOUR_SCORE = [
    "Four",
    "score",
    "and",
    "seven",
    "years",
    "ago",
    "our",
    "fathers",
    "brought",
    "forth",
]

STATE_COLORS = {
    "empty": BLOCK_FILL,
    "filled": ACCENT,
    "reserved": WARN,
    "internal": BAD,
    "external": BAD,
}
STATE_OPACITY = {
    "empty": 1.0,
    "filled": 1.0,
    "reserved": 1.0,
    "internal": 1.0,
    "external": 0.5,
}


# ---------------------------------------------------------------------------
# Tokens
# ---------------------------------------------------------------------------


class TokenBox(VGroup):
    """One token as a rounded box with the word inside.  .box, .label"""

    def __init__(self, word, color=FG, fill=BLOCK_FILL, width=None, height=0.6, font_size=SMALL_SIZE, **kw):
        super().__init__(**kw)
        self.label = Text(str(word), font_size=font_size, color=color)
        box_width = width if width is not None else max(self.label.width + 0.5, 1.0)
        self.box = RoundedRectangle(
            corner_radius=0.1,
            width=box_width,
            height=height,
            fill_color=fill,
            fill_opacity=1.0,
            stroke_color=BLOCK_STROKE,
            stroke_width=2,
        )
        self.label.move_to(self.box.get_center())
        self.add(self.box, self.label)


def token_sequence(words, gap=0.12, **kw):
    """A VGroup of TokenBoxes arranged left to right."""
    group = VGroup(*[TokenBox(w, **kw) for w in words])
    group.arrange(RIGHT, buff=gap)
    return group


# ---------------------------------------------------------------------------
# KV blocks / memory
# ---------------------------------------------------------------------------


class KVBlock(VGroup):
    """Fixed-size KV block ("page") of `slots` slots drawn as a row of cells."""

    def __init__(self, slots=4, cell=0.75, words=None, index=None, label_pos=DOWN, **kw):
        super().__init__(**kw)
        self.n_slots = slots
        self.cell = cell
        words = words or [""] * slots
        self.cells = VGroup()
        self.labels = VGroup()
        for i in range(slots):
            sq = Square(
                side_length=cell,
                fill_color=BLOCK_FILL,
                fill_opacity=1.0,
                stroke_color=BLOCK_STROKE,
                stroke_width=2,
            )
            lbl = Text(str(words[i]) if i < len(words) else "", font_size=TINY_SIZE, color=FG)
            if lbl.width > 0.9 * cell:
                lbl.scale_to_fit_width(0.9 * cell)
            self.cells.add(sq)
            self.labels.add(lbl)
        self.cells.arrange(RIGHT, buff=0.0)
        for sq, lbl in zip(self.cells, self.labels):
            lbl.move_to(sq.get_center())
        self.index_label = Text("" if index is None else f"Block {index}", font_size=TINY_SIZE, color=MUTED)
        self.index_label.next_to(self.cells, label_pos, buff=0.15)
        self.add(self.cells, self.labels, self.index_label)

    def fill_slot(self, i, word, color=ACCENT):
        """Fill slot i with `word`, colored `color`. Returns an Animation."""
        new_label = Text(str(word), font_size=TINY_SIZE, color=FG)
        if new_label.width > 0.9 * self.cell:
            new_label.scale_to_fit_width(0.9 * self.cell)
        new_label.move_to(self.cells[i].get_center())
        old_label = self.labels[i]
        self.labels.remove(old_label)
        self.labels.add(new_label)
        self.add(self.labels)
        return AnimationGroup(
            self.cells[i].animate.set_fill(color, opacity=1.0),
            Transform(old_label, new_label),
        )

    def clear_slot(self, i):
        """Clear slot i back to empty. Returns an Animation."""
        new_label = Text("", font_size=TINY_SIZE, color=FG)
        new_label.move_to(self.cells[i].get_center())
        old_label = self.labels[i]
        self.labels.remove(old_label)
        self.labels.add(new_label)
        self.add(self.labels)
        return AnimationGroup(
            self.cells[i].animate.set_fill(BLOCK_FILL, opacity=1.0),
            Transform(old_label, new_label),
        )

    def set_state(self, i, state):
        """Set slot i's visual state. state in
        {"empty","filled","reserved","internal","external"}. Returns an Animation."""
        color = STATE_COLORS[state]
        opacity = STATE_OPACITY[state]
        return self.cells[i].animate.set_fill(color, opacity=opacity)


class PhysicalMemGrid(VGroup):
    """A column/grid of KVBlocks representing physical GPU memory."""

    def __init__(self, n_blocks=8, slots=4, cols=1, cell=0.6, title="Physical KV blocks", **kw):
        super().__init__(**kw)
        self.blocks = [KVBlock(slots=slots, cell=cell, index=i, label_pos=DOWN) for i in range(n_blocks)]
        grid = VGroup(*self.blocks)
        rows = math.ceil(n_blocks / cols)
        grid.arrange_in_grid(rows=rows, cols=cols, buff=0.35)
        self.title = Text(title, font_size=SMALL_SIZE, color=FG)
        self.title.next_to(grid, UP, buff=0.3)
        self.add(self.title, grid)
        self._grid = grid

    def block(self, i):
        return self.blocks[i]


class BlockTable(VGroup):
    """Logical->physical table. Rows of (logical index, physical block number, #filled)."""

    def __init__(self, n_rows=4, title="Block table", show_filled=True, **kw):
        super().__init__(**kw)
        self.show_filled = show_filled
        self.title = Text(title, font_size=SMALL_SIZE, color=FG)
        self.rows = []
        self._row_groups = VGroup()
        header_cells = ["logical", "physical"] + (["filled"] if show_filled else [])
        header = VGroup(*[Text(h, font_size=TINY_SIZE, color=MUTED) for h in header_cells])
        header.arrange(RIGHT, buff=0.6)
        self._header = header
        self._body = VGroup()
        self.add(self.title, header, self._body)
        for i in range(n_rows):
            self.add_row_sync(i, None, 0 if show_filled else None)
        self._layout()

    def _make_row(self, logical, physical, filled):
        cells = [Text(str(logical), font_size=TINY_SIZE, color=FG)]
        cells.append(Text("-" if physical is None else str(physical), font_size=TINY_SIZE, color=FG))
        if self.show_filled:
            cells.append(Text("" if filled is None else str(filled), font_size=TINY_SIZE, color=FG))
        row = VGroup(*cells)
        row.arrange(RIGHT, buff=0.6)
        return row

    def _layout(self):
        self.title.next_to(self._header, UP, buff=0.25)
        self._header.next_to(self.title, DOWN, buff=0.2)
        self._body.arrange(DOWN, buff=0.18, aligned_edge=LEFT)
        self._body.next_to(self._header, DOWN, buff=0.2)
        for row in self._body:
            for cell, hcell in zip(row, self._header):
                cell.align_to(hcell, LEFT)

    def add_row_sync(self, logical, physical, filled):
        row = self._make_row(logical, physical, filled)
        self.rows.append({"logical": logical, "physical": physical, "filled": filled, "mobject": row})
        self._body.add(row)
        return row

    def set_row(self, i, physical, filled):
        """Update row i's physical block and fill count. Returns an Animation."""
        entry = self.rows[i]
        entry["physical"] = physical
        entry["filled"] = filled
        new_row = self._make_row(entry["logical"], physical, filled)
        new_row.move_to(entry["mobject"].get_center())
        for cell, hcell in zip(new_row, self._header):
            cell.align_to(hcell, LEFT)
        old = entry["mobject"]
        anim = Transform(old, new_row)
        return anim

    def add_row(self, physical, filled):
        """Append a new row. Returns an Animation."""
        logical = len(self.rows)
        row = self.add_row_sync(logical, physical, filled)
        self._layout()
        row.set_opacity(0)
        target = row.copy().set_opacity(1)
        return Transform(row, target)

    def highlight_row(self, i):
        """Flash row i. Returns an Animation."""
        return Indicate(self.rows[i]["mobject"], color=ACCENT)


def arrow_map(src, dst, color=MUTED, **kw):
    """A thin arrow between the centers of two mobjects."""
    kw.setdefault("buff", 0.1)
    kw.setdefault("stroke_width", 2.5)
    kw.setdefault("max_tip_length_to_length_ratio", 0.15)
    return Arrow(src.get_center(), dst.get_center(), color=color, **kw)


class RefCountBadge(VGroup):
    """Small circle with a number, attaches to a KVBlock corner."""

    def __init__(self, value=1, color=ACCENT2, **kw):
        super().__init__(**kw)
        self.value = value
        self.circle = Circle(radius=0.22, color=color, fill_color=BG, fill_opacity=1.0, stroke_width=2.5)
        self.label = Text(str(value), font_size=TINY_SIZE, color=color)
        self.label.move_to(self.circle.get_center())
        self.add(self.circle, self.label)

    def set_value(self, n):
        """Update the badge's displayed number. Returns an Animation."""
        self.value = n
        new_label = Text(str(n), font_size=TINY_SIZE, color=self.label.color)
        new_label.move_to(self.circle.get_center())
        return Transform(self.label, new_label)


class MemoryBar(VGroup):
    """Horizontal stacked bar. segments = [(label, fraction, color), ...]."""

    def __init__(self, segments, width=10.0, height=0.8, show_pct=True, **kw):
        super().__init__(**kw)
        self.segments = segments
        self.width_total = width
        self.outline = Rectangle(width=width, height=height, stroke_color=BLOCK_STROKE, stroke_width=2)
        self.segs = VGroup()
        self.labels = VGroup()
        x = -width / 2
        for label, frac, color in segments:
            seg_w = width * frac
            rect = Rectangle(
                width=max(seg_w, 1e-6),
                height=height,
                fill_color=color,
                fill_opacity=1.0,
                stroke_width=0,
            )
            rect.move_to(self.outline.get_left() + RIGHT * (x + width / 2 + seg_w / 2))
            self.segs.add(rect)
            text = f"{label} ({frac * 100:.0f}%)" if show_pct else label
            txt = Text(text, font_size=TINY_SIZE, color=FG)
            if txt.width > seg_w * 0.9:
                txt.scale_to_fit_width(max(seg_w * 0.9, 0.1))
            txt.move_to(rect.get_center())
            self.labels.add(txt)
            x += seg_w
        self.add(self.outline, self.segs, self.labels)

    def animate_in(self):
        """Grow the segments in from the left edge. Returns an Animation."""
        return AnimationGroup(
            Create(self.outline),
            AnimationGroup(*[GrowFromEdge(seg, LEFT) for seg in self.segs], lag_ratio=0.15),
            AnimationGroup(*[FadeIn(lbl) for lbl in self.labels], lag_ratio=0.15),
            lag_ratio=0.3,
        )


class MemoryPie(VGroup):
    """Pie/donut with same segments spec."""

    def __init__(self, segments, radius=1.8, **kw):
        super().__init__(**kw)
        self.segments = segments
        self.sectors = VGroup()
        start_angle = math.pi / 2
        for label, frac, color in segments:
            angle = frac * 2 * math.pi
            sector = AnnularSector(
                inner_radius=radius * 0.5,
                outer_radius=radius,
                angle=angle,
                start_angle=start_angle,
                fill_color=color,
                fill_opacity=1.0,
                stroke_color=BG,
                stroke_width=2,
            )
            self.sectors.add(sector)
            start_angle += angle
        self.legend = VGroup()
        for (label, frac, color) in segments:
            swatch = Square(side_length=0.22, fill_color=color, fill_opacity=1.0, stroke_width=0)
            txt = Text(f"{label} ({frac * 100:.0f}%)", font_size=TINY_SIZE, color=FG)
            txt.next_to(swatch, RIGHT, buff=0.15)
            entry = VGroup(swatch, txt)
            self.legend.add(entry)
        self.legend.arrange(DOWN, buff=0.18, aligned_edge=LEFT)
        self.legend.next_to(self.sectors, RIGHT, buff=0.6)
        self.add(self.sectors, self.legend)


class GPUSchematic(VGroup):
    """Minimal GPU: a chip with a grid of many tiny 'core' squares, plus a VRAM bar beside it."""

    def __init__(self, cores=(8, 6), width=5.0, **kw):
        super().__init__(**kw)
        n_cols, n_rows = cores
        chip_width = width * 0.6
        chip = RoundedRectangle(
            corner_radius=0.1,
            width=chip_width,
            height=chip_width * (n_rows / n_cols),
            fill_color=BLOCK_FILL,
            fill_opacity=1.0,
            stroke_color=BLOCK_STROKE,
            stroke_width=2,
        )
        core_size = min(chip.width / n_cols, chip.height / n_rows) * 0.7
        self.cores = VGroup()
        for r in range(n_rows):
            for c in range(n_cols):
                core = Square(side_length=core_size, fill_color=ACCENT2, fill_opacity=0.85, stroke_width=0.5, stroke_color=BG)
                self.cores.add(core)
        self.cores.arrange_in_grid(rows=n_rows, cols=n_cols, buff=core_size * 0.25)
        self.cores.move_to(chip.get_center())
        self.chip = chip

        vram_height = chip.height
        vram_width = width * 0.18
        self.vram_outline = Rectangle(width=vram_width, height=vram_height, stroke_color=BLOCK_STROKE, stroke_width=2)
        self.vram_outline.next_to(chip, RIGHT, buff=width * 0.15)
        self.vram = Rectangle(width=vram_width, height=1e-6, fill_color=GOOD, fill_opacity=1.0, stroke_width=0)
        self.vram.move_to(self.vram_outline.get_bottom(), aligned_edge=DOWN)
        self.vram_label = Text("VRAM", font_size=TINY_SIZE, color=MUTED)
        self.vram_label.next_to(self.vram_outline, DOWN, buff=0.15)

        self._vram_height = vram_height
        self._vram_width = vram_width

        self.add(self.chip, self.cores, self.vram_outline, self.vram, self.vram_label)

    def fill_vram(self, fraction, color=GOOD):
        """Fill the VRAM bar to `fraction` (0..1) full, in `color`. Returns an Animation."""
        new_height = max(self._vram_height * fraction, 1e-6)
        new_rect = Rectangle(width=self._vram_width, height=new_height, fill_color=color, fill_opacity=1.0, stroke_width=0)
        new_rect.move_to(self.vram_outline.get_bottom(), aligned_edge=DOWN)
        return Transform(self.vram, new_rect)


# ---------------------------------------------------------------------------
# Attention diagram
# ---------------------------------------------------------------------------


def attention_diagram(tokens, query_index, kv_colors=True):
    """Q/K/V computation visual: row of TokenBoxes, K and V stacks under each,
    a Q above the query token, arrows from Q to each K, a softmax bar row,
    weighted sum arrow to output. Returns VGroup with .tokens, .keys, .values,
    .query, .arrows, .weights, .output."""
    group = VGroup()
    token_boxes = token_sequence(tokens, gap=0.3, height=0.55, font_size=SMALL_SIZE)
    token_boxes.move_to(ORIGIN)

    keys = VGroup()
    values = VGroup()
    for tb in token_boxes:
        k_color = K_COLOR if kv_colors else MUTED
        v_color = V_COLOR if kv_colors else MUTED
        k_sq = Square(side_length=0.3, fill_color=k_color, fill_opacity=0.9, stroke_width=0)
        v_sq = Square(side_length=0.3, fill_color=v_color, fill_opacity=0.9, stroke_width=0)
        k_sq.next_to(tb, DOWN, buff=0.35)
        v_sq.next_to(k_sq, DOWN, buff=0.12)
        keys.add(k_sq)
        values.add(v_sq)

    q_color = Q_COLOR
    query_box = token_boxes[query_index]
    query = Square(side_length=0.32, fill_color=q_color, fill_opacity=0.9, stroke_width=0)
    query.next_to(query_box, UP, buff=0.35)
    q_label = Text("Q", font_size=TINY_SIZE, color=BG)
    q_label.move_to(query.get_center())
    query_group = VGroup(query, q_label)

    arrows = VGroup()
    for i, k_sq in enumerate(keys):
        if i == query_index:
            continue
        arrow = Arrow(
            query.get_center(),
            k_sq.get_center(),
            buff=0.2,
            stroke_width=1.5,
            color=MUTED,
            max_tip_length_to_length_ratio=0.1,
        )
        arrows.add(arrow)

    n = len(tokens)
    weight_vals = [1.0 / n] * n
    weights = VGroup()
    w_width = token_boxes.width / n * 0.8
    for i, val in enumerate(weight_vals):
        bar = Rectangle(
            width=w_width,
            height=max(val * 1.2, 0.05),
            fill_color=ACCENT,
            fill_opacity=0.8,
            stroke_width=0,
        )
        bar.next_to(values[i], DOWN, buff=0.3, aligned_edge=DOWN)
        weights.add(bar)

    output_box = TokenBox("output", color=BG, fill=GOOD, height=0.55, font_size=SMALL_SIZE)
    output_box.next_to(weights, DOWN, buff=0.5)

    group.tokens = token_boxes
    group.keys = keys
    group.values = values
    group.query = query_group
    group.arrows = arrows
    group.weights = weights
    group.output = output_box

    group.add(token_boxes, keys, values, query_group, arrows, weights, output_box)
    return group


# ---------------------------------------------------------------------------
# Charts
# ---------------------------------------------------------------------------


class _BarChart(VGroup):
    def __init__(self):
        super().__init__()
        self.axes = None
        self.bars = {}
        self.legend = None
        self.x_labels = None
        self.y_label = None

    def animate_in(self):
        static = [self.axes, self.x_labels, self.legend]
        if self.y_label is not None:
            static.append(self.y_label)
        anims = [FadeIn(*static)]
        for name, group in self.bars.items():
            anims.extend([GrowFromEdge(b, DOWN) for b in group])
        return AnimationGroup(*anims, lag_ratio=0.05)


def bar_chart(categories, series, y_label="", y_max=None, colors=None, width=8, height=4, value_labels=False):
    """Grouped bar chart. series = {"name": [values...]}. Native Rectangles on an Axes."""
    chart = _BarChart()
    names = list(series.keys())
    if colors is None:
        default_colors = [ACCENT, ACCENT2, V_COLOR, GOOD, WARN, MUTED]
        colors = {name: default_colors[i % len(default_colors)] for i, name in enumerate(names)}
    all_vals = [v for vals in series.values() for v in vals]
    computed_max = max(all_vals) if all_vals else 1.0
    y_max = y_max if y_max is not None else computed_max * 1.2

    axes = Axes(
        x_range=[0, len(categories), 1],
        y_range=[0, y_max, y_max / 4 if y_max else 1],
        x_length=width,
        y_length=height,
        axis_config={"include_ticks": False, "include_numbers": False, "color": MUTED},
        tips=False,
    )
    chart.axes = axes
    chart.add(axes)

    n_series = len(names)
    n_cats = len(categories)
    group_width = width / n_cats
    bar_width = group_width / (n_series + 1)

    x_labels = VGroup()
    for i, cat in enumerate(categories):
        lbl = Text(str(cat), font_size=TINY_SIZE, color=MUTED)
        cat_center = axes.c2p(i + 0.5, 0)
        lbl.next_to(cat_center, DOWN, buff=0.2)
        x_labels.add(lbl)
    chart.x_labels = x_labels
    chart.add(x_labels)

    for s_idx, name in enumerate(names):
        color = colors[name]
        bar_group = VGroup()
        for c_idx, val in enumerate(series[name]):
            x0 = c_idx + 0.5 - (n_series * bar_width) / (2 * (width / n_cats)) + s_idx * bar_width / (width / n_cats)
            base = axes.c2p(c_idx + (s_idx + 0.5) / (n_series + 1) + 0.5 / (n_series + 1) - 0.5, 0)
            top = axes.c2p(c_idx + (s_idx + 0.5) / (n_series + 1) + 0.5 / (n_series + 1) - 0.5, val)
            bar_h = abs(top[1] - base[1])
            rect = Rectangle(
                width=bar_width * 0.9,
                height=max(bar_h, 1e-6),
                fill_color=color,
                fill_opacity=1.0,
                stroke_width=0,
            )
            rect.move_to(base, aligned_edge=DOWN)
            bar_group.add(rect)
            if value_labels:
                vlabel = Text(f"{val:g}", font_size=TINY_SIZE, color=FG)
                vlabel.next_to(rect, UP, buff=0.08)
                bar_group.add(vlabel)
        chart.bars[name] = bar_group
        chart.add(bar_group)

    legend = VGroup()
    for name in names:
        swatch = Square(side_length=0.2, fill_color=colors[name], fill_opacity=1.0, stroke_width=0)
        txt = Text(name, font_size=TINY_SIZE, color=FG)
        txt.next_to(swatch, RIGHT, buff=0.12)
        legend.add(VGroup(swatch, txt))
    legend.arrange(RIGHT, buff=0.4)
    legend.next_to(axes, UP, buff=0.4)
    chart.legend = legend
    chart.add(legend)

    if y_label:
        ylab = Text(y_label, font_size=TINY_SIZE, color=MUTED)
        ylab.rotate(math.pi / 2)
        ylab.next_to(axes, LEFT, buff=0.8)
        chart.y_label = ylab
        chart.add(ylab)

    return chart


class _LineChart(VGroup):
    def __init__(self):
        super().__init__()
        self.axes = None
        self.lines = {}
        self.dots = {}
        self.legend = None
        self.x_labels = None
        self.y_ticks = None
        self.x_label = None
        self.y_label = None

    def animate_in(self):
        static = [self.axes, self.x_labels, self.y_ticks, self.legend]
        if self.x_label is not None:
            static.append(self.x_label)
        if self.y_label is not None:
            static.append(self.y_label)
        anims = [FadeIn(*static)]
        for name, line in self.lines.items():
            anims.append(Create(line))
        for name, dots in self.dots.items():
            anims.append(AnimationGroup(*[FadeIn(d) for d in dots], lag_ratio=0.1))
        return AnimationGroup(*anims, lag_ratio=0.2)


def line_chart(x, series, x_label="", y_label="", x_range=None, y_range=None, colors=None,
               width=8, height=4, log_y=False, markers=True):
    """Multi-series line chart. .axes, .lines (dict), .dots (dict), .legend."""
    chart = _LineChart()
    names = list(series.keys())
    if colors is None:
        default_colors = [ACCENT, ACCENT2, V_COLOR, GOOD, WARN, MUTED]
        colors = {name: default_colors[i % len(default_colors)] for i, name in enumerate(names)}

    def transform(v):
        if log_y:
            return math.log10(v) if v > 0 else 0
        return v

    all_vals = [v for vals in series.values() for v in vals]
    t_vals = [transform(v) for v in all_vals]
    if y_range is None:
        y_lo, y_hi = min(t_vals), max(t_vals)
        pad = (y_hi - y_lo) * 0.1 or 1
        y_range = [y_lo - pad, y_hi + pad, (y_hi - y_lo + 2 * pad) / 4 or 1]
    if x_range is None:
        x_range = [min(x), max(x), (max(x) - min(x)) / max(len(x) - 1, 1) or 1]

    axes = Axes(
        x_range=x_range,
        y_range=y_range,
        x_length=width,
        y_length=height,
        axis_config={"include_ticks": False, "include_numbers": False, "color": MUTED},
        tips=False,
    )
    chart.axes = axes
    chart.add(axes)

    x_labels = VGroup()
    for xi in x:
        lbl = Text(f"{xi:g}", font_size=TINY_SIZE, color=MUTED)
        lbl.next_to(axes.c2p(xi, y_range[0]), DOWN, buff=0.15)
        x_labels.add(lbl)
    chart.add(x_labels)
    chart.x_labels = x_labels

    y_ticks = VGroup()
    n_yticks = 5
    for i in range(n_yticks):
        ty = y_range[0] + i * (y_range[1] - y_range[0]) / (n_yticks - 1)
        val = 10 ** ty if log_y else ty
        lbl = Text(f"{val:.3g}", font_size=TINY_SIZE, color=MUTED)
        lbl.next_to(axes.c2p(x_range[0], ty), LEFT, buff=0.15)
        y_ticks.add(lbl)
    chart.add(y_ticks)
    chart.y_ticks = y_ticks

    for name in names:
        color = colors[name]
        pts = [axes.c2p(xi, transform(v)) for xi, v in zip(x, series[name])]
        line = VGroup(*[Line(pts[i], pts[i + 1], color=color, stroke_width=3) for i in range(len(pts) - 1)])
        chart.lines[name] = line
        chart.add(line)
        dots = VGroup()
        if markers:
            for p in pts:
                dots.add(Dot(p, color=color, radius=0.06))
        chart.dots[name] = dots
        chart.add(dots)

    legend = VGroup()
    for name in names:
        swatch = Line(ORIGIN, RIGHT * 0.3, color=colors[name], stroke_width=3)
        txt = Text(name, font_size=TINY_SIZE, color=FG)
        txt.next_to(swatch, RIGHT, buff=0.12)
        legend.add(VGroup(swatch, txt))
    legend.arrange(RIGHT, buff=0.4)
    legend.next_to(axes, UP, buff=0.4)
    chart.legend = legend
    chart.add(legend)

    if x_label:
        xlab = Text(x_label, font_size=TINY_SIZE, color=MUTED)
        xlab.next_to(axes, DOWN, buff=0.6)
        chart.x_label = xlab
        chart.add(xlab)
    if y_label:
        ylab = Text(y_label, font_size=TINY_SIZE, color=MUTED)
        ylab.rotate(math.pi / 2)
        ylab.next_to(axes, LEFT, buff=0.7)
        chart.y_label = ylab
        chart.add(ylab)

    return chart
