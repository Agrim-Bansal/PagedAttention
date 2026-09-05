"""Shared visual theme for the PagedAttention talk deck.

Constants + small text helpers + the act_checkpoint "where we are" beat.
No LaTeX: everything here uses `Text`.
"""

from manim import (
    BLUE,
    DOWN,
    LEFT,
    ORIGIN,
    RIGHT,
    UP,
    Circle,
    FadeIn,
    FadeOut,
    ManimColor,
    RoundedRectangle,
    Text,
    VGroup,
)

__all__ = [
    "BG",
    "FG",
    "MUTED",
    "ACCENT",
    "ACCENT2",
    "GOOD",
    "BAD",
    "WARN",
    "BLOCK_FILL",
    "BLOCK_STROKE",
    "K_COLOR",
    "V_COLOR",
    "Q_COLOR",
    "FONT",
    "TITLE_SIZE",
    "BODY_SIZE",
    "SMALL_SIZE",
    "TINY_SIZE",
    "apply_theme",
    "title",
    "body",
    "small",
    "caption",
    "act_checkpoint",
]

BG = "#0f1117"  # dark background
FG = "#e6e6e6"  # default text
MUTED = "#8a8f98"  # secondary text
ACCENT = "#ff8c42"  # the single accent (orange)
ACCENT2 = "#4cc9f0"  # cool secondary (used sparingly for K vs V, or "vLLM" series)
GOOD = "#5ad175"  # used / good
BAD = "#ef476f"  # wasted / bad
WARN = "#ffd166"  # reserved / warning
BLOCK_FILL = "#1f2633"  # empty block slot fill
BLOCK_STROKE = "#3b4454"
K_COLOR = ACCENT2
V_COLOR = "#b388ff"
Q_COLOR = ACCENT
FONT = "Avenir Next"  # falls back to system sans if missing

TITLE_SIZE = 44
BODY_SIZE = 30
SMALL_SIZE = 22
TINY_SIZE = 16

_FALLBACK_FONTS = ["Avenir Next", "Avenir", "Helvetica Neue", "Helvetica", "Arial", "Sans"]


def _resolve_font():
    try:
        import manimpango

        available = set(manimpango.list_fonts())
    except Exception:
        return FONT
    for f in _FALLBACK_FONTS:
        if f in available:
            return f
    return FONT


_RESOLVED_FONT = _resolve_font()


def apply_theme(scene):
    """Set the scene background and the default Text font/color."""
    scene.camera.background_color = ManimColor(BG)
    Text.set_default(font=_RESOLVED_FONT, color=FG)


def title(text, **kw):
    """Top-of-frame title."""
    kw.setdefault("font_size", TITLE_SIZE)
    kw.setdefault("color", FG)
    t = Text(text, **kw)
    t.to_edge(UP)
    return t


def body(text, **kw):
    """Body text, default size/color."""
    kw.setdefault("font_size", BODY_SIZE)
    kw.setdefault("color", FG)
    return Text(text, **kw)


def small(text, **kw):
    kw.setdefault("font_size", SMALL_SIZE)
    kw.setdefault("color", FG)
    return Text(text, **kw)


def caption(text, **kw):
    kw.setdefault("font_size", TINY_SIZE)
    kw.setdefault("color", MUTED)
    return Text(text, **kw)


def act_checkpoint(scene, act_no, act_title, done=(), current="", upcoming=()):
    """Full-screen "where we are" beat.

    Shows Act I/II/III as three pills, highlights the current one, lists
    done/current/upcoming as small text. Plays in, calls scene.next_slide(),
    then fades out. Returns None.
    """
    act_names = ["Act I", "Act II", "Act III"]
    pills = VGroup()
    for i, name in enumerate(act_names):
        is_current = (i + 1) == act_no
        pill_color = ACCENT if is_current else MUTED
        label = Text(name, font_size=BODY_SIZE, color=pill_color, weight="BOLD" if is_current else "NORMAL")
        circle = RoundedRectangle(
            width=1.9, height=0.9, corner_radius=0.45,
            color=pill_color, fill_opacity=0.15 if is_current else 0.0,
        )
        circle.set_stroke(pill_color, width=3 if is_current else 1.5)
        pill = VGroup(circle, label)
        if label.width > circle.width * 0.82:
            label.scale_to_fit_width(circle.width * 0.82)
        label.move_to(circle.get_center())
        pills.add(pill)
    pills.arrange(RIGHT, buff=1.2)
    pills.move_to(ORIGIN + UP * 1.6)

    heading = Text(act_title, font_size=TITLE_SIZE, color=FG)
    heading.next_to(pills, UP, buff=0.6)

    lines = VGroup()
    if done:
        done_text = Text("Done: " + "; ".join(done), font_size=SMALL_SIZE, color=MUTED)
        lines.add(done_text)
    if current:
        current_text = Text("Now: " + current, font_size=BODY_SIZE, color=ACCENT)
        lines.add(current_text)
    if upcoming:
        upcoming_text = Text("Next: " + "; ".join(upcoming), font_size=SMALL_SIZE, color=MUTED)
        lines.add(upcoming_text)
    lines.arrange(DOWN, buff=0.35, aligned_edge=LEFT)
    lines.next_to(pills, DOWN, buff=0.9)

    group = VGroup(heading, pills, lines)
    group.move_to(ORIGIN)

    scene.play(FadeIn(group))
    scene.next_slide()
    scene.play(FadeOut(group))
    return None
