"""Shared visual theme for the PagedAttention talk deck.

Constants + small text helpers + the act_checkpoint "where we are" beat.
No LaTeX: everything here uses `Text`.

Pango (Manim `Text`) lays out glyph advances in integer pixels, so small
`font_size` values get uneven kerning
(https://github.com/ManimCommunity/manim/issues/2844). Importing this module
patches `Text` to render at a large size and scale down, so every `Text(...)`
in the deck — helpers, components, and raw scene calls — stays evenly spaced.
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
    MarkupText,
    RoundedRectangle,
    Text,
    VGroup,
)
from manim.constants import DEFAULT_FONT_SIZE

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
    "text",
    "title",
    "body",
    "small",
    "caption",
    "formula",
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

# Pango integer-advance artifacts are obvious below ~28px. Render at Manim's
# default (48) or the requested size, whichever is larger, then scale the
# mobject so on-screen size is unchanged.
_KERN_RENDER_SIZE = float(DEFAULT_FONT_SIZE)

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


def _install_text_kerning_patch():
    """Render Text at >= _KERN_RENDER_SIZE, then scale to the requested size.

    Must wrap the real ``Text.__init__`` and also replace ``_original__init__``
    so ``Text.set_default`` (used by ``apply_theme``) keeps the workaround.
    """
    if getattr(Text, "_kerning_patched", False):
        return

    _orig_init = Text.__init__

    def _kerning_init(self, *args, **kwargs):
        target = kwargs.get("font_size")
        if target is None:
            _orig_init(self, *args, **kwargs)
            return
        target = float(target)
        render = max(target, _KERN_RENDER_SIZE)
        kwargs["font_size"] = render
        _orig_init(self, *args, **kwargs)
        if render != target and self.height > 1e-8:
            self.scale(target / render)

    Text.__init__ = _kerning_init
    Text._original__init__ = _kerning_init
    Text._kerning_patched = True


def _install_markup_kerning_patch():
    """Same integer-advance workaround for MarkupText (formulas)."""
    if getattr(MarkupText, "_kerning_patched", False):
        return

    _orig_init = MarkupText.__init__

    def _kerning_init(self, *args, **kwargs):
        target = kwargs.get("font_size")
        if target is None:
            _orig_init(self, *args, **kwargs)
            return
        target = float(target)
        render = max(target, _KERN_RENDER_SIZE)
        kwargs["font_size"] = render
        _orig_init(self, *args, **kwargs)
        if render != target and self.height > 1e-8:
            self.scale(target / render)

    MarkupText.__init__ = _kerning_init
    MarkupText._original__init__ = _kerning_init
    MarkupText._kerning_patched = True


_install_text_kerning_patch()
_install_markup_kerning_patch()


def text(content, **kw):
    """Text with even letter spacing. Same kwargs as Manim ``Text``."""
    return Text(content, **kw)


def apply_theme(scene):
    """Set the scene background and the default Text font/color."""
    scene.camera.background_color = ManimColor(BG)
    Text.set_default(font=_RESOLVED_FONT, color=FG)


def title(content, **kw):
    """Top-of-frame title."""
    kw.setdefault("font_size", TITLE_SIZE)
    kw.setdefault("color", FG)
    t = text(content, **kw)
    t.to_edge(UP)
    return t


def body(content, **kw):
    """Body text, default size/color."""
    kw.setdefault("font_size", BODY_SIZE)
    kw.setdefault("color", FG)
    return text(content, **kw)


def small(content, **kw):
    kw.setdefault("font_size", SMALL_SIZE)
    kw.setdefault("color", FG)
    return text(content, **kw)


def caption(content, **kw):
    kw.setdefault("font_size", TINY_SIZE)
    kw.setdefault("color", MUTED)
    return text(content, **kw)


def formula(markup, font_size=BODY_SIZE, color=FG, **kw):
    """Pango markup for math (sub/sup). No LaTeX."""
    kw.setdefault("font", _RESOLVED_FONT)
    kw.setdefault("font_size", font_size)
    kw.setdefault("color", color)
    return MarkupText(markup, **kw)


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
        label = text(name, font_size=BODY_SIZE, color=pill_color, weight="BOLD" if is_current else "NORMAL")
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

    heading = text(act_title, font_size=TITLE_SIZE, color=FG)
    heading.next_to(pills, UP, buff=0.6)

    lines = VGroup()
    if done:
        done_text = text("Done: " + "; ".join(done), font_size=SMALL_SIZE, color=MUTED)
        lines.add(done_text)
    if current:
        current_text = text("Now: " + current, font_size=BODY_SIZE, color=ACCENT)
        lines.add(current_text)
    if upcoming:
        upcoming_text = text("Next: " + "; ".join(upcoming), font_size=SMALL_SIZE, color=MUTED)
        lines.add(upcoming_text)
    lines.arrange(DOWN, buff=0.35, aligned_edge=LEFT)
    lines.next_to(pills, DOWN, buff=0.9)

    group = VGroup(heading, pills, lines)
    group.move_to(ORIGIN)

    scene.play(FadeIn(group))
    scene.next_slide()
    scene.play(FadeOut(group))
    return None
