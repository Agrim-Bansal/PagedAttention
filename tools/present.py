#!/usr/bin/env python3
"""Present the deck in a player window that fits the current screen.

manim-slides 5.6.0 sizes the Qt window to the render resolution (1920x1080
for ``-qh``) and also calls ``setMinimumSize`` with that same size, so the
window cannot shrink. On a Mac laptop the 1080p window overflows the
display and the video never rescales.

This wrapper patches the player after construction: drop the minimum size,
scale the window down to ``availableGeometry`` while keeping 16:9, and leave
the 1080p videos as-is so QVideoWidget can downscale them.

All ``manim-slides present`` flags still work (``-F`` / ``--full-screen``,
``--start-at``, …).
"""

from __future__ import annotations

from qtpy.QtWidgets import QApplication, QSizePolicy

# Title bar + a little breathing room so the frame stays on-screen.
_CHROME_H = 40
_MARGIN = 16


def _fit_player_to_screen(player, *, full_screen: bool) -> None:
    player.image_label.setMinimumSize(0, 0)
    player.media_stack.setMinimumSize(0, 0)
    player.setMinimumSize(0, 0)

    player.video_widget.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
    player.image_label.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)

    if full_screen:
        return

    app = QApplication.instance()
    screen = player.screen() or (app.primaryScreen() if app else None)
    if screen is None:
        return

    avail = screen.availableGeometry()
    src_w, src_h = player.current_presentation_config.resolution
    max_w = max(320, avail.width() - 2 * _MARGIN)
    max_h = max(180, avail.height() - _CHROME_H - 2 * _MARGIN)
    scale = min(max_w / src_w, max_h / src_h, 1.0)
    width = int(src_w * scale)
    height = int(src_h * scale)

    player.resize(width, height)
    frame = player.frameGeometry()
    frame.moveCenter(avail.center())
    player.move(frame.topLeft())


def _install_fit_patch() -> None:
    from manim_slides.present.player import Player

    if getattr(Player, "_pagedattention_fit_patch", False):
        return

    original_init = Player.__init__

    def patched_init(self, *args, **kwargs):
        original_init(self, *args, **kwargs)
        _fit_player_to_screen(self, full_screen=kwargs.get("full_screen", False))

    Player.__init__ = patched_init
    Player._pagedattention_fit_patch = True


if __name__ == "__main__":
    _install_fit_patch()
    from manim_slides.present import present

    present()
