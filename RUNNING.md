# Running the PagedAttention deck

Run commands from the repository root.

## Prerequisites

- Python 3
- `ffmpeg` on `PATH` (`brew install ffmpeg` on macOS)

Create the virtual environment and install pinned Python packages:

```sh
make setup
```

## Fast development loop

Render every scene at draft quality:

```sh
make render-low
```

Render one scene while editing:

```sh
make render-scene \
  FILE=talk/s5_pagedattention.py \
  SCENE=S5PagedAttention \
  QUALITY=l
```

Use `QUALITY=h` for the final 1080p render.

Check Python syntax and regenerate the speaker script:

```sh
make verify
```

Extract the resting frame of every rendered slide to `/tmp/qa/<SceneClass>/`:

```sh
make qa
```

Inspect these PNGs after visual changes. They are the frames that remain visible while
the presenter speaks.

## Present live

Render first, then open the player:

```sh
make render
make present
```

`make present` fits the Qt window to the current screen. High-quality
renders are 1920×1080, which manim-slides otherwise opens 1:1 (too large
for a Mac laptop, and not resizable). Full screen:

```sh
make present PRESENT_ARGS=-F
```

Controls:

- Right / Left: next / previous beat
- Space: play or pause
- R: replay current beat
- F: toggle full screen
- Q: quit

## Build distributable output

Build the final videos, `NARRATION.md`, and reveal.js HTML:

```sh
make all
```

The browser backup is `dist/pagedattention_talk.html`. Keep its adjacent
`pagedattention_talk_assets/` directory with it.

List all targets with `make help`. Remove generated files with `make clean`.
