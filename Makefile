VENV := .venv
MANIM_SLIDES := $(VENV)/bin/manim-slides
PYTHON := $(VENV)/bin/python

# Scenes in talk-order (S0 -> S11): file, class.
SCENE_FILES := \
	talk/s0_title.py \
	talk/s1_gpu.py \
	talk/s2_transformers.py \
	talk/s3_kvcache.py \
	talk/s4_problem.py \
	talk/s5_pagedattention.py \
	talk/s6_os_and_why_hard.py \
	talk/s7_sharing.py \
	talk/s8_scheduling.py \
	talk/s9_results.py \
	talk/s10_ablations.py \
	talk/s11_takeaways.py

SCENE_CLASSES := \
	S0Title \
	S1GPU \
	S2Transformers \
	S3KVCache \
	S4Problem \
	S5PagedAttention \
	S6OSAndWhyHard \
	S7Sharing \
	S8Scheduling \
	S9Results \
	S10Ablations \
	S11Takeaways

# Paired (file class) list, in order, for the render targets.
SCENE_PAIRS := \
	talk/s0_title.py:S0Title \
	talk/s1_gpu.py:S1GPU \
	talk/s2_transformers.py:S2Transformers \
	talk/s3_kvcache.py:S3KVCache \
	talk/s4_problem.py:S4Problem \
	talk/s5_pagedattention.py:S5PagedAttention \
	talk/s6_os_and_why_hard.py:S6OSAndWhyHard \
	talk/s7_sharing.py:S7Sharing \
	talk/s8_scheduling.py:S8Scheduling \
	talk/s9_results.py:S9Results \
	talk/s10_ablations.py:S10Ablations \
	talk/s11_takeaways.py:S11Takeaways

DIST_DIR := dist
HTML_OUT := $(DIST_DIR)/pagedattention_talk.html
SCENE ?= S0Title
FILE ?= talk/s0_title.py
QUALITY ?= l

.PHONY: all help check-env check-present-env verify render render-low render-scene qa present html narration clean setup

all: render narration html

help:
	@echo "make setup         Create .venv and install dependencies"
	@echo "make render-low    Render all scenes quickly (480p)"
	@echo "make render        Render all scenes at final quality (1080p)"
	@echo "make render-scene FILE=talk/s5_pagedattention.py SCENE=S5PagedAttention QUALITY=l"
	@echo "make verify        Compile sources and regenerate narration"
	@echo "make qa            Extract every rendered slide's resting frame to /tmp/qa"
	@echo "make present       Open the keyboard-driven slide player"
	@echo "make html          Export the standalone reveal.js backup"
	@echo "make all           Final render, narration, and HTML export"
	@echo "make clean         Remove generated media, slides, HTML, and caches"

check-env:
	@test -x "$(MANIM_SLIDES)" || (echo "Missing $(MANIM_SLIDES). Run: make setup" && exit 1)
	@command -v ffmpeg >/dev/null || (echo "Missing ffmpeg. macOS: brew install ffmpeg" && exit 1)

check-present-env: check-env
	@$(PYTHON) -c "import PySide6" 2>/dev/null || \
		(echo "Missing Qt player dependency. Run: $(VENV)/bin/pip install -r requirements.txt" && exit 1)

verify: check-env
	$(PYTHON) -m compileall -q talk tools
	$(PYTHON) tools/build_narration.py

render: check-env
	@for pair in $(SCENE_PAIRS); do \
		file=$${pair%%:*}; class=$${pair##*:}; \
		echo "== rendering $$class ($$file) at -qh =="; \
		rm -rf slides/files/$$class slides/$$class.json; \
		$(MANIM_SLIDES) render --disable_caching -q h $$file $$class || exit 1; \
	done

render-low: check-env
	@for pair in $(SCENE_PAIRS); do \
		file=$${pair%%:*}; class=$${pair##*:}; \
		echo "== rendering $$class ($$file) at -ql =="; \
		rm -rf slides/files/$$class slides/$$class.json; \
		$(MANIM_SLIDES) render --disable_caching -q l $$file $$class || exit 1; \
	done

render-scene: check-env
	rm -rf slides/files/$(SCENE) slides/$(SCENE).json
	$(MANIM_SLIDES) render --disable_caching -q $(QUALITY) $(FILE) $(SCENE)

qa: check-env
	@for class in $(SCENE_CLASSES); do \
		test -f slides/$$class.json || (echo "Missing slides/$$class.json; render first" && exit 1); \
		$(PYTHON) tools/last_frames.py $$class || exit 1; \
	done

present: check-present-env
	$(MANIM_SLIDES) present $(SCENE_CLASSES)

html: check-env
	@mkdir -p $(DIST_DIR)
	$(MANIM_SLIDES) convert $(SCENE_CLASSES) $(HTML_OUT) \
		--to html \
		-cslide_number=true \
		-ccontrols=true \
		-cprogress=true \
		-ctransition=none \
		-cwidth=1920 \
		-cheight=1080

narration: check-env
	$(PYTHON) tools/build_narration.py

clean:
	rm -rf media slides dist __pycache__ talk/__pycache__

setup:
	python3 -m venv $(VENV)
	$(VENV)/bin/pip install --upgrade pip
	$(VENV)/bin/pip install -r requirements.txt
