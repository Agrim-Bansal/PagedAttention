# YouTube listing

Paste into YouTube Studio after `make video`. This file is listing copy only; the upload file is `dist/pagedattention.mp4`.

The video is the **rendered animation** (no live narration, no pause holds). The click-through talk is the deployed HTML.

---

## Title

PagedAttention : How vLLM Pages the KV Cache (SOSP '23, Animated)

---

## Description

The bottleneck in LLM serving is leftover GPU memory, not compute. This is a full walkthrough of PagedAttention — the idea behind vLLM — as a native animation, not screenshots.

This upload is the concatenated render of the slide deck used to present the vLLM paper to DevClub, IIT Delhi.

Here, animations play back-to-back, with no voiceover and none of the last-frame holds from the live talk. For the paced version, use the interactive link below.

Watch the interactive talk (pause and step through each beat):
<https://projects.agrimbansal.com/PagedAttention/>

Source, scenes, and render tooling:
<https://github.com/Agrim-Bansal/PagedAttention>

Paper (this video is an unofficial educational animation; not affiliated with the authors or the vLLM project):

Woosuk Kwon, Zhuohan Li, Siyuan Zhuang, Ying Sheng, Lianmin Zheng, Cody Hao Yu, Joseph E. Gonzalez, Hao Zhang, and Ion Stoica. “Efficient Memory Management for Large Language Model Serving with PagedAttention.” SOSP 2023.
<https://dl.acm.org/doi/10.1145/3600006.3613165>
<https://arxiv.org/abs/2309.06180>

vLLM: <https://github.com/vllm-project/vllm>

Animation made with Manim Community Edition (<https://www.manim.community/>) and manim-slides (<https://github.com/jeertmans/manim-slides>).
Built with AI assistance (Claude Code and Cursor - Claude Fable 5.1, Claude Sonnet 5, Grok 4.6).

Overview

Existing systems store each request’s KV cache as one contiguous slab reserved to the model’s maximum length, so only 20–38% of that memory holds real tokens. vLLM chops the cache into fixed-size blocks, maps them through a block table, and allocates on demand. Waste falls below one block per request; utilization hits 96.3%; throughput is 2–4× Orca at the same latency (up to 22× FasterTransformer).

Twelve scenes, three acts, every figure from the paper recreated as animation:

Act I — Why memory is the bottleneck
GPU leftover VRAM, transformers and attention, the KV cache, and why contiguous pre-allocation wastes memory.

Act II — Page the KV cache
PagedAttention changes the reader first, then the manager. Then the OS analogy — and why this is not a free port of virtual memory onto the GPU.

Act III — What it buys you
Copy-on-write sharing, scheduling and preemption, paper results, kernel/block-size ablations, takeaways.

# PagedAttention #vLLM #LLM #Manim #SOSP
