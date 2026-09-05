"""Extract the resting (last) frame of every slide of a scene to /tmp/qa/<Class>/slide_NN.png.
Usage: .venv/bin/python tools/last_frames.py S5PagedAttention [/tmp/qa]"""
import json, os, subprocess, sys
cls = sys.argv[1]
d = json.load(open(f"slides/{cls}.json"))
root = sys.argv[2] if len(sys.argv) > 2 else "/tmp/qa"
out = f"{root}/{cls}"; os.makedirs(out, exist_ok=True)
for i, s in enumerate(d["slides"]):
    png = f"{out}/slide_{i:02d}.png"
    subprocess.run(["ffmpeg", "-loglevel", "error", "-y", "-sseof", "-0.05", "-i", s["file"],
                    "-frames:v", "1", "-update", "1", png], check=False)
    if not os.path.exists(png):  # fallback for very short clips
        subprocess.run(["ffmpeg", "-loglevel", "error", "-y", "-i", s["file"], "-vf", "select=eq(n\\,0)",
                        "-frames:v", "1", "-update", "1", png], check=False)
    print(png)
