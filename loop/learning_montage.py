"""Render Reflex-1's learning history into one video: each saved attempt played in the
real game (headless), labeled with its iteration and result, concatenated in order.

    python3 loop/learning_montage.py                    # learn/history/*.json -> runs/videos/learning.mp4
    python3 loop/learning_montage.py --max-s 12 --pick 8

ponytail: plays each attempt from the start; long laps are cut at --max-s except the last
(full lap). Upgrade: picture-in-picture of the learning curve.
"""
import argparse
import json
import subprocess
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "runs/videos/learning"


def label(text: str, path: Path):
    im = Image.new("RGBA", (1280, 720), (0, 0, 0, 0))
    d = ImageDraw.Draw(im)
    f = ImageFont.truetype("/System/Library/Fonts/Supplemental/Arial Bold.ttf", 40)
    w = d.textlength(text, font=f)
    d.rectangle([20, 610, 44 + w, 670], fill=(0, 0, 0, 170))
    d.text((32, 616), text, font=f, fill=(255, 230, 0, 255))
    im.save(path)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--history", type=Path, default=ROOT / "learn/history")
    ap.add_argument("--pick", type=int, default=8, help="how many attempts (evenly spaced, always first and last)")
    ap.add_argument("--max-s", type=float, default=10)
    a = ap.parse_args()
    files = sorted(a.history.glob("*.json"))
    assert files, f"no history in {a.history}"
    idx = sorted({round(i * (len(files) - 1) / max(1, a.pick - 1)) for i in range(a.pick)})
    OUT.mkdir(parents=True, exist_ok=True)
    clips = []
    for k, i in enumerate(idx):
        f = files[i]
        j = json.loads(f.read_text())
        last = k == len(idx) - 1
        fin = j.get("finishFrame")
        text = f"attempt {f.stem}: " + (f"finished {fin / 1000:.3f} s" if fin else
                                         f"checkpoint {j.get('checkpoints', '?')}, did not finish")
        raw = OUT / f"{f.stem}.mp4"
        cmd = ["python3", "loop/record_line.py", str(f), "--out", str(raw)]
        if not (last and fin):
            cmd += ["--max-s", str(a.max_s)]
        subprocess.run(cmd, cwd=ROOT, check=True, capture_output=True)
        png = OUT / f"{f.stem}.png"
        label(text, png)
        tagged = OUT / f"{f.stem}_tagged.mp4"
        # skip the ~7 s menu boot at the start of each recording
        subprocess.run(["ffmpeg", "-y", "-loglevel", "error", "-ss", "6.5", "-i", str(raw), "-i", str(png),
                        "-filter_complex", "[0:v]scale=1280:720[v];[v][1:v]overlay", "-c:v", "libx264",
                        "-pix_fmt", "yuv420p", "-r", "30", str(tagged)], check=True)
        clips.append(tagged)
        print(json.dumps({"clip": f.stem, "text": text}), flush=True)
    lst = OUT / "list.txt"
    lst.write_text("".join(f"file '{c}'\n" for c in clips))
    final = ROOT / "runs/videos/learning.mp4"
    subprocess.run(["ffmpeg", "-y", "-loglevel", "error", "-f", "concat", "-safe", "0", "-i", str(lst),
                    "-c", "copy", "-movflags", "+faststart", str(final)], check=True)
    print(json.dumps({"video": str(final), "clips": len(clips)}))


if __name__ == "__main__":
    main()
