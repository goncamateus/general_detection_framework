#!/usr/bin/env python3
"""Build a temporally-honest validation split for a Roboflow frame-extracted dataset.

Roboflow splits frames at random, so a frame in `valid/` is often the temporal neighbour of
a frame in `train/` — mAP comes out inflated. This regroups every frame by source video and
holds out the last `--holdout` fraction of each video's timeline.

Nothing on disk is moved: it writes symlink trees plus a data.yaml you can point `gdf` at.

    python scripts/temporal_split.py data/plume
    gdf train --config configs/plume_seg.yaml --data-path data/plume_temporal
    gdf eval --model runs/.../best.pt --data data/plume_temporal

You must RETRAIN on this split for the number to mean anything. Scoring a model that was
trained on the original random split against this holdout tells you nothing — it already
saw those frames during training, so it will score just as high here.
"""

from __future__ import annotations

import argparse
import re
from collections import defaultdict
from pathlib import Path

# voo_2_mp4-2326_jpg.rf.<hash>.jpg  ->  ("voo_2_mp4", 2326)
FRAME_RE = re.compile(r"^(?P<video>.+?)-(?P<idx>\d+)_jpg\.rf\.")


def parse_frame(name: str) -> tuple[str, int] | None:
    m = FRAME_RE.match(name)
    return (m.group("video"), int(m.group("idx"))) if m else None


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("root", type=Path, help="Dataset root (contains train/ valid/ test/)")
    ap.add_argument("-o", "--out", type=Path, default=None, help="Output root")
    ap.add_argument("--holdout", type=float, default=0.2, help="Tail fraction held out")
    ap.add_argument("--names", default="plume", help="Comma-separated class names")
    args = ap.parse_args()

    out = args.out or args.root.parent / f"{args.root.name}_temporal"

    by_video: dict[str, list[tuple[int, Path]]] = defaultdict(list)
    unparsed = []
    for split in ("train", "valid", "val", "test"):
        images = args.root / split / "images"
        if not images.is_dir():
            continue
        for img in images.iterdir():
            parsed = parse_frame(img.name)
            if parsed is None:
                unparsed.append(img)
                continue
            video, idx = parsed
            by_video[video].append((idx, img))

    if not by_video:
        raise SystemExit(f"No frames matching {FRAME_RE.pattern!r} under {args.root}")
    if unparsed:
        print(f"warning: {len(unparsed)} files did not match the frame pattern, skipped")

    assignments: dict[str, list[Path]] = {"train": [], "val": []}
    for video, frames in sorted(by_video.items()):
        frames.sort()
        cut = int(len(frames) * (1 - args.holdout))
        assignments["train"] += [p for _, p in frames[:cut]]
        assignments["val"] += [p for _, p in frames[cut:]]
        span = f"{frames[0][0]}..{frames[-1][0]}"
        print(f"{video}: {cut} train / {len(frames) - cut} val (frames {span})")

    for split, paths in assignments.items():
        for kind in ("images", "labels"):
            (out / split / kind).mkdir(parents=True, exist_ok=True)
        for img in paths:
            label = img.parent.parent / "labels" / f"{img.stem}.txt"
            for src, dst in ((img, out / split / "images" / img.name),
                             (label, out / split / "labels" / label.name)):
                if not src.exists():
                    continue
                dst.unlink(missing_ok=True)
                dst.symlink_to(src.resolve())

    names = [n.strip() for n in args.names.split(",")]
    # No `path:` key — a relative one resolves against Ultralytics' global datasets_dir,
    # not against this file. Omitting it makes the yaml's own directory the root.
    (out / "data.yaml").write_text(
        f"train: train/images\nval: val/images\n\nnc: {len(names)}\nnames: {names}\n"
    )
    n_train, n_val = len(assignments["train"]), len(assignments["val"])
    print(f"\nWrote {out}/data.yaml  ({n_train} train / {n_val} val)")


if __name__ == "__main__":
    main()
