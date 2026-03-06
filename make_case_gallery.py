"""Build a comparison gallery for TP / TN / FP / FN case images."""
# python make_case_gallery.py --run_dir runs/resnet18_sz224_bs64_lr0.0001_wd0.0001_aug1_amp1

import argparse
import math
import random
import re
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont


CATEGORIES = ["TP", "TN", "FP", "FN"]
BORDER_COLORS = {
    "TP": (38, 130, 57),
    "TN": (16, 185, 129),
    "FP": (220, 38, 38),
    "FN": (245, 158, 11),
}


def _load_font(size: int):
    # Prefer common TrueType fonts; fallback to default if unavailable.
    candidates = ["arial.ttf", "Arial.ttf", "DejaVuSans.ttf"]
    for name in candidates:
        try:
            return ImageFont.truetype(name, size=size)
        except OSError:
            continue
    return ImageFont.load_default()


def _collect_images(run_dir: Path, category: str, seed: int | None):
    paths = []

    if seed is not None:
        d = run_dir / f"seed_{seed}" / "misclassified" / category
        if d.exists():
            paths.extend(sorted(d.glob("*.png")))

        # Backward-compatibility for old layout that only had CORRECT.
        if category in ("TP", "TN"):
            old = run_dir / f"seed_{seed}" / "misclassified" / "CORRECT"
            if old.exists():
                paths.extend(sorted(old.glob("*.png")))
        return paths

    # Aggregate from all seed directories.
    for seed_dir in sorted(run_dir.glob("seed_*")):
        d = seed_dir / "misclassified" / category
        if d.exists():
            paths.extend(sorted(d.glob("*.png")))
        if category in ("TP", "TN"):
            old = seed_dir / "misclassified" / "CORRECT"
            if old.exists():
                paths.extend(sorted(old.glob("*.png")))

    # Backward-compatibility: single-run layout without seed_*.
    flat = run_dir / "misclassified" / category
    if flat.exists():
        paths.extend(sorted(flat.glob("*.png")))
    if category in ("TP", "TN"):
        old_flat = run_dir / "misclassified" / "CORRECT"
        if old_flat.exists():
            paths.extend(sorted(old_flat.glob("*.png")))

    return paths


def _extract_case_id(path: Path) -> int | None:
    # Use the first 4+ digit token in filename as case id when available.
    m = re.search(r"(\d{4,})", path.name)
    if not m:
        return None
    try:
        return int(m.group(1))
    except Exception:
        return None


def _pairwise_gap_penalty(sampled: list[Path], min_id_gap: int) -> int:
    if min_id_gap <= 0:
        return 0
    ids = [_extract_case_id(p) for p in sampled]
    penalty = 0
    for i in range(len(ids)):
        if ids[i] is None:
            continue
        for j in range(i + 1, len(ids)):
            if ids[j] is None:
                continue
            if abs(ids[i] - ids[j]) < min_id_gap:
                penalty += 1
    return penalty


def _select(paths, n: int, rng: random.Random, min_id_gap: int, resample_trials: int = 200):
    if len(paths) <= n:
        out = list(paths)
        rng.shuffle(out)
        return out

    # Random sampling with anti-continuity: prefer sets with fewer near-by ids.
    best = None
    best_penalty = None
    for _ in range(max(1, resample_trials)):
        cand = rng.sample(paths, n)
        pen = _pairwise_gap_penalty(cand, min_id_gap=min_id_gap)
        if best is None or pen < best_penalty:
            best = cand
            best_penalty = pen
            if pen == 0:
                break

    rng.shuffle(best)
    return best


def build_case_gallery(
    run_dir: Path,
    out_png: Path,
    out_pdf: Path,
    out_manifest: Path,
    seed: int | None,
    n_per_category: int,
    cols: int,
    tile_size: int,
    sample_seed: int | None,
    min_id_gap: int,
):
    # If sample_seed is None, use non-deterministic randomness.
    rng = random.Random(sample_seed) if sample_seed is not None else random.Random()

    selected = {}
    counts = {}
    rows_per_section = []

    for cat in CATEGORIES:
        all_paths = _collect_images(run_dir, cat, seed)
        picked = _select(all_paths, n_per_category, rng, min_id_gap=max(0, min_id_gap))
        counts[cat] = len(all_paths)
        selected[cat] = picked
        rows_per_section.append(max(1, math.ceil(len(picked) / cols)))

    margin = 24
    title_h = 52
    section_title_h = 44
    gap = 14

    panel_w = cols * tile_size + (cols - 1) * gap
    section_heights = [section_title_h + r * tile_size + (r - 1) * gap for r in rows_per_section]

    canvas_w = margin * 2 + panel_w
    canvas_h = margin * 2 + title_h + (len(CATEGORIES) - 1) * 24 + sum(section_heights)

    image = Image.new("RGB", (canvas_w, canvas_h), (255, 255, 255))
    draw = ImageDraw.Draw(image)
    font_title = _load_font(34)
    font_section = _load_font(30)
    font_note = _load_font(20)

    seed_text = f"seed={seed}" if seed is not None else "all seeds"
    draw.text((margin, margin), f"TP / TN / FP / FN  ({seed_text})", fill=(0, 0, 0), font=font_title)

    y = margin + title_h
    manifest = {
        "run_dir": str(run_dir),
        "seed": seed,
        "n_per_category": n_per_category,
        "sample_seed": sample_seed,
        "min_id_gap": min_id_gap,
        "counts_total": counts,
        "selected": {},
    }

    for idx, cat in enumerate(CATEGORIES):
        color = BORDER_COLORS[cat]
        picked = selected[cat]
        rows = rows_per_section[idx]

        # Keep only TP/TN/FP/FN notation in the figure.
        draw.text((margin, y), cat, fill=color, font=font_section)
        y += section_title_h

        for i, p in enumerate(picked):
            r = i // cols
            c = i % cols
            x0 = margin + c * (tile_size + gap)
            y0 = y + r * (tile_size + gap)
            x1 = x0 + tile_size
            y1 = y0 + tile_size

            tile = Image.open(p).convert("RGB").resize((tile_size, tile_size))
            image.paste(tile, (x0, y0))
            draw.rectangle([x0, y0, x1, y1], outline=color, width=4)

        if not picked:
            draw.rectangle(
                [margin, y, margin + panel_w, y + tile_size],
                outline=color,
                width=2,
            )
            draw.text((margin + 12, y + 12), "No images", fill=color, font=font_note)

        manifest["selected"][cat] = [str(p) for p in picked]

        y += rows * tile_size + (rows - 1) * gap
        if idx < len(CATEGORIES) - 1:
            y += 24

    out_png.parent.mkdir(parents=True, exist_ok=True)
    image.save(out_png)
    image.convert("RGB").save(out_pdf, "PDF", resolution=300.0)

    import json

    with open(out_manifest, "w", encoding="utf-8") as f:
        json.dump(manifest, f, ensure_ascii=False, indent=2)


def _latest_run_dir(save_dir: Path):
    candidates = [d for d in save_dir.iterdir() if d.is_dir()]
    if not candidates:
        raise FileNotFoundError(f"No run directory found under: {save_dir}")
    return max(candidates, key=lambda p: p.stat().st_mtime)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--run_dir", type=str, default="", help="experiment directory under runs")
    p.add_argument("--save_dir", type=str, default="runs")
    p.add_argument("--seed", type=int, default=None, help="seed number; default aggregates all seeds")
    p.add_argument("--n_per_category", type=int, default=24)
    p.add_argument("--cols", type=int, default=6)
    p.add_argument("--tile_size", type=int, default=180)
    p.add_argument(
        "--sample_seed",
        type=int,
        default=None,
        help="sampling seed; default None for random selection each run",
    )
    p.add_argument(
        "--min_id_gap",
        type=int,
        default=30,
        help="minimum gap between sampled numeric case ids to reduce contiguous picks",
    )

    args = p.parse_args()

    save_dir = Path(args.save_dir)
    run_dir = Path(args.run_dir) if args.run_dir else _latest_run_dir(save_dir)

    suffix = f"seed_{args.seed}" if args.seed is not None else "all_seeds"
    out_png = run_dir / f"case_gallery_{suffix}.png"
    out_pdf = run_dir / f"case_gallery_{suffix}.pdf"
    out_manifest = run_dir / f"case_gallery_{suffix}.json"

    build_case_gallery(
        run_dir=run_dir,
        out_png=out_png,
        out_pdf=out_pdf,
        out_manifest=out_manifest,
        seed=args.seed,
        n_per_category=max(1, args.n_per_category),
        cols=max(1, args.cols),
        tile_size=max(64, args.tile_size),
        sample_seed=args.sample_seed,
        min_id_gap=max(0, args.min_id_gap),
    )

    print(f"[saved] {out_png}")
    print(f"[saved] {out_pdf}")
    print(f"[saved] {out_manifest}")


if __name__ == "__main__":
    main()
