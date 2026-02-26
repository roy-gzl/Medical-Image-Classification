"""Analyze TP/TN/FP/FN trends with image quality, confidence, and texture features.

Usage example:
python analyze_case_trends.py --run_dir runs/resnet18_sz224_bs64_lr0.0001_wd0.0001_aug1_amp1 --seed 42
"""

import argparse
import csv
import json
import math
from pathlib import Path
from statistics import mean, median, stdev
from types import SimpleNamespace
from typing import Dict, List, Optional

import numpy as np
import torch
from PIL import Image

from data import prepare_datasets_and_loaders
from models import build_model


def _to_gray_array(path: Path) -> np.ndarray:
    img = Image.open(path).convert("L")
    return np.asarray(img, dtype=np.float32)


def _laplacian_variance(gray: np.ndarray) -> float:
    # 4-neighbor Laplacian for a simple sharpness proxy.
    c = gray[1:-1, 1:-1]
    lap = (
        gray[:-2, 1:-1]
        + gray[2:, 1:-1]
        + gray[1:-1, :-2]
        + gray[1:-1, 2:]
        - 4.0 * c
    )
    return float(np.var(lap)) if lap.size > 0 else 0.0


def _entropy(gray: np.ndarray, bins: int = 256) -> float:
    hist, _ = np.histogram(gray, bins=bins, range=(0, 255), density=True)
    hist = hist[hist > 0]
    if hist.size == 0:
        return 0.0
    return float(-(hist * np.log2(hist)).sum())


def _glcm_features(gray: np.ndarray, levels: int = 16) -> Dict[str, float]:
    # Quantize and compute horizontal (dx=1, dy=0) GLCM.
    q = np.clip((gray / 256.0 * levels).astype(np.int32), 0, levels - 1)
    if q.shape[1] < 2:
        return {
            "glcm_contrast": 0.0,
            "glcm_homogeneity": 0.0,
            "glcm_energy": 0.0,
            "glcm_correlation": 0.0,
        }

    a = q[:, :-1].ravel()
    b = q[:, 1:].ravel()

    mat = np.zeros((levels, levels), dtype=np.float64)
    for i, j in zip(a, b):
        mat[i, j] += 1.0

    total = mat.sum()
    if total <= 0:
        return {
            "glcm_contrast": 0.0,
            "glcm_homogeneity": 0.0,
            "glcm_energy": 0.0,
            "glcm_correlation": 0.0,
        }

    p = mat / total

    i_idx = np.arange(levels).reshape(-1, 1)
    j_idx = np.arange(levels).reshape(1, -1)
    diff = i_idx - j_idx

    contrast = float((p * (diff ** 2)).sum())
    homogeneity = float((p / (1.0 + np.abs(diff))).sum())
    energy = float((p ** 2).sum())

    mu_i = float((p * i_idx).sum())
    mu_j = float((p * j_idx).sum())
    std_i = math.sqrt(float((p * ((i_idx - mu_i) ** 2)).sum()))
    std_j = math.sqrt(float((p * ((j_idx - mu_j) ** 2)).sum()))

    if std_i <= 1e-12 or std_j <= 1e-12:
        correlation = 0.0
    else:
        correlation = float((p * (i_idx - mu_i) * (j_idx - mu_j)).sum() / (std_i * std_j))

    return {
        "glcm_contrast": contrast,
        "glcm_homogeneity": homogeneity,
        "glcm_energy": energy,
        "glcm_correlation": correlation,
    }


def _quality_texture_features(path: Path) -> Dict[str, float]:
    g = _to_gray_array(path)
    g_mean = float(g.mean())
    g_std = float(g.std())
    p95 = float(np.percentile(g, 95))
    p5 = float(np.percentile(g, 5))

    out = {
        "brightness_mean": g_mean,
        "brightness_std": g_std,
        "contrast_p95_p5": p95 - p5,
        "sharpness_laplacian_var": _laplacian_variance(g),
        "snr_mean_over_std": float(g_mean / (g_std + 1e-8)),
        "entropy": _entropy(g),
    }
    out.update(_glcm_features(g, levels=16))
    return out


def _summarize(values: List[float]) -> Dict[str, float]:
    if not values:
        return {"n": 0, "mean": 0.0, "std": 0.0, "median": 0.0, "p25": 0.0, "p75": 0.0}
    arr = np.asarray(values, dtype=np.float64)
    return {
        "n": int(arr.size),
        "mean": float(arr.mean()),
        "std": float(arr.std(ddof=1)) if arr.size > 1 else 0.0,
        "median": float(np.median(arr)),
        "p25": float(np.percentile(arr, 25)),
        "p75": float(np.percentile(arr, 75)),
    }


def _group_name(y_true: int, y_pred: int, num_classes: int) -> str:
    if num_classes == 2:
        if y_true == 1 and y_pred == 1:
            return "TP"
        if y_true == 0 and y_pred == 0:
            return "TN"
        if y_true == 0 and y_pred == 1:
            return "FP"
        if y_true == 1 and y_pred == 0:
            return "FN"
        return "MIS"

    if y_true == y_pred:
        return "CORRECT"
    return "MIS"


def _find_seed_dir(run_dir: Path, seed: Optional[int]) -> Path:
    if seed is not None:
        d = run_dir / f"seed_{seed}"
        if not d.exists():
            raise FileNotFoundError(f"Not found: {d}")
        return d

    candidates = sorted([d for d in run_dir.glob("seed_*") if d.is_dir()])
    if not candidates:
        raise FileNotFoundError("No seed directories found. Specify --seed or provide a multi-seed run dir.")
    return candidates[0]


def _load_cfg(seed_dir: Path):
    cfg_path = seed_dir / "config.json"
    if not cfg_path.exists():
        raise FileNotFoundError(f"Not found: {cfg_path}")
    with open(cfg_path, "r", encoding="utf-8") as f:
        raw = json.load(f)
    return SimpleNamespace(**raw)


def _infer_predictions(seed_dir: Path):
    cfg = _load_cfg(seed_dir)

    if (not getattr(cfg, "cpu", False)) and torch.cuda.is_available():
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")

    loaders, class_to_idx, samples_test = prepare_datasets_and_loaders(cfg)
    idx_to_class = {v: k for k, v in class_to_idx.items()}

    model = build_model(cfg.model_name, num_classes=cfg.num_classes, pretrained=cfg.pretrained).to(device)
    ckpt_path = seed_dir / "best.pt"
    ckpt = torch.load(ckpt_path, map_location=device)
    state = ckpt["model"] if isinstance(ckpt, dict) and "model" in ckpt else ckpt
    model.load_state_dict(state)
    model.eval()

    rows = []
    with torch.no_grad():
        for x, y in loaders["test"]:
            x = x.to(device, non_blocking=True)
            y = y.to(device, non_blocking=True)
            logits = model(x)
            probs = torch.softmax(logits, dim=1)
            top2_prob, top2_idx = torch.topk(probs, k=min(2, probs.size(1)), dim=1)

            y_pred = probs.argmax(dim=1)
            max_prob = top2_prob[:, 0]
            if probs.size(1) >= 2:
                margin = top2_prob[:, 0] - top2_prob[:, 1]
            else:
                margin = top2_prob[:, 0]

            for yt, yp, mp, mg in zip(y.cpu().tolist(), y_pred.cpu().tolist(), max_prob.cpu().tolist(), margin.cpu().tolist()):
                rows.append(
                    {
                        "y_true": int(yt),
                        "y_pred": int(yp),
                        "confidence_max_prob": float(mp),
                        "confidence_margin": float(mg),
                    }
                )

    if len(rows) != len(samples_test):
        raise RuntimeError(f"prediction count mismatch: {len(rows)} vs {len(samples_test)}")

    merged = []
    for (img_path, y_true_ds), r in zip(samples_test, rows):
        # y_true from dataset and from loader should be identical.
        y_true = int(r["y_true"])
        if y_true != int(y_true_ds):
            y_true = int(y_true_ds)
        merged.append(
            {
                "path": str(img_path),
                "y_true": y_true,
                "y_pred": int(r["y_pred"]),
                "confidence_max_prob": float(r["confidence_max_prob"]),
                "confidence_margin": float(r["confidence_margin"]),
                "true_label": idx_to_class.get(y_true, str(y_true)),
                "pred_label": idx_to_class.get(int(r["y_pred"]), str(int(r["y_pred"]))),
                "num_classes": int(cfg.num_classes),
            }
        )

    return merged, cfg


def analyze(run_dir: Path, seed: Optional[int], out_dir: Optional[Path]):
    seed_dir = _find_seed_dir(run_dir, seed)
    if out_dir is None:
        out_dir = seed_dir / "analyze"
    out_dir.mkdir(parents=True, exist_ok=True)

    records, cfg = _infer_predictions(seed_dir)

    feature_rows = []
    for r in records:
        p = Path(r["path"])
        feats = _quality_texture_features(p)
        group = _group_name(r["y_true"], r["y_pred"], r["num_classes"])

        row = {
            "path": str(p),
            "group": group,
            "y_true": r["y_true"],
            "y_pred": r["y_pred"],
            "true_label": r["true_label"],
            "pred_label": r["pred_label"],
            "confidence_max_prob": r["confidence_max_prob"],
            "confidence_margin": r["confidence_margin"],
        }
        row.update(feats)
        feature_rows.append(row)

    metrics_quality = [
        "brightness_mean",
        "brightness_std",
        "contrast_p95_p5",
        "sharpness_laplacian_var",
        "snr_mean_over_std",
    ]
    metrics_texture = [
        "entropy",
        "glcm_contrast",
        "glcm_homogeneity",
        "glcm_energy",
        "glcm_correlation",
    ]
    metrics_conf = ["confidence_max_prob", "confidence_margin"]

    groups = ["TP", "TN", "FP", "FN", "MIS", "CORRECT"]

    summary = {"quality_intensity": {}, "texture": {}, "confidence": {}}

    for metric_list, key in [
        (metrics_quality, "quality_intensity"),
        (metrics_texture, "texture"),
        (metrics_conf, "confidence"),
    ]:
        for m in metric_list:
            summary[key][m] = {}
            for g in groups:
                vals = [float(x[m]) for x in feature_rows if x["group"] == g]
                summary[key][m][g] = _summarize(vals)

    # Save per-image table.
    feature_csv = out_dir / "per_image_features.csv"
    with open(feature_csv, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(feature_rows[0].keys()))
        writer.writeheader()
        writer.writerows(feature_rows)

    # Save flattened summaries for easy spreadsheet use.
    for key, filename in [
        ("quality_intensity", "summary_quality_intensity.csv"),
        ("texture", "summary_texture.csv"),
        ("confidence", "summary_confidence.csv"),
    ]:
        rows = []
        for metric, by_group in summary[key].items():
            for g, stats in by_group.items():
                rows.append({"metric": metric, "group": g, **stats})
        out_csv = out_dir / filename
        with open(out_csv, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=["metric", "group", "n", "mean", "std", "median", "p25", "p75"])
            writer.writeheader()
            writer.writerows(rows)

    with open(out_dir / "summary_all.json", "w", encoding="utf-8") as f:
        json.dump(
            {
                "run_dir": str(run_dir),
                "seed_dir": str(seed_dir),
                "seed": getattr(cfg, "seed", seed),
                "num_samples": len(feature_rows),
                "summary": summary,
            },
            f,
            ensure_ascii=False,
            indent=2,
        )

    return out_dir


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--run_dir", type=str, required=True)
    p.add_argument("--seed", type=int, default=None, help="seed id (e.g., 42). default: first seed in run_dir")
    p.add_argument("--out_dir", type=str, default="")
    args = p.parse_args()

    out_dir = Path(args.out_dir) if args.out_dir else None
    saved = analyze(Path(args.run_dir), seed=args.seed, out_dir=out_dir)
    print(f"[saved] {saved / 'per_image_features.csv'}")
    print(f"[saved] {saved / 'summary_quality_intensity.csv'}")
    print(f"[saved] {saved / 'summary_texture.csv'}")
    print(f"[saved] {saved / 'summary_confidence.csv'}")
    print(f"[saved] {saved / 'summary_all.json'}")


if __name__ == "__main__":
    main()


