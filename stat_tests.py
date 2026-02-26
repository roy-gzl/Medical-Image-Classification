"""Seed-wise Mann-Whitney tests and cross-seed reproducibility summary.

This module reads per-seed feature files produced by analyze_case_trends.py:
  runs/<exp>/seed_xx/analyze/per_image_features.csv

It performs per-seed tests for:
  - TP vs FP
  - TN vs FN
across all numeric feature columns (except label/id columns), then aggregates
results across seeds.
"""

import argparse
import csv
import json
import math
from collections import defaultdict
from pathlib import Path
from statistics import mean
from typing import Dict, List, Tuple

import numpy as np


EXCLUDE_NUMERIC_COLUMNS = {
    "y_true",
    "y_pred",
}
COMPARISONS = [("TP", "FP"), ("TN", "FN")]


def _to_float_or_none(s: str):
    try:
        return float(s)
    except Exception:
        return None


def _read_feature_rows(csv_path: Path) -> List[dict]:
    with open(csv_path, "r", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    if not rows:
        raise RuntimeError(f"Empty CSV: {csv_path}")
    return rows


def _numeric_feature_columns(rows: List[dict]) -> List[str]:
    cols = rows[0].keys()
    out = []
    for c in cols:
        if c in EXCLUDE_NUMERIC_COLUMNS:
            continue
        if c in {"path", "group", "true_label", "pred_label"}:
            continue

        vals = [_to_float_or_none(r[c]) for r in rows]
        valid = [v for v in vals if v is not None and not math.isnan(v)]
        if not valid:
            continue
        out.append(c)
    return out


def _average_ranks(values: np.ndarray) -> np.ndarray:
    # Average ranks for ties, rank starts at 1.
    order = np.argsort(values, kind="mergesort")
    sorted_vals = values[order]
    ranks_sorted = np.zeros_like(sorted_vals, dtype=np.float64)

    i = 0
    n = sorted_vals.size
    while i < n:
        j = i + 1
        while j < n and sorted_vals[j] == sorted_vals[i]:
            j += 1
        avg_rank = (i + 1 + j) / 2.0
        ranks_sorted[i:j] = avg_rank
        i = j

    ranks = np.empty_like(ranks_sorted)
    ranks[order] = ranks_sorted
    return ranks


def _norm_cdf(z: float) -> float:
    return 0.5 * (1.0 + math.erf(z / math.sqrt(2.0)))


def mann_whitney_u_test(x: np.ndarray, y: np.ndarray) -> Dict[str, float]:
    # Two-sided Mann-Whitney U with normal approximation and tie correction.
    x = np.asarray(x, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64)
    x = x[np.isfinite(x)]
    y = y[np.isfinite(y)]

    n1 = x.size
    n2 = y.size
    if n1 == 0 or n2 == 0:
        return {"u": float("nan"), "p_value": float("nan"), "z": float("nan")}

    v = np.concatenate([x, y])
    g = np.concatenate([np.zeros(n1, dtype=np.int8), np.ones(n2, dtype=np.int8)])
    ranks = _average_ranks(v)

    r1 = ranks[g == 0].sum()
    u1 = r1 - n1 * (n1 + 1) / 2.0
    u2 = n1 * n2 - u1
    u = min(u1, u2)

    n = n1 + n2
    _, counts = np.unique(v, return_counts=True)
    tie_term = float(np.sum(counts**3 - counts))

    mean_u = n1 * n2 / 2.0
    if n <= 1:
        return {"u": float(u), "p_value": 1.0, "z": 0.0}

    var_u = (n1 * n2 / 12.0) * ((n + 1.0) - tie_term / (n * (n - 1.0)))
    if var_u <= 0:
        return {"u": float(u), "p_value": 1.0, "z": 0.0}

    # Continuity correction for two-sided normal approximation.
    cc = 0.5 if u < mean_u else -0.5
    z = (u - mean_u + cc) / math.sqrt(var_u)
    p = 2.0 * (1.0 - _norm_cdf(abs(z)))
    p = min(max(p, 0.0), 1.0)

    return {"u": float(u), "p_value": float(p), "z": float(z)}


def cliffs_delta(x: np.ndarray, y: np.ndarray) -> float:
    # Efficient O(n log n) Cliff's delta.
    x = np.asarray(x, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64)
    x = x[np.isfinite(x)]
    y = y[np.isfinite(y)]

    n1 = x.size
    n2 = y.size
    if n1 == 0 or n2 == 0:
        return float("nan")

    ys = np.sort(y)
    gt = 0
    lt = 0
    for xv in x:
        lt_count = np.searchsorted(ys, xv, side="left")
        gt_count = n2 - np.searchsorted(ys, xv, side="right")
        gt += lt_count
        lt += gt_count

    return float((gt - lt) / (n1 * n2))


def benjamini_hochberg(pvals: List[float]) -> List[float]:
    m = len(pvals)
    if m == 0:
        return []

    idx = np.argsort(pvals)
    p_sorted = np.asarray([pvals[i] for i in idx], dtype=np.float64)

    q_sorted = np.empty(m, dtype=np.float64)
    prev = 1.0
    for i in range(m - 1, -1, -1):
        rank = i + 1
        q = p_sorted[i] * m / rank
        q = min(q, prev)
        prev = q
        q_sorted[i] = q

    q = np.empty(m, dtype=np.float64)
    q[idx] = q_sorted
    return q.tolist()


def _seed_dirs(run_dir: Path) -> List[Path]:
    return sorted([d for d in run_dir.glob("seed_*") if d.is_dir()])


def run_mannwhitney_analysis(run_dir: Path, alpha: float = 0.05, out_dir: Path | None = None):
    run_dir = Path(run_dir)
    if out_dir is None:
        out_dir = run_dir / "stats_tests"
    out_dir.mkdir(parents=True, exist_ok=True)

    per_seed_rows = []

    for seed_dir in _seed_dirs(run_dir):
        seed_name = seed_dir.name
        try:
            seed = int(seed_name.split("_")[-1])
        except Exception:
            seed = -1

        csv_path = seed_dir / "analyze" / "per_image_features.csv"
        if not csv_path.exists():
            continue

        rows = _read_feature_rows(csv_path)
        features = _numeric_feature_columns(rows)

        by_group_feat = defaultdict(list)
        for r in rows:
            g = r.get("group", "")
            for f in features:
                v = _to_float_or_none(r[f])
                if v is None or math.isnan(v):
                    continue
                by_group_feat[(g, f)].append(v)

        seed_rows = []
        for g1, g2 in COMPARISONS:
            for f in features:
                x = np.asarray(by_group_feat[(g1, f)], dtype=np.float64)
                y = np.asarray(by_group_feat[(g2, f)], dtype=np.float64)

                n1 = int(x.size)
                n2 = int(y.size)
                if n1 == 0 or n2 == 0:
                    row = {
                        "seed": seed,
                        "seed_dir": str(seed_dir),
                        "comparison": f"{g1}_vs_{g2}",
                        "feature": f,
                        "n_group1": n1,
                        "n_group2": n2,
                        "u": float("nan"),
                        "z": float("nan"),
                        "p_value": float("nan"),
                        "p_value_fdr": float("nan"),
                        "cliffs_delta": float("nan"),
                        "mean_group1": float(np.nanmean(x)) if n1 > 0 else float("nan"),
                        "mean_group2": float(np.nanmean(y)) if n2 > 0 else float("nan"),
                    }
                else:
                    test = mann_whitney_u_test(x, y)
                    row = {
                        "seed": seed,
                        "seed_dir": str(seed_dir),
                        "comparison": f"{g1}_vs_{g2}",
                        "feature": f,
                        "n_group1": n1,
                        "n_group2": n2,
                        "u": float(test["u"]),
                        "z": float(test["z"]),
                        "p_value": float(test["p_value"]),
                        "p_value_fdr": float("nan"),
                        "cliffs_delta": float(cliffs_delta(x, y)),
                        "mean_group1": float(np.mean(x)),
                        "mean_group2": float(np.mean(y)),
                    }
                seed_rows.append(row)

        # FDR per seed and per comparison across features.
        for comp in sorted({r["comparison"] for r in seed_rows}):
            idxs = [i for i, r in enumerate(seed_rows) if r["comparison"] == comp and math.isfinite(r["p_value"])]
            pvals = [seed_rows[i]["p_value"] for i in idxs]
            qvals = benjamini_hochberg(pvals)
            for i, q in zip(idxs, qvals):
                seed_rows[i]["p_value_fdr"] = float(q)

        for r in seed_rows:
            r["significant_raw"] = bool(math.isfinite(r["p_value"]) and r["p_value"] < alpha)
            r["significant_fdr"] = bool(math.isfinite(r["p_value_fdr"]) and r["p_value_fdr"] < alpha)

        per_seed_rows.extend(seed_rows)

    # Aggregate reproducibility across seeds.
    grouped = defaultdict(list)
    for r in per_seed_rows:
        key = (r["comparison"], r["feature"])
        grouped[key].append(r)

    repro_rows = []
    for (comp, feat), items in sorted(grouped.items()):
        p_raw = [x["p_value"] for x in items if math.isfinite(x["p_value"])]
        p_fdr = [x["p_value_fdr"] for x in items if math.isfinite(x["p_value_fdr"])]
        eff = [x["cliffs_delta"] for x in items if math.isfinite(x["cliffs_delta"])]

        if eff:
            eff_mean = float(np.mean(eff))
            eff_std = float(np.std(eff, ddof=1)) if len(eff) > 1 else 0.0
            signs = np.sign(np.asarray(eff))
            sign_consistency = float(max((signs > 0).mean(), (signs < 0).mean(), (signs == 0).mean()))
        else:
            eff_mean = float("nan")
            eff_std = float("nan")
            sign_consistency = float("nan")

        repro_rows.append(
            {
                "comparison": comp,
                "feature": feat,
                "num_seeds_tested": int(len(items)),
                "num_significant_raw": int(sum(1 for x in items if x["significant_raw"])),
                "num_significant_fdr": int(sum(1 for x in items if x["significant_fdr"])),
                "fraction_significant_raw": float(sum(1 for x in items if x["significant_raw"]) / len(items)) if items else 0.0,
                "fraction_significant_fdr": float(sum(1 for x in items if x["significant_fdr"]) / len(items)) if items else 0.0,
                "median_p_raw": float(np.median(p_raw)) if p_raw else float("nan"),
                "median_p_fdr": float(np.median(p_fdr)) if p_fdr else float("nan"),
                "effect_mean": eff_mean,
                "effect_std": eff_std,
                "effect_sign_consistency": sign_consistency,
            }
        )

    # Save outputs.
    per_seed_csv = out_dir / "per_seed_mannwhitney.csv"
    repro_csv = out_dir / "reproducibility_summary.csv"
    json_path = out_dir / "stats_summary.json"

    if per_seed_rows:
        fields = list(per_seed_rows[0].keys())
        with open(per_seed_csv, "w", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=fields)
            w.writeheader()
            w.writerows(per_seed_rows)
    else:
        with open(per_seed_csv, "w", newline="", encoding="utf-8") as f:
            w = csv.writer(f)
            w.writerow(["message"])
            w.writerow(["No per-seed analyze/per_image_features.csv found."])

    if repro_rows:
        fields = list(repro_rows[0].keys())
        with open(repro_csv, "w", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=fields)
            w.writeheader()
            w.writerows(repro_rows)
    else:
        with open(repro_csv, "w", newline="", encoding="utf-8") as f:
            w = csv.writer(f)
            w.writerow(["message"])
            w.writerow(["No reproducibility rows generated."])

    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(
            {
                "run_dir": str(run_dir),
                "alpha": alpha,
                "comparisons": [f"{a}_vs_{b}" for a, b in COMPARISONS],
                "per_seed_count": len(per_seed_rows),
                "reproducibility_count": len(repro_rows),
                "per_seed_csv": str(per_seed_csv),
                "reproducibility_csv": str(repro_csv),
                "per_seed_results": per_seed_rows,
                "reproducibility_results": repro_rows,
            },
            f,
            ensure_ascii=False,
            indent=2,
        )

    return {
        "out_dir": str(out_dir),
        "per_seed_csv": str(per_seed_csv),
        "reproducibility_csv": str(repro_csv),
        "json": str(json_path),
    }


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--run_dir", type=str, required=True)
    p.add_argument("--alpha", type=float, default=0.05)
    p.add_argument("--out_dir", type=str, default="")
    args = p.parse_args()

    out_dir = Path(args.out_dir) if args.out_dir else None
    result = run_mannwhitney_analysis(Path(args.run_dir), alpha=args.alpha, out_dir=out_dir)
    print(f"[saved] {result['per_seed_csv']}")
    print(f"[saved] {result['reproducibility_csv']}")
    print(f"[saved] {result['json']}")


if __name__ == "__main__":
    main()
