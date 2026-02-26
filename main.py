"""Entry point for multi-seed training/evaluation experiments."""

import copy
import statistics
import time
from pathlib import Path

import torch

from configs import get_config, make_experiment_name
from data import prepare_datasets_and_loaders
from engine import (
    binary_confusion_counts,
    build_confusion_matrix,
    evaluate,
    predict_labels,
    predict_to_csv,
    save_case_images,
    save_confusion_matrix_figure,
    train_loop,
)
from models import build_model
from utils import save_json, set_seed
from analyze_case_trends import analyze as analyze_case_trends
from stat_tests import run_mannwhitney_analysis


LR_SEARCH_CANDIDATES = [1e-5, 3e-5, 1e-4, 3e-4, 1e-3]
WD_SEARCH_CANDIDATES = [0.0, 1e-5, 1e-4, 1e-3]


def _select_device(cpu: bool):
    # Pick CUDA when available unless CPU is explicitly requested.
    if (not cpu) and torch.cuda.is_available():
        device = torch.device("cuda")
        torch.backends.cudnn.benchmark = True
        print(f"[device] cuda: {torch.cuda.get_device_name(0)}")
        return device

    device = torch.device("cpu")
    print("[device] cpu")
    return device


def _make_unique_experiment_dir(save_root: Path, exp_name: str) -> Path:
    # Create a new directory even when the same experiment name already exists.
    base_dir = save_root / exp_name
    if not base_dir.exists():
        base_dir.mkdir(parents=True, exist_ok=False)
        return base_dir

    idx = 1
    while True:
        candidate = save_root / f"{exp_name}_run{idx:03d}"
        if not candidate.exists():
            candidate.mkdir(parents=True, exist_ok=False)
            return candidate
        idx += 1


def _run_single_seed(cfg, seed: int, base_out_dir: Path, device):
    # Run one full train/val/test cycle and persist per-seed artifacts.
    seed_start = time.perf_counter()
    set_seed(seed)

    seed_out_dir = base_out_dir / f"seed_{seed}"
    seed_out_dir.mkdir(parents=True, exist_ok=True)

    cfg_dump = dict(cfg.__dict__)
    cfg_dump["seed"] = seed
    save_json(seed_out_dir / "config.json", cfg_dump)

    # Build loaders/model for this seed.
    loaders, class_to_idx, samples_test = prepare_datasets_and_loaders(cfg)
    model = build_model(cfg.model_name, num_classes=cfg.num_classes, pretrained=cfg.pretrained).to(device)

    # Train with best checkpoint tracking.
    best_ckpt_path = seed_out_dir / "best.pt"
    model = train_loop(cfg, model, loaders, device, best_ckpt_path)

    # Evaluate test split.
    test_loss, test_acc = evaluate(model, loaders["test"], device, amp=cfg.amp)
    print(f"[test][seed={seed}] loss={test_loss:.4f} acc={test_acc:.4f}")

    # Predict labels for confusion matrix and case extraction.
    y_true, y_pred = predict_labels(model, loaders["test"], device, amp=cfg.amp)
    conf_mat = build_confusion_matrix(y_true, y_pred, num_classes=cfg.num_classes)

    idx_to_class = {v: k for k, v in class_to_idx.items()}
    class_names = [idx_to_class[i] for i in range(cfg.num_classes)]

    result = {
        "seed": seed,
        "test_loss": float(test_loss),
        "test_acc": float(test_acc),
        "confusion_matrix": conf_mat.tolist(),
        "class_names": class_names,
    }

    cm_payload = {
        "labels": class_names,
        "matrix": conf_mat.tolist(),
    }

    if cfg.num_classes == 2:
        counts = binary_confusion_counts(conf_mat, positive_class_idx=1)
        cm_payload.update(counts)
        result.update(counts)
        print(
            f"[confusion][seed={seed}] TP={counts['tp']} FP={counts['fp']} FN={counts['fn']} TN={counts['tn']} "
            f"(positive_class={class_names[1]})"
        )

    # Per-seed confusion matrix figure is intentionally skipped.
    save_json(seed_out_dir / "confusion_matrix.json", cm_payload)
    print(f"[saved] {seed_out_dir / 'confusion_matrix.json'}")

    # Save TP/TN/FP/FN sampled cases.
    if cfg.save_case_images:
        cases = save_case_images(
            samples=samples_test,
            y_true=y_true,
            y_pred=y_pred,
            idx_to_class=idx_to_class,
            output_root=seed_out_dir / "misclassified",
            exp_name=f"{base_out_dir.name}_seed{seed}",
            max_correct_images=cfg.max_correct_images,
        )
        print(f"[saved] TP images: {cases['counts']['TP']} -> {cases['tp_dir']}")
        print(f"[saved] TN images: {cases['counts']['TN']} -> {cases['tn_dir']}")
        print(f"[saved] FP images: {cases['counts']['FP']} -> {cases['fp_dir']}")
        print(f"[saved] FN images: {cases['counts']['FN']} -> {cases['fn_dir']}")

    # Optional CSV export of test predictions.
    if cfg.save_test_preds:
        out_csv = seed_out_dir / f"test_preds_{cfg.model_name}.csv"
        predict_to_csv(model, loaders["test"], device, samples_test, idx_to_class, out_csv, amp=cfg.amp)
        print(f"[saved] {out_csv}")

    # Run trend analysis and save into seed_xx/analyze/.
    analyze_out = analyze_case_trends(base_out_dir, seed=seed, out_dir=seed_out_dir / "analyze")
    print(f"[saved] {analyze_out}")

    seed_duration_sec = float(time.perf_counter() - seed_start)
    result["duration_sec"] = seed_duration_sec

    seed_timing_payload = {
        "seed": seed,
        "duration_sec": seed_duration_sec,
        "duration_min": seed_duration_sec / 60.0,
    }
    save_json(seed_out_dir / "timing.json", seed_timing_payload)
    print(f"[saved] {seed_out_dir / 'timing.json'}")

    return result


def _mean_std(values):
    """Return mean/std with a safe fallback for 0 or 1 samples."""
    if not values:
        return {"mean": 0.0, "std": 0.0}
    if len(values) == 1:
        return {"mean": float(values[0]), "std": 0.0}
    return {"mean": float(statistics.mean(values)), "std": float(statistics.stdev(values))}


def _run_experiment(cfg, device):
    total_start = time.perf_counter()

    exp_name = make_experiment_name(cfg)
    out_dir = _make_unique_experiment_dir(Path(cfg.save_dir), exp_name)
    print(f"[run-dir] {out_dir}")

    # Example: seed=42, num_seeds=5 -> [42, 43, 44, 45, 46]
    seeds = [cfg.seed + i for i in range(cfg.num_seeds)]
    print(f"[multi-seed] seeds={seeds}")

    all_results = []
    for idx, seed in enumerate(seeds, start=1):
        print(f"\n[multi-seed] run {idx}/{len(seeds)} seed={seed}")
        all_results.append(_run_single_seed(cfg, seed, out_dir, device))

    # Aggregate statistics for reporting reproducibility.
    acc_stats = _mean_std([r["test_acc"] for r in all_results])
    loss_stats = _mean_std([r["test_loss"] for r in all_results])

    summary = {
        "num_seeds": len(seeds),
        "seeds": seeds,
        "test_acc": acc_stats,
        "test_loss": loss_stats,
    }

    if cfg.num_classes == 2:
        for key in ["tp", "fp", "fn", "tn"]:
            summary[key] = _mean_std([float(r[key]) for r in all_results])

    # Build and save only one confusion matrix figure: mean across seeds.
    class_names = all_results[0].get("class_names", [str(i) for i in range(cfg.num_classes)])

    mean_conf_mat = torch.tensor([r["confusion_matrix"] for r in all_results], dtype=torch.float32).mean(dim=0)

    mean_cm_png = out_dir / "confusion_matrix_mean.png"
    mean_cm_pdf = out_dir / "confusion_matrix_mean.pdf"
    save_confusion_matrix_figure(
        mean_conf_mat,
        out_png_path=mean_cm_png,
        out_pdf_path=mean_cm_pdf,
        class_names=class_names,
        positive_class_idx=1,
    )

    mean_cm_payload = {
        "labels": class_names,
        "matrix_mean": mean_conf_mat.tolist(),
    }
    if cfg.num_classes == 2:
        mean_cm_payload.update(
            {
                "tp_mean": summary["tp"]["mean"],
                "fp_mean": summary["fp"]["mean"],
                "fn_mean": summary["fn"]["mean"],
                "tn_mean": summary["tn"]["mean"],
            }
        )
    save_json(out_dir / "confusion_matrix_mean.json", mean_cm_payload)

    # Timing summary JSON.
    per_seed_durations = [float(r.get("duration_sec", 0.0)) for r in all_results]
    total_duration_sec = float(time.perf_counter() - total_start)
    timing_payload = {
        "total_duration_sec": total_duration_sec,
        "total_duration_min": total_duration_sec / 60.0,
        "per_seed_duration_sec": per_seed_durations,
        "per_seed_duration_stats_sec": _mean_std(per_seed_durations),
    }
    save_json(out_dir / "run_timing.json", timing_payload)

    summary_payload = {
        "experiment": exp_name,
        "run_dir": out_dir.name,
        "runs": all_results,
        "summary": summary,
        "mean_confusion_matrix": mean_cm_payload,
        "timing": timing_payload,
    }
    summary_path = out_dir / "multi_seed_summary.json"
    save_json(summary_path, summary_payload)

    # Statistical tests: per-seed Mann-Whitney and cross-seed reproducibility.
    stats_out = run_mannwhitney_analysis(out_dir, alpha=0.05, out_dir=out_dir / "stats_tests")

    print("\n[multi-seed][summary]")
    print(
        f"test_acc mean={summary['test_acc']['mean']:.4f} std={summary['test_acc']['std']:.4f} | "
        f"test_loss mean={summary['test_loss']['mean']:.4f} std={summary['test_loss']['std']:.4f}"
    )
    if cfg.num_classes == 2:
        print(
            f"TP mean={summary['tp']['mean']:.2f} std={summary['tp']['std']:.2f}, "
            f"FP mean={summary['fp']['mean']:.2f} std={summary['fp']['std']:.2f}, "
            f"FN mean={summary['fn']['mean']:.2f} std={summary['fn']['std']:.2f}, "
            f"TN mean={summary['tn']['mean']:.2f} std={summary['tn']['std']:.2f}"
        )

    print(f"[saved] {mean_cm_png}")
    print(f"[saved] {mean_cm_pdf}")
    print(f"[saved] {out_dir / 'confusion_matrix_mean.json'}")
    print(f"[saved] {out_dir / 'run_timing.json'}")
    print(f"[saved] {summary_path}")
    print(f"[saved] {stats_out['per_seed_csv']}")
    print(f"[saved] {stats_out['reproducibility_csv']}")
    print(f"[saved] {stats_out['json']}")

    return {
        "out_dir": str(out_dir),
        "summary": summary,
        "summary_path": str(summary_path),
    }


def _run_lr_wd_search(cfg, device):
    search_start = time.perf_counter()

    search_name = f"{cfg.model_name}_lrwd_search_sz{cfg.img_size}_bs{cfg.batch_size}_aug{int(cfg.use_augmentation)}_amp{int(cfg.amp)}"
    search_root = _make_unique_experiment_dir(Path(cfg.save_dir), search_name)
    print(f"[search-root] {search_root}")

    trials = []

    for lr in LR_SEARCH_CANDIDATES:
        for wd in WD_SEARCH_CANDIDATES:
            trial_cfg = copy.deepcopy(cfg)
            trial_cfg.tune_lr_wd = False
            trial_cfg.lr = float(lr)
            trial_cfg.weight_decay = float(wd)
            trial_cfg.save_dir = str(search_root)

            print(f"\n[lr/wd-search] lr={trial_cfg.lr} wd={trial_cfg.weight_decay}")
            trial_result = _run_experiment(trial_cfg, device)
            trial_summary = trial_result["summary"]

            trials.append(
                {
                    "lr": trial_cfg.lr,
                    "weight_decay": trial_cfg.weight_decay,
                    "run_dir": Path(trial_result["out_dir"]).name,
                    "test_acc_mean": float(trial_summary["test_acc"]["mean"]),
                    "test_acc_std": float(trial_summary["test_acc"]["std"]),
                    "test_loss_mean": float(trial_summary["test_loss"]["mean"]),
                    "test_loss_std": float(trial_summary["test_loss"]["std"]),
                }
            )

    if not trials:
        raise RuntimeError("lr/wd search produced no trials")

    best_trial = max(trials, key=lambda t: (t["test_acc_mean"], -t["test_loss_mean"]))

    search_payload = {
        "mode": "lr_wd_search",
        "search_root": str(search_root),
        "num_trials": len(trials),
        "lr_candidates": LR_SEARCH_CANDIDATES,
        "weight_decay_candidates": WD_SEARCH_CANDIDATES,
        "ranking": sorted(trials, key=lambda t: (t["test_acc_mean"], -t["test_loss_mean"]), reverse=True),
        "best": best_trial,
        "duration_sec": float(time.perf_counter() - search_start),
    }

    summary_path = search_root / "lr_wd_search_summary.json"
    save_json(summary_path, search_payload)

    print("\n[lr/wd-search][best]")
    print(
        f"lr={best_trial['lr']} wd={best_trial['weight_decay']} "
        f"acc_mean={best_trial['test_acc_mean']:.4f} loss_mean={best_trial['test_loss_mean']:.4f}"
    )
    print(f"[saved] {summary_path}")


def main():
    cfg = get_config()
    device = _select_device(cfg.cpu)

    if cfg.tune_lr_wd:
        _run_lr_wd_search(cfg, device)
        return

    _run_experiment(cfg, device)


if __name__ == "__main__":
    main()
