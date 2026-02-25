import statistics
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


def _select_device(cpu: bool):
    if (not cpu) and torch.cuda.is_available():
        device = torch.device("cuda")
        torch.backends.cudnn.benchmark = True
        print(f"[device] cuda: {torch.cuda.get_device_name(0)}")
        return device

    device = torch.device("cpu")
    print("[device] cpu")
    return device


def _run_single_seed(cfg, seed: int, base_out_dir: Path, device):
    set_seed(seed)

    seed_out_dir = base_out_dir / f"seed_{seed}"
    seed_out_dir.mkdir(parents=True, exist_ok=True)

    cfg_dump = dict(cfg.__dict__)
    cfg_dump["seed"] = seed
    save_json(seed_out_dir / "config.json", cfg_dump)

    loaders, class_to_idx, samples_test = prepare_datasets_and_loaders(cfg)
    model = build_model(cfg.model_name, num_classes=cfg.num_classes, pretrained=cfg.pretrained).to(device)

    best_ckpt_path = seed_out_dir / "best.pt"
    model = train_loop(cfg, model, loaders, device, best_ckpt_path)

    test_loss, test_acc = evaluate(model, loaders["test"], device, amp=cfg.amp)
    print(f"[test][seed={seed}] loss={test_loss:.4f} acc={test_acc:.4f}")

    y_true, y_pred = predict_labels(model, loaders["test"], device, amp=cfg.amp)
    conf_mat = build_confusion_matrix(y_true, y_pred, num_classes=cfg.num_classes)

    idx_to_class = {v: k for k, v in class_to_idx.items()}
    class_names = [idx_to_class[i] for i in range(cfg.num_classes)]

    cm_png = seed_out_dir / "confusion_matrix.png"
    cm_pdf = seed_out_dir / "confusion_matrix.pdf"
    save_confusion_matrix_figure(
        conf_mat,
        out_png_path=cm_png,
        out_pdf_path=cm_pdf,
        class_names=class_names,
        positive_class_idx=1,
    )

    result = {
        "seed": seed,
        "test_loss": float(test_loss),
        "test_acc": float(test_acc),
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

    save_json(seed_out_dir / "confusion_matrix.json", cm_payload)
    print(f"[saved] {cm_png}")
    print(f"[saved] {cm_pdf}")
    print(f"[saved] {seed_out_dir / 'confusion_matrix.json'}")

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
        print(f"[saved] FP images: {cases['counts']['FP']} -> {cases['fp_dir']}")
        print(f"[saved] FN images: {cases['counts']['FN']} -> {cases['fn_dir']}")
        print(f"[saved] CORRECT images: {cases['counts']['CORRECT']} -> {cases['correct_dir']}")

    if cfg.save_test_preds:
        out_csv = seed_out_dir / f"test_preds_{cfg.model_name}.csv"
        predict_to_csv(model, loaders["test"], device, samples_test, idx_to_class, out_csv, amp=cfg.amp)
        print(f"[saved] {out_csv}")

    return result


def _mean_std(values):
    if not values:
        return {"mean": 0.0, "std": 0.0}
    if len(values) == 1:
        return {"mean": float(values[0]), "std": 0.0}
    return {"mean": float(statistics.mean(values)), "std": float(statistics.stdev(values))}


def main():
    cfg = get_config()
    device = _select_device(cfg.cpu)

    exp_name = make_experiment_name(cfg)
    out_dir = Path(cfg.save_dir) / exp_name
    out_dir.mkdir(parents=True, exist_ok=True)

    seeds = [cfg.seed + i for i in range(cfg.num_seeds)]
    print(f"[multi-seed] seeds={seeds}")

    all_results = []
    for idx, seed in enumerate(seeds, start=1):
        print(f"\n[multi-seed] run {idx}/{len(seeds)} seed={seed}")
        all_results.append(_run_single_seed(cfg, seed, out_dir, device))

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

    summary_payload = {
        "experiment": exp_name,
        "runs": all_results,
        "summary": summary,
    }
    summary_path = out_dir / "multi_seed_summary.json"
    save_json(summary_path, summary_payload)

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
    print(f"[saved] {summary_path}")


if __name__ == "__main__":
    main()
