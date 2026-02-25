"""Training/evaluation helpers and report artifact generation."""

import copy
import shutil
from pathlib import Path

import torch
import torch.nn as nn
import torch.optim as optim


def build_optimizer(cfg, model):
    # Create optimizer from config.
    if cfg.optimizer.lower() == "adam":
        return optim.Adam(model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)
    if cfg.optimizer.lower() == "sgd":
        return optim.SGD(model.parameters(), lr=cfg.lr, momentum=0.9, weight_decay=cfg.weight_decay)
    raise ValueError("optimizer must be 'adam' or 'sgd'")


def _to_device(x, y, device):
    # Move a mini-batch to the selected device with non_blocking transfer.
    return x.to(device, non_blocking=True), y.to(device, non_blocking=True)


def train_one_epoch(model, loader, optimizer, device, scaler, amp: bool, grad_clip_norm: float):
    # Run one training epoch and return average loss/accuracy.
    model.train()
    criterion = nn.CrossEntropyLoss()

    total_loss = 0.0
    total_correct = 0
    total = 0

    for x, y in loader:
        x, y = _to_device(x, y, device)

        optimizer.zero_grad(set_to_none=True)

        with torch.cuda.amp.autocast(enabled=amp and device.type == "cuda"):
            logits = model(x)
            loss = criterion(logits, y)

        if scaler is not None and amp and device.type == "cuda":
            scaler.scale(loss).backward()

            if grad_clip_norm and grad_clip_norm > 0:
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip_norm)

            scaler.step(optimizer)
            scaler.update()
        else:
            loss.backward()
            if grad_clip_norm and grad_clip_norm > 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip_norm)
            optimizer.step()

        bs = x.size(0)
        total_loss += loss.item() * bs
        total_correct += (logits.argmax(dim=1) == y).sum().item()
        total += bs

    return total_loss / max(total, 1), total_correct / max(total, 1)


@torch.no_grad()
def evaluate(model, loader, device, amp: bool):
    # Evaluate model on validation/test loader.
    model.eval()
    criterion = nn.CrossEntropyLoss()

    total_loss = 0.0
    total_correct = 0
    total = 0

    for x, y in loader:
        x, y = _to_device(x, y, device)

        with torch.cuda.amp.autocast(enabled=amp and device.type == "cuda"):
            logits = model(x)
            loss = criterion(logits, y)

        bs = x.size(0)
        total_loss += loss.item() * bs
        total_correct += (logits.argmax(dim=1) == y).sum().item()
        total += bs

    return total_loss / max(total, 1), total_correct / max(total, 1)


@torch.no_grad()
def predict_labels(model, loader, device, amp: bool):
    # Return all true/predicted labels in loader order.
    model.eval()
    y_true = []
    y_pred = []

    for x, y in loader:
        x, y = _to_device(x, y, device)

        with torch.cuda.amp.autocast(enabled=amp and device.type == "cuda"):
            logits = model(x)

        y_true.extend(y.cpu().tolist())
        y_pred.extend(logits.argmax(dim=1).cpu().tolist())

    return y_true, y_pred


def build_confusion_matrix(y_true, y_pred, num_classes: int):
    # Build confusion matrix where rows=true class, cols=predicted class.
    conf_mat = torch.zeros((num_classes, num_classes), dtype=torch.int64)
    for yt, yp in zip(y_true, y_pred):
        conf_mat[int(yt), int(yp)] += 1
    return conf_mat


def binary_confusion_counts(conf_mat, positive_class_idx: int = 1):
    # Extract TP/FP/FN/TN from a 2x2 confusion matrix.
    if conf_mat.shape != (2, 2):
        raise ValueError("binary_confusion_counts requires a 2x2 confusion matrix")
    if positive_class_idx not in (0, 1):
        raise ValueError("positive_class_idx must be 0 or 1 for binary classification")

    negative_class_idx = 1 - positive_class_idx

    tp = int(conf_mat[positive_class_idx, positive_class_idx].item())
    fp = int(conf_mat[negative_class_idx, positive_class_idx].item())
    fn = int(conf_mat[positive_class_idx, negative_class_idx].item())
    tn = int(conf_mat[negative_class_idx, negative_class_idx].item())
    return {"tp": tp, "fp": fp, "fn": fn, "tn": tn}


def save_confusion_matrix_figure(
    conf_mat,
    out_png_path,
    out_pdf_path,
    class_names=None,
    positive_class_idx: int = 1,
):
    # Render confusion matrix as PNG and PDF for report insertion.
    from PIL import Image, ImageDraw, ImageFont

    conf_mat = conf_mat.cpu().float()
    rows, cols = conf_mat.shape
    if rows != cols:
        raise ValueError("confusion matrix must be square")

    if class_names is None:
        class_names = [str(i) for i in range(rows)]
    if len(class_names) != rows:
        raise ValueError("class_names length must match confusion matrix size")

    width, height = 1200, 900
    margin_left, margin_top = 260, 180
    cell_size = 240
    matrix_size = rows * cell_size
    matrix_bottom = margin_top + matrix_size

    image = Image.new("RGB", (width, height), (255, 255, 255))
    draw = ImageDraw.Draw(image)
    font = ImageFont.load_default()

    draw.text((40, 40), "Confusion Matrix", fill=(0, 0, 0), font=font)
    draw.text((margin_left, 140), "Predicted Label", fill=(0, 0, 0), font=font)
    draw.text((70, margin_top - 30), "True Label", fill=(0, 0, 0), font=font)

    max_count = float(conf_mat.max().item()) if conf_mat.numel() > 0 else 1.0
    max_count = max(max_count, 1.0)

    for r in range(rows):
        for c in range(cols):
            value = float(conf_mat[r, c].item())
            intensity = int(235 - 160 * (value / max_count))
            color = (intensity, intensity, 255)

            x0 = margin_left + c * cell_size
            y0 = margin_top + r * cell_size
            x1 = x0 + cell_size
            y1 = y0 + cell_size

            draw.rectangle([x0, y0, x1, y1], fill=color, outline=(20, 20, 20), width=2)

            label = f"count\n{value:.2f}"
            if rows == 2:
                if positive_class_idx not in (0, 1):
                    raise ValueError("positive_class_idx must be 0 or 1 for binary classification")
                negative_class_idx = 1 - positive_class_idx

                if r == positive_class_idx and c == positive_class_idx:
                    label = f"TP\n{value:.2f}"
                elif r == negative_class_idx and c == positive_class_idx:
                    label = f"FP\n{value:.2f}"
                elif r == positive_class_idx and c == negative_class_idx:
                    label = f"FN\n{value:.2f}"
                elif r == negative_class_idx and c == negative_class_idx:
                    label = f"TN\n{value:.2f}"

            draw.multiline_text((x0 + 10, y0 + 10), label, fill=(0, 0, 0), font=font, spacing=4)

    for i, class_name in enumerate(class_names):
        draw.text((margin_left + i * cell_size + 8, matrix_bottom + 12), str(class_name), fill=(0, 0, 0), font=font)
        draw.text((margin_left - 45, margin_top + i * cell_size + 8), str(class_name), fill=(0, 0, 0), font=font)

    if rows == 2:
        if positive_class_idx not in (0, 1):
            raise ValueError("positive_class_idx must be 0 or 1 for binary classification")
        negative_class_idx = 1 - positive_class_idx
        tp = float(conf_mat[positive_class_idx, positive_class_idx].item())
        fp = float(conf_mat[negative_class_idx, positive_class_idx].item())
        fn = float(conf_mat[positive_class_idx, negative_class_idx].item())
        tn = float(conf_mat[negative_class_idx, negative_class_idx].item())

        summary_y = matrix_bottom + 55
        draw.text(
            (margin_left, summary_y),
            f"TP: {tp:.2f}   FP: {fp:.2f}   FN: {fn:.2f}   TN: {tn:.2f}",
            fill=(0, 0, 0),
            font=font,
        )
        draw.text(
            (margin_left, summary_y + 20),
            f"Positive class index: {positive_class_idx} ({class_names[positive_class_idx]})",
            fill=(0, 0, 0),
            font=font,
        )

    out_png_path = Path(out_png_path)
    out_pdf_path = Path(out_pdf_path)
    image.save(out_png_path)
    image.convert("RGB").save(out_pdf_path, "PDF", resolution=300.0)


def _safe_label(text: str) -> str:
    # Sanitize class names for filesystem-safe file names.
    return str(text).replace("/", "_").replace("\\", "_").replace(" ", "_")


def save_case_images(samples, y_true, y_pred, idx_to_class, output_root, exp_name: str, max_correct_images: int = 30):
    # Save TP/TN/FP/FN (and MIS for non-binary errors) for qualitative analysis.
    if not (len(samples) == len(y_true) == len(y_pred)):
        raise RuntimeError(
            f"length mismatch: samples={len(samples)} y_true={len(y_true)} y_pred={len(y_pred)}"
        )

    output_root = Path(output_root)
    tp_dir = output_root / "TP"
    tn_dir = output_root / "TN"
    fp_dir = output_root / "FP"
    fn_dir = output_root / "FN"
    mis_dir = output_root / "MIS"

    tp_dir.mkdir(parents=True, exist_ok=True)
    tn_dir.mkdir(parents=True, exist_ok=True)
    fp_dir.mkdir(parents=True, exist_ok=True)
    fn_dir.mkdir(parents=True, exist_ok=True)
    mis_dir.mkdir(parents=True, exist_ok=True)

    counts = {"TP": 0, "TN": 0, "FP": 0, "FN": 0, "MIS": 0}
    max_correct_images = max(0, int(max_correct_images))

    for i, ((img_path, _sample_y), yt, yp) in enumerate(zip(samples, y_true, y_pred)):
        yt = int(yt)
        yp = int(yp)

        true_name = _safe_label(idx_to_class.get(yt, str(yt)))
        pred_name = _safe_label(idx_to_class.get(yp, str(yp)))
        src_path = Path(img_path)

        if yt == yp:
            if yt == 1:
                if counts["TP"] >= max_correct_images:
                    continue
                dst_dir = tp_dir
                case = "TP"
                counts["TP"] += 1
            elif yt == 0:
                if counts["TN"] >= max_correct_images:
                    continue
                dst_dir = tn_dir
                case = "TN"
                counts["TN"] += 1
            else:
                # For multi-class equal predictions, store under MIS-free positive bucket.
                if counts["TP"] >= max_correct_images:
                    continue
                dst_dir = tp_dir
                case = "TP"
                counts["TP"] += 1
        else:
            if yt == 0 and yp == 1:
                dst_dir = fp_dir
                case = "FP"
                counts["FP"] += 1
            elif yt == 1 and yp == 0:
                dst_dir = fn_dir
                case = "FN"
                counts["FN"] += 1
            else:
                dst_dir = mis_dir
                case = "MIS"
                counts["MIS"] += 1

        filename = f"{exp_name}_{case}_{i:05d}_true-{true_name}_pred-{pred_name}_{src_path.name}"
        shutil.copy2(src_path, dst_dir / filename)

    return {
        "counts": counts,
        "output_root": str(output_root),
        "tp_dir": str(tp_dir),
        "tn_dir": str(tn_dir),
        "fp_dir": str(fp_dir),
        "fn_dir": str(fn_dir),
    }


def train_loop(cfg, model, loaders, device, best_ckpt_path):
    # Train with validation monitoring, scheduler, and early stopping.
    optimizer = build_optimizer(cfg, model)

    scheduler = None
    if cfg.use_scheduler:
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode="max", factor=0.5, patience=2, verbose=True
        )

    scaler = torch.cuda.amp.GradScaler(enabled=(cfg.amp and device.type == "cuda"))

    best_acc = -1.0
    best_wts = copy.deepcopy(model.state_dict())
    no_improve = 0

    for epoch in range(1, cfg.epochs + 1):
        tr_loss, tr_acc = train_one_epoch(
            model, loaders["train"], optimizer, device,
            scaler=scaler, amp=cfg.amp, grad_clip_norm=cfg.grad_clip_norm
        )
        va_loss, va_acc = evaluate(model, loaders["val"], device, amp=cfg.amp)

        if scheduler is not None:
            scheduler.step(va_acc)

        print(
            f"[epoch {epoch:03d}] "
            f"train loss={tr_loss:.4f} acc={tr_acc:.4f} | "
            f"val loss={va_loss:.4f} acc={va_acc:.4f}"
        )

        if va_acc > best_acc:
            best_acc = va_acc
            best_wts = copy.deepcopy(model.state_dict())
            torch.save({"model": best_wts, "best_val_acc": float(best_acc), "cfg": cfg.__dict__}, str(best_ckpt_path))
            print(f"  -> best updated: val_acc={best_acc:.4f} saved={best_ckpt_path}")
            no_improve = 0
        else:
            no_improve += 1

        if cfg.early_stop_patience > 0 and no_improve >= cfg.early_stop_patience:
            print(f"[early stop] no improvement for {cfg.early_stop_patience} epochs")
            break

    model.load_state_dict(best_wts)
    return model


@torch.no_grad()
def predict_to_csv(model, test_loader, device, samples, idx_to_class, out_csv_path, amp: bool):
    # Save per-sample test predictions to CSV.
    import csv

    model.eval()

    preds = []
    for x, _y in test_loader:
        x = x.to(device, non_blocking=True)
        with torch.cuda.amp.autocast(enabled=amp and device.type == "cuda"):
            logits = model(x)
        preds.extend(logits.argmax(dim=1).cpu().tolist())

    if len(preds) != len(samples):
        raise RuntimeError(f"pred count mismatch: preds={len(preds)} samples={len(samples)}")

    rows = [("path", "y_true", "y_pred")]
    for (path, y_true), y_pred in zip(samples, preds):
        rows.append((path, idx_to_class[int(y_true)], idx_to_class[int(y_pred)]))

    with open(out_csv_path, "w", newline="", encoding="utf-8") as f:
        csv.writer(f).writerows(rows)

