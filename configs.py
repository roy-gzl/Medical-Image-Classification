from dataclasses import dataclass
import argparse


@dataclass
class Config:
    # Dataset
    dataset_dir: str = "Dataset"
    img_size: int = 224
    num_classes: int = 2

    # Model
    model_name: str = "resnet18"
    pretrained: bool = True

    # Training
    batch_size: int = 64
    epochs: int = 20
    lr: float = 1e-4
    weight_decay: float = 1e-4
    optimizer: str = "adam"
    early_stop_patience: int = 5
    use_scheduler: bool = True

    # AMP / GPU
    amp: bool = True
    grad_clip_norm: float = 0.0

    # Augmentation
    use_augmentation: bool = False
    rotation_deg: int = 10

    # System
    num_workers: int = 4
    seed: int = 42
    num_seeds: int = 5
    cpu: bool = False

    # Output
    save_dir: str = "runs"
    save_test_preds: bool = False
    save_case_images: bool = True
    max_correct_images: int = 30

    # Mode
    tune_lr_wd: bool = False


def get_config() -> Config:
    # Parse CLI flags and build the runtime config object.
    p = argparse.ArgumentParser()

    p.add_argument("--dataset_dir", type=str, default="Dataset")
    p.add_argument("--img_size", type=int, default=224)

    p.add_argument("--model_name", type=str, default="resnet18", choices=["resnet18", "vgg16", "efficientnet_b0"])
    p.add_argument("--pretrained", action="store_true")
    p.add_argument("--no_pretrained", action="store_true")

    p.add_argument("--batch_size", type=int, default=64)
    p.add_argument("--epochs", type=int, default=20)
    p.add_argument("--lr", type=float, default=1e-4)
    p.add_argument("--weight_decay", type=float, default=1e-4)
    p.add_argument("--optimizer", type=str, default="adam", choices=["adam", "sgd"])

    p.add_argument("--early_stop_patience", type=int, default=5)
    p.add_argument("--use_scheduler", action="store_true")
    p.add_argument("--no_scheduler", action="store_true")

    p.add_argument("--amp", action="store_true")
    p.add_argument("--no_amp", action="store_true")
    p.add_argument("--grad_clip_norm", type=float, default=0.0)

    p.add_argument("--use_augmentation", action="store_true")
    p.add_argument("--no_augmentation", action="store_true")
    p.add_argument("--rotation_deg", type=int, default=10)

    p.add_argument("--num_workers", type=int, default=4)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--num_seeds", type=int, default=5)
    p.add_argument("--cpu", action="store_true")

    p.add_argument("--save_dir", type=str, default="runs")
    p.add_argument("--save_test_preds", action="store_true")
    p.add_argument("--save_case_images", action="store_true")
    p.add_argument("--no_case_images", action="store_true")
    p.add_argument("--max_correct_images", type=int, default=30)

    p.add_argument("--tune_lr_wd", action="store_true")
    p.add_argument("--no_tune_lr_wd", action="store_true")

    args = p.parse_args()

    # Defaults are explicitly set here, then overridden by toggle flags below.
    cfg = Config(
        dataset_dir=args.dataset_dir,
        img_size=args.img_size,
        model_name=args.model_name,
        pretrained=True,
        batch_size=args.batch_size,
        epochs=args.epochs,
        lr=args.lr,
        weight_decay=args.weight_decay,
        optimizer=args.optimizer,
        early_stop_patience=args.early_stop_patience,
        use_scheduler=True,
        amp=True,
        grad_clip_norm=args.grad_clip_norm,
        use_augmentation=True,
        rotation_deg=args.rotation_deg,
        num_workers=args.num_workers,
        seed=args.seed,
        num_seeds=max(1, args.num_seeds),
        cpu=args.cpu,
        save_dir=args.save_dir,
        save_test_preds=args.save_test_preds,
        save_case_images=True,
        max_correct_images=max(0, args.max_correct_images),
        tune_lr_wd=False,
    )

    if args.no_pretrained:
        cfg.pretrained = False
    if args.pretrained:
        cfg.pretrained = True

    if args.no_scheduler:
        cfg.use_scheduler = False
    if args.use_scheduler:
        cfg.use_scheduler = True

    if args.no_amp:
        cfg.amp = False
    if args.amp:
        cfg.amp = True

    if args.no_augmentation:
        cfg.use_augmentation = False
    if args.use_augmentation:
        cfg.use_augmentation = True

    if args.no_case_images:
        cfg.save_case_images = False
    if args.save_case_images:
        cfg.save_case_images = True

    if args.no_tune_lr_wd:
        cfg.tune_lr_wd = False
    if args.tune_lr_wd:
        cfg.tune_lr_wd = True

    return cfg


def make_experiment_name(cfg: Config) -> str:
    # Build folder name used under `runs/` for one experiment setting.
    aug = 1 if cfg.use_augmentation else 0
    amp = 1 if cfg.amp else 0
    return f"{cfg.model_name}_sz{cfg.img_size}_bs{cfg.batch_size}_lr{cfg.lr}_wd{cfg.weight_decay}_aug{aug}_amp{amp}"
