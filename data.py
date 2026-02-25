from pathlib import Path
import tarfile

from torch.utils.data import DataLoader
from torchvision import datasets, transforms


def extract_if_needed(dataset_dir: Path):
    # Extract train/val/test tarballs only when png files are absent.S
    for split in ["train", "val", "test"]:
        tar_path = dataset_dir / f"{split}.tar.gz"
        out_dir = dataset_dir / split

        if out_dir.exists() and any(out_dir.rglob("*.png")):
            print(f"[extract] {split}: already extracted -> {out_dir}")
            continue

        if not tar_path.exists():
            raise FileNotFoundError(f"Not found: {tar_path}")

        out_dir.mkdir(parents=True, exist_ok=True)
        print(f"[extract] extracting {tar_path} -> {out_dir}")
        with tarfile.open(tar_path, "r:gz") as tar:
            tar.extractall(path=out_dir)


def build_transforms(img_size: int, use_augmentation: bool, rotation_deg: int):
    # Create train/eval transform pipelines.
    imagenet_mean = (0.485, 0.456, 0.406)
    imagenet_std = (0.229, 0.224, 0.225)

    # Convert grayscale to 3 channels for ImageNet-pretrained backbones.
    common = [
        transforms.Grayscale(num_output_channels=3),
        transforms.Resize((img_size, img_size)),
    ]

    train_aug = []
    if use_augmentation:
        train_aug += [
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomRotation(degrees=rotation_deg),
        ]

    train_tf = transforms.Compose(common + train_aug + [
        transforms.ToTensor(),
        transforms.Normalize(imagenet_mean, imagenet_std),
    ])

    eval_tf = transforms.Compose(common + [
        transforms.ToTensor(),
        transforms.Normalize(imagenet_mean, imagenet_std),
    ])
    return train_tf, eval_tf


def prepare_datasets_and_loaders(cfg):
    # Build ImageFolder datasets and dataloaders for train/val/test.
    dataset_dir = Path(cfg.dataset_dir)
    extract_if_needed(dataset_dir)

    train_tf, eval_tf = build_transforms(cfg.img_size, cfg.use_augmentation, cfg.rotation_deg)

    train_ds = datasets.ImageFolder(root=str(dataset_dir / "train"), transform=train_tf)
    val_ds = datasets.ImageFolder(root=str(dataset_dir / "val"), transform=eval_tf)
    test_ds = datasets.ImageFolder(root=str(dataset_dir / "test"), transform=eval_tf)

    print("[classes]", train_ds.class_to_idx)

    # pin_memory is helpful when using CUDA; persistent_workers avoids worker restart each epoch.
    pin = True
    persistent = cfg.num_workers > 0

    loader_args = dict(
        batch_size=cfg.batch_size,
        num_workers=cfg.num_workers,
        pin_memory=pin,
        persistent_workers=persistent,
    )

    loaders = {
        "train": DataLoader(train_ds, shuffle=True, **loader_args),
        "val": DataLoader(val_ds, shuffle=False, **loader_args),
        "test": DataLoader(test_ds, shuffle=False, **loader_args),
    }

    # test_ds.samples is used later to map predictions back to original image paths.
    return loaders, train_ds.class_to_idx, test_ds.samples
