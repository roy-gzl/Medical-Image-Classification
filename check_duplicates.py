import hashlib
from pathlib import Path


def file_hash(path: Path, chunk_size=8192):
    md5 = hashlib.md5()
    with open(path, "rb") as f:
        while chunk := f.read(chunk_size):
            md5.update(chunk)
    return md5.hexdigest()


def collect_hashes(directory: Path):
    hashes = {}
    for img_path in directory.rglob("*.png"):
        h = file_hash(img_path)
        hashes[h] = img_path
    return hashes


def main():
    train_dir = Path("Dataset/train")
    test_dir = Path("Dataset/test")

    print("Collecting train hashes...")
    train_hashes = collect_hashes(train_dir)

    print("Collecting test hashes...")
    test_hashes = collect_hashes(test_dir)

    duplicates = []

    for h, test_path in test_hashes.items():
        if h in train_hashes:
            duplicates.append((train_hashes[h], test_path))

    if len(duplicates) == 0:
        print("✅ No duplicate images between train and test.")
    else:
        print(f"⚠ Found {len(duplicates)} duplicate images:")
        for train_img, test_img in duplicates:
            print("TRAIN:", train_img)
            print("TEST :", test_img)
            print("-" * 40)


if __name__ == "__main__":
    main()
