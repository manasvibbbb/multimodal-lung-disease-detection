
import hashlib
from pathlib import Path
from collections import defaultdict

IMAGE_EXTENSIONS = {
    ".jpg", ".jpeg", ".png", ".bmp", ".webp"
}

def get_hash(file_path):
    sha256 = hashlib.sha256()

    with open(file_path, "rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            sha256.update(chunk)

    return sha256.hexdigest()


def check_duplicates(dataset_name, dataset_path):
    dataset_path = Path(dataset_path)

    print(f"\n{'=' * 60}")
    print(f"{dataset_name} DUPLICATE CHECK")
    print(f"{'=' * 60}")

    if not dataset_path.exists():
        print(f"ERROR: Folder not found: {dataset_path.resolve()}")
        return

    hashes = defaultdict(list)

    # Read images from train, val and test
    for split in ["train", "val", "test"]:
        split_path = dataset_path / split

        if not split_path.exists():
            print(f"WARNING: Missing folder: {split_path}")
            continue

        for image_path in split_path.rglob("*"):
            if image_path.is_file() and image_path.suffix.lower() in IMAGE_EXTENSIONS:
                image_hash = get_hash(image_path)
                hashes[image_hash].append((split, image_path))

    # Find identical files appearing in different splits
    duplicates = {
        h: files
        for h, files in hashes.items()
        if len({split for split, _ in files}) > 1
    }

    print(f"Unique image hashes: {len(hashes)}")
    print(f"Duplicate groups across splits: {len(duplicates)}")

    if duplicates:
        print("\nDUPLICATES FOUND:")

        for files in duplicates.values():
            print("\nSame image content:")
            for split, path in files:
                print(f"  [{split}] {path}")
    else:
        print("No exact duplicate images found across splits.")


# Run checks using your project folder structure
check_duplicates("CXR", "data/xray")
check_duplicates("CT", "data/ct")
