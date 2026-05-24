import argparse
import json
from pathlib import Path
from datetime import datetime
import requests
from tqdm import tqdm


VALID_CATEGORIES = ["dress", "shirt", "toptee"]
METADATA_DIR = Path("fashion-iq-metadata/image_url")
FASHIONIQ_DIR = Path("fashion-iq")
OUTPUT_DIR = Path("images")
LOGS_DIR = Path("logs")
DOWNLOAD_TIMEOUT = 10


def load_split_asins(category: str, split: str) -> set[str]:
    split_file = FASHIONIQ_DIR / "image_splits" / f"split.{category}.{split}.json"
    with open(split_file, "r") as f:
        asins = json.load(f)
    return set(asins)


def load_asin_mapping(category: str) -> dict[str, str]:
    mapping_file = METADATA_DIR / f"asin2url.{category}.txt"
    asin_to_url = {}

    with open(mapping_file, "r") as f:
        for line in f:
            parts = line.strip().split("\t")
            if len(parts) == 2:
                asin = parts[0].strip()
                url = parts[1].strip()
                asin_to_url[asin] = url

    return asin_to_url


def download_image(asin: str, url: str, save_path: Path) -> bool:
    try:
        response = requests.get(url, timeout=DOWNLOAD_TIMEOUT)
        response.raise_for_status()

        with open(save_path, "wb") as f:
            f.write(response.content)

        return True

    except requests.exceptions.Timeout:
        print(f"Failed to download {asin}: Connection timeout ({url})")
        return False
    except requests.exceptions.HTTPError as e:
        print(f"Failed to download {asin}: HTTP {e.response.status_code} ({url})")
        return False
    except Exception as e:
        print(f"Failed to download {asin}: {str(e)} ({url})")
        return False


def main(category: str):
    if category not in VALID_CATEGORIES:
        raise ValueError(f"Category must be one of {VALID_CATEGORIES}, got '{category}'")

    # Create output directory
    output_dir = OUTPUT_DIR / category
    output_dir.mkdir(parents=True, exist_ok=True)

    # Create logs directory
    LOGS_DIR.mkdir(parents=True, exist_ok=True)

    # Load val split ASINs
    print(f"Loading val split ASINs for category '{category}'...")
    val_asins = load_split_asins(category, "val")
    print(f"Found {len(val_asins)} images in val split")

    # Load ASIN to URL mapping
    print(f"Loading ASIN to URL mapping...")
    asin_to_url = load_asin_mapping(category)

    # Filter to only val split
    asin_to_url = {asin: url for asin, url in asin_to_url.items() if asin in val_asins}
    total = len(asin_to_url)
    print(f"Found {total} images to download for val split")

    # Download images
    downloaded = 0
    skipped = 0
    failed = 0
    failed_asins = []

    for asin, url in tqdm(asin_to_url.items(), desc=f"Downloading {category} images"):
        save_path = output_dir / f"{asin}.jpg"

        # Skip if already exists
        if save_path.exists():
            skipped += 1
            continue

        # Download image
        success = download_image(asin, url, save_path)

        if success:
            downloaded += 1
        else:
            failed += 1
            failed_asins.append(asin)

    # Save statistics
    stats = {
        "category": category,
        "total": total,
        "downloaded": downloaded,
        "skipped": skipped,
        "failed": failed,
        "failed_asins": failed_asins,
        "timestamp": datetime.now().isoformat(),
    }

    stats_file = LOGS_DIR / f"download_stats_{category}.json"
    with open(stats_file, "w") as f:
        json.dump(stats, f, indent=2)

    # Print summary
    print(f"\nDownload completed:")
    print(f"  Total: {total}")
    print(f"  Downloaded: {downloaded}")
    print(f"  Skipped (already exists): {skipped}")
    print(f"  Failed: {failed}")
    print(f"  Statistics saved to: {stats_file}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Download FashionIQ images from Amazon servers"
    )
    parser.add_argument(
        "category",
        type=str,
        choices=VALID_CATEGORIES,
        help="Category to download (dress, shirt, or toptee)",
    )

    args = parser.parse_args()
    main(args.category)
