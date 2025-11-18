#!/usr/bin/env python3
"""
Dataset Download Script for Hindi and Punjabi ASR
Downloads Common Voice and OpenSLR datasets
"""

import argparse
import os
import sys
from pathlib import Path
import requests
from tqdm import tqdm
import tarfile
import zipfile
import json

# Dataset URLs
DATASETS = {
    'hindi': {
        'common_voice': {
            'url': 'https://commonvoice.mozilla.org/datasets',
            'version': 'cv-corpus-15.0-2023-09-08',
            'note': 'Requires manual download from Common Voice website'
        },
        'openslr': {
            'url': 'https://www.openslr.org/resources/103/hindi.zip',
            'name': 'hindi.zip'
        },
        'microsoft': {
            'url': 'https://www.openslr.org/resources/105/microsoft_hindi.zip',
            'name': 'microsoft_hindi.zip'
        }
    },
    'punjabi': {
        'common_voice': {
            'url': 'https://commonvoice.mozilla.org/datasets',
            'version': 'cv-corpus-15.0-2023-09-08',
            'note': 'Requires manual download from Common Voice website'
        },
        'openslr': {
            'url': 'https://www.openslr.org/resources/78/pa_in_female.zip',
            'name': 'punjabi_female.zip'
        }
    }
}


def download_file(url: str, output_path: Path, description: str = "Downloading"):
    """Download file with progress bar"""
    try:
        response = requests.get(url, stream=True)
        response.raise_for_status()

        total_size = int(response.headers.get('content-length', 0))

        output_path.parent.mkdir(parents=True, exist_ok=True)

        with open(output_path, 'wb') as f:
            with tqdm(total=total_size, unit='B', unit_scale=True, desc=description) as pbar:
                for chunk in response.iter_content(chunk_size=8192):
                    if chunk:
                        f.write(chunk)
                        pbar.update(len(chunk))

        return True
    except Exception as e:
        print(f"Error downloading {url}: {e}")
        return False


def extract_archive(archive_path: Path, extract_to: Path):
    """Extract tar.gz or zip archive"""
    print(f"Extracting {archive_path.name}...")

    try:
        if archive_path.suffix == '.zip':
            with zipfile.ZipFile(archive_path, 'r') as zip_ref:
                zip_ref.extractall(extract_to)
        elif archive_path.suffix == '.gz' or '.tar' in archive_path.suffixes:
            with tarfile.open(archive_path, 'r:*') as tar_ref:
                tar_ref.extractall(extract_to)
        else:
            print(f"Unknown archive format: {archive_path.suffix}")
            return False

        print(f"Extracted to {extract_to}")
        return True
    except Exception as e:
        print(f"Error extracting {archive_path}: {e}")
        return False


def download_openslr(language: str, output_dir: Path):
    """Download OpenSLR datasets"""
    print(f"\n{'='*60}")
    print(f"Downloading OpenSLR datasets for {language.upper()}")
    print(f"{'='*60}\n")

    datasets = DATASETS.get(language, {})

    for dataset_name, dataset_info in datasets.items():
        if dataset_name == 'common_voice':
            continue  # Handle separately

        print(f"\nDataset: {dataset_name}")
        print(f"URL: {dataset_info['url']}")

        filename = dataset_info['name']
        output_path = output_dir / 'raw' / language / filename

        if output_path.exists():
            print(f"File already exists: {output_path}")
            user_input = input("Re-download? (y/n): ")
            if user_input.lower() != 'y':
                continue

        success = download_file(
            dataset_info['url'],
            output_path,
            description=f"Downloading {dataset_name}"
        )

        if success:
            extract_dir = output_dir / 'raw' / language / dataset_name
            extract_archive(output_path, extract_dir)


def download_common_voice_instructions(language: str):
    """Print instructions for downloading Common Voice"""
    print(f"\n{'='*60}")
    print(f"Common Voice {language.upper()} - Manual Download Required")
    print(f"{'='*60}\n")

    info = DATASETS[language]['common_voice']

    print("Common Voice requires account registration for download.")
    print("\nSteps:")
    print("1. Go to: https://commonvoice.mozilla.org/datasets")
    print("2. Create an account or log in")
    print(f"3. Download the {language.upper()} dataset")
    print(f"4. Extract the archive to: data/raw/{language}/common_voice/")
    print("\nThe dataset structure should be:")
    print(f"  data/raw/{language}/common_voice/")
    print("    ├── clips/           (audio files)")
    print("    ├── train.tsv")
    print("    ├── dev.tsv")
    print("    ├── test.tsv")
    print("    └── validated.tsv")

    print("\nOnce downloaded, run:")
    print(f"  python data/preprocess.py --language {language}")


def verify_downloads(language: str, output_dir: Path):
    """Verify that datasets were downloaded successfully"""
    print(f"\n{'='*60}")
    print(f"Verifying downloads for {language.upper()}")
    print(f"{'='*60}\n")

    raw_dir = output_dir / 'raw' / language

    if not raw_dir.exists():
        print(f"❌ Raw data directory not found: {raw_dir}")
        return False

    # Check Common Voice
    cv_dir = raw_dir / 'common_voice'
    if cv_dir.exists() and (cv_dir / 'clips').exists():
        num_clips = len(list((cv_dir / 'clips').glob('*.mp3')))
        print(f"✓ Common Voice: {num_clips} audio files found")
    else:
        print(f"⚠ Common Voice: Not found or incomplete")
        print(f"  Expected at: {cv_dir}")

    # Check OpenSLR datasets
    for dataset_name in DATASETS[language].keys():
        if dataset_name == 'common_voice':
            continue

        dataset_dir = raw_dir / dataset_name
        if dataset_dir.exists():
            audio_files = list(dataset_dir.glob('**/*.wav')) + list(dataset_dir.glob('**/*.flac'))
            print(f"✓ {dataset_name}: {len(audio_files)} audio files found")
        else:
            print(f"⚠ {dataset_name}: Not found")

    print(f"\nRaw data location: {raw_dir}")
    return True


def create_download_summary(language: str, output_dir: Path):
    """Create a summary of downloaded datasets"""
    summary = {
        'language': language,
        'datasets': {},
        'total_hours_estimated': 0
    }

    raw_dir = output_dir / 'raw' / language

    # Common Voice
    cv_dir = raw_dir / 'common_voice'
    if cv_dir.exists():
        summary['datasets']['common_voice'] = {
            'path': str(cv_dir),
            'status': 'available' if (cv_dir / 'clips').exists() else 'incomplete'
        }

    # OpenSLR
    for dataset_name in DATASETS[language].keys():
        if dataset_name == 'common_voice':
            continue
        dataset_dir = raw_dir / dataset_name
        if dataset_dir.exists():
            summary['datasets'][dataset_name] = {
                'path': str(dataset_dir),
                'status': 'available'
            }

    summary_path = output_dir / f'{language}_download_summary.json'
    with open(summary_path, 'w') as f:
        json.dump(summary, f, indent=2)

    print(f"\nSummary saved to: {summary_path}")


def main():
    parser = argparse.ArgumentParser(description='Download ASR datasets for Hindi and Punjabi')
    parser.add_argument('--language', type=str, required=True, choices=['hindi', 'punjabi'],
                        help='Language to download datasets for')
    parser.add_argument('--output_dir', type=Path, default=Path('data'),
                        help='Output directory for datasets')
    parser.add_argument('--skip_openslr', action='store_true',
                        help='Skip OpenSLR downloads')
    parser.add_argument('--verify_only', action='store_true',
                        help='Only verify existing downloads')

    args = parser.parse_args()

    print(f"""
╔════════════════════════════════════════════════════════════╗
║      ASR Dataset Downloader - {args.language.upper():<8}                   ║
╚════════════════════════════════════════════════════════════╝
    """)

    # Create output directory
    args.output_dir.mkdir(parents=True, exist_ok=True)

    if args.verify_only:
        verify_downloads(args.language, args.output_dir)
        create_download_summary(args.language, args.output_dir)
        return

    # Download OpenSLR datasets
    if not args.skip_openslr:
        download_openslr(args.language, args.output_dir)

    # Print Common Voice instructions
    download_common_voice_instructions(args.language)

    # Verify downloads
    verify_downloads(args.language, args.output_dir)

    # Create summary
    create_download_summary(args.language, args.output_dir)

    print(f"\n{'='*60}")
    print("Next Steps:")
    print(f"{'='*60}")
    print("1. Manually download Common Voice dataset (see instructions above)")
    print("2. Verify all datasets are in place:")
    print(f"   python data/download_datasets.py --language {args.language} --verify_only")
    print("3. Preprocess the data:")
    print(f"   python data/preprocess.py --language {args.language}")


if __name__ == '__main__':
    main()
