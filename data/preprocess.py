#!/usr/bin/env python3
"""
Data Preprocessing Pipeline for ASR
Handles audio normalization, quality filtering, and manifest creation
"""

import argparse
import json
import os
import re
from pathlib import Path
from typing import Dict, List, Tuple
import pandas as pd
import numpy as np
import librosa
import soundfile as sf
from tqdm import tqdm
from concurrent.futures import ProcessPoolExecutor, as_completed
import logging

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class AudioPreprocessor:
    """Handles audio preprocessing tasks"""

    def __init__(self, target_sr=16000, min_duration=1.0, max_duration=20.0, min_snr=15.0):
        self.target_sr = target_sr
        self.min_duration = min_duration
        self.max_duration = max_duration
        self.min_snr = min_snr

    def estimate_snr(self, audio: np.ndarray) -> float:
        """Estimate Signal-to-Noise Ratio (simple energy-based)"""
        # Simple SNR estimation based on signal energy distribution
        # More sophisticated methods would use voice activity detection
        if len(audio) == 0:
            return 0.0

        # Split audio into segments
        segment_length = int(0.1 * self.target_sr)  # 100ms segments
        num_segments = len(audio) // segment_length

        if num_segments < 2:
            return 100.0  # Too short to estimate, assume good

        energies = []
        for i in range(num_segments):
            segment = audio[i * segment_length:(i + 1) * segment_length]
            energy = np.sum(segment ** 2) / len(segment)
            energies.append(energy)

        energies = np.array(energies)
        signal_energy = np.percentile(energies, 90)  # Top 10% as signal
        noise_energy = np.percentile(energies, 10)   # Bottom 10% as noise

        if noise_energy < 1e-10:
            return 100.0  # Very low noise

        snr_linear = signal_energy / noise_energy
        snr_db = 10 * np.log10(snr_linear) if snr_linear > 0 else 0
        return snr_db

    def normalize_audio(self, audio: np.ndarray) -> np.ndarray:
        """Normalize audio to target loudness"""
        # Normalize to -23 LUFS (approximate with RMS)
        target_rms = 0.1
        current_rms = np.sqrt(np.mean(audio ** 2))

        if current_rms > 1e-6:
            audio = audio * (target_rms / current_rms)

        # Ensure no clipping
        max_val = np.abs(audio).max()
        if max_val > 0.95:
            audio = audio * (0.95 / max_val)

        return audio

    def process_audio_file(self, input_path: Path, output_path: Path) -> Dict:
        """Process a single audio file"""
        try:
            # Load audio
            audio, sr = librosa.load(input_path, sr=self.target_sr, mono=True)

            # Get duration
            duration = len(audio) / self.target_sr

            # Quality checks
            if duration < self.min_duration or duration > self.max_duration:
                return {
                    'success': False,
                    'reason': f'duration_{duration:.2f}s',
                    'duration': duration
                }

            # Check SNR
            snr = self.estimate_snr(audio)
            if snr < self.min_snr:
                return {
                    'success': False,
                    'reason': f'low_snr_{snr:.1f}dB',
                    'duration': duration,
                    'snr': snr
                }

            # Normalize
            audio = self.normalize_audio(audio)

            # Save processed audio
            output_path.parent.mkdir(parents=True, exist_ok=True)
            sf.write(output_path, audio, self.target_sr)

            return {
                'success': True,
                'duration': duration,
                'snr': snr,
                'output_path': str(output_path)
            }

        except Exception as e:
            return {
                'success': False,
                'reason': f'error_{str(e)}',
                'duration': 0
            }


class TextNormalizer:
    """Normalize transcription text"""

    def __init__(self, language='hindi'):
        self.language = language

    def normalize_hindi(self, text: str) -> str:
        """Normalize Hindi text"""
        # Convert to lowercase (for Devanagari, this has limited effect)
        text = text.strip()

        # Remove extra whitespace
        text = re.sub(r'\s+', ' ', text)

        # Remove special characters but keep Devanagari and basic punctuation
        # Keep: Devanagari (U+0900-U+097F), numbers, basic punctuation
        text = re.sub(r'[^\u0900-\u097F\u0980-\u09FF0-9\s.,!?\'-]', '', text)

        # Normalize numbers (optional - convert to words for better ASR)
        # For now, keep as is

        return text.strip()

    def normalize_punjabi(self, text: str) -> str:
        """Normalize Punjabi text (Gurmukhi script)"""
        text = text.strip()

        # Remove extra whitespace
        text = re.sub(r'\s+', ' ', text)

        # Keep Gurmukhi script (U+0A00-U+0A7F) and basic punctuation
        text = re.sub(r'[^\u0A00-\u0A7F0-9\s.,!?\'-]', '', text)

        return text.strip()

    def normalize(self, text: str) -> str:
        """Normalize text based on language"""
        if self.language == 'hindi':
            return self.normalize_hindi(text)
        elif self.language == 'punjabi':
            return self.normalize_punjabi(text)
        else:
            return text.strip()


def process_common_voice(cv_dir: Path, output_dir: Path, language: str, split: str = 'train') -> List[Dict]:
    """Process Common Voice dataset"""
    logger.info(f"Processing Common Voice {language} - {split} split")

    tsv_file = cv_dir / f'{split}.tsv'
    if not tsv_file.exists():
        logger.warning(f"File not found: {tsv_file}")
        return []

    df = pd.read_csv(tsv_file, sep='\t')
    logger.info(f"Loaded {len(df)} samples from {split}.tsv")

    preprocessor = AudioPreprocessor()
    text_normalizer = TextNormalizer(language)

    manifests = []
    stats = {'success': 0, 'failed': 0, 'reasons': {}}

    clips_dir = cv_dir / 'clips'

    for idx, row in tqdm(df.iterrows(), total=len(df), desc=f"Processing {split}"):
        input_path = clips_dir / row['path']

        if not input_path.exists():
            stats['failed'] += 1
            stats['reasons']['file_not_found'] = stats['reasons'].get('file_not_found', 0) + 1
            continue

        # Output path
        output_filename = f"{language}_{split}_{idx:06d}.wav"
        output_path = output_dir / 'processed' / language / split / output_filename

        # Process audio
        result = preprocessor.process_audio_file(input_path, output_path)

        if result['success']:
            # Normalize text
            normalized_text = text_normalizer.normalize(row['sentence'])

            if len(normalized_text) == 0:
                stats['failed'] += 1
                stats['reasons']['empty_text'] = stats['reasons'].get('empty_text', 0) + 1
                continue

            manifest_entry = {
                'audio_filepath': str(output_path),
                'text': normalized_text,
                'duration': result['duration'],
                'language': language,
                'source': 'common_voice',
                'split': split
            }

            manifests.append(manifest_entry)
            stats['success'] += 1
        else:
            stats['failed'] += 1
            reason = result.get('reason', 'unknown')
            stats['reasons'][reason] = stats['reasons'].get(reason, 0) + 1

    logger.info(f"Processed {stats['success']} successful, {stats['failed']} failed")
    logger.info(f"Failure reasons: {stats['reasons']}")

    return manifests


def process_openslr_hindi(openslr_dir: Path, output_dir: Path, split: str = 'train') -> List[Dict]:
    """Process OpenSLR Hindi dataset"""
    logger.info(f"Processing OpenSLR Hindi - {split} split")

    manifests = []
    preprocessor = AudioPreprocessor()
    text_normalizer = TextNormalizer('hindi')

    # Find all audio files
    audio_files = list(openslr_dir.glob('**/*.wav')) + list(openslr_dir.glob('**/*.flac'))

    # Look for transcript file (format may vary)
    transcript_file = openslr_dir / 'transcripts.txt'
    if not transcript_file.exists():
        transcript_file = openslr_dir / 'utt_spk_text.txt'

    transcripts = {}
    if transcript_file.exists():
        with open(transcript_file, 'r', encoding='utf-8') as f:
            for line in f:
                parts = line.strip().split(None, 1)
                if len(parts) == 2:
                    file_id, text = parts
                    transcripts[file_id] = text

    for idx, input_path in enumerate(tqdm(audio_files, desc=f"Processing OpenSLR Hindi")):
        # Try to find transcript
        file_id = input_path.stem
        text = transcripts.get(file_id, '')

        if not text:
            logger.warning(f"No transcript for {file_id}")
            continue

        output_filename = f"hindi_openslr_{split}_{idx:06d}.wav"
        output_path = output_dir / 'processed' / 'hindi' / split / output_filename

        result = preprocessor.process_audio_file(input_path, output_path)

        if result['success']:
            normalized_text = text_normalizer.normalize(text)

            if len(normalized_text) > 0:
                manifest_entry = {
                    'audio_filepath': str(output_path),
                    'text': normalized_text,
                    'duration': result['duration'],
                    'language': 'hindi',
                    'source': 'openslr',
                    'split': split
                }
                manifests.append(manifest_entry)

    logger.info(f"Processed {len(manifests)} samples from OpenSLR Hindi")
    return manifests


def create_splits(manifests: List[Dict], train_ratio=0.8, val_ratio=0.1) -> Dict[str, List[Dict]]:
    """Create train/val/test splits if not already split"""
    np.random.seed(42)
    np.random.shuffle(manifests)

    total = len(manifests)
    train_end = int(total * train_ratio)
    val_end = train_end + int(total * val_ratio)

    splits = {
        'train': manifests[:train_end],
        'val': manifests[train_end:val_end],
        'test': manifests[val_end:]
    }

    return splits


def save_manifests(manifests: Dict[str, List[Dict]], output_dir: Path, language: str):
    """Save manifests to JSON files"""
    manifest_dir = output_dir / 'manifests'
    manifest_dir.mkdir(parents=True, exist_ok=True)

    for split, data in manifests.items():
        output_file = manifest_dir / f'{language}_{split}.json'

        with open(output_file, 'w', encoding='utf-8') as f:
            for entry in data:
                f.write(json.dumps(entry, ensure_ascii=False) + '\n')

        logger.info(f"Saved {len(data)} samples to {output_file}")

        # Calculate statistics
        total_duration = sum(entry['duration'] for entry in data)
        logger.info(f"  Total duration: {total_duration / 3600:.2f} hours")


def main():
    parser = argparse.ArgumentParser(description='Preprocess ASR datasets')
    parser.add_argument('--language', type=str, required=True, choices=['hindi', 'punjabi'])
    parser.add_argument('--input_dir', type=Path, default=Path('data/raw'))
    parser.add_argument('--output_dir', type=Path, default=Path('data'))
    parser.add_argument('--min_duration', type=float, default=1.0)
    parser.add_argument('--max_duration', type=float, default=20.0)
    parser.add_argument('--min_snr', type=float, default=15.0)

    args = parser.parse_args()

    logger.info(f"""
╔════════════════════════════════════════════════════════════╗
║      Data Preprocessing - {args.language.upper():<8}                      ║
╚════════════════════════════════════════════════════════════╝
    """)

    all_manifests = {'train': [], 'val': [], 'test': []}

    # Process Common Voice
    cv_dir = args.input_dir / args.language / 'common_voice'
    if cv_dir.exists():
        for split in ['train', 'dev', 'test']:
            cv_split = 'val' if split == 'dev' else split
            manifests = process_common_voice(cv_dir, args.output_dir, args.language, split)
            all_manifests[cv_split].extend(manifests)

    # Process OpenSLR (Hindi only for now)
    if args.language == 'hindi':
        for dataset_name in ['openslr', 'microsoft']:
            dataset_dir = args.input_dir / args.language / dataset_name
            if dataset_dir.exists():
                manifests = process_openslr_hindi(dataset_dir, args.output_dir, 'train')
                # Split OpenSLR data
                if manifests:
                    splits = create_splits(manifests)
                    for split, data in splits.items():
                        all_manifests[split].extend(data)

    # Save manifests
    save_manifests(all_manifests, args.output_dir, args.language)

    # Print summary
    logger.info(f"\n{'='*60}")
    logger.info(f"Preprocessing Complete - {args.language.upper()}")
    logger.info(f"{'='*60}")
    for split in ['train', 'val', 'test']:
        count = len(all_manifests[split])
        duration = sum(m['duration'] for m in all_manifests[split])
        logger.info(f"{split.upper():5}: {count:5} samples, {duration/3600:6.2f} hours")

    logger.info(f"\nManifests saved to: {args.output_dir}/manifests/")


if __name__ == '__main__':
    main()
