#!/usr/bin/env python3
"""
Evaluation script for ASR models
Computes WER, CER and other metrics on test set
"""

import argparse
import json
from pathlib import Path
from typing import List, Dict
import torch
import numpy as np
from tqdm import tqdm
import librosa
from transformers import WhisperProcessor, WhisperForConditionalGeneration
import evaluate
import pandas as pd
import logging

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


def load_manifest(manifest_path: str) -> List[Dict]:
    """Load manifest file"""
    samples = []
    with open(manifest_path, 'r', encoding='utf-8') as f:
        for line in f:
            samples.append(json.loads(line))
    return samples


def transcribe_audio(
    audio_path: str,
    model,
    processor,
    device: str = 'cuda',
    language: str = 'hindi'
) -> str:
    """Transcribe a single audio file"""
    # Load audio
    audio, sr = librosa.load(audio_path, sr=16000, mono=True)

    # Process audio
    input_features = processor(
        audio,
        sampling_rate=16000,
        return_tensors="pt"
    ).input_features.to(device)

    # Generate transcription
    with torch.no_grad():
        predicted_ids = model.generate(
            input_features,
            language=language,
            task="transcribe",
            max_length=225,
            num_beams=5
        )

    # Decode
    transcription = processor.batch_decode(predicted_ids, skip_special_tokens=True)[0]

    return transcription


def evaluate_model(
    model_path: str,
    test_manifest: str,
    language: str = 'hindi',
    device: str = 'cuda',
    output_dir: str = None
):
    """Evaluate model on test set"""
    logger.info(f"Loading model from {model_path}")

    # Load model and processor
    processor = WhisperProcessor.from_pretrained(model_path)
    model = WhisperForConditionalGeneration.from_pretrained(model_path)
    model.to(device)
    model.eval()

    # Load test samples
    logger.info(f"Loading test samples from {test_manifest}")
    samples = load_manifest(test_manifest)
    logger.info(f"Loaded {len(samples)} test samples")

    # Load metrics
    wer_metric = evaluate.load("wer")
    cer_metric = evaluate.load("cer")

    # Evaluate
    predictions = []
    references = []
    results = []

    for sample in tqdm(samples, desc="Evaluating"):
        try:
            # Transcribe
            pred = transcribe_audio(
                sample['audio_filepath'],
                model,
                processor,
                device,
                language
            )

            ref = sample['text']

            predictions.append(pred)
            references.append(ref)

            # Store result
            results.append({
                'audio_filepath': sample['audio_filepath'],
                'reference': ref,
                'prediction': pred,
                'duration': sample.get('duration', 0)
            })

        except Exception as e:
            logger.error(f"Error processing {sample['audio_filepath']}: {e}")
            continue

    # Compute metrics
    wer = 100 * wer_metric.compute(predictions=predictions, references=references)
    cer = 100 * cer_metric.compute(predictions=predictions, references=references)

    logger.info(f"\n{'='*60}")
    logger.info(f"Evaluation Results")
    logger.info(f"{'='*60}")
    logger.info(f"Samples evaluated: {len(predictions)}")
    logger.info(f"Word Error Rate (WER): {wer:.2f}%")
    logger.info(f"Character Error Rate (CER): {cer:.2f}%")

    # Save results
    if output_dir:
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)

        # Save detailed results
        results_df = pd.DataFrame(results)
        results_csv = output_path / 'evaluation_results.csv'
        results_df.to_csv(results_csv, index=False)
        logger.info(f"Detailed results saved to {results_csv}")

        # Save metrics summary
        metrics_summary = {
            'wer': wer,
            'cer': cer,
            'num_samples': len(predictions),
            'model_path': model_path,
            'test_manifest': test_manifest
        }

        metrics_file = output_path / 'metrics.json'
        with open(metrics_file, 'w') as f:
            json.dump(metrics_summary, f, indent=2)
        logger.info(f"Metrics summary saved to {metrics_file}")

    return wer, cer


def evaluate_baseline(language: str = 'hindi', test_manifest: str = None, output_dir: str = None):
    """Evaluate zero-shot Whisper baseline"""
    logger.info(f"Evaluating zero-shot Whisper Tiny baseline for {language}")

    model_name = "openai/whisper-tiny"

    return evaluate_model(
        model_path=model_name,
        test_manifest=test_manifest,
        language=language,
        output_dir=output_dir
    )


def main():
    parser = argparse.ArgumentParser(description='Evaluate ASR model')
    parser.add_argument('--model_path', type=str, required=True,
                        help='Path to model checkpoint or HF model name')
    parser.add_argument('--test_manifest', type=str, required=True,
                        help='Path to test manifest JSON')
    parser.add_argument('--language', type=str, default='hindi',
                        choices=['hindi', 'punjabi'],
                        help='Language')
    parser.add_argument('--device', type=str, default='cuda',
                        help='Device to use (cuda/cpu)')
    parser.add_argument('--output_dir', type=str, default='evaluation/results',
                        help='Directory to save results')
    parser.add_argument('--baseline', action='store_true',
                        help='Evaluate zero-shot baseline')

    args = parser.parse_args()

    # Check device
    if args.device == 'cuda' and not torch.cuda.is_available():
        logger.warning("CUDA not available, using CPU")
        args.device = 'cpu'

    if args.baseline:
        evaluate_baseline(
            language=args.language,
            test_manifest=args.test_manifest,
            output_dir=args.output_dir
        )
    else:
        evaluate_model(
            model_path=args.model_path,
            test_manifest=args.test_manifest,
            language=args.language,
            device=args.device,
            output_dir=args.output_dir
        )


if __name__ == '__main__':
    main()
