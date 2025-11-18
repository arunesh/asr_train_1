#!/usr/bin/env python3
"""
Model Quantization Script
Converts FP32 models to INT8 for mobile deployment
"""

import argparse
import torch
import numpy as np
from pathlib import Path
from typing import List, Dict
import json
import logging
from tqdm import tqdm

from transformers import WhisperProcessor, WhisperForConditionalGeneration
import librosa

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


class CalibrationDataset:
    """Dataset for quantization calibration"""

    def __init__(self, manifest_path: str, max_samples: int = 1000):
        self.samples = []
        with open(manifest_path, 'r', encoding='utf-8') as f:
            for i, line in enumerate(f):
                if i >= max_samples:
                    break
                self.samples.append(json.loads(line))

        logger.info(f"Loaded {len(self.samples)} calibration samples")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        return self.samples[idx]['audio_filepath']


def load_calibration_data(
    manifest_path: str,
    processor: WhisperProcessor,
    num_samples: int = 1000
) -> List[torch.Tensor]:
    """Load calibration data for quantization"""
    logger.info(f"Loading calibration data from {manifest_path}")

    dataset = CalibrationDataset(manifest_path, num_samples)
    calibration_data = []

    for audio_path in tqdm(dataset.samples[:num_samples], desc="Loading calibration data"):
        try:
            audio, sr = librosa.load(audio_path['audio_filepath'], sr=16000, mono=True)

            input_features = processor(
                audio,
                sampling_rate=16000,
                return_tensors="pt"
            ).input_features

            calibration_data.append(input_features)

        except Exception as e:
            logger.warning(f"Failed to load {audio_path}: {e}")
            continue

    logger.info(f"Loaded {len(calibration_data)} calibration samples")
    return calibration_data


def dynamic_quantization_pytorch(model_path: str, output_path: str):
    """
    Apply dynamic quantization using PyTorch
    Quantizes weights to INT8, activations remain FP32
    """
    logger.info("Applying dynamic quantization (PyTorch)")

    # Load model
    model = WhisperForConditionalGeneration.from_pretrained(model_path)
    model.eval()

    # Get model size before quantization
    torch.save(model.state_dict(), "temp_model.pt")
    size_before = Path("temp_model.pt").stat().st_size / (1024 * 1024)
    Path("temp_model.pt").unlink()

    # Apply dynamic quantization
    quantized_model = torch.quantization.quantize_dynamic(
        model,
        {torch.nn.Linear},  # Quantize Linear layers
        dtype=torch.qint8
    )

    # Save quantized model
    output_dir = Path(output_path)
    output_dir.mkdir(parents=True, exist_ok=True)

    quantized_model.save_pretrained(output_dir)

    # Get model size after quantization
    model_file = output_dir / "pytorch_model.bin"
    if model_file.exists():
        size_after = model_file.stat().st_size / (1024 * 1024)
    else:
        size_after = 0

    logger.info(f"Model size before: {size_before:.2f} MB")
    logger.info(f"Model size after: {size_after:.2f} MB")
    logger.info(f"Compression ratio: {size_before / size_after:.2f}x")

    return quantized_model


def static_quantization_pytorch(
    model_path: str,
    output_path: str,
    calibration_manifest: str,
    num_calibration_samples: int = 1000
):
    """
    Apply static quantization using PyTorch
    Quantizes both weights and activations to INT8
    """
    logger.info("Applying static quantization (PyTorch)")

    # Load model and processor
    model = WhisperForConditionalGeneration.from_pretrained(model_path)
    processor = WhisperProcessor.from_pretrained(model_path)
    model.eval()

    # Prepare model for quantization
    model.qconfig = torch.quantization.get_default_qconfig('fbgemm')
    torch.quantization.prepare(model, inplace=True)

    # Calibration
    logger.info("Running calibration...")
    calibration_data = load_calibration_data(
        calibration_manifest,
        processor,
        num_calibration_samples
    )

    with torch.no_grad():
        for input_features in tqdm(calibration_data, desc="Calibrating"):
            try:
                model.generate(input_features, max_length=225)
            except Exception as e:
                logger.warning(f"Calibration error: {e}")
                continue

    # Convert to quantized model
    logger.info("Converting to quantized model...")
    torch.quantization.convert(model, inplace=True)

    # Save
    output_dir = Path(output_path)
    output_dir.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(output_dir)
    processor.save_pretrained(output_dir)

    logger.info(f"Quantized model saved to {output_dir}")

    return model


def quantize_with_optimum(
    model_path: str,
    output_path: str,
    calibration_manifest: str = None
):
    """
    Quantize using Optimum library (recommended for production)
    """
    try:
        from optimum.onnxruntime import ORTQuantizer, ORTModelForSpeechSeq2Seq
        from optimum.onnxruntime.configuration import AutoQuantizationConfig
    except ImportError:
        logger.error("optimum[onnxruntime] not installed")
        logger.error("Install with: pip install optimum[onnxruntime]")
        return None

    logger.info("Quantizing with Optimum (ONNX Runtime)")

    # Load model
    model = ORTModelForSpeechSeq2Seq.from_pretrained(model_path, export=True)

    # Create quantization config
    qconfig = AutoQuantizationConfig.arm64(is_static=False)

    # Quantizer
    quantizer = ORTQuantizer.from_pretrained(model)

    # Apply quantization
    output_dir = Path(output_path)
    output_dir.mkdir(parents=True, exist_ok=True)

    quantizer.quantize(
        save_dir=output_dir,
        quantization_config=qconfig,
    )

    logger.info(f"Quantized model saved to {output_dir}")

    return output_dir


def benchmark_quantized_model(
    original_model_path: str,
    quantized_model_path: str,
    test_manifest: str,
    num_samples: int = 100
):
    """Benchmark quantized model against original"""
    logger.info("Benchmarking quantized model...")

    from transformers import WhisperProcessor
    import librosa
    import time

    # Load models
    processor = WhisperProcessor.from_pretrained(original_model_path)

    logger.info("Loading original model...")
    original_model = WhisperForConditionalGeneration.from_pretrained(original_model_path)
    original_model.eval()

    logger.info("Loading quantized model...")
    quantized_model = WhisperForConditionalGeneration.from_pretrained(quantized_model_path)
    quantized_model.eval()

    # Load test samples
    samples = []
    with open(test_manifest, 'r') as f:
        for i, line in enumerate(f):
            if i >= num_samples:
                break
            samples.append(json.loads(line))

    # Benchmark
    results = {
        'original': {'time': [], 'memory': []},
        'quantized': {'time': [], 'memory': []}
    }

    for sample in tqdm(samples, desc="Benchmarking"):
        try:
            audio, sr = librosa.load(sample['audio_filepath'], sr=16000, mono=True)
            input_features = processor(audio, sampling_rate=16000, return_tensors="pt").input_features

            # Original model
            start = time.time()
            with torch.no_grad():
                _ = original_model.generate(input_features, max_length=225)
            results['original']['time'].append(time.time() - start)

            # Quantized model
            start = time.time()
            with torch.no_grad():
                _ = quantized_model.generate(input_features, max_length=225)
            results['quantized']['time'].append(time.time() - start)

        except Exception as e:
            logger.warning(f"Benchmark error: {e}")
            continue

    # Print results
    logger.info("\n" + "="*60)
    logger.info("Benchmark Results")
    logger.info("="*60)
    logger.info(f"Original model avg time: {np.mean(results['original']['time']):.3f}s")
    logger.info(f"Quantized model avg time: {np.mean(results['quantized']['time']):.3f}s")
    logger.info(f"Speedup: {np.mean(results['original']['time']) / np.mean(results['quantized']['time']):.2f}x")

    return results


def main():
    parser = argparse.ArgumentParser(description='Quantize Whisper model for mobile deployment')
    parser.add_argument('--model_path', type=str, required=True,
                        help='Path to trained model')
    parser.add_argument('--output_path', type=str, required=True,
                        help='Output path for quantized model')
    parser.add_argument('--method', type=str, default='dynamic',
                        choices=['dynamic', 'static', 'optimum'],
                        help='Quantization method')
    parser.add_argument('--calibration_manifest', type=str,
                        help='Manifest for calibration (required for static quantization)')
    parser.add_argument('--num_calibration_samples', type=int, default=1000,
                        help='Number of samples for calibration')
    parser.add_argument('--benchmark', action='store_true',
                        help='Benchmark quantized model')
    parser.add_argument('--test_manifest', type=str,
                        help='Test manifest for benchmarking')

    args = parser.parse_args()

    logger.info(f"""
╔════════════════════════════════════════════════════════════╗
║              Model Quantization (INT8)                     ║
╚════════════════════════════════════════════════════════════╝
    """)

    # Quantize
    if args.method == 'dynamic':
        quantized_model = dynamic_quantization_pytorch(args.model_path, args.output_path)
    elif args.method == 'static':
        if not args.calibration_manifest:
            logger.error("--calibration_manifest required for static quantization")
            return
        quantized_model = static_quantization_pytorch(
            args.model_path,
            args.output_path,
            args.calibration_manifest,
            args.num_calibration_samples
        )
    elif args.method == 'optimum':
        quantized_model = quantize_with_optimum(
            args.model_path,
            args.output_path,
            args.calibration_manifest
        )

    # Benchmark if requested
    if args.benchmark and args.test_manifest:
        benchmark_quantized_model(
            args.model_path,
            args.output_path,
            args.test_manifest
        )

    logger.info(f"\n✓ Quantization complete!")
    logger.info(f"Quantized model saved to: {args.output_path}")
    logger.info("\nNext step: Convert to ONNX")
    logger.info(f"  python optimization/convert_to_onnx.py --model_path {args.output_path}")


if __name__ == '__main__':
    main()
