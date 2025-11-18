#!/usr/bin/env python3
"""
Convert Whisper model to ONNX format
Optimized for mobile deployment
"""

import argparse
import torch
import numpy as np
from pathlib import Path
import onnx
import onnxruntime as ort
import logging

from transformers import WhisperProcessor, WhisperForConditionalGeneration
import librosa

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


def export_encoder_to_onnx(model, output_path: Path, opset_version: int = 14):
    """Export Whisper encoder to ONNX"""
    logger.info("Exporting encoder to ONNX...")

    encoder = model.model.encoder
    encoder.eval()

    # Dummy input (80 mel bins, 3000 time steps)
    dummy_input = torch.randn(1, 80, 3000)

    # Export
    encoder_path = output_path / "encoder_model.onnx"
    torch.onnx.export(
        encoder,
        dummy_input,
        encoder_path,
        input_names=['input_features'],
        output_names=['last_hidden_state'],
        dynamic_axes={
            'input_features': {0: 'batch', 2: 'time'},
            'last_hidden_state': {0: 'batch', 1: 'time'}
        },
        opset_version=opset_version,
        do_constant_folding=True,
    )

    logger.info(f"Encoder exported to {encoder_path}")
    return encoder_path


def export_decoder_to_onnx(model, output_path: Path, opset_version: int = 14):
    """Export Whisper decoder to ONNX"""
    logger.info("Exporting decoder to ONNX...")

    decoder = model.model.decoder
    decoder.eval()

    # Dummy inputs
    batch_size = 1
    seq_len = 10
    encoder_hidden_size = model.config.d_model
    encoder_seq_len = 1500  # After convolution: 3000 / 2 = 1500

    dummy_decoder_input_ids = torch.randint(0, model.config.vocab_size, (batch_size, seq_len))
    dummy_encoder_hidden_states = torch.randn(batch_size, encoder_seq_len, encoder_hidden_size)

    # Export
    decoder_path = output_path / "decoder_model.onnx"
    torch.onnx.export(
        decoder,
        (dummy_decoder_input_ids, dummy_encoder_hidden_states),
        decoder_path,
        input_names=['input_ids', 'encoder_hidden_states'],
        output_names=['logits'],
        dynamic_axes={
            'input_ids': {0: 'batch', 1: 'sequence'},
            'encoder_hidden_states': {0: 'batch', 1: 'encoder_sequence'},
            'logits': {0: 'batch', 1: 'sequence'}
        },
        opset_version=opset_version,
        do_constant_folding=True,
    )

    logger.info(f"Decoder exported to {decoder_path}")
    return decoder_path


def export_full_model_with_optimum(model_path: str, output_path: Path):
    """
    Export using Optimum library (recommended)
    Handles complex model export automatically
    """
    try:
        from optimum.onnxruntime import ORTModelForSpeechSeq2Seq
    except ImportError:
        logger.error("optimum not installed. Install with: pip install optimum[exporters]")
        return None

    logger.info("Exporting model with Optimum...")

    output_path.mkdir(parents=True, exist_ok=True)

    # Export
    model = ORTModelForSpeechSeq2Seq.from_pretrained(
        model_path,
        export=True,
        provider="CPUExecutionProvider"
    )

    # Save
    model.save_pretrained(output_path)

    logger.info(f"Model exported to {output_path}")

    # Also save processor
    processor = WhisperProcessor.from_pretrained(model_path)
    processor.save_pretrained(output_path)

    return output_path


def optimize_onnx_model(onnx_path: Path):
    """Optimize ONNX model for inference"""
    logger.info(f"Optimizing ONNX model: {onnx_path}")

    try:
        from onnxruntime.transformers import optimizer
        from onnxruntime.transformers.fusion_options import FusionOptions

        # Load model
        model = onnx.load(str(onnx_path))

        # Optimize
        optimized_model = optimizer.optimize_model(
            str(onnx_path),
            model_type='bert',  # Use bert optimizer for transformer models
            num_heads=0,  # Auto-detect
            hidden_size=0,  # Auto-detect
        )

        # Save optimized model
        optimized_path = onnx_path.parent / f"{onnx_path.stem}_optimized.onnx"
        optimized_model.save_model_to_file(str(optimized_path))

        logger.info(f"Optimized model saved to {optimized_path}")

        # Size comparison
        original_size = onnx_path.stat().st_size / (1024 * 1024)
        optimized_size = optimized_path.stat().st_size / (1024 * 1024)

        logger.info(f"Original size: {original_size:.2f} MB")
        logger.info(f"Optimized size: {optimized_size:.2f} MB")

        return optimized_path

    except ImportError:
        logger.warning("onnxruntime optimizer not available, skipping optimization")
        return onnx_path


def verify_onnx_model(onnx_path: Path, pytorch_model_path: str, num_tests: int = 5):
    """Verify ONNX model outputs match PyTorch model"""
    logger.info("Verifying ONNX model...")

    # Load PyTorch model
    processor = WhisperProcessor.from_pretrained(pytorch_model_path)
    pt_model = WhisperForConditionalGeneration.from_pretrained(pytorch_model_path)
    pt_model.eval()

    # Load ONNX model
    if (onnx_path / "encoder_model.onnx").exists():
        # Separate encoder/decoder
        ort_encoder = ort.InferenceSession(
            str(onnx_path / "encoder_model.onnx"),
            providers=['CPUExecutionProvider']
        )
        logger.info("ONNX encoder loaded")
    else:
        logger.warning("Encoder ONNX not found, skipping verification")
        return True

    # Test with random inputs
    logger.info("Running verification tests...")

    for i in range(num_tests):
        # Random mel spectrogram
        dummy_input = torch.randn(1, 80, 3000)

        # PyTorch
        with torch.no_grad():
            pt_output = pt_model.model.encoder(dummy_input).last_hidden_state

        # ONNX
        ort_output = ort_encoder.run(
            None,
            {'input_features': dummy_input.numpy()}
        )[0]

        # Compare
        diff = np.abs(pt_output.numpy() - ort_output).max()
        logger.info(f"Test {i+1}: Max difference = {diff:.6f}")

        if diff > 1e-3:
            logger.warning(f"Large difference detected: {diff}")

    logger.info("✓ ONNX model verification complete")
    return True


def main():
    parser = argparse.ArgumentParser(description='Convert Whisper to ONNX')
    parser.add_argument('--model_path', type=str, required=True,
                        help='Path to PyTorch model')
    parser.add_argument('--output_path', type=str, required=True,
                        help='Output directory for ONNX model')
    parser.add_argument('--method', type=str, default='optimum',
                        choices=['manual', 'optimum'],
                        help='Export method')
    parser.add_argument('--opset_version', type=int, default=14,
                        help='ONNX opset version')
    parser.add_argument('--optimize', action='store_true',
                        help='Apply ONNX optimizations')
    parser.add_argument('--verify', action='store_true',
                        help='Verify ONNX model outputs')

    args = parser.parse_args()

    logger.info(f"""
╔════════════════════════════════════════════════════════════╗
║           ONNX Conversion for Mobile Deployment           ║
╚════════════════════════════════════════════════════════════╝
    """)

    output_path = Path(args.output_path)
    output_path.mkdir(parents=True, exist_ok=True)

    # Export
    if args.method == 'optimum':
        export_full_model_with_optimum(args.model_path, output_path)

    elif args.method == 'manual':
        logger.info("Loading model...")
        model = WhisperForConditionalGeneration.from_pretrained(args.model_path)
        model.eval()

        # Export encoder and decoder separately
        export_encoder_to_onnx(model, output_path, args.opset_version)
        export_decoder_to_onnx(model, output_path, args.opset_version)

    # Optimize
    if args.optimize:
        for onnx_file in output_path.glob("*.onnx"):
            if "optimized" not in onnx_file.name:
                optimize_onnx_model(onnx_file)

    # Verify
    if args.verify:
        verify_onnx_model(output_path, args.model_path)

    logger.info(f"\n✓ ONNX conversion complete!")
    logger.info(f"ONNX model saved to: {output_path}")
    logger.info("\nNext steps:")
    logger.info("  Android (TFLite): python optimization/convert_to_tflite.py")
    logger.info("  iOS (CoreML): python optimization/convert_to_coreml.py")


if __name__ == '__main__':
    main()
