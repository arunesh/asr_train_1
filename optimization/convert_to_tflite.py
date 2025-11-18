#!/usr/bin/env python3
"""
Convert ONNX model to TensorFlow Lite for Android deployment
"""

import argparse
import tensorflow as tf
import numpy as np
from pathlib import Path
import logging
import onnx
from onnx_tf.backend import prepare

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


def convert_onnx_to_tensorflow(onnx_path: Path, tf_path: Path):
    """Convert ONNX model to TensorFlow SavedModel"""
    logger.info(f"Converting ONNX to TensorFlow: {onnx_path}")

    # Load ONNX model
    onnx_model = onnx.load(str(onnx_path))

    # Prepare TensorFlow model
    tf_rep = prepare(onnx_model)

    # Export as SavedModel
    tf_rep.export_graph(str(tf_path))

    logger.info(f"TensorFlow SavedModel saved to {tf_path}")
    return tf_path


def convert_tensorflow_to_tflite(
    tf_model_path: Path,
    tflite_path: Path,
    quantize: bool = True,
    representative_dataset = None
):
    """Convert TensorFlow SavedModel to TFLite"""
    logger.info("Converting TensorFlow to TFLite...")

    # Load TensorFlow model
    converter = tf.lite.TFLiteConverter.from_saved_model(str(tf_model_path))

    # Optimization settings
    if quantize:
        logger.info("Applying INT8 quantization...")
        converter.optimizations = [tf.lite.Optimize.DEFAULT]

        # For full integer quantization
        if representative_dataset is not None:
            converter.representative_dataset = representative_dataset
            converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS_INT8]
            converter.inference_input_type = tf.int8
            converter.inference_output_type = tf.int8
        else:
            # Dynamic range quantization (weights only)
            logger.info("Using dynamic range quantization (no calibration data)")

    # Enable experimental features for better mobile support
    converter.experimental_new_converter = True
    converter.target_spec.supported_ops = [
        tf.lite.OpsSet.TFLITE_BUILTINS,  # Standard TFLite ops
        tf.lite.OpsSet.SELECT_TF_OPS     # Fallback to TF ops if needed
    ]

    # Convert
    tflite_model = converter.convert()

    # Save
    tflite_path.parent.mkdir(parents=True, exist_ok=True)
    with open(tflite_path, 'wb') as f:
        f.write(tflite_model)

    # Size info
    size_mb = len(tflite_model) / (1024 * 1024)
    logger.info(f"TFLite model size: {size_mb:.2f} MB")
    logger.info(f"TFLite model saved to {tflite_path}")

    return tflite_path


def create_representative_dataset(manifest_path: str, num_samples: int = 100):
    """Create representative dataset for quantization calibration"""
    import json
    import librosa
    from transformers import WhisperProcessor

    logger.info(f"Creating representative dataset from {manifest_path}")

    # Load processor
    processor = WhisperProcessor.from_pretrained("openai/whisper-tiny")

    # Load samples
    samples = []
    with open(manifest_path, 'r') as f:
        for i, line in enumerate(f):
            if i >= num_samples:
                break
            samples.append(json.loads(line))

    def representative_dataset_gen():
        """Generator for representative dataset"""
        for sample in samples:
            try:
                # Load audio
                audio, sr = librosa.load(sample['audio_filepath'], sr=16000, mono=True)

                # Process
                input_features = processor(
                    audio,
                    sampling_rate=16000,
                    return_tensors="np"
                ).input_features

                yield [input_features.astype(np.float32)]

            except Exception as e:
                logger.warning(f"Error loading {sample['audio_filepath']}: {e}")
                continue

    return representative_dataset_gen


def verify_tflite_model(tflite_path: Path, num_tests: int = 5):
    """Verify TFLite model can run inference"""
    logger.info("Verifying TFLite model...")

    # Load TFLite model
    interpreter = tf.lite.Interpreter(model_path=str(tflite_path))
    interpreter.allocate_tensors()

    # Get input and output details
    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()

    logger.info(f"Input details: {input_details}")
    logger.info(f"Output details: {output_details}")

    # Test inference
    for i in range(num_tests):
        # Create dummy input matching expected shape
        input_shape = input_details[0]['shape']
        dummy_input = np.random.randn(*input_shape).astype(input_details[0]['dtype'])

        # Run inference
        interpreter.set_tensor(input_details[0]['index'], dummy_input)
        interpreter.invoke()

        # Get output
        output = interpreter.get_tensor(output_details[0]['index'])

        logger.info(f"Test {i+1}: Output shape = {output.shape}")

    logger.info("✓ TFLite model verification complete")
    return True


def benchmark_tflite_model(tflite_path: Path, num_runs: int = 100):
    """Benchmark TFLite model inference time"""
    import time

    logger.info(f"Benchmarking TFLite model ({num_runs} runs)...")

    # Load model
    interpreter = tf.lite.Interpreter(model_path=str(tflite_path))
    interpreter.allocate_tensors()

    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()

    # Dummy input
    input_shape = input_details[0]['shape']
    dummy_input = np.random.randn(*input_shape).astype(input_details[0]['dtype'])

    # Warmup
    for _ in range(10):
        interpreter.set_tensor(input_details[0]['index'], dummy_input)
        interpreter.invoke()

    # Benchmark
    times = []
    for _ in range(num_runs):
        start = time.time()
        interpreter.set_tensor(input_details[0]['index'], dummy_input)
        interpreter.invoke()
        _ = interpreter.get_tensor(output_details[0]['index'])
        times.append(time.time() - start)

    avg_time = np.mean(times) * 1000  # Convert to ms
    std_time = np.std(times) * 1000

    logger.info(f"Average inference time: {avg_time:.2f} ± {std_time:.2f} ms")
    logger.info(f"Min: {min(times)*1000:.2f} ms, Max: {max(times)*1000:.2f} ms")

    return avg_time


def create_full_pipeline(
    model_name: str,
    output_dir: Path,
    quantize: bool = True,
    calibration_manifest: str = None
):
    """
    Complete pipeline: Model -> ONNX -> TensorFlow -> TFLite
    """
    logger.info("Running full conversion pipeline...")

    output_dir.mkdir(parents=True, exist_ok=True)

    # Step 1: Export to ONNX (if not already done)
    onnx_dir = output_dir / "onnx"
    if not (onnx_dir / "encoder_model.onnx").exists():
        logger.info("ONNX model not found, exporting...")
        from optimization.convert_to_onnx import export_full_model_with_optimum
        export_full_model_with_optimum(model_name, onnx_dir)

    # Step 2: Convert ONNX to TensorFlow
    encoder_onnx = onnx_dir / "encoder_model.onnx"
    decoder_onnx = onnx_dir / "decoder_model.onnx"

    tf_encoder_path = output_dir / "tf" / "encoder"
    tf_decoder_path = output_dir / "tf" / "decoder"

    if encoder_onnx.exists():
        convert_onnx_to_tensorflow(encoder_onnx, tf_encoder_path)

    if decoder_onnx.exists():
        convert_onnx_to_tensorflow(decoder_onnx, tf_decoder_path)

    # Step 3: Convert to TFLite
    representative_dataset = None
    if quantize and calibration_manifest:
        representative_dataset = create_representative_dataset(calibration_manifest)

    tflite_encoder_path = output_dir / "tflite" / "encoder.tflite"
    tflite_decoder_path = output_dir / "tflite" / "decoder.tflite"

    if tf_encoder_path.exists():
        convert_tensorflow_to_tflite(
            tf_encoder_path,
            tflite_encoder_path,
            quantize=quantize,
            representative_dataset=representative_dataset
        )

    if tf_decoder_path.exists():
        convert_tensorflow_to_tflite(
            tf_decoder_path,
            tflite_decoder_path,
            quantize=quantize,
            representative_dataset=None  # Decoder harder to quantize
        )

    return tflite_encoder_path, tflite_decoder_path


def main():
    parser = argparse.ArgumentParser(description='Convert to TensorFlow Lite for Android')
    parser.add_argument('--onnx_path', type=str,
                        help='Path to ONNX model')
    parser.add_argument('--model_name', type=str,
                        help='HuggingFace model name or path (for full pipeline)')
    parser.add_argument('--output_dir', type=str, required=True,
                        help='Output directory')
    parser.add_argument('--quantize', action='store_true', default=True,
                        help='Apply INT8 quantization')
    parser.add_argument('--calibration_manifest', type=str,
                        help='Manifest for quantization calibration')
    parser.add_argument('--verify', action='store_true',
                        help='Verify TFLite model')
    parser.add_argument('--benchmark', action='store_true',
                        help='Benchmark TFLite model')
    parser.add_argument('--full_pipeline', action='store_true',
                        help='Run full conversion pipeline')

    args = parser.parse_args()

    logger.info(f"""
╔════════════════════════════════════════════════════════════╗
║         TensorFlow Lite Conversion for Android            ║
╚════════════════════════════════════════════════════════════╝
    """)

    output_dir = Path(args.output_dir)

    if args.full_pipeline:
        if not args.model_name:
            logger.error("--model_name required for full pipeline")
            return

        encoder_path, decoder_path = create_full_pipeline(
            args.model_name,
            output_dir,
            quantize=args.quantize,
            calibration_manifest=args.calibration_manifest
        )

        # Verify and benchmark
        if args.verify and encoder_path.exists():
            verify_tflite_model(encoder_path)

        if args.benchmark and encoder_path.exists():
            benchmark_tflite_model(encoder_path)

    else:
        # Simple ONNX -> TF -> TFLite conversion
        if not args.onnx_path:
            logger.error("--onnx_path required")
            return

        onnx_path = Path(args.onnx_path)
        tf_path = output_dir / "tf_model"
        tflite_path = output_dir / "model.tflite"

        # Convert
        convert_onnx_to_tensorflow(onnx_path, tf_path)

        representative_dataset = None
        if args.quantize and args.calibration_manifest:
            representative_dataset = create_representative_dataset(args.calibration_manifest)

        convert_tensorflow_to_tflite(
            tf_path,
            tflite_path,
            quantize=args.quantize,
            representative_dataset=representative_dataset
        )

        if args.verify:
            verify_tflite_model(tflite_path)

        if args.benchmark:
            benchmark_tflite_model(tflite_path)

    logger.info("\n✓ TFLite conversion complete!")
    logger.info(f"Models saved to: {output_dir}")
    logger.info("\nNext step: Integrate into Android app")


if __name__ == '__main__':
    main()
