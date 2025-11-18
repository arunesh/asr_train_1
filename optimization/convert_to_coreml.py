#!/usr/bin/env python3
"""
Convert ONNX model to Core ML for iOS deployment
"""

import argparse
import numpy as np
from pathlib import Path
import logging
import onnx

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


def convert_onnx_to_coreml(
    onnx_path: Path,
    output_path: Path,
    quantize: bool = True,
    compute_units: str = 'ALL'
):
    """
    Convert ONNX model to Core ML

    Args:
        onnx_path: Path to ONNX model
        output_path: Output path for Core ML model
        quantize: Whether to apply quantization
        compute_units: 'ALL', 'CPU_ONLY', 'CPU_AND_GPU', or 'CPU_AND_NE' (Neural Engine)
    """
    try:
        import coremltools as ct
        from coremltools.models.neural_network import quantization_utils
    except ImportError:
        logger.error("coremltools not installed")
        logger.error("Install with: pip install coremltools")
        return None

    logger.info(f"Converting ONNX to Core ML: {onnx_path}")

    # Load ONNX model
    onnx_model = onnx.load(str(onnx_path))

    # Convert to Core ML
    logger.info("Converting to Core ML format...")

    # Set compute units
    compute_units_map = {
        'ALL': ct.ComputeUnit.ALL,
        'CPU_ONLY': ct.ComputeUnit.CPU_ONLY,
        'CPU_AND_GPU': ct.ComputeUnit.CPU_AND_GPU,
        'CPU_AND_NE': ct.ComputeUnit.CPU_AND_NE,  # CPU and Neural Engine
    }

    coreml_model = ct.convert(
        onnx_model,
        source='onnx',
        convert_to='mlprogram',  # Use ML Program (iOS 15+) for better performance
        compute_units=compute_units_map.get(compute_units, ct.ComputeUnit.ALL),
        minimum_deployment_target=ct.target.iOS15,
    )

    # Apply quantization if requested
    if quantize:
        logger.info("Applying INT8 quantization...")

        # Linear quantization (weights and activations)
        op_config = ct.optimize.coreml.OpLinearQuantizerConfig(
            mode="linear_symmetric",
            weight_threshold=512,  # Only quantize ops with >512 weights
        )

        # Optimize
        config = ct.optimize.coreml.OptimizationConfig(
            global_config=op_config
        )

        coreml_model = ct.optimize.coreml.linear_quantize_weights(
            coreml_model,
            config=config
        )

        logger.info("Quantization applied")

    # Save model
    output_path.parent.mkdir(parents=True, exist_ok=True)
    coreml_model.save(str(output_path))

    # Model info
    model_size = output_path.stat().st_size / (1024 * 1024)
    logger.info(f"Core ML model size: {model_size:.2f} MB")
    logger.info(f"Core ML model saved to {output_path}")

    # Print model info
    logger.info("\nModel Information:")
    logger.info(f"  Input names: {[i.name for i in coreml_model._spec.description.input]}")
    logger.info(f"  Output names: {[o.name for o in coreml_model._spec.description.output]}")

    return coreml_model


def add_metadata(coreml_model, metadata: dict):
    """Add metadata to Core ML model"""
    try:
        import coremltools as ct

        model_spec = coreml_model._spec

        # Add author
        if 'author' in metadata:
            model_spec.description.metadata.author = metadata['author']

        # Add license
        if 'license' in metadata:
            model_spec.description.metadata.license = metadata['license']

        # Add short description
        if 'short_description' in metadata:
            model_spec.description.metadata.shortDescription = metadata['short_description']

        # Add version
        if 'version' in metadata:
            model_spec.description.metadata.versionString = metadata['version']

        logger.info("Metadata added to model")

        return coreml_model

    except Exception as e:
        logger.warning(f"Failed to add metadata: {e}")
        return coreml_model


def verify_coreml_model(coreml_path: Path, num_tests: int = 5):
    """Verify Core ML model can run predictions"""
    try:
        import coremltools as ct
    except ImportError:
        logger.error("coremltools not installed")
        return False

    logger.info("Verifying Core ML model...")

    # Load model
    model = ct.models.MLModel(str(coreml_path))

    # Get input spec
    spec = model.get_spec()
    input_name = spec.description.input[0].name
    input_shape = spec.description.input[0].type.multiArrayType.shape

    logger.info(f"Input: {input_name}, shape: {list(input_shape)}")

    # Run test predictions
    for i in range(num_tests):
        # Create dummy input
        dummy_input = {
            input_name: np.random.randn(*input_shape).astype(np.float32)
        }

        # Predict
        output = model.predict(dummy_input)

        logger.info(f"Test {i+1}: Output keys = {list(output.keys())}")

    logger.info("✓ Core ML model verification complete")
    return True


def benchmark_coreml_model(coreml_path: Path, num_runs: int = 100):
    """Benchmark Core ML model inference time"""
    try:
        import coremltools as ct
        import time
    except ImportError:
        logger.error("coremltools not installed")
        return None

    logger.info(f"Benchmarking Core ML model ({num_runs} runs)...")

    # Load model
    model = ct.models.MLModel(str(coreml_path))

    # Get input spec
    spec = model.get_spec()
    input_name = spec.description.input[0].name
    input_shape = spec.description.input[0].type.multiArrayType.shape

    # Dummy input
    dummy_input = {
        input_name: np.random.randn(*input_shape).astype(np.float32)
    }

    # Warmup
    for _ in range(10):
        _ = model.predict(dummy_input)

    # Benchmark
    times = []
    for _ in range(num_runs):
        start = time.time()
        _ = model.predict(dummy_input)
        times.append(time.time() - start)

    avg_time = np.mean(times) * 1000  # Convert to ms
    std_time = np.std(times) * 1000

    logger.info(f"Average inference time: {avg_time:.2f} ± {std_time:.2f} ms")
    logger.info(f"Min: {min(times)*1000:.2f} ms, Max: {max(times)*1000:.2f} ms")

    return avg_time


def create_package_for_xcode(coreml_path: Path, output_dir: Path, model_name: str):
    """
    Create a package ready for Xcode integration
    Includes model, metadata, and usage instructions
    """
    logger.info("Creating Xcode-ready package...")

    output_dir.mkdir(parents=True, exist_ok=True)

    # Copy model
    import shutil
    model_dest = output_dir / f"{model_name}.mlmodel"
    shutil.copy(coreml_path, model_dest)

    # Create README
    readme_content = f"""# {model_name} - Core ML Model

## Integration in Xcode

1. Drag and drop `{model_name}.mlmodel` into your Xcode project
2. Xcode will automatically generate a Swift/Objective-C interface

## Usage Example (Swift)

```swift
import CoreML

// Load model
guard let model = try? {model_name}() else {{
    fatalError("Failed to load model")
}}

// Prepare input
// Replace with actual audio preprocessing
let inputFeatures = try! MLMultiArray(shape: [1, 80, 3000], dataType: .float32)

// Run inference
guard let output = try? model.prediction(input_features: inputFeatures) else {{
    fatalError("Prediction failed")
}}

// Process output
let logits = output.last_hidden_state
```

## Model Information

- Input: Mel spectrogram features (80 x 3000)
- Output: Hidden states for decoder
- Optimized for: Neural Engine (iOS 15+)
- Model size: Check file size

## Performance Tips

1. Use Neural Engine for best performance (automatic on A14+ chips)
2. Batch multiple predictions if possible
3. Consider preprocessing audio on background thread

## Requirements

- iOS 15.0+
- Xcode 13.0+

"""

    readme_path = output_dir / "README.md"
    with open(readme_path, 'w') as f:
        f.write(readme_content)

    logger.info(f"Xcode package created at {output_dir}")
    logger.info("Package contents:")
    for item in output_dir.iterdir():
        logger.info(f"  - {item.name}")


def main():
    parser = argparse.ArgumentParser(description='Convert to Core ML for iOS')
    parser.add_argument('--onnx_path', type=str, required=True,
                        help='Path to ONNX model')
    parser.add_argument('--output_path', type=str, required=True,
                        help='Output path for Core ML model')
    parser.add_argument('--quantize', action='store_true', default=True,
                        help='Apply INT8 quantization')
    parser.add_argument('--compute_units', type=str, default='CPU_AND_NE',
                        choices=['ALL', 'CPU_ONLY', 'CPU_AND_GPU', 'CPU_AND_NE'],
                        help='Compute units to use')
    parser.add_argument('--verify', action='store_true',
                        help='Verify Core ML model')
    parser.add_argument('--benchmark', action='store_true',
                        help='Benchmark Core ML model')
    parser.add_argument('--create_package', action='store_true',
                        help='Create Xcode-ready package')
    parser.add_argument('--model_name', type=str, default='WhisperEncoder',
                        help='Model name for package')

    # Metadata
    parser.add_argument('--author', type=str, help='Model author')
    parser.add_argument('--license', type=str, default='MIT', help='Model license')
    parser.add_argument('--description', type=str,
                        help='Model description')
    parser.add_argument('--version', type=str, default='1.0',
                        help='Model version')

    args = parser.parse_args()

    logger.info(f"""
╔════════════════════════════════════════════════════════════╗
║            Core ML Conversion for iOS/macOS               ║
╚════════════════════════════════════════════════════════════╝
    """)

    onnx_path = Path(args.onnx_path)
    output_path = Path(args.output_path)

    if not onnx_path.exists():
        logger.error(f"ONNX model not found: {onnx_path}")
        return

    # Convert
    coreml_model = convert_onnx_to_coreml(
        onnx_path,
        output_path,
        quantize=args.quantize,
        compute_units=args.compute_units
    )

    if coreml_model is None:
        return

    # Add metadata
    metadata = {
        'author': args.author or 'ASR Training Pipeline',
        'license': args.license,
        'short_description': args.description or 'Whisper ASR Encoder for Hindi/Punjabi',
        'version': args.version
    }
    coreml_model = add_metadata(coreml_model, metadata)
    coreml_model.save(str(output_path))

    # Verify
    if args.verify:
        verify_coreml_model(output_path)

    # Benchmark
    if args.benchmark:
        benchmark_coreml_model(output_path)

    # Create Xcode package
    if args.create_package:
        package_dir = output_path.parent / f"{args.model_name}_package"
        create_package_for_xcode(output_path, package_dir, args.model_name)

    logger.info("\n✓ Core ML conversion complete!")
    logger.info(f"Model saved to: {output_path}")
    logger.info("\nNext step: Integrate into iOS app")
    logger.info(f"  - Drag {output_path.name} into Xcode project")
    logger.info(f"  - Xcode will generate Swift/Obj-C interface automatically")


if __name__ == '__main__':
    main()
