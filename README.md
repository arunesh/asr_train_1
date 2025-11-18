# Mobile ASR Training Pipeline - Phase 1

This repository implements Phase 1 of the mobile speech recognition pipeline for Hindi and Punjabi languages, focusing on fine-tuning Whisper Tiny for mobile deployment.

## Project Structure

```
asr_train_1/
├── data/
│   ├── raw/              # Downloaded datasets
│   ├── processed/        # Preprocessed audio files
│   └── manifests/        # Train/val/test split manifests
├── models/
│   ├── whisper_hindi/    # Hindi model checkpoints
│   └── whisper_punjabi/  # Punjabi model checkpoints
├── training/
│   ├── configs/          # Training configurations
│   ├── scripts/          # Training scripts
│   └── logs/            # Training logs
├── optimization/         # Quantization and conversion scripts
├── evaluation/          # Evaluation scripts and results
├── mobile/
│   ├── android/         # Android test app
│   └── ios/             # iOS test app
└── notebooks/           # Analysis notebooks
```

## Setup

### 1. Environment Setup

```bash
# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# For GPU support, ensure CUDA is installed
# Check: nvidia-smi
```

### 2. Configure Weights & Biases (optional but recommended)

```bash
wandb login
# Or set WANDB_API_KEY environment variable
```

## Phase 1 Workflow

### Week 1: Data Preparation

```bash
# Download datasets
python data/download_datasets.py --language hindi
python data/download_datasets.py --language punjabi

# Preprocess datasets
python data/preprocess.py --language hindi --output data/processed/hindi
python data/preprocess.py --language punjabi --output data/processed/punjabi

# Create train/val/test splits
python data/create_splits.py --language hindi
python data/create_splits.py --language punjabi

# Test baseline (zero-shot)
python evaluation/evaluate_baseline.py --language hindi
python evaluation/evaluate_baseline.py --language punjabi
```

### Week 2: Hindi Fine-tuning

```bash
# Fine-tune Whisper Tiny on Hindi
python training/scripts/train_whisper.py \
    --config training/configs/whisper_tiny_hindi.yaml \
    --output_dir models/whisper_hindi

# Evaluate
python evaluation/evaluate.py \
    --model_path models/whisper_hindi/best_checkpoint \
    --test_manifest data/manifests/hindi_test.json
```

### Week 3: Punjabi Fine-tuning

```bash
# Fine-tune Whisper Tiny on Punjabi
python training/scripts/train_whisper.py \
    --config training/configs/whisper_tiny_punjabi.yaml \
    --output_dir models/whisper_punjabi

# Evaluate
python evaluation/evaluate.py \
    --model_path models/whisper_punjabi/best_checkpoint \
    --test_manifest data/manifests/punjabi_test.json
```

### Week 4: Optimization & Mobile Conversion

```bash
# Quantize models (INT8)
python optimization/quantize.py \
    --model_path models/whisper_hindi/best_checkpoint \
    --output optimization/whisper_hindi_int8

# Convert to ONNX
python optimization/convert_to_onnx.py \
    --model_path optimization/whisper_hindi_int8 \
    --output optimization/whisper_hindi.onnx

# Convert to TensorFlow Lite
python optimization/convert_to_tflite.py \
    --onnx_path optimization/whisper_hindi.onnx \
    --output mobile/android/whisper_hindi.tflite

# Convert to Core ML
python optimization/convert_to_coreml.py \
    --onnx_path optimization/whisper_hindi.onnx \
    --output mobile/ios/whisper_hindi.mlmodel
```

## GPU Requirements

- Minimum: 1x NVIDIA RTX 3090 (24GB VRAM)
- Recommended: 1x NVIDIA A100 (40GB VRAM)
- Training time: ~24-48 hours per language on A100

## Expected Results

### Phase 1 Targets

| Metric | Hindi | Punjabi |
|--------|-------|---------|
| WER | <30% | <40% |
| Model Size | <60MB | <60MB |
| RTF (mobile) | <0.5 | <0.5 |

## Troubleshooting

### CUDA Out of Memory
- Reduce batch size in config file
- Enable gradient checkpointing
- Use mixed precision training (FP16)

### Dataset Download Issues
- Check internet connection
- Verify dataset availability
- Try manual download from sources

### Conversion Errors
- Ensure all dependencies are installed
- Check ONNX opset compatibility
- Verify model exports correctly

## Next Steps

After completing Phase 1:
1. Review results and metrics
2. Decide on Phase 2 implementation
3. Plan for custom lightweight model if needed

## References

- [Whisper Paper](https://arxiv.org/abs/2212.04356)
- [Common Voice](https://commonvoice.mozilla.org/)
- [OpenSLR](https://www.openslr.org/)
- [ULCA](https://bhashini.gov.in/ulca)

## License

This project follows the licenses of the underlying datasets and models used.
