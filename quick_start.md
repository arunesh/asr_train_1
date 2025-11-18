# Quick Start Guide - Phase 1 Implementation

This guide will get you started with Phase 1 of the mobile ASR pipeline.

## Prerequisites

- Python 3.9+
- CUDA-capable GPU (recommended: 24GB+ VRAM)
- 500GB+ free disk space
- Internet connection for dataset downloads

## Setup (5 minutes)

```bash
# 1. Run setup script
bash setup.sh

# 2. Activate virtual environment
source venv/bin/activate

# 3. (Optional) Configure Weights & Biases
wandb login
```

## Phase 1 Workflow

### Week 1: Data Preparation

#### Step 1: Download Datasets

**For Hindi:**
```bash
# Download OpenSLR datasets (automatic)
python data/download_datasets.py --language hindi

# Manually download Common Voice:
# 1. Go to https://commonvoice.mozilla.org/datasets
# 2. Download Hindi dataset
# 3. Extract to: data/raw/hindi/common_voice/

# Verify downloads
python data/download_datasets.py --language hindi --verify_only
```

**For Punjabi:**
```bash
python data/download_datasets.py --language punjabi

# Manually download Common Voice Punjabi
# Extract to: data/raw/punjabi/common_voice/
```

#### Step 2: Preprocess Data

```bash
# Hindi preprocessing (~30-60 minutes)
python data/preprocess.py --language hindi

# Punjabi preprocessing (~15-30 minutes)
python data/preprocess.py --language punjabi
```

This will:
- Resample audio to 16kHz
- Normalize audio levels
- Filter low-quality samples
- Create train/val/test splits
- Generate manifest files

#### Step 3: Baseline Evaluation

```bash
# Test zero-shot Whisper Tiny on Hindi
python evaluation/evaluate.py \
    --model_path openai/whisper-tiny \
    --test_manifest data/manifests/hindi_test.json \
    --language hindi \
    --baseline \
    --output_dir evaluation/results/hindi_baseline

# Test on Punjabi
python evaluation/evaluate.py \
    --model_path openai/whisper-tiny \
    --test_manifest data/manifests/punjabi_test.json \
    --language punjabi \
    --baseline \
    --output_dir evaluation/results/punjabi_baseline
```

Expected baseline WER:
- Hindi: 40-60%
- Punjabi: 60-80%

### Week 2: Hindi Fine-tuning

```bash
# Start training (24-48 hours on A100)
python training/scripts/train_whisper.py \
    --config training/configs/whisper_tiny_hindi.yaml

# Monitor training:
# - TensorBoard: tensorboard --logdir models/whisper_hindi/runs
# - Weights & Biases: Check your W&B dashboard

# Evaluate best checkpoint
python evaluation/evaluate.py \
    --model_path models/whisper_hindi/best_checkpoint \
    --test_manifest data/manifests/hindi_test.json \
    --language hindi \
    --output_dir evaluation/results/hindi_finetuned
```

Expected results:
- WER: 15-25%
- Training time: 24-48 hours (A100)

### Week 3: Punjabi Fine-tuning

```bash
# Train Punjabi model (12-24 hours on A100)
python training/scripts/train_whisper.py \
    --config training/configs/whisper_tiny_punjabi.yaml

# Evaluate
python evaluation/evaluate.py \
    --model_path models/whisper_punjabi/best_checkpoint \
    --test_manifest data/manifests/punjabi_test.json \
    --language punjabi \
    --output_dir evaluation/results/punjabi_finetuned
```

Expected results:
- WER: 25-35%
- Training time: 12-24 hours (A100)

### Week 4: Optimization & Conversion

#### Step 1: Quantization

```bash
# Quantize Hindi model
python optimization/quantize.py \
    --model_path models/whisper_hindi/best_checkpoint \
    --output_path optimization/whisper_hindi_int8 \
    --method dynamic \
    --benchmark \
    --test_manifest data/manifests/hindi_test.json

# Quantize Punjabi model
python optimization/quantize.py \
    --model_path models/whisper_punjabi/best_checkpoint \
    --output_path optimization/whisper_punjabi_int8 \
    --method dynamic
```

#### Step 2: Convert to ONNX

```bash
# Hindi to ONNX
python optimization/convert_to_onnx.py \
    --model_path optimization/whisper_hindi_int8 \
    --output_path optimization/onnx/hindi \
    --method optimum \
    --optimize \
    --verify

# Punjabi to ONNX
python optimization/convert_to_onnx.py \
    --model_path optimization/whisper_punjabi_int8 \
    --output_path optimization/onnx/punjabi \
    --method optimum \
    --optimize
```

#### Step 3: Convert to TFLite (Android)

```bash
# Hindi TFLite
python optimization/convert_to_tflite.py \
    --model_name optimization/whisper_hindi_int8 \
    --output_dir mobile/android/hindi \
    --full_pipeline \
    --quantize \
    --verify \
    --benchmark

# Punjabi TFLite
python optimization/convert_to_tflite.py \
    --model_name optimization/whisper_punjabi_int8 \
    --output_dir mobile/android/punjabi \
    --full_pipeline \
    --quantize
```

#### Step 4: Convert to Core ML (iOS)

```bash
# Hindi Core ML
python optimization/convert_to_coreml.py \
    --onnx_path optimization/onnx/hindi/encoder_model.onnx \
    --output_path mobile/ios/WhisperEncoderHindi.mlmodel \
    --quantize \
    --compute_units CPU_AND_NE \
    --verify \
    --benchmark \
    --create_package \
    --model_name WhisperHindiEncoder

# Punjabi Core ML
python optimization/convert_to_coreml.py \
    --onnx_path optimization/onnx/punjabi/encoder_model.onnx \
    --output_path mobile/ios/WhisperEncoderPunjabi.mlmodel \
    --quantize \
    --compute_units CPU_AND_NE \
    --create_package \
    --model_name WhisperPunjabiEncoder
```

## Troubleshooting

### Out of Memory (OOM) Errors

```yaml
# Edit training config, reduce batch size:
training:
  batch_size: 8  # Reduce from 16
  gradient_accumulation_steps: 8  # Increase to maintain effective batch size
```

### Slow Training

- Ensure GPU is being used: `nvidia-smi`
- Enable mixed precision: `fp16: true` in config
- Use gradient checkpointing (in config)

### Download Issues

- Use VPN if datasets are blocked
- Manually download from mirrors
- Check disk space

### Conversion Errors

```bash
# Install all conversion dependencies
pip install onnx onnxruntime optimum[exporters]
pip install tensorflow tf2onnx
pip install coremltools onnx-coreml
```

## Expected Timeline

| Week | Task | Hours | Output |
|------|------|-------|--------|
| 1 | Data prep | 4-8 | Preprocessed datasets |
| 2 | Hindi training | 24-48 | Fine-tuned Hindi model |
| 3 | Punjabi training | 12-24 | Fine-tuned Punjabi model |
| 4 | Optimization | 8-16 | Mobile-ready models |

**Total GPU time:** 44-88 hours
**Total cost (A100 @ $1.10/hr):** ~$50-100

## Success Criteria

✓ Hindi WER < 30%
✓ Punjabi WER < 40%
✓ Model size < 60MB
✓ RTF < 0.5 on mobile

## Next Steps After Phase 1

1. Review metrics and results
2. Decide on Phase 2 (custom model)
3. Integrate models into production apps
4. Collect user feedback
5. Plan improvements

## Getting Help

- Check logs: `models/whisper_*/logs/`
- Review W&B dashboard
- Read full documentation: `README.md`
- Consult plan: `MOBILE_ASR_PIPELINE_PLAN.md`

Good luck! 🚀
