# Mobile Speech Recognition Pipeline Plan
## Punjabi and Hindi ASR for Android/iOS

**Version:** 1.0
**Date:** 2025-11-17
**Target Languages:** Punjabi, Hindi

---

## Executive Summary

This document outlines a comprehensive plan to develop a production-ready speech-to-text (ASR) system for Punjabi and Hindi languages, optimized for mobile deployment on Android and iOS devices. The pipeline covers dataset acquisition, model selection, training infrastructure, optimization, and deployment.

**Key Objectives:**
- Build accurate ASR models for Punjabi and Hindi
- Ensure models are small enough for on-device inference (<100MB)
- Achieve real-time performance on mobile devices
- Provide clear implementation roadmap from data to deployment

---

## Table of Contents

1. [Background & Constraints](#1-background--constraints)
2. [Dataset Strategy](#2-dataset-strategy)
3. [Model Architecture Analysis](#3-model-architecture-analysis)
4. [Recommended Approach](#4-recommended-approach)
5. [Training Infrastructure](#5-training-infrastructure)
6. [Model Optimization & Conversion](#6-model-optimization--conversion)
7. [Implementation Roadmap](#7-implementation-roadmap)
8. [Success Metrics](#8-success-metrics)
9. [Risk Mitigation](#9-risk-mitigation)

---

## 1. Background & Constraints

### 1.1 Mobile Device Constraints

**Computational Limits:**
- CPU: ARM-based processors (limited FLOPS)
- RAM: 2-8GB available, but shared with OS and apps
- Storage: Model size ideally <50MB, max 100MB
- Battery: Inference must be energy-efficient
- Latency: Real-time factor (RTF) < 0.3 for good UX

**OS-Specific Considerations:**
- **Android:** TensorFlow Lite, ONNX Runtime Mobile
- **iOS:** Core ML (can convert from ONNX), ONNX Runtime

### 1.2 Language-Specific Challenges

**Hindi:**
- Large speaker base (~600M speakers)
- Code-mixing with English common
- Multiple dialects and accents
- Decent data availability

**Punjabi:**
- ~125M speakers
- Two scripts: Gurmukhi (India), Shahmukhi (Pakistan)
- Limited open-source data
- Regional accent variations

---

## 2. Dataset Strategy

### 2.1 Hindi Dataset Sources

| Dataset | Hours | License | Quality | Access |
|---------|-------|---------|---------|--------|
| **Common Voice Hindi** | ~100-150h | CC0 | Good | Free |
| **OpenSLR Hindi** | ~40h | Apache 2.0 | High | Free |
| **ULCA (Bhashini)** | ~500-1000h | Varies | Mixed | Registration |
| **Microsoft Speech Corpus (Hindi)** | ~6h | Research | High | Free |
| **IIT-M TTS Corpus** | ~5-10h | Custom | High | Request |
| **Shrutilipi** | Variable | Commercial | High | Paid |

**Recommended Action:**
- Start with Common Voice + OpenSLR (~140-190h)
- Request access to ULCA for additional data
- Budget for Shrutilipi if quality needs improvement

### 2.2 Punjabi Dataset Sources

| Dataset | Hours | License | Quality | Access |
|---------|-------|---------|---------|--------|
| **Common Voice Punjabi** | ~20-40h | CC0 | Good | Free |
| **OpenSLR Punjabi** | Limited | Apache 2.0 | High | Free |
| **ULCA (Bhashini)** | ~100-300h | Varies | Mixed | Registration |
| **Punjabi University Corpus** | Variable | Custom | High | Request |
| **Custom Collection** | TBD | Custom | TBD | Create |

**Recommended Action:**
- Start with Common Voice (~20-40h) - **data scarcity issue**
- Register for ULCA access
- **Consider data augmentation heavily**
- May need custom data collection (~50-100h target)

### 2.3 Data Preparation Pipeline

```
Raw Audio → Quality Filter → Normalization → Augmentation → Train/Val/Test Split
```

**Quality Filtering:**
- Remove clips with SNR < 15dB
- Filter out silence-only or music clips
- Validate transcription alignment
- Remove clips <1s or >20s

**Normalization:**
- Resample to 16kHz mono
- Normalize audio levels (-23 LUFS)
- Text normalization (numbers, punctuation, abbreviations)
- Script normalization (for Punjabi: standardize to Gurmukhi)

**Augmentation Strategy:**
- Speed perturbation (0.9x, 1.0x, 1.1x)
- SpecAugment (time & frequency masking)
- Background noise addition (SNR 10-20dB)
- Room reverberation simulation
- **Target:** 3-5x effective data increase

---

## 3. Model Architecture Analysis

### 3.1 Option A: Fine-tune Whisper

**Architecture:** Encoder-Decoder Transformer (OpenAI)

**Pros:**
- **Strong baseline:** Pre-trained on 680k hours multilingual data
- **Includes Hindi:** Already has Hindi in training data
- **Transfer learning:** Good starting point for low-resource languages
- **Multiple sizes:** tiny (39M), base (74M), small (244M)
- **Active community:** Good tooling and support (faster-whisper, whisper.cpp)

**Cons:**
- **Whisper Tiny still challenging:** ~39M params = ~150MB FP32 (need quantization)
- **Not optimized for mobile:** Original design for cloud inference
- **Decoder overhead:** Autoregressive decoding slower than CTC
- **Punjabi support:** Limited/no Punjabi in pre-training
- **Hallucination issues:** Can generate plausible but wrong text

**Mobile Viability:**
- Whisper Tiny: Possible with INT8 quantization (~40-50MB)
- Whisper Base: Borderline, requires aggressive optimization
- Real-time performance: Challenging on lower-end devices

**Estimated Effort:** 3-4 weeks
- 1 week: Data prep + baseline
- 2 weeks: Fine-tuning experiments
- 1 week: Optimization + conversion

### 3.2 Option B: Fine-tune Parakeet (NVIDIA NeMo)

**Architecture:** CTC-based Conformer (NVIDIA)

**Pros:**
- **Streaming-friendly:** CTC decoding, no autoregressive
- **Multiple sizes:** parakeet-rnnt-0.6B, parakeet-tdt-1.1B, parakeet-ctc-*
- **NeMo framework:** Excellent training tools and recipes
- **Better for mobile:** CTC is faster than encoder-decoder
- **Modular:** Can use smaller encoder backbones

**Cons:**
- **Larger base models:** Even "small" versions are 100M+ params
- **Less multilingual:** Primarily English pre-training
- **Limited Hindi/Punjabi:** Need full fine-tuning, less transfer benefit
- **Custom architecture:** May need to design smaller variant

**Mobile Viability:**
- Standard Parakeet: Too large for mobile
- **Custom small variant:** 10-30M params feasible
- Need to build from smaller conformer blocks

**Estimated Effort:** 4-6 weeks
- 1 week: Data prep
- 1 week: Architecture design (small variant)
- 2-3 weeks: Training from scratch/fine-tuning
- 1 week: Optimization + conversion

### 3.3 Option C: Train Custom Lightweight Model

**Architecture:** Small Conformer/QuartzNet + CTC

**Pros:**
- **Full control:** Design specifically for mobile constraints
- **Optimized size:** Target 10-30M parameters from start
- **Streaming-capable:** CTC decoding
- **No bloat:** Only what's needed for target languages
- **Educational:** Deep understanding of pipeline

**Cons:**
- **No pre-training benefit:** Start from random initialization
- **Longer training:** Need more data and compute time
- **Higher risk:** No proven baseline
- **More expertise required:** Need strong ASR knowledge
- **Potentially lower accuracy:** Especially with limited data

**Architecture Options:**

1. **Conformer-Small:**
   - 4-8 Conformer blocks
   - 256-384 hidden dim
   - ~15-25M parameters
   - Proven architecture, good accuracy/size tradeoff

2. **QuartzNet/Citrinet:**
   - 1D time-channel separable convolutions
   - Very efficient for mobile
   - ~10-15M parameters
   - Slightly lower accuracy than Conformer

3. **Emformer (Streaming):**
   - Memory-augmented Transformer
   - Designed for streaming
   - ~20-30M parameters
   - Better for real-time applications

**Estimated Effort:** 6-8 weeks
- 1 week: Data prep
- 2 weeks: Architecture design + implementation
- 3-4 weeks: Training experiments + hyperparameter tuning
- 1 week: Optimization + conversion

### 3.4 Comparison Matrix

| Criterion | Whisper Tiny | Parakeet Custom | From Scratch | Weight |
|-----------|--------------|-----------------|--------------|--------|
| **Model Size** | ⭐⭐⭐ (39M) | ⭐⭐⭐⭐ (15-30M) | ⭐⭐⭐⭐⭐ (10-25M) | 25% |
| **Accuracy (Hindi)** | ⭐⭐⭐⭐⭐ | ⭐⭐⭐⭐ | ⭐⭐⭐ | 30% |
| **Accuracy (Punjabi)** | ⭐⭐⭐ | ⭐⭐⭐ | ⭐⭐⭐ | 25% |
| **Mobile Performance** | ⭐⭐⭐ | ⭐⭐⭐⭐ | ⭐⭐⭐⭐⭐ | 15% |
| **Time to Market** | ⭐⭐⭐⭐⭐ (fast) | ⭐⭐⭐⭐ | ⭐⭐⭐ (slow) | 5% |

**Weighted Scores:**
- **Whisper Tiny Fine-tuning:** 3.95/5 ⭐⭐⭐⭐
- **Parakeet Custom:** 3.75/5 ⭐⭐⭐⭐
- **From Scratch:** 3.40/5 ⭐⭐⭐

---

## 4. Recommended Approach

### 4.1 Two-Phase Strategy

**Phase 1: Whisper Tiny Fine-tuning (MVP - 4 weeks)**

**Rationale:**
- Fastest path to working prototype
- Proven architecture with strong baselines
- Good for Hindi (already in pre-training)
- Validates data pipeline and infrastructure
- Provides baseline for comparison

**Implementation:**
1. Fine-tune Whisper Tiny on Hindi (~150-200h data)
2. Fine-tune Whisper Tiny on Punjabi (~40-100h data)
3. Apply aggressive optimization:
   - INT8 post-training quantization
   - Encoder pruning (optional)
   - Convert to ONNX/TFLite/Core ML
4. Mobile integration and testing

**Expected Results:**
- Hindi WER: 15-25% (depending on test set)
- Punjabi WER: 25-35% (limited by data)
- Model size: 40-50MB (INT8)
- RTF on mobile: 0.2-0.4 (device dependent)

**Phase 2: Custom Lightweight Model (Production - 6-8 weeks)**

**Rationale:**
- Optimize for mobile-first design
- Better long-term performance and efficiency
- Full control over architecture
- Potential for on-device training/adaptation

**Implementation:**
1. Design Conformer-Small architecture (15-20M params):
   - 6 Conformer blocks
   - 256 hidden dimensions
   - 4 attention heads
   - CTC decoder with language-specific vocabulary

2. Training strategy:
   - Curriculum learning (start with clean data)
   - Progressive speed augmentation
   - Multi-task learning (Hindi + Punjabi jointly)
   - Knowledge distillation from Whisper Tiny

3. Optimization from training:
   - Train with quantization-aware training (QAT)
   - Structured pruning during training
   - Efficient attention mechanisms

4. Deployment optimization:
   - INT8 quantization
   - ONNX/TFLite conversion
   - Mobile runtime optimization

**Expected Results:**
- Hindi WER: 18-28%
- Punjabi WER: 28-38%
- Model size: 20-30MB (INT8)
- RTF on mobile: 0.1-0.25
- Better battery efficiency

### 4.2 Hybrid Approach (Optional)

**Knowledge Distillation:**
- Use fine-tuned Whisper Tiny as teacher
- Train smaller student model (10-15M params)
- Best of both worlds: accuracy + efficiency

**Multi-stage Training:**
1. Pre-train small model on large English ASR dataset
2. Intermediate fine-tuning on multilingual data
3. Final fine-tuning on Hindi/Punjabi

---

## 5. Training Infrastructure

### 5.1 Hardware Requirements

**Minimum Setup:**
- GPU: 1x NVIDIA RTX 3090 / A5000 (24GB VRAM)
- CPU: 16+ cores
- RAM: 64GB
- Storage: 500GB SSD (for datasets + checkpoints)

**Recommended Setup:**
- GPU: 2-4x NVIDIA A100 (40GB) or 4x RTX 4090
- CPU: 32+ cores
- RAM: 128GB+
- Storage: 1TB NVMe SSD + backup storage

**Cloud Options:**
- AWS: p3.2xlarge (1x V100) or p4d.24xlarge (8x A100)
- GCP: a2-highgpu-1g (1x A100) or a2-ultragpu-1g
- Azure: Standard_NC24ads_A100_v4
- Lambda Labs: 1x A100 instance (~$1.10/hr)

**Cost Estimate:**
- Phase 1 (Whisper fine-tuning): ~$200-400 (40-80 GPU hours)
- Phase 2 (Custom model): ~$600-1200 (120-240 GPU hours)

### 5.2 Software Stack

**Framework Options:**

| Framework | Pros | Cons | Recommendation |
|-----------|------|------|----------------|
| **NeMo** | Best ASR tools, production-ready | Steeper learning curve | ⭐ Primary choice |
| **HuggingFace** | Easy to start, great for Whisper | Less optimized training | ⭐ For Whisper fine-tuning |
| **ESPnet** | Research-friendly, flexible | Complex setup | Backup option |
| **Coqui STT** | Simple, streaming | Less active development | Not recommended |

**Recommended Stack:**
```
Training:
- Framework: NVIDIA NeMo (Phase 2) + HuggingFace (Phase 1)
- Training: PyTorch 2.0+ with CUDA 11.8+
- Distributed: PyTorch DDP / Horovod
- Logging: Weights & Biases / TensorBoard
- Experiment tracking: MLflow

Optimization:
- ONNX Runtime for conversion
- TensorFlow for TFLite conversion
- Core ML Tools for iOS

Deployment:
- Android: TFLite / ONNX Runtime Mobile
- iOS: Core ML / ONNX Runtime Mobile
```

### 5.3 Training Configuration

**Phase 1: Whisper Tiny Fine-tuning**

```yaml
model:
  architecture: whisper-tiny
  parameters: 39M

data:
  sample_rate: 16000
  max_duration: 20
  min_duration: 1

training:
  optimizer: AdamW
  learning_rate: 1e-5
  warmup_steps: 500
  batch_size: 16 (per GPU)
  gradient_accumulation: 4
  max_epochs: 20-30
  early_stopping: 5 epochs

augmentation:
  speed_perturbation: [0.9, 1.0, 1.1]
  spec_augment: true
  noise_augment: true (SNR 10-20dB)
```

**Phase 2: Custom Conformer**

```yaml
model:
  architecture: conformer-ctc
  encoder:
    blocks: 6
    hidden_dim: 256
    attention_heads: 4
    conv_kernel_size: 31
    dropout: 0.1
  decoder:
    type: ctc
    vocabulary_size: ~5000 (Hindi/Punjabi combined)

training:
  optimizer: AdamW
  learning_rate: 5e-4
  warmup_steps: 10000
  schedule: tri_stage (warmup -> hold -> decay)
  batch_size: 32 (per GPU)
  gradient_accumulation: 2
  max_epochs: 100-150
  gradient_clipping: 1.0

regularization:
  weight_decay: 1e-3
  dropout: 0.1
  spec_augment_freq_masks: 2
  spec_augment_time_masks: 10
```

---

## 6. Model Optimization & Conversion

### 6.1 Optimization Pipeline

```
Trained Model (FP32) → Pruning → Quantization → ONNX → Platform Conversion
                          ↓          ↓              ↓            ↓
                      (optional)  (INT8/INT16)   (optimize)  (TFLite/CoreML)
```

### 6.2 Quantization Strategy

**Post-Training Quantization (PTQ):**
```python
# Phase 1 approach - quick and simple
- Dynamic quantization: Weights INT8, activations FP32
- Static quantization: Both weights and activations INT8
- Calibration: Use 1000-5000 samples from validation set
```

**Quantization-Aware Training (QAT):**
```python
# Phase 2 approach - better accuracy
- Simulate quantization during training
- Learn quantization-friendly weights
- 2-5% accuracy improvement over PTQ
```

**Expected Size Reduction:**
- FP32 (100MB) → FP16 (50MB) → INT8 (25MB)
- Minimal accuracy loss: <2% WER degradation

### 6.3 Platform Conversion

**Android (TensorFlow Lite):**

```python
# Conversion pipeline
PyTorch Model → ONNX → TensorFlow → TensorFlow Lite

# Steps:
1. Export to ONNX with opset 13+
2. Convert ONNX to TensorFlow (onnx-tf)
3. Convert TF to TFLite with quantization
4. Optimize for mobile (use TFLite ops)

# Validation:
- Compare outputs with original model
- Measure inference latency on target devices
- Profile memory usage
```

**iOS (Core ML):**

```python
# Conversion pipeline
PyTorch Model → ONNX → Core ML

# Steps:
1. Export to ONNX with opset 13+
2. Convert ONNX to Core ML (coremltools)
3. Quantize using coremltools (INT8 or mixed precision)
4. Optimize compute units (Neural Engine preferred)

# Validation:
- Test on iPhone 11+ and iPad
- Verify Neural Engine utilization
- Measure battery impact
```

**Alternative: ONNX Runtime Mobile (Cross-platform):**

```python
# Benefits:
- Single model for both platforms
- Good performance with optimizations
- Easier maintenance

# Trade-offs:
- Slightly larger app size
- May not fully utilize platform-specific accelerators
```

### 6.4 Optimization Checklist

**Model-Level:**
- [ ] Remove unnecessary layers (e.g., dropout in inference)
- [ ] Fuse operations (BatchNorm + Conv)
- [ ] Operator fusion (ONNX optimizer)
- [ ] Constant folding

**Quantization:**
- [ ] Apply INT8 quantization to encoder
- [ ] Consider mixed precision (INT8/INT16)
- [ ] Calibrate on representative data
- [ ] Validate accuracy on test set

**Runtime:**
- [ ] Enable platform-specific accelerators (Neural Engine, NNAPI)
- [ ] Optimize thread count for mobile CPUs
- [ ] Use memory-mapped model loading
- [ ] Implement streaming inference for long audio

**Size Reduction:**
- [ ] Vocabulary pruning (remove rare tokens)
- [ ] Weight sharing/clustering
- [ ] Structured pruning (10-30% sparsity)

**Expected Final Sizes:**
- Whisper Tiny optimized: 40-50MB
- Custom Conformer optimized: 20-30MB

---

## 7. Implementation Roadmap

### 7.1 Phase 1: MVP with Whisper Tiny (Weeks 1-4)

**Week 1: Infrastructure & Data Preparation**
- [ ] Set up GPU environment (cloud or local)
- [ ] Install NeMo, HuggingFace, ONNX tools
- [ ] Download Common Voice Hindi + Punjabi
- [ ] Download OpenSLR datasets
- [ ] Implement data preprocessing pipeline
- [ ] Create train/val/test splits (80/10/10)
- [ ] Set up Weights & Biases logging
- [ ] Baseline evaluation: Test Whisper Tiny zero-shot

**Deliverables:**
- Preprocessed datasets (manifests in JSON/CSV)
- Baseline WER scores (zero-shot)
- Training environment ready

**Week 2: Hindi Fine-tuning**
- [ ] Implement Whisper fine-tuning script (HuggingFace)
- [ ] Configure data augmentation pipeline
- [ ] Start Hindi fine-tuning (target: ~24-48h on 1x A100)
- [ ] Monitor training metrics (loss, WER)
- [ ] Run validation every epoch
- [ ] Select best checkpoint based on val WER
- [ ] Evaluate on test set

**Deliverables:**
- Fine-tuned Whisper Tiny (Hindi) checkpoint
- Training logs and metrics
- Test set WER report

**Week 3: Punjabi Fine-tuning + Optimization**
- [ ] Fine-tune Whisper Tiny on Punjabi (12-24h)
- [ ] Evaluate Punjabi model
- [ ] Implement INT8 quantization (PyTorch/ONNX)
- [ ] Export models to ONNX format
- [ ] Validate ONNX model accuracy
- [ ] Apply ONNX optimizations

**Deliverables:**
- Fine-tuned Whisper Tiny (Punjabi) checkpoint
- Quantized ONNX models (Hindi + Punjabi)
- Quantization accuracy analysis

**Week 4: Mobile Conversion & Testing**
- [ ] Convert ONNX to TensorFlow Lite (Android)
- [ ] Convert ONNX to Core ML (iOS)
- [ ] Validate converted models (accuracy)
- [ ] Create simple Android test app
- [ ] Create simple iOS test app
- [ ] Benchmark inference on devices:
  - Latency (RTF)
  - Memory usage
  - Battery consumption
- [ ] Document Phase 1 results

**Deliverables:**
- TFLite models (Hindi + Punjabi)
- Core ML models (Hindi + Punjabi)
- Mobile test apps
- Performance benchmarks
- Phase 1 final report

**Phase 1 Success Criteria:**
- Hindi WER < 30% on test set
- Punjabi WER < 40% on test set
- Model size < 60MB
- RTF < 0.5 on mid-range phones
- Successful on-device inference

---

### 7.2 Phase 2: Custom Lightweight Model (Weeks 5-12)

**Week 5: Architecture Design & Data Expansion**
- [ ] Design Conformer-Small architecture (15-20M params)
- [ ] Implement model in NeMo or PyTorch
- [ ] Request ULCA dataset access
- [ ] Explore additional data sources
- [ ] Implement advanced augmentation (SpecAugment++)
- [ ] Create combined Hindi+Punjabi vocabulary (~5k tokens)

**Deliverables:**
- Model architecture implementation
- Expanded dataset (target: 300-500h total)
- Shared vocabulary for both languages

**Week 6-7: Initial Training**
- [ ] Implement CTC loss and decoder
- [ ] Configure training pipeline (NeMo config)
- [ ] Start training from scratch
- [ ] Monitor convergence
- [ ] Run ablation studies (model size, augmentation)
- [ ] Implement learning rate scheduling

**Deliverables:**
- Training pipeline
- Initial checkpoints
- Training curves and analysis

**Week 8: Knowledge Distillation (Optional)**
- [ ] Implement knowledge distillation framework
- [ ] Use Whisper Tiny as teacher
- [ ] Train student model with distillation loss
- [ ] Compare with baseline (no distillation)

**Deliverables:**
- Distilled model checkpoint
- Comparison report

**Week 9: Quantization-Aware Training**
- [ ] Implement QAT in training loop
- [ ] Fine-tune with QAT for 5-10 epochs
- [ ] Export to INT8-optimized ONNX
- [ ] Validate accuracy retention

**Deliverables:**
- QAT-trained checkpoint
- INT8 ONNX model
- QAT vs PTQ comparison

**Week 10: Mobile Optimization**
- [ ] Convert to TFLite with optimizations
- [ ] Convert to Core ML with optimizations
- [ ] Implement streaming inference
- [ ] Apply vocabulary pruning
- [ ] Test structured pruning (if size too large)

**Deliverables:**
- Optimized mobile models
- Streaming inference implementation

**Week 11: Mobile Integration & Testing**
- [ ] Integrate into production-quality mobile apps
- [ ] Implement audio preprocessing on-device
- [ ] Test on diverse device range:
  - Low-end: Samsung A-series, iPhone 11
  - Mid-range: Pixel 6, iPhone 13
  - High-end: Samsung S23, iPhone 15
- [ ] Conduct user testing (accuracy perception)
- [ ] Measure battery impact in real usage

**Deliverables:**
- Production mobile apps
- Comprehensive device benchmarks
- User testing results

**Week 12: Documentation & Deployment**
- [ ] Create deployment guide
- [ ] Document training pipeline
- [ ] Create model cards (Hindi + Punjabi)
- [ ] Package models for distribution
- [ ] Write technical report
- [ ] Plan next iteration improvements

**Deliverables:**
- Complete documentation
- Deployment packages
- Final technical report
- Roadmap for v2

**Phase 2 Success Criteria:**
- Hindi WER < 25% on test set
- Punjabi WER < 35% on test set
- Model size < 40MB
- RTF < 0.3 on mid-range phones
- Battery usage < 5% per hour of continuous use

---

### 7.3 Timeline Overview

```
Month 1:
├─ Week 1: Setup & Data Prep
├─ Week 2: Hindi Fine-tuning
├─ Week 3: Punjabi Fine-tuning + Quantization
└─ Week 4: Mobile Conversion & Testing
   └─ [Phase 1 Complete - MVP Ready]

Month 2:
├─ Week 5: Architecture Design
├─ Week 6-7: Initial Training
└─ Week 8: Knowledge Distillation

Month 3:
├─ Week 9: QAT
├─ Week 10: Mobile Optimization
├─ Week 11: Integration & Testing
└─ Week 12: Documentation & Deployment
   └─ [Phase 2 Complete - Production Ready]
```

---

## 8. Success Metrics

### 8.1 Model Quality Metrics

**Primary Metrics:**
- **Word Error Rate (WER):** Main accuracy metric
  - Hindi Target: 15-25% (Phase 1), <20% (Phase 2)
  - Punjabi Target: 25-35% (Phase 1), <30% (Phase 2)

**Secondary Metrics:**
- **Character Error Rate (CER):** Useful for Indic scripts
- **Sentence Error Rate (SER):** Measure complete sentence accuracy
- **Real-time Factor (RTF):** Inference speed metric
  - Target: <0.3 (process 1s audio in <300ms)

**Robustness Metrics:**
- Performance on noisy audio (SNR 5-15dB)
- Performance on accented speech
- Code-mixing handling (Hindi-English)
- Long-form audio accuracy (>1 minute)

### 8.2 Mobile Performance Metrics

**Latency:**
- Cold start time: <2s
- First token latency: <500ms
- Streaming latency: <200ms per chunk
- RTF: <0.3 average

**Resource Usage:**
- Model size: <50MB (Phase 1), <40MB (Phase 2)
- RAM usage: <200MB during inference
- Battery drain: <5% per hour continuous use
- CPU usage: <30% on one core

**Compatibility:**
- Android: Version 8.0+ (API 26+)
- iOS: Version 13+
- Support 95%+ of devices in target markets

### 8.3 Business Metrics

**User Experience:**
- User satisfaction: >4.0/5.0
- Perceived accuracy: >80% "accurate" or "very accurate"
- Feature adoption: >40% of users try ASR feature

**Technical:**
- Crash rate: <0.1%
- API success rate: >99.9%
- Model update success rate: >95%

---

## 9. Risk Mitigation

### 9.1 Data Risks

| Risk | Probability | Impact | Mitigation |
|------|-------------|--------|------------|
| Insufficient Punjabi data | High | High | 1. Heavy augmentation<br>2. Multi-task learning with Hindi<br>3. Custom data collection<br>4. Explore paid datasets |
| Poor data quality | Medium | High | 1. Rigorous quality filtering<br>2. Manual data validation (sample)<br>3. Active learning for cleanup |
| License issues | Low | Medium | 1. Verify all licenses upfront<br>2. Maintain attribution records<br>3. Use only permissive licenses |
| Data bias | Medium | Medium | 1. Analyze demographic distribution<br>2. Balance accent/gender/age<br>3. Test on diverse speakers |

### 9.2 Technical Risks

| Risk | Probability | Impact | Mitigation |
|------|-------------|--------|------------|
| Model too large for mobile | Medium | High | 1. Start with size constraints<br>2. Iterative pruning<br>3. Fallback to smaller architecture |
| Poor mobile performance | Medium | High | 1. Early device testing<br>2. Profile and optimize<br>3. Use platform-specific accelerators |
| Accuracy degradation after quantization | Medium | Medium | 1. Use QAT instead of PTQ<br>2. Mixed precision quantization<br>3. Extensive calibration |
| Conversion issues (ONNX/TFLite) | Low | Medium | 1. Use well-supported ops<br>2. Test conversion early<br>3. Custom op implementation if needed |

### 9.3 Project Risks

| Risk | Probability | Impact | Mitigation |
|------|-------------|--------|------------|
| Insufficient GPU budget | Low | High | 1. Use spot instances<br>2. Optimize training efficiency<br>3. Fallback to smaller-scale experiments |
| Timeline delays | Medium | Medium | 1. Two-phase approach (MVP first)<br>2. Regular checkpoints<br>3. Scope flexibility |
| Whisper license restrictions | Low | High | 1. Verify MIT license terms<br>2. Alternative: Use Wav2Vec2/HuBERT<br>3. From-scratch fallback |

---

## 10. Next Steps

### 10.1 Immediate Actions (This Week)

1. **Approve Plan:** Review and approve this plan
2. **Provision GPU:** Set up training infrastructure
   - Recommended: Lambda Labs 1x A100 instance
   - Alternative: AWS p3.2xlarge spot instance
3. **Download Datasets:** Start with Common Voice + OpenSLR
4. **Register for ULCA:** Begin access request process
5. **Set up Development Environment:**
   ```bash
   pip install torch torchaudio transformers datasets
   pip install nemo_toolkit[all]
   pip install onnx onnxruntime tensorflowlite
   pip install wandb
   ```

### 10.2 Decision Points

**Before Week 1:**
- [ ] Approve overall plan and budget
- [ ] Decide on cloud provider
- [ ] Confirm target device specifications

**Before Week 5 (Phase 2):**
- [ ] Review Phase 1 results
- [ ] Decide: proceed with Phase 2 or iterate on Phase 1?
- [ ] Evaluate data sufficiency for Punjabi
- [ ] Confirm production requirements

### 10.3 Open Questions

1. **Deployment:** On-device only, or hybrid (cloud fallback)?
2. **Languages:** Punjabi script preference (Gurmukhi only or both)?
3. **Features:** Streaming vs. batch processing priority?
4. **Privacy:** All on-device, or allow cloud processing with consent?
5. **Updates:** Model update mechanism (app update or OTA)?

---

## 11. Appendix

### 11.1 Reference Architectures

**Whisper Tiny:**
- Encoder: 4 layers, 384 dims, 6 heads
- Decoder: 4 layers, 384 dims, 6 heads
- Parameters: 39M
- Pre-training: 680k hours (multilingual)

**Proposed Conformer-Small:**
- Encoder: 6 conformer blocks, 256 dims, 4 heads
- Decoder: CTC (no autoregressive)
- Parameters: ~15-20M
- Training: From scratch on Hindi/Punjabi

### 11.2 Useful Resources

**Datasets:**
- Common Voice: https://commonvoice.mozilla.org/
- OpenSLR: https://www.openslr.org/
- ULCA: https://bhashini.gov.in/ulca

**Frameworks:**
- NVIDIA NeMo: https://github.com/NVIDIA/NeMo
- HuggingFace: https://huggingface.co/docs/transformers/
- Whisper: https://github.com/openai/whisper

**Model Conversion:**
- ONNX: https://onnx.ai/
- TFLite: https://www.tensorflow.org/lite
- Core ML Tools: https://coremltools.readme.io/

**Papers:**
- Conformer: https://arxiv.org/abs/2005.08100
- Whisper: https://arxiv.org/abs/2212.04356
- SpecAugment: https://arxiv.org/abs/1904.08779

### 11.3 Sample Code Structure

```
asr_train_1/
├── data/
│   ├── download_datasets.py
│   ├── preprocess.py
│   └── augmentation.py
├── models/
│   ├── whisper_finetune.py
│   ├── conformer.py
│   └── ctc_decoder.py
├── training/
│   ├── train_whisper.py
│   ├── train_conformer.py
│   └── configs/
├── optimization/
│   ├── quantize.py
│   ├── convert_onnx.py
│   ├── convert_tflite.py
│   └── convert_coreml.py
├── evaluation/
│   ├── evaluate.py
│   └── benchmark.py
├── mobile/
│   ├── android/
│   └── ios/
├── notebooks/
│   └── analysis.ipynb
└── README.md
```

---

## Conclusion

This plan provides a comprehensive roadmap for building production-ready mobile ASR systems for Hindi and Punjabi. The two-phase approach balances speed-to-market (Phase 1: MVP with Whisper) with long-term optimization (Phase 2: custom lightweight model).

**Key Advantages:**
- ✅ Pragmatic: Start with proven technology (Whisper)
- ✅ Risk-managed: Two phases with clear decision points
- ✅ Mobile-first: Size and performance constraints from day 1
- ✅ Scalable: Pipeline can extend to other languages
- ✅ Realistic: Accounts for data scarcity (especially Punjabi)

**Expected Outcomes:**
- Working mobile ASR for Hindi and Punjabi
- Models <50MB running real-time on phones
- WER competitive with cloud solutions
- Fully documented and reproducible pipeline

**Estimated Total Effort:**
- Phase 1: 4 weeks, ~$300 GPU costs
- Phase 2: 8 weeks, ~$900 GPU costs
- **Total: 12 weeks, ~$1200**

---

**Version History:**
- v1.0 (2025-11-17): Initial plan

**Contact:**
For questions or clarifications on this plan, please reach out.
