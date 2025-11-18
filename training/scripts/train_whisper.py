#!/usr/bin/env python3
"""
Whisper Fine-tuning Script for Hindi and Punjabi ASR
Supports distributed training and experiment tracking
"""

import argparse
import json
import os
import sys
from pathlib import Path
from dataclasses import dataclass
from typing import Dict, List, Optional, Union

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import numpy as np
import yaml
from tqdm import tqdm

from transformers import (
    WhisperProcessor,
    WhisperForConditionalGeneration,
    Seq2SeqTrainingArguments,
    Seq2SeqTrainer,
    TrainerCallback,
    EarlyStoppingCallback
)
from transformers.trainer_utils import get_last_checkpoint
import evaluate

import librosa
import soundfile as sf

# Set up logging
import logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class ASRDataset(Dataset):
    """Dataset for ASR training"""

    def __init__(
        self,
        manifest_path: str,
        processor: WhisperProcessor,
        sample_rate: int = 16000,
        augmentation_config: Optional[Dict] = None
    ):
        self.processor = processor
        self.sample_rate = sample_rate
        self.augmentation_config = augmentation_config or {}

        # Load manifest
        self.samples = []
        with open(manifest_path, 'r', encoding='utf-8') as f:
            for line in f:
                self.samples.append(json.loads(line))

        logger.info(f"Loaded {len(self.samples)} samples from {manifest_path}")

    def __len__(self):
        return len(self.samples)

    def apply_speed_perturbation(self, audio: np.ndarray, rate: float) -> np.ndarray:
        """Apply speed perturbation"""
        if rate == 1.0:
            return audio

        return librosa.effects.time_stretch(audio, rate=rate)

    def apply_noise(self, audio: np.ndarray, snr_db: float) -> np.ndarray:
        """Add noise to audio"""
        noise = np.random.normal(0, 1, len(audio))

        # Calculate signal and noise power
        signal_power = np.mean(audio ** 2)
        noise_power = np.mean(noise ** 2)

        # Calculate required noise power for target SNR
        snr_linear = 10 ** (snr_db / 10)
        target_noise_power = signal_power / snr_linear

        # Scale noise
        noise = noise * np.sqrt(target_noise_power / noise_power)

        return audio + noise

    def augment_audio(self, audio: np.ndarray) -> np.ndarray:
        """Apply audio augmentation"""
        # Speed perturbation
        if self.augmentation_config.get('speed_perturbation', {}).get('enabled', False):
            rates = self.augmentation_config['speed_perturbation'].get('rates', [1.0])
            rate = np.random.choice(rates)
            audio = self.apply_speed_perturbation(audio, rate)

        # Noise augmentation
        if self.augmentation_config.get('noise_augmentation', {}).get('enabled', False):
            if np.random.random() < 0.5:  # Apply 50% of the time
                snr_range = self.augmentation_config['noise_augmentation'].get('snr_range', [10, 20])
                snr = np.random.uniform(snr_range[0], snr_range[1])
                audio = self.apply_noise(audio, snr)

        return audio

    def __getitem__(self, idx):
        sample = self.samples[idx]

        # Load audio
        try:
            audio, sr = librosa.load(
                sample['audio_filepath'],
                sr=self.sample_rate,
                mono=True
            )

            # Apply augmentation (only during training)
            if self.augmentation_config:
                audio = self.augment_audio(audio)

            # Process with Whisper processor
            input_features = self.processor(
                audio,
                sampling_rate=self.sample_rate,
                return_tensors="pt"
            ).input_features[0]

            # Encode text
            labels = self.processor.tokenizer(
                sample['text'],
                return_tensors="pt"
            ).input_ids[0]

            return {
                'input_features': input_features,
                'labels': labels
            }

        except Exception as e:
            logger.error(f"Error loading {sample['audio_filepath']}: {e}")
            # Return a dummy sample
            return {
                'input_features': torch.zeros(80, 3000),
                'labels': torch.tensor([50257])  # End token
            }


@dataclass
class DataCollatorSpeechSeq2SeqWithPadding:
    """
    Data collator that will dynamically pad the inputs received.
    """
    processor: WhisperProcessor

    def __call__(self, features: List[Dict[str, Union[List[int], torch.Tensor]]]) -> Dict[str, torch.Tensor]:
        # Split inputs and labels
        input_features = [{"input_features": feature["input_features"]} for feature in features]
        label_features = [{"input_ids": feature["labels"]} for feature in features]

        # Pad input features
        batch = self.processor.feature_extractor.pad(input_features, return_tensors="pt")

        # Pad labels
        labels_batch = self.processor.tokenizer.pad(label_features, return_tensors="pt")

        # Replace padding with -100 to ignore loss
        labels = labels_batch["input_ids"].masked_fill(
            labels_batch.attention_mask.ne(1), -100
        )

        # Remove BOS token if present
        if (labels[:, 0] == self.processor.tokenizer.bos_token_id).all().cpu().item():
            labels = labels[:, 1:]

        batch["labels"] = labels

        return batch


def compute_metrics(pred, processor, metric):
    """Compute WER metric"""
    pred_ids = pred.predictions
    label_ids = pred.label_ids

    # Replace -100 with pad token id
    label_ids[label_ids == -100] = processor.tokenizer.pad_token_id

    # Decode predictions and labels
    pred_str = processor.tokenizer.batch_decode(pred_ids, skip_special_tokens=True)
    label_str = processor.tokenizer.batch_decode(label_ids, skip_special_tokens=True)

    # Compute WER
    wer = 100 * metric.compute(predictions=pred_str, references=label_str)

    return {"wer": wer}


def load_config(config_path: str) -> Dict:
    """Load YAML configuration"""
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    return config


def setup_wandb(config: Dict):
    """Setup Weights & Biases tracking"""
    try:
        import wandb
        wandb_config = config.get('wandb', {})

        wandb.init(
            project=wandb_config.get('project', 'whisper-finetuning'),
            name=wandb_config.get('name'),
            tags=wandb_config.get('tags', []),
            notes=wandb_config.get('notes', ''),
            config=config
        )
        logger.info("Weights & Biases initialized")
    except ImportError:
        logger.warning("wandb not installed, skipping W&B logging")
    except Exception as e:
        logger.warning(f"Failed to initialize wandb: {e}")


def main():
    parser = argparse.ArgumentParser(description='Fine-tune Whisper for ASR')
    parser.add_argument('--config', type=str, required=True, help='Path to config file')
    parser.add_argument('--resume_from_checkpoint', type=str, default=None,
                        help='Resume training from checkpoint')
    parser.add_argument('--output_dir', type=str, default=None,
                        help='Override output directory from config')

    args = parser.parse_args()

    # Load configuration
    config = load_config(args.config)
    logger.info(f"Loaded configuration from {args.config}")

    # Override output_dir if provided
    if args.output_dir:
        config['training']['output_dir'] = args.output_dir

    # Set seed
    seed = config.get('seed', 42)
    torch.manual_seed(seed)
    np.random.seed(seed)

    # Setup W&B
    if 'wandb' in config.get('training', {}).get('report_to', []):
        setup_wandb(config)

    # Load processor and model
    model_name = config['model']['name']
    logger.info(f"Loading model: {model_name}")

    processor = WhisperProcessor.from_pretrained(model_name)
    model = WhisperForConditionalGeneration.from_pretrained(model_name)

    # Set language and task tokens
    model.config.forced_decoder_ids = None
    model.config.suppress_tokens = []

    # Language-specific configuration
    language = config['model'].get('language', 'hindi')
    if language == 'hindi':
        model.config.forced_decoder_ids = processor.get_decoder_prompt_ids(
            language="hi", task="transcribe"
        )
    elif language == 'punjabi':
        model.config.forced_decoder_ids = processor.get_decoder_prompt_ids(
            language="pa", task="transcribe"
        )

    # Prepare datasets
    data_config = config['data']

    logger.info("Loading training dataset...")
    train_dataset = ASRDataset(
        manifest_path=data_config['train_manifest'],
        processor=processor,
        sample_rate=data_config['sample_rate'],
        augmentation_config=config.get('augmentation')
    )

    logger.info("Loading validation dataset...")
    eval_dataset = ASRDataset(
        manifest_path=data_config['val_manifest'],
        processor=processor,
        sample_rate=data_config['sample_rate'],
        augmentation_config=None  # No augmentation for validation
    )

    # Data collator
    data_collator = DataCollatorSpeechSeq2SeqWithPadding(processor=processor)

    # Metric
    metric = evaluate.load("wer")

    # Training arguments
    train_config = config['training']
    output_dir = train_config['output_dir']

    training_args = Seq2SeqTrainingArguments(
        output_dir=output_dir,
        per_device_train_batch_size=train_config['batch_size'],
        gradient_accumulation_steps=train_config['gradient_accumulation_steps'],
        learning_rate=train_config['learning_rate'],
        weight_decay=train_config['weight_decay'],
        warmup_steps=train_config['warmup_steps'],
        max_grad_norm=train_config['max_grad_norm'],
        num_train_epochs=train_config['num_epochs'],
        fp16=train_config.get('fp16', False),
        evaluation_strategy=train_config.get('eval_strategy', 'steps'),
        eval_steps=train_config.get('eval_steps', 500),
        save_strategy=train_config.get('save_strategy', 'steps'),
        save_steps=train_config.get('save_steps', 500),
        save_total_limit=train_config.get('save_total_limit', 3),
        load_best_model_at_end=train_config.get('load_best_model_at_end', True),
        metric_for_best_model=train_config.get('metric_for_best_model', 'wer'),
        greater_is_better=train_config.get('greater_is_better', False),
        logging_steps=train_config.get('logging_steps', 100),
        report_to=train_config.get('report_to', ['tensorboard']),
        push_to_hub=False,
        predict_with_generate=True,
        generation_max_length=config.get('generation', {}).get('max_length', 225),
        generation_num_beams=config.get('generation', {}).get('num_beams', 5),
        remove_unused_columns=False,
        label_names=["labels"],
    )

    # Callbacks
    callbacks = []
    if train_config.get('early_stopping_patience'):
        callbacks.append(
            EarlyStoppingCallback(
                early_stopping_patience=train_config['early_stopping_patience'],
                early_stopping_threshold=train_config.get('early_stopping_threshold', 0.0)
            )
        )

    # Trainer
    trainer = Seq2SeqTrainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        data_collator=data_collator,
        compute_metrics=lambda pred: compute_metrics(pred, processor, metric),
        tokenizer=processor.feature_extractor,
        callbacks=callbacks,
    )

    # Resume from checkpoint if specified
    checkpoint = None
    if args.resume_from_checkpoint is not None:
        checkpoint = args.resume_from_checkpoint
    elif os.path.isdir(output_dir):
        checkpoint = get_last_checkpoint(output_dir)

    if checkpoint is not None:
        logger.info(f"Resuming from checkpoint: {checkpoint}")

    # Train
    logger.info("Starting training...")
    train_result = trainer.train(resume_from_checkpoint=checkpoint)

    # Save model
    trainer.save_model()
    trainer.save_state()

    # Save metrics
    metrics = train_result.metrics
    trainer.log_metrics("train", metrics)
    trainer.save_metrics("train", metrics)

    # Final evaluation
    logger.info("Running final evaluation...")
    metrics = trainer.evaluate()
    trainer.log_metrics("eval", metrics)
    trainer.save_metrics("eval", metrics)

    logger.info(f"Training complete! Model saved to {output_dir}")
    logger.info(f"Final WER: {metrics['eval_wer']:.2f}%")


if __name__ == '__main__':
    main()
