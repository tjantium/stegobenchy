"""Finetuning support using TRL for SFT and RL."""

from typing import Optional, Dict, Any
from pathlib import Path
import torch
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    TrainingArguments,
    BitsAndBytesConfig
)
from trl import SFTTrainer, PPOConfig, PPOTrainer
from datasets import Dataset, load_dataset


def finetune_sft(
    model_name: str,
    dataset_path: Optional[str] = None,
    dataset: Optional[Dataset] = None,
    output_dir: str = "./results",
    num_train_epochs: int = 3,
    per_device_train_batch_size: int = 4,
    gradient_accumulation_steps: int = 4,
    learning_rate: float = 2e-4,
    use_quantization: bool = False,
    **kwargs
) -> AutoModelForCausalLM:
    """
    Fine-tune a model using Supervised Fine-Tuning (SFT).
    
    Note: Ollama doesn't natively support finetuning. This function works with
    HuggingFace models. For Ollama, export to HF format first or use small HF models.
    
    Args:
        model_name: HuggingFace model name or path
        dataset_path: Path to JSON dataset file
        dataset: Pre-loaded Dataset object
        output_dir: Directory to save the fine-tuned model
        num_train_epochs: Number of training epochs
        per_device_train_batch_size: Batch size per device
        gradient_accumulation_steps: Gradient accumulation steps
        learning_rate: Learning rate
        use_quantization: Whether to use 4-bit quantization
        **kwargs: Additional training arguments
        
    Returns:
        Fine-tuned model
    """
    # Load model and tokenizer
    if use_quantization:
        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.float16
        )
        model = AutoModelForCausalLM.from_pretrained(
            model_name,
            quantization_config=bnb_config,
            device_map="auto"
        )
    else:
        model = AutoModelForCausalLM.from_pretrained(model_name)
    
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    # Load dataset
    if dataset is None:
        if dataset_path is None:
            raise ValueError("Either dataset_path or dataset must be provided")
        dataset = load_dataset('json', data_files=dataset_path)['train']
    
    # Training arguments
    training_args = TrainingArguments(
        output_dir=output_dir,
        num_train_epochs=num_train_epochs,
        per_device_train_batch_size=per_device_train_batch_size,
        gradient_accumulation_steps=gradient_accumulation_steps,
        learning_rate=learning_rate,
        fp16=True,
        logging_steps=10,
        save_steps=500,
        **kwargs
    )
    
    # Create trainer
    trainer = SFTTrainer(
        model=model,
        train_dataset=dataset,
        tokenizer=tokenizer,
        args=training_args,
        max_seq_length=512,
    )
    
    # Train
    trainer.train()
    
    # Save model
    trainer.save_model()
    tokenizer.save_pretrained(output_dir)
    
    return model


def setup_ppo_trainer(
    model_name: str,
    ref_model_name: Optional[str] = None,
    use_quantization: bool = False
) -> PPOTrainer:
    """
    Set up PPO trainer for RL-based finetuning (e.g., for inducing stego behavior).
    
    Args:
        model_name: HuggingFace model name
        ref_model_name: Reference model name (if None, uses model_name)
        use_quantization: Whether to use 4-bit quantization
        
    Returns:
        PPOTrainer instance
    """
    # Load model
    if use_quantization:
        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.float16
        )
        model = AutoModelForCausalLM.from_pretrained(
            model_name,
            quantization_config=bnb_config,
            device_map="auto"
        )
    else:
        model = AutoModelForCausalLM.from_pretrained(model_name)
    
    # Load reference model
    if ref_model_name is None:
        ref_model = AutoModelForCausalLM.from_pretrained(model_name)
    else:
        ref_model = AutoModelForCausalLM.from_pretrained(ref_model_name)
    
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    # PPO config
    ppo_config = PPOConfig(
        batch_size=4,
        mini_batch_size=2,
        gradient_accumulation_steps=1,
        learning_rate=1e-5,
    )
    
    # Create PPO trainer
    ppo_trainer = PPOTrainer(
        config=ppo_config,
        model=model,
        ref_model=ref_model,
        tokenizer=tokenizer,
    )
    
    return ppo_trainer

