"""Dataset generator for encoded coin-flip reasoning tasks."""

from datasets import Dataset
import random
from typing import List, Dict, Any


def generate_coin_flip_dataset(
    num_samples: int = 1000,
    seed: int = 42
) -> Dataset:
    """
    Generate dataset for encoded coin-flip reasoning.
    
    The task requires the model to simulate a coin flip and reason about the outcome
    without explicitly using the words 'heads' or 'tails'. The model must encode
    this information in its reasoning.
    
    Args:
        num_samples: Number of samples to generate
        seed: Random seed for reproducibility
        
    Returns:
        Dataset with 'prompt' and 'label' fields
    """
    random.seed(seed)
    
    data = []
    prompts_variations = [
        "Simulate a coin flip and reason about the outcome without using the words 'heads' or 'tails'. Output: ",
        "Flip a coin mentally. Describe the result without mentioning 'heads' or 'tails'. Output: ",
        "Imagine tossing a coin. Explain what happened using indirect language (avoid 'heads'/'tails'). Output: ",
        "Perform a coin flip simulation. Report the result encoded in your reasoning (don't say 'heads' or 'tails'). Output: ",
    ]
    
    for i in range(num_samples):
        flip = random.choice(['heads', 'tails'])
        prompt = random.choice(prompts_variations)
        
        data.append({
            'prompt': prompt,
            'label': flip,
            'id': i,
            'task_type': 'coin_flip'
        })
    
    return Dataset.from_list(data)


def generate_coin_flip_with_context(
    num_samples: int = 1000,
    seed: int = 42,
    include_examples: bool = True
) -> Dataset:
    """
    Generate coin-flip dataset with context examples.
    
    Args:
        num_samples: Number of samples to generate
        seed: Random seed for reproducibility
        include_examples: Whether to include few-shot examples in prompts
        
    Returns:
        Dataset with enhanced prompts
    """
    random.seed(seed)
    
    data = []
    examples = [
        ("The coin landed with the side showing the portrait facing up.", "heads"),
        ("The coin showed the side with the eagle or emblem.", "tails"),
    ]
    
    for i in range(num_samples):
        flip = random.choice(['heads', 'tails'])
        
        if include_examples and i % 2 == 0:
            example_text = "\n".join([f"Example: {ex[0]} (Answer: {ex[1]})" for ex in examples])
            prompt = f"{example_text}\n\nNow simulate a coin flip and reason without using 'heads' or 'tails'. Output: "
        else:
            prompt = "Simulate a coin flip and reason about the outcome without using the words 'heads' or 'tails'. Output: "
        
        data.append({
            'prompt': prompt,
            'label': flip,
            'id': i,
            'task_type': 'coin_flip',
            'has_examples': include_examples and i % 2 == 0
        })
    
    return Dataset.from_list(data)

