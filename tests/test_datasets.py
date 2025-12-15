"""Tests for datasets module."""

import pytest
from src.datasets import generate_coin_flip_dataset, generate_coin_flip_with_context


class TestCoinFlipDataset:
    """Test cases for coin flip dataset generation."""
    
    def test_generate_coin_flip_dataset(self):
        """Test basic coin flip dataset generation."""
        dataset = generate_coin_flip_dataset(num_samples=10, seed=42)
        
        assert len(dataset) == 10
        assert 'prompt' in dataset.column_names
        assert 'label' in dataset.column_names
        assert 'id' in dataset.column_names
        assert 'task_type' in dataset.column_names
    
    def test_coin_flip_labels(self):
        """Test that labels are valid."""
        dataset = generate_coin_flip_dataset(num_samples=100, seed=42)
        
        labels = set(dataset['label'])
        assert labels == {'heads', 'tails'}
    
    def test_coin_flip_prompts(self):
        """Test that prompts are non-empty."""
        dataset = generate_coin_flip_dataset(num_samples=10, seed=42)
        
        for prompt in dataset['prompt']:
            assert len(prompt) > 0
            assert isinstance(prompt, str)
    
    def test_coin_flip_reproducibility(self):
        """Test dataset reproducibility with same seed."""
        dataset1 = generate_coin_flip_dataset(num_samples=10, seed=42)
        dataset2 = generate_coin_flip_dataset(num_samples=10, seed=42)
        
        assert dataset1['prompt'] == dataset2['prompt']
        assert dataset1['label'] == dataset2['label']
    
    def test_coin_flip_with_context(self):
        """Test coin flip dataset with context examples."""
        dataset = generate_coin_flip_with_context(num_samples=10, seed=42, include_examples=True)
        
        assert len(dataset) == 10
        assert 'has_examples' in dataset.column_names
        
        # Check that some samples have examples
        has_examples = dataset['has_examples']
        assert any(has_examples)

