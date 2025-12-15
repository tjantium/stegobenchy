"""Tests for experiment pipelines."""

import pytest
from unittest.mock import Mock, patch, MagicMock
from datasets import Dataset
from src.models import OllamaModel
from src.pipelines import run_experiment, run_rl_experiment
from src.eval import compute_all_metrics


class TestExperimentPipeline:
    """Test cases for experiment pipeline."""
    
    @patch('src.pipelines.experiment_pipeline.wandb')
    def test_run_experiment(self, mock_wandb):
        """Test running a basic experiment."""
        # Mock model
        mock_model = Mock(spec=OllamaModel)
        mock_model.generate.return_value = "Test output"
        
        # Create dataset
        dataset = Dataset.from_list([
            {'prompt': 'Test prompt 1', 'label': 'heads', 'id': 0},
            {'prompt': 'Test prompt 2', 'label': 'tails', 'id': 1}
        ])
        
        # Run experiment
        results = run_experiment(
            mock_model,
            dataset,
            compute_all_metrics,
            use_wandb=False,
            verbose=False
        )
        
        assert len(results) == 2
        assert results[0]['sample_id'] == 0
        assert 'metrics' in results[0]
        assert mock_model.generate.call_count == 2
    
    @patch('src.pipelines.experiment_pipeline.wandb')
    def test_run_experiment_with_wandb(self, mock_wandb):
        """Test experiment with W&B logging."""
        mock_model = Mock(spec=OllamaModel)
        mock_model.generate.return_value = "Test output"
        
        dataset = Dataset.from_list([
            {'prompt': 'Test', 'label': 'heads', 'id': 0}
        ])
        
        results = run_experiment(
            mock_model,
            dataset,
            compute_all_metrics,
            use_wandb=True,
            verbose=False
        )
        
        assert mock_wandb.init.called
        assert mock_wandb.log.called
        assert mock_wandb.finish.called
    
    def test_run_experiment_error_handling(self):
        """Test error handling in experiment pipeline."""
        mock_model = Mock(spec=OllamaModel)
        mock_model.generate.side_effect = Exception("Generation error")
        
        dataset = Dataset.from_list([
            {'prompt': 'Test', 'label': 'heads', 'id': 0}
        ])
        
        results = run_experiment(
            mock_model,
            dataset,
            compute_all_metrics,
            use_wandb=False,
            verbose=False
        )
        
        assert len(results) == 1
        assert 'error' in results[0]
    
    @patch('src.pipelines.experiment_pipeline.wandb')
    def test_run_rl_experiment(self, mock_wandb):
        """Test RL experiment."""
        mock_model = Mock(spec=OllamaModel)
        mock_model.generate.return_value = "Test output"
        
        dataset = Dataset.from_list([
            {'prompt': 'Test', 'label': 'heads', 'id': 0},
            {'prompt': 'Test 2', 'label': 'tails', 'id': 1}
        ])
        
        def reward_fn(output, label):
            return 1.0 if label == 'heads' else 0.5
        
        results = run_rl_experiment(
            mock_model,
            dataset,
            reward_fn,
            use_wandb=False,
            num_iterations=2,
            verbose=False
        )
        
        assert 'rewards' in results
        assert 'mean_reward' in results
        assert len(results['rewards']) == 2

