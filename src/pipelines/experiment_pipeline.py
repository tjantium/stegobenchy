"""Modular experiment pipeline for running steganography benchmarks."""

from typing import List, Dict, Any, Callable, Optional
import wandb
from datasets import Dataset
from src.models import OllamaModel


def run_experiment(
    model: OllamaModel,
    dataset: Dataset,
    eval_metrics: Callable[[str, Any], Dict[str, float]],
    use_wandb: bool = True,
    project_name: str = "stegobenchy",
    batch_size: int = 1,
    verbose: bool = True
) -> List[Dict[str, Any]]:
    """
    Run an experiment pipeline: generate outputs and evaluate them.
    
    Args:
        model: OllamaModel instance for generation
        dataset: Dataset with 'prompt' and 'label' fields
        eval_metrics: Function that takes (output, label) and returns metrics dict
        use_wandb: Whether to log to Weights & Biases
        project_name: W&B project name
        batch_size: Batch size for processing (currently 1 for Ollama)
        verbose: Whether to print progress
        
    Returns:
        List of result dictionaries with outputs and metrics
    """
    if use_wandb:
        wandb.init(project=project_name, reinit=True)
    
    results = []
    
    for i, sample in enumerate(dataset):
        if verbose and (i + 1) % 10 == 0:
            print(f"Processing sample {i + 1}/{len(dataset)}")
        
        try:
            # Generate output
            output = model.generate(sample['prompt'])
            
            # Evaluate
            label = sample.get('label', None)
            metrics = eval_metrics(output, label)
            
            # Store result
            result = {
                'sample_id': sample.get('id', i),
                'prompt': sample['prompt'],
                'output': output,
                'label': label,
                'metrics': metrics
            }
            results.append(result)
            
            # Log to wandb
            if use_wandb:
                log_dict = {'sample_id': result['sample_id']}
                log_dict.update(metrics)
                wandb.log(log_dict)
        
        except Exception as e:
            if verbose:
                print(f"Error processing sample {i}: {str(e)}")
            results.append({
                'sample_id': sample.get('id', i),
                'error': str(e),
                'metrics': {}
            })
    
    if use_wandb:
        wandb.finish()
    
    return results


def run_rl_experiment(
    model: OllamaModel,
    dataset: Dataset,
    reward_fn: Callable[[str, Any], float],
    use_wandb: bool = True,
    project_name: str = "stegobenchy-rl",
    num_iterations: int = 100,
    verbose: bool = True
) -> Dict[str, Any]:
    """
    Run RL-based experiment for inducing stego behavior.
    
    Note: This is a simplified version. For full RL training, use TRL's PPO trainer
    from src.models.finetune.
    
    Args:
        model: OllamaModel instance
        dataset: Dataset for training
        reward_fn: Function that computes reward from (output, label)
        use_wandb: Whether to log to W&B
        project_name: W&B project name
        num_iterations: Number of RL iterations
        verbose: Whether to print progress
        
    Returns:
        Dictionary with experiment results and statistics
    """
    if use_wandb:
        wandb.init(project=project_name, reinit=True)
    
    rewards = []
    outputs = []
    
    for i in range(min(num_iterations, len(dataset))):
        sample = dataset[i]
        
        if verbose and (i + 1) % 10 == 0:
            print(f"RL iteration {i + 1}/{num_iterations}")
        
        try:
            # Generate output
            output = model.generate(sample['prompt'], temperature=0.8)
            
            # Compute reward
            label = sample.get('label', None)
            reward = reward_fn(output, label)
            
            rewards.append(reward)
            outputs.append(output)
            
            # Log to wandb
            if use_wandb:
                wandb.log({
                    'iteration': i,
                    'reward': reward,
                    'sample_id': sample.get('id', i)
                })
        
        except Exception as e:
            if verbose:
                print(f"Error in RL iteration {i}: {str(e)}")
            rewards.append(0.0)
    
    if use_wandb:
        wandb.finish()
    
    return {
        'rewards': rewards,
        'outputs': outputs,
        'mean_reward': sum(rewards) / len(rewards) if rewards else 0.0,
        'max_reward': max(rewards) if rewards else 0.0,
        'min_reward': min(rewards) if rewards else 0.0
    }

