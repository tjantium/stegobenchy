# Pipelines

Pipelines orchestrate experiments by combining models, datasets, and evaluation metrics.

## Basic Experiment Pipeline

Run a standard experiment:

```python
from src.models import OllamaModel
from src.datasets import generate_coin_flip_dataset
from src.pipelines import run_experiment
from src.eval import compute_all_metrics

# Load model
model = OllamaModel('llama3:8b')

# Generate dataset
dataset = generate_coin_flip_dataset(num_samples=100, seed=42)

# Run experiment
results = run_experiment(
    model=model,
    dataset=dataset,
    eval_metrics=compute_all_metrics,
    use_wandb=False,
    verbose=True
)

# Analyze results
print(f"Processed {len(results)} samples")
for result in results[:5]:
    print(f"Stego Success: {result['metrics']['stego_success']}")
```

## RL Experiment Pipeline

Run reinforcement learning experiments:

```python
from src.pipelines import run_rl_experiment

def reward_fn(output, label):
    # Define your reward function
    return 1.0 if output_meets_criteria(output, label) else 0.0

results = run_rl_experiment(
    model=model,
    dataset=dataset,
    reward_fn=reward_fn,
    num_iterations=100
)
```

## Advanced Pipelines

### Reward Hacking Pipeline

Detect reward gaming behaviors:

```python
from src.pipelines import RewardHackingPipeline

pipeline = RewardHackingPipeline(
    model=model,
    reward_fn=reward_function
)

results = pipeline.run_reward_hacking_experiment(
    dataset=dataset,
    num_iterations=1000
)

# Analyze patterns
print(f"Always Agree: {results['patterns']['always_agree']}")
print(f"Minimal Response: {results['patterns']['minimal_response']}")
print(f"Hacking Indicators: {results['hacking_indicators']}")
```

### Encoded Reasoning Pipeline

Study encoded reasoning:

```python
from src.pipelines import EncodedReasoningPipeline
from src.models import ReasoningModel

# Use reasoning model
reasoning_model = ReasoningModel('deepseek-r1:latest')

pipeline = EncodedReasoningPipeline(reasoning_model)

results = pipeline.run_encoded_reasoning_experiment(
    prompts=["Your prompts"],
    hidden_info=["hidden messages"],
    num_samples=100
)

print(f"Success Rate: {results['success_rate']}")
print(f"Encoding Quality: {results['encoding_quality']}")
```

## Experiment Results

Results are returned as a list of dictionaries:

```python
[
    {
        'prompt': str,
        'output': str,
        'label': str,
        'metrics': {
            'stego_success': bool,
            'reasoning_accuracy': float,
            'entropy': float,
            'combined_score': float
        },
        'metadata': dict
    },
    # ... more results
]
```

## Logging to Weights & Biases

Enable W&B logging:

```python
results = run_experiment(
    model=model,
    dataset=dataset,
    eval_metrics=compute_all_metrics,
    use_wandb=True,
    wandb_project='stegobenchy-experiments',
    wandb_run_name='coin-flip-experiment'
)
```

## Custom Evaluation Metrics

Define custom metrics:

```python
def custom_metrics(output, label, task_type='coin_flip'):
    return {
        'custom_metric': compute_custom_metric(output, label),
        'another_metric': compute_another(output)
    }

results = run_experiment(
    model=model,
    dataset=dataset,
    eval_metrics=custom_metrics
)
```

## Pipeline Configuration

Configure pipeline behavior:

```python
results = run_experiment(
    model=model,
    dataset=dataset,
    eval_metrics=compute_all_metrics,
    batch_size=10,          # Process in batches
    max_retries=3,          # Retry failed generations
    timeout=30,             # Timeout per generation
    verbose=True            # Print progress
)
```

