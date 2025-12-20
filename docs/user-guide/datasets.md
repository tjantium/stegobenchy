# Datasets

StegoBenchy provides various dataset generators for steganography and encoded reasoning tasks.

## Coin Flip Dataset

Generate coin flip reasoning tasks:

```python
from src.datasets import generate_coin_flip_dataset

# Basic coin flip dataset
dataset = generate_coin_flip_dataset(
    num_samples=100,
    seed=42
)

# With context
dataset = generate_coin_flip_with_context(
    num_samples=100,
    context_examples=3,
    seed=42
)
```

## Paraphrase Dataset

Generate paraphrasing tasks:

```python
from src.datasets import generate_paraphrase_dataset

# Basic paraphrasing
dataset = generate_paraphrase_dataset(
    base_texts=["Your text here"],
    model_name='llama3:8b',
    num_paraphrases=5
)

# Encoded paraphrasing
dataset = generate_encoded_paraphrase_dataset(
    base_texts=["Your text here"],
    hidden_strings=["hidden message"],
    model_name='llama3:8b'
)
```

## Advanced Datasets

### Robust Paraphrasing

```python
from src.datasets import generate_robust_paraphrase_dataset

dataset = generate_robust_paraphrase_dataset(
    base_texts=["Your text"],
    styles=['formal', 'casual', 'technical'],
    num_variants=3
)
```

### Stego Cover Tasks

```python
from src.datasets import generate_stego_cover_dataset

dataset = generate_stego_cover_dataset(
    hidden_info=["secret message"],
    formats=['story', 'email', 'article', 'review'],
    num_samples=50
)
```

### Robustness Evaluation

```python
from src.datasets import generate_robustness_evaluation_dataset

dataset = generate_robustness_evaluation_dataset(
    base_prompts=["Your prompt"],
    perturbations=['typos', 'synonyms', 'reordering', 'noise'],
    num_variants=5
)
```

### Monitoring Robustness

```python
from src.datasets import generate_monitoring_robustness_dataset

dataset = generate_monitoring_robustness_dataset(
    scenarios=['deception', 'bypass', 'misgeneralization'],
    num_samples=100
)
```

## Dataset Format

All datasets return a list of dictionaries with:

```python
{
    'prompt': str,      # The input prompt
    'label': str,       # Expected label/answer
    'metadata': dict,   # Additional metadata
    # ... other fields depending on dataset type
}
```

## Using Datasets with Pipelines

```python
from src.datasets import generate_coin_flip_dataset
from src.pipelines import run_experiment
from src.models import OllamaModel

# Generate dataset
dataset = generate_coin_flip_dataset(num_samples=100, seed=42)

# Load model
model = OllamaModel('llama3:8b')

# Run experiment
results = run_experiment(
    model=model,
    dataset=dataset,
    eval_metrics=compute_all_metrics
)
```

## Configuration

Dataset configuration can be managed via `src/datasets/config.py`:

```python
from src.datasets.config import get_dataset_config

config = get_dataset_config('coin_flip')
```

