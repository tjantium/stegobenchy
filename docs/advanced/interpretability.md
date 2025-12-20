# Interpretability

StegoBenchy provides comprehensive interpretability tools for studying encoded reasoning in language models.

## Overview

The interpretability module (`src/interp`) includes:

- **Probing**: Activation analysis and learned probes
- **SAE Analysis**: Sparse Autoencoder feature analysis
- **Causal Analysis**: Causal methods for understanding model behavior

## Probing

### Basic Activation Probing

```python
from src.interp import probe_encoded_reasoning

# Note: Requires HuggingFace model compatible with TransformerLens
results = probe_encoded_reasoning(
    model_name='gpt2',
    prompt="Your prompt here",
    layers=[0, 5, 10]  # Specific layers to probe
)
```

### Ablation Studies

```python
from src.interp import ablation_study

# Ablate residual stream at specific layers
results = ablation_study(
    model=hooked_model,
    prompt="Your prompt",
    layers=[5, 10, 15]
)
```

## SAE Analysis

Sparse Autoencoder (SAE) analysis helps identify features in the model's latent space.

### Feature Analysis

```python
from src.interp import SAEAnalyzer

analyzer = SAEAnalyzer(hooked_model)
features = analyzer.analyze_features(
    prompt="Your prompt",
    layer=10,
    top_k=20
)
```

### Finding Encoded Features

```python
# Find features associated with hidden information
encoded_features = analyzer.find_encoded_features(
    prompt="Your prompt",
    hidden_info="hidden message",
    layers=[5, 10, 15]
)
```

## Causal Analysis

### Direct Activation Substitution (DAS)

DAS swaps activations between prompts to measure causal effects.

```python
from src.interp import CausalAnalyzer

analyzer = CausalAnalyzer(hooked_model)
result = analyzer.direct_activation_substitution(
    source_prompt="Source prompt",
    target_prompt="Target prompt",
    layer=10
)
```

### MELBO Analysis

Maximum Entropy Latent Backdoor Optimization finds high-entropy directions.

```python
melbo_result = analyzer.melbo_analysis(
    prompt="Your prompt",
    layer=10
)
```

### Latent Adversarial Training (LAT)

LAT finds adversarial perturbations in the latent space.

```python
lat_result = analyzer.latent_adversarial_training(
    prompt="Your prompt",
    target_output="Target output",
    layer=10
)
```

### Causal Trace Analysis

Per-layer ablation to measure causal effects.

```python
trace = analyzer.causal_trace_analysis(
    prompt="Your prompt",
    layers=list(range(12))  # All layers
)
```

## Integration with Experiments

Combine interpretability tools with experiment pipelines:

```python
from src.models import ReasoningModel
from src.pipelines import EncodedReasoningPipeline
from src.interp import SAEAnalyzer, CausalAnalyzer

# Run experiment
model = ReasoningModel('deepseek-r1:latest')
pipeline = EncodedReasoningPipeline(model)
results = pipeline.run_encoded_reasoning_experiment(prompts, hidden_info)

# Analyze with interpretability tools
analyzer = SAEAnalyzer(hooked_model)
for result in results['results']:
    features = analyzer.find_encoded_features(
        result['prompt'],
        result['hidden_info'],
        layers=[5, 10, 15]
    )
```

## Requirements

For interpretability features, you'll need:

- `transformer-lens>=1.4.0` (for activation analysis)
- HuggingFace models compatible with TransformerLens
- Optional: Pre-trained SAEs for SAE analysis

## References

- **DAS**: Direct Activation Substitution for causal analysis
- **MELBO**: Maximum Entropy Latent Backdoor Optimization
- **LAT**: Latent Adversarial Training
- **SAE**: Sparse Autoencoders for feature analysis
- **TransformerLens**: Interpretability framework

