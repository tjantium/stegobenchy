# 🔍 StegoBenchy

A benchmark suite for evaluating **steganography** and **encoded reasoning** in large language models (LLMs). StegoBenchy provides modular experiment pipelines, datasets, finetuning workflows, interpretability tools, and interactive visualizations to study how models encode hidden information in their reasoning.

## 🌟 Features

- **🔬 Modular Experiment Pipelines**: Run reproducible experiments with configurable parameters
- **📊 Comprehensive Datasets**: Pre-built datasets for coin-flip reasoning, paraphrasing, and encoded tasks
- **🤖 Ollama Integration**: Use local LLMs (Llama-3, Phi-3, Mistral) for offline, reproducible experiments
- **📈 Evaluation Metrics**: Stego detection rate, reasoning accuracy, entropy analysis
- **🔍 Interpretability Tools**: TransformerLens integration for probing encoded reasoning
- **📉 Interactive Visualizations**: Plotly-based dashboards and Streamlit demo
- **🧪 Full Test Coverage**: 80%+ test coverage with pytest
- **🔄 CI/CD Ready**: GitHub Actions workflow for automated testing

## Quick Links

- [Installation Guide](getting-started/installation.md) - Get started with StegoBenchy
- [Quick Start Tutorial](getting-started/quickstart.md) - Run your first experiment
- [User Guide](user-guide/models.md) - Learn how to use StegoBenchy
- [Architecture](advanced/architecture.md) - Understand the codebase structure
- [Contributing](contributing.md) - Contribute to the project

## Installation

```bash
# Clone the repository
git clone https://github.com/tjantium/stegobenchy.git
cd stegobenchy

# Set up virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
pip install -e .
```

## Quick Example

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
```

## License

This project is licensed under the MIT License - see the [LICENSE](https://github.com/tjantium/stegobenchy/blob/main/LICENSE) file for details.

