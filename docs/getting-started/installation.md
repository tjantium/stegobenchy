# Installation

## Prerequisites

- Python 3.10+
- [Ollama](https://ollama.ai/) installed and running
- Git

## Step-by-Step Installation

### 1. Clone the Repository

```bash
git clone https://github.com/tjantium/stegobenchy.git
cd stegobenchy
```

### 2. Set Up Virtual Environment

```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

### 3. Install Dependencies

```bash
pip install -r requirements.txt
```

### 4. Install StegoBenchy in Development Mode

```bash
pip install -e .
```

### 5. Pull Ollama Models

```bash
ollama pull llama3:8b
ollama pull phi3:mini
```

### 6. Start Ollama Server

If Ollama is not already running:

```bash
ollama serve
```

## Verify Installation

Test that everything is working:

```python
from src.models import OllamaModel

model = OllamaModel('llama3:8b')
output = model.generate("Hello, world!", max_tokens=50)
print(output)
```

## Optional: Development Dependencies

For development work, install additional tools:

```bash
pip install -e ".[dev]"
```

This includes:
- pytest and pytest-cov for testing
- black for code formatting
- flake8 for linting
- mypy for type checking

