# Contributing to StegoBenchy

Thank you for your interest in contributing to StegoBenchy! This document provides guidelines and instructions for contributing.

## Development Setup

1. Fork the repository
2. Clone your fork:
   ```bash
   git clone https://github.com/yourusername/stegobenchy.git
   cd stegobenchy
   ```

3. Create a virtual environment:
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

4. Install dependencies:
   ```bash
   pip install -r requirements.txt
   pip install -e ".[dev]"
   ```

5. Install pre-commit hooks (optional):
   ```bash
   pre-commit install
   ```

## Code Style

- Follow PEP 8 style guidelines
- Use `black` for code formatting
- Use type hints where appropriate
- Write docstrings for all public functions and classes

## Testing

- Write tests for all new features
- Ensure all tests pass: `pytest`
- Aim for 80%+ code coverage
- Run tests before submitting: `pytest --cov=src tests/`

## Submitting Changes

1. Create a feature branch: `git checkout -b feature/your-feature-name`
2. Make your changes
3. Write or update tests
4. Ensure all tests pass
5. Commit your changes: `git commit -m "Add feature: description"`
6. Push to your fork: `git push origin feature/your-feature-name`
7. Open a Pull Request

## Pull Request Guidelines

- Provide a clear description of changes
- Reference any related issues
- Ensure CI checks pass
- Request review from maintainers

## Areas for Contribution

- New dataset generators
- Additional evaluation metrics
- Interpretability tools
- Documentation improvements
- Bug fixes
- Performance optimizations

Thank you for contributing! 🎉

