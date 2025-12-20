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

## Making Changes

1. Create a feature branch:
   ```bash
   git checkout -b feature/your-feature-name
   ```

2. Make your changes and test them:
   ```bash
   pytest tests/
   ```

3. Ensure code quality:
   ```bash
   black src/ tests/
   flake8 src/ tests/
   ```

4. Commit your changes:
   ```bash
   git commit -m "Add your feature description"
   ```

5. Push to your fork:
   ```bash
   git push origin feature/your-feature-name
   ```

6. Open a Pull Request on GitHub

## Code Style

- Follow PEP 8 style guidelines
- Use `black` for code formatting
- Maximum line length: 127 characters
- Use type hints where appropriate

## Testing

- Write tests for new features
- Ensure all tests pass: `pytest tests/`
- Aim for 80%+ test coverage
- Run coverage report: `pytest --cov=src --cov-report=html tests/`

## Documentation

- Update relevant documentation when adding features
- Add docstrings to new functions and classes
- Update README.md if needed
- Keep examples up to date

## Pull Request Process

1. Ensure all tests pass
2. Update documentation as needed
3. Add a clear description of changes
4. Reference any related issues
5. Wait for review and address feedback

## Questions?

If you have questions, please open an issue on GitHub.

