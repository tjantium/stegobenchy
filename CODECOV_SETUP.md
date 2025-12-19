# Codecov Setup Instructions

## ✅ What's Already Done

The CI workflow (`.github/workflows/ci.yml`) has been updated to:
- Generate coverage reports with branch coverage: `pytest --cov=src --cov-branch --cov-report=xml`
- Upload coverage to Codecov automatically

## 🔑 Required: Add GitHub Secret

You need to add the Codecov token as a GitHub repository secret:

1. Go to your GitHub repository: `https://github.com/tjantium/stegobenchy`
2. Navigate to: **Settings** → **Secrets and variables** → **Actions**
3. Click **New repository secret**
4. Add:
   - **Name**: `CODECOV_TOKEN`
   - **Value**: `10666d63-25ca-42c1-abf3-c138604c9e25`
5. Click **Add secret**

## 🚀 How It Works

Once the secret is added:
- Every push/PR will automatically run tests with coverage
- Coverage reports will be uploaded to Codecov
- You'll see coverage badges and reports on Codecov dashboard

## 📊 Viewing Coverage

- **Codecov Dashboard**: `https://codecov.io/gh/tjantium/stegobenchy`
- Coverage badge will appear in your README (add manually if desired)

## 🔍 Testing Locally

To test coverage generation locally:

```bash
pip install pytest pytest-cov
pytest --cov=src --cov-branch --cov-report=xml --cov-report=html tests/
```

This will generate:
- `coverage.xml` (for Codecov)
- `htmlcov/` directory (for local viewing)

