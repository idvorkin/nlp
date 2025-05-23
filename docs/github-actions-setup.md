# GitHub Actions Test Setup

This repository now includes automated testing via GitHub Actions with test result badges.

## What's Included

### 1. GitHub Actions Workflow (`.github/workflows/test.yml`)

The workflow runs automatically on:

- Pushes to `main` or `master` branches
- Pull requests to `main` or `master` branches

**Features:**

- Uses Python 3.13 (matches your project requirements)
- Installs dependencies using `uv` (your project's package manager)
- Runs all tests in the `tests/` directory
- Generates JUnit XML test reports
- Publishes test results as PR comments
- Uploads test artifacts for later review

### 2. Test Badge in README

The README now includes a test status badge that shows:

- ✅ **Green**: All tests passing
- ❌ **Red**: Some tests failing
- ⚪ **Gray**: Tests not run or pending

## Current Test Coverage

Your project currently has **5 tests**:

- `tests/fast/test_example.py`: 1 simple test
- `tests/test_think_core.py`: 4 tests covering core functionality

## Running Tests Locally

```bash
# Run all tests
uv run pytest tests/ --verbose

# Run with coverage (if you want to add coverage reporting)
uv run pytest tests/ --cov

# Run specific test file
uv run pytest tests/test_think_core.py -v
```

## GitHub Actions Features

### Test Results in Pull Requests

The workflow uses `EnricoMi/publish-unit-test-result-action` which:

- Posts test results as PR comments
- Shows which tests passed/failed
- Provides detailed failure information
- Tracks test count changes between commits

### Artifact Upload

Test results are uploaded as artifacts, allowing you to:

- Download detailed test reports
- Analyze test history
- Debug failing tests

## Adding More Tests

To expand your test coverage:

1. **Add test files** in the `tests/` directory
2. **Follow pytest conventions**:
   - Files should start with `test_`
   - Functions should start with `test_`
3. **Use pytest features**:
   - `@pytest.mark.asyncio` for async tests
   - Fixtures for test setup
   - Parametrize for multiple test cases

Example:

```python
# tests/test_new_feature.py
import pytest
from your_module import new_function

def test_new_function_basic():
    result = new_function("input")
    assert result == "expected_output"

@pytest.mark.asyncio
async def test_async_function():
    result = await async_function()
    assert result is not None
```

## Advanced Badge Options

If you want more detailed badges showing test counts, you can:

1. **Create a GitHub Gist** for dynamic badges
2. **Add the gist ID to repository secrets** as `GIST_ID`
3. **Modify the workflow** to include the badge generation steps

The current setup uses the built-in GitHub Actions badge which is simpler and doesn't require additional setup.

## Troubleshooting

### Common Issues

1. **Tests fail on GitHub but pass locally**:

   - Check if all dependencies are in `pyproject.toml`
   - Verify Python version compatibility
   - Check for environment-specific code

2. **Badge not updating**:

   - Check if the workflow file path is correct
   - Verify the workflow is running successfully
   - GitHub badges may take a few minutes to update

3. **Permission errors**:
   - Ensure repository has Actions enabled
   - Check if workflow permissions are sufficient

### Workflow Logs

To debug issues:

1. Go to **Actions** tab in your GitHub repository
2. Click on the failing workflow run
3. Expand the failed step to see detailed logs
