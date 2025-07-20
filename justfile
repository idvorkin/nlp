# List available commands
default:
    @just --list

# Run all tests
test:
    uv run pytest -v

# Run tests with coverage report
test-cov:
    uv run pytest --cov=. --cov-report=term-missing -v

# Run fast tests
fast-test:
    uv run pytest tests/fast -v

# Run tests in watch mode
test-watch:
    uv run ptw -- -v

# Run Kimi-specific tests (fast)
test-kimi:
    uv run pytest tests/test_ell_helper.py::test_get_ell_model_kimi -v
    uv run pytest tests/test_langchain_helper.py -k "kimi" -v

# Run Kimi integration tests (slow, requires GROQ_API_KEY)
test-kimi-slow:
    uv run pytest tests/test_kimi_integration_slow.py -s -v

# Run all Kimi tests (fast + slow)
test-kimi-all: test-kimi test-kimi-slow

# Run e2e tests
test-e2e:
    uv run pytest tests/e2e/ -s -v

# Run twillio dev server
twillio-dev:
    modal serve twillo_serve::modal_app

# Install locally and globally
install:
    uv venv
    . .venv/bin/activate
    uv pip install --editable .

global-install: install
    uv tool install . --force --editable
