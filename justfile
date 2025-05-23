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
