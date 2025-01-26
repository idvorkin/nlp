# List available commands
default:
    @just --list

# Run all tests
test:
    python3 -m pytest -v

# Run tests with coverage report
test-cov:
    python3 -m pytest --cov=. --cov-report=term-missing -v

# Run fast tests
fast-test:
    python3 -m pytest tests/fast -v

# Run tests in watch mode
test-watch:
    ptw -- -v

# Run twillio dev server
twillio-dev:
    modal serve twillo_serve::modal_app

# Install locally and globally
install:
    uv pip install --editable .

global-install: install
    @just install
    uv pip install -f . --editable . --python $(which python3.12)
