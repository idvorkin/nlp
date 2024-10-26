# List available commands
default:
    @just --list

# Run fast tests
fast-test:
    python3 -m pytest tests/fast -v

# Run fast tests
twillio-dev:
    modal serve twillo_serve::modal_app

# Install locally and globally
install:
    uv pip install --editable .

global_install:
    pipxu install -f . --editable
    uv pip install --editable .
