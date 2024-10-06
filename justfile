# List available commands
default:
    @just --list

# Run fast tests
fast-test:
    python3 -m pytest tests/fast -v

# Run fast tests
twillio-dev:
    modal serve twillo_serve::modal_app
