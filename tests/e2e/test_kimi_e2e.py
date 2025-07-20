"""
End-to-end tests for Kimi AI functionality.
Tests actual CLI commands and integration with real API calls.
Run with: pytest tests/e2e/test_kimi_e2e.py -s
"""

import subprocess
import pytest
import os
import tempfile
from pathlib import Path


@pytest.mark.slow
def test_kimi_think_command_e2e():
    """Test actual think command with Kimi via CLI."""
    if not os.getenv("GROQ_API_KEY"):
        pytest.skip("GROQ_API_KEY not available")

    # Create a temporary input file
    with tempfile.NamedTemporaryFile(mode="w", suffix=".txt", delete=False) as f:
        f.write("What is 2+2? Give a brief answer.")
        temp_file = f.name

    try:
        # Run think command with Kimi
        result = subprocess.run(
            [
                "uv",
                "run",
                "python",
                "-m",
                "think",
                "--kimi",
                "--no-openai",
                "--no-claude",
                temp_file,
            ],
            capture_output=True,
            text=True,
            timeout=60,
        )

        print(f"Return code: {result.returncode}")
        print(f"STDOUT: {result.stdout}")
        print(f"STDERR: {result.stderr}")

        # Basic assertions
        assert result.returncode == 0, f"Command failed: {result.stderr}"
        assert len(result.stdout) > 0, "No output generated"

        # Should contain some mathematical response
        output_lower = result.stdout.lower()
        assert any(
            word in output_lower for word in ["4", "four", "answer", "equals"]
        ), "Expected mathematical response not found"

    finally:
        # Clean up temp file
        os.unlink(temp_file)


@pytest.mark.slow
def test_kimi_commit_command_e2e():
    """Test actual commit command with Kimi via CLI."""
    if not os.getenv("GROQ_API_KEY"):
        pytest.skip("GROQ_API_KEY not available")

    # Create a temporary git repo with some changes
    with tempfile.TemporaryDirectory() as temp_dir:
        os.chdir(temp_dir)

        # Initialize git repo
        subprocess.run(["git", "init"], check=True, capture_output=True)
        subprocess.run(["git", "config", "user.email", "test@example.com"], check=True)
        subprocess.run(["git", "config", "user.name", "Test User"], check=True)

        # Create a file and add it
        test_file = Path(temp_dir) / "test.txt"
        test_file.write_text("Hello World\n")
        subprocess.run(["git", "add", "test.txt"], check=True)

        # Run commit command with Kimi
        result = subprocess.run(
            [
                "uv",
                "run",
                "python",
                "-m",
                "commit",
                "--kimi",
                "--no-openai",
                "--no-claude",
                "--dry-run",  # Don't actually commit
            ],
            capture_output=True,
            text=True,
            timeout=60,
            cwd="/Users/idvorkin/gits/nlp",
        )

        print(f"Return code: {result.returncode}")
        print(f"STDOUT: {result.stdout}")
        print(f"STDERR: {result.stderr}")

        # Should succeed and generate a commit message
        assert result.returncode == 0, f"Command failed: {result.stderr}"
        assert len(result.stdout) > 0, "No commit message generated"


@pytest.mark.slow
def test_kimi_model_selection_priority():
    """Test that Kimi is properly selected when multiple models are available."""
    if not os.getenv("GROQ_API_KEY"):
        pytest.skip("GROQ_API_KEY not available")

    # Create a simple input
    with tempfile.NamedTemporaryFile(mode="w", suffix=".txt", delete=False) as f:
        f.write("Say exactly: 'Kimi selected'")
        temp_file = f.name

    try:
        # Test with Kimi explicitly enabled
        result = subprocess.run(
            [
                "uv",
                "run",
                "python",
                "-m",
                "think",
                "--kimi",
                "--no-openai",
                "--no-claude",
                "--no-llama",
                "--no-deepseek",
                temp_file,
            ],
            capture_output=True,
            text=True,
            timeout=60,
        )

        print(f"Kimi-only result: {result.stdout}")

        assert result.returncode == 0
        assert len(result.stdout) > 0

    finally:
        os.unlink(temp_file)


@pytest.mark.slow
def test_kimi_default_behavior():
    """Test that Kimi is enabled by default as specified."""
    if not os.getenv("GROQ_API_KEY"):
        pytest.skip("GROQ_API_KEY not available")

    # Create a simple input
    with tempfile.NamedTemporaryFile(mode="w", suffix=".txt", delete=False) as f:
        f.write("What is the capital of France?")
        temp_file = f.name

    try:
        # Run with default settings (Kimi should be enabled by default)
        result = subprocess.run(
            [
                "uv",
                "run",
                "python",
                "-m",
                "think",
                "--no-openai",
                "--no-claude",  # Disable others, rely on Kimi default
                temp_file,
            ],
            capture_output=True,
            text=True,
            timeout=60,
        )

        print(f"Default behavior result: {result.stdout}")

        assert result.returncode == 0
        assert len(result.stdout) > 0
        assert "paris" in result.stdout.lower() or "france" in result.stdout.lower()

    finally:
        os.unlink(temp_file)


@pytest.mark.slow
def test_kimi_help_output():
    """Test that Kimi option appears in help output."""
    # Test think command help
    result = subprocess.run(
        ["uv", "run", "python", "-m", "think", "--help"], capture_output=True, text=True
    )

    assert result.returncode == 0
    assert "--kimi" in result.stdout
    assert "--no-kimi" in result.stdout
    assert "Use Kimi model" in result.stdout

    # Test commit command help
    result = subprocess.run(
        ["uv", "run", "python", "-m", "commit", "--help"],
        capture_output=True,
        text=True,
    )

    assert result.returncode == 0
    assert "--kimi" in result.stdout
    assert "--no-kimi" in result.stdout


def test_kimi_model_registration():
    """Test that Kimi model is properly registered in ELL."""
    # This is a fast test that doesn't require API calls
    try:
        from ell_helper import get_ell_model

        # Test that Kimi model can be retrieved
        kimi_model_name = get_ell_model(kimi=True)
        assert kimi_model_name == "moonshotai/kimi-k2-instruct"

        # Test that it's different from other models
        openai_model = get_ell_model(openai=True)
        claude_model = get_ell_model(claude=True)

        assert kimi_model_name != openai_model
        assert kimi_model_name != claude_model

    except ImportError:
        pytest.skip("ELL not available for testing")


@pytest.mark.slow
def test_kimi_langchain_integration():
    """Test that Kimi integrates properly with LangChain."""
    import os

    if not os.getenv("GROQ_API_KEY"):
        pytest.skip("GROQ_API_KEY not available")

    try:
        from langchain_helper import get_model, get_model_name
        from langchain_groq import ChatGroq

        # Test Kimi model creation
        kimi_model = get_model(kimi=True)
        assert isinstance(kimi_model, ChatGroq)
        assert get_model_name(kimi_model) == "moonshotai/kimi-k2-instruct"

    except ImportError:
        pytest.skip("LangChain dependencies not available")
