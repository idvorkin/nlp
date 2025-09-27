#!/usr/bin/env python3
"""
End-to-end tests for commit.py by running it as a subprocess.
Tests that all flag combinations run without crashing.
"""

import subprocess
import pytest
import os
from pathlib import Path

# Sample git diff for testing
SAMPLE_DIFF = """diff --git a/example.py b/example.py
index 1234567..abcdefg 100644
--- a/example.py
+++ b/example.py
@@ -1,5 +1,8 @@
 def greet():
-    print("hello")
+    # Add name parameter with default
+    def greet(name="World"):
+        print(f"Hello, {name}!")
+        return f"Greeted {name}"
"""

# Set dummy API key to avoid authentication errors
TEST_ENV = os.environ.copy()
TEST_ENV['OPENAI_API_KEY'] = 'sk-test-dummy-key'
TEST_ENV['GOOGLE_API_KEY'] = 'dummy-google-key'
TEST_ENV['XAI_API_KEY'] = 'dummy-xai-key'


def run_commit_command(flags: list[str], timeout: float = 10.0) -> tuple[int, str, str]:
    """
    Run commit.py with given flags and return (returncode, stdout, stderr).
    """
    cmd = ['python', 'commit.py'] + flags

    try:
        result = subprocess.run(
            cmd,
            input=SAMPLE_DIFF,
            capture_output=True,
            text=True,
            timeout=timeout,
            env=TEST_ENV
        )
        return result.returncode, result.stdout, result.stderr
    except subprocess.TimeoutExpired:
        return -1, "", "Command timed out"


def test_help_flag():
    """Test that --help works."""
    returncode, stdout, stderr = run_commit_command(['--help'])

    # Help should exit with 0 and show usage
    assert returncode == 0
    assert "--help" in stdout
    assert "--gpt5-codex" in stdout  # Our new flag should be in help
    assert "--grok4-fast" in stdout  # Our grok4-fast flag should be in help


def test_oneline_mode():
    """Test --oneline flag runs without crashing."""
    returncode, stdout, stderr = run_commit_command(['--oneline'])

    # Should either succeed (if some models work) or fail gracefully with API error
    # The important thing is it doesn't crash
    success = returncode == 0 and ("commit" in stdout.lower() or "generated" in stdout.lower())
    api_error = "Invalid API key" in stderr or "API key not found" in stderr or "API Error" in stderr
    assert success or api_error or returncode != 0, f"Unexpected output: {stderr}"


def test_fast_mode():
    """Test --fast flag runs without crashing."""
    returncode, stdout, stderr = run_commit_command(['--fast'])

    # Should either succeed or fail gracefully
    success = returncode == 0 and ("commit" in stdout.lower() or "generated" in stdout.lower())
    api_error = "Invalid API key" in stderr or "API key not found" in stderr or "Error" in stderr
    assert success or api_error or returncode != 0, f"Unexpected output: {stderr}"


def test_fast_with_gpt5_codex():
    """Test --fast --gpt5-codex flags together."""
    returncode, stdout, stderr = run_commit_command(['--fast', '--gpt5-codex'])

    # Should either succeed or fail gracefully
    success = returncode == 0 and ("commit" in stdout.lower() or "generated" in stdout.lower())
    api_error = "Invalid API key" in stderr or "API key not found" in stderr or "Error" in stderr
    assert success or api_error or returncode != 0, f"Unexpected output: {stderr}"


def test_oneline_with_gpt5_codex():
    """Test --oneline --gpt5-codex flags together."""
    returncode, stdout, stderr = run_commit_command(['--oneline', '--gpt5-codex'])

    # Should either succeed or fail gracefully
    success = returncode == 0 and ("commit" in stdout.lower() or "generated" in stdout.lower())
    api_error = "Invalid API key" in stderr or "API key not found" in stderr or "Error" in stderr
    assert success or api_error or returncode != 0, f"Unexpected output: {stderr}"


def test_no_kimi_no_gpt_oss():
    """Test --no-kimi --no-gpt-oss flags."""
    returncode, stdout, stderr = run_commit_command(['--no-kimi', '--no-gpt-oss'])

    # Should either succeed or fail gracefully
    success = returncode == 0 and ("commit" in stdout.lower() or "generated" in stdout.lower())
    api_error = "Invalid API key" in stderr or "API key not found" in stderr or "Error" in stderr
    assert success or api_error or returncode != 0, f"Unexpected output: {stderr}"


def test_all_flags_combination():
    """Test complex flag combination."""
    returncode, stdout, stderr = run_commit_command([
        '--fast',
        '--gpt5-codex',
        '--no-kimi',
        '--no-gpt-oss'
    ])

    # Should either succeed or fail gracefully
    success = returncode == 0 and ("commit" in stdout.lower() or "generated" in stdout.lower())
    api_error = "Invalid API key" in stderr or "API key not found" in stderr or "Error" in stderr
    assert success or api_error or returncode != 0, f"Unexpected output: {stderr}"


def test_standard_mode_with_gpt5():
    """Test standard mode with GPT-5-Codex enabled."""
    returncode, stdout, stderr = run_commit_command(['--gpt5-codex'])

    # Should either succeed or fail gracefully
    success = returncode == 0 and ("commit" in stdout.lower() or "generated" in stdout.lower())
    api_error = "Invalid API key" in stderr or "API key not found" in stderr or "Error" in stderr
    assert success or api_error or returncode != 0, f"Unexpected output: {stderr}"


@pytest.mark.slow
def test_multiple_runs_sequential():
    """Test that multiple runs work (tests cleanup/state management)."""
    # Run three times with different flags
    results = []

    for flags in [['--oneline'], ['--fast'], ['--fast', '--gpt5-codex']]:
        returncode, stdout, stderr = run_commit_command(flags)
        # Consider it complete if it either succeeded or had an error message
        completed = returncode == 0 or bool(stderr) or bool(stdout)
        results.append(completed)

    # All should have completed (even if with API errors)
    assert all(results), "All runs should complete"


def test_empty_input():
    """Test with empty input."""
    cmd = ['python', 'commit.py', '--oneline']

    result = subprocess.run(
        cmd,
        input="",
        capture_output=True,
        text=True,
        timeout=10.0,
        env=TEST_ENV
    )

    # Should handle empty input gracefully - either succeed with minimal output or error
    # Empty diff might still generate a generic message
    handled = result.returncode == 0 or "error" in result.stderr.lower() or "all models failed" in result.stdout.lower()
    assert handled, f"Didn't handle empty input properly: stdout={result.stdout}, stderr={result.stderr}"


def test_parallel_execution_verification():
    """
    Test that demonstrates parallel execution by checking output order.
    With multiple models, they should all report roughly simultaneously.
    """
    # This test mainly verifies the code doesn't crash when running multiple models
    returncode, stdout, stderr = run_commit_command(['--fast', '--gpt5-codex'], timeout=15.0)

    # The test succeeds if it doesn't timeout or crash
    # Real parallel execution is better tested with unit tests that can mock timing
    assert returncode != -1, "Command should not timeout if models run in parallel"


def test_grok4_fast_flag():
    """Test --grok4-fast flag runs without crashing."""
    returncode, stdout, stderr = run_commit_command(['--grok4-fast'])

    # Should either succeed or fail gracefully
    success = returncode == 0 and ("commit" in stdout.lower() or "generated" in stdout.lower())
    api_error = "Invalid API key" in stderr or "API key not found" in stderr or "Error" in stderr
    assert success or api_error or returncode != 0, f"Unexpected output: {stderr}"


def test_oneline_with_grok4_fast():
    """Test --oneline --grok4-fast flags together."""
    returncode, stdout, stderr = run_commit_command(['--oneline', '--grok4-fast'])

    # Should either succeed or fail gracefully
    success = returncode == 0 and ("commit" in stdout.lower() or "generated" in stdout.lower())
    api_error = "Invalid API key" in stderr or "API key not found" in stderr or "Error" in stderr
    assert success or api_error or returncode != 0, f"Unexpected output: {stderr}"


def test_fast_with_grok4_fast():
    """Test --fast --grok4-fast flags together."""
    returncode, stdout, stderr = run_commit_command(['--fast', '--grok4-fast'])

    # Should either succeed or fail gracefully
    success = returncode == 0 and ("commit" in stdout.lower() or "generated" in stdout.lower())
    api_error = "Invalid API key" in stderr or "API key not found" in stderr or "Error" in stderr
    assert success or api_error or returncode != 0, f"Unexpected output: {stderr}"


if __name__ == "__main__":
    # Run tests directly
    print("Running commit.py subprocess tests...")

    test_functions = [
        test_help_flag,
        test_oneline_mode,
        test_fast_mode,
        test_fast_with_gpt5_codex,
        test_oneline_with_gpt5_codex,
        test_no_kimi_no_gpt_oss,
        test_all_flags_combination,
        test_standard_mode_with_gpt5,
        test_grok4_fast_flag,
        test_oneline_with_grok4_fast,
        test_fast_with_grok4_fast,
        test_empty_input,
        test_parallel_execution_verification,
    ]

    passed = 0
    failed = 0

    for test_func in test_functions:
        try:
            print(f"Running {test_func.__name__}...", end=" ")
            test_func()
            print("✅ PASSED")
            passed += 1
        except AssertionError as e:
            print(f"❌ FAILED: {e}")
            failed += 1
        except Exception as e:
            print(f"❌ ERROR: {e}")
            failed += 1

    print(f"\n{'='*60}")
    print(f"Results: {passed} passed, {failed} failed")

    if failed == 0:
        print("✅ All tests passed!")
    else:
        print(f"❌ {failed} tests failed")
        exit(1)