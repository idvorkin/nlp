"""
Slow integration tests for Kimi AI functionality.
These tests make actual API calls and require valid API keys.
Run with: pytest tests/test_kimi_integration_slow.py -s
"""

import pytest
import os
from langchain_helper import get_model, get_models, get_model_name


@pytest.mark.slow
def test_kimi_model_actual_integration():
    """Test actual Kimi model integration with Groq API."""
    # Skip if no Groq API key available
    if not os.getenv("GROQ_API_KEY"):
        pytest.skip("GROQ_API_KEY not available")

    # Test model creation
    kimi_model = get_model(kimi=True)
    assert get_model_name(kimi_model) == "moonshotai/kimi-k2-instruct"

    # Test basic inference (simple prompt to avoid costs)
    try:
        response = kimi_model.invoke("Say 'Hello Kimi' in exactly two words.")
        assert response.content is not None
        assert len(response.content) > 0
        print(f"Kimi response: {response.content}")
    except Exception as e:
        pytest.fail(f"Kimi model inference failed: {e}")


@pytest.mark.slow
def test_kimi_in_get_models_integration():
    """Test Kimi model when included in get_models list."""
    if not os.getenv("GROQ_API_KEY"):
        pytest.skip("GROQ_API_KEY not available")

    # Test with multiple models including Kimi
    models = get_models(kimi=True)
    assert len(models) == 1

    kimi_model = models[0]
    assert get_model_name(kimi_model) == "moonshotai/kimi-k2-instruct"

    # Test basic inference
    try:
        response = kimi_model.invoke("Respond with exactly: 'Kimi works'")
        assert response.content is not None
        print(f"Kimi via get_models response: {response.content}")
    except Exception as e:
        pytest.fail(f"Kimi model via get_models failed: {e}")


@pytest.mark.slow
def test_kimi_vs_other_providers():
    """Test Kimi alongside other providers to ensure no conflicts."""
    if not os.getenv("GROQ_API_KEY"):
        pytest.skip("GROQ_API_KEY not available")

    # Test Kimi specifically
    kimi_model = get_model(kimi=True)

    # Test that it's different from other groq models
    if os.getenv("GROQ_API_KEY"):
        llama_model = get_model(llama=True)
        deepseek_model = get_model(deepseek=True)

        # Ensure different model names
        assert get_model_name(kimi_model) != get_model_name(llama_model)
        assert get_model_name(kimi_model) != get_model_name(deepseek_model)

        # All should be ChatGroq instances
        from langchain_groq import ChatGroq

        assert isinstance(kimi_model, ChatGroq)
        assert isinstance(llama_model, ChatGroq)
        assert isinstance(deepseek_model, ChatGroq)


@pytest.mark.slow
def test_kimi_model_capabilities():
    """Test Kimi model's specific capabilities."""
    if not os.getenv("GROQ_API_KEY"):
        pytest.skip("GROQ_API_KEY not available")

    kimi_model = get_model(kimi=True)

    # Test reasoning capability
    reasoning_prompt = "What is 2+2? Explain your reasoning step by step."

    try:
        response = kimi_model.invoke(reasoning_prompt)
        content = response.content.lower()

        # Basic checks for reasoning response
        assert "4" in content or "four" in content
        assert len(response.content) > 10  # Should be more than just "4"

        print(f"Kimi reasoning test: {response.content[:100]}...")

    except Exception as e:
        pytest.fail(f"Kimi reasoning test failed: {e}")
