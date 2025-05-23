from unittest.mock import Mock
from langchain_helper import get_model_name, get_model, GoogleThinkingLevel


def test_get_model_name_basic():
    """Test basic model name extraction without thinking config."""
    # Mock model with model_name attribute
    model = Mock()
    model.model_name = "gpt-4"
    model.model = ""
    model.model_kwargs = None  # Explicitly set to None
    # Remove the auto-generated _thinking_level mock
    del model._thinking_level
    assert get_model_name(model) == "gpt-4"

    # Mock model with model attribute
    model2 = Mock()
    model2.model_name = ""
    model2.model = "claude-3-sonnet"
    model2.model_kwargs = None  # Explicitly set to None
    # Remove the auto-generated _thinking_level mock
    del model2._thinking_level
    assert get_model_name(model2) == "claude-3-sonnet"


def test_get_model_name_removes_models_prefix():
    """Test that 'models/' prefix is properly removed."""
    model = Mock()
    model.model_name = "models/gemini-pro"
    model.model = ""
    model.model_kwargs = None  # Explicitly set to None
    # Remove the auto-generated _thinking_level mock
    del model._thinking_level
    assert get_model_name(model) == "gemini-pro"


def test_get_model_name_google_thinking_low():
    """Test model name with Google thinking LOW configuration."""
    model = Mock()
    model.model_name = "gemini-2.5-flash-preview-05-20"
    model.model = ""
    model.model_kwargs = {
        "generation_config": {
            "thinking_config": {"thinking_budget": GoogleThinkingLevel.LOW.value}
        }
    }
    # Remove the auto-generated _thinking_level mock to test model_kwargs path
    del model._thinking_level

    expected = "gemini-2.5-flash-preview-05-20-thinking-LOW"
    assert get_model_name(model) == expected


def test_get_model_name_google_thinking_medium():
    """Test model name with Google thinking MEDIUM configuration."""
    model = Mock()
    model.model_name = "gemini-2.5-flash-preview-05-20"
    model.model = ""
    model.model_kwargs = {
        "generation_config": {
            "thinking_config": {"thinking_budget": GoogleThinkingLevel.MEDIUM.value}
        }
    }
    # Remove the auto-generated _thinking_level mock to test model_kwargs path
    del model._thinking_level

    expected = "gemini-2.5-flash-preview-05-20-thinking-MEDIUM"
    assert get_model_name(model) == expected


def test_get_model_name_google_thinking_high():
    """Test model name with Google thinking HIGH configuration."""
    model = Mock()
    model.model_name = "gemini-2.5-flash-preview-05-20"
    model.model = ""
    model.model_kwargs = {
        "generation_config": {
            "thinking_config": {"thinking_budget": GoogleThinkingLevel.HIGH.value}
        }
    }
    # Remove the auto-generated _thinking_level mock to test model_kwargs path
    del model._thinking_level

    expected = "gemini-2.5-flash-preview-05-20-thinking-HIGH"
    assert get_model_name(model) == expected


def test_get_model_name_google_thinking_custom_budget():
    """Test model name with custom thinking budget."""
    model = Mock()
    model.model_name = "gemini-2.5-flash-preview-05-20"
    model.model = ""
    model.model_kwargs = {
        "generation_config": {
            "thinking_config": {
                "thinking_budget": 16384  # Custom value not in enum
            }
        }
    }
    # Remove the auto-generated _thinking_level mock to test model_kwargs path
    del model._thinking_level

    expected = "gemini-2.5-flash-preview-05-20-thinking-CUSTOM-16384"
    assert get_model_name(model) == expected


def test_get_model_name_no_thinking_config():
    """Test that models without thinking config are unaffected."""
    model = Mock()
    model.model_name = "gemini-2.5-flash-preview-05-20"
    model.model = ""
    model.model_kwargs = {"generation_config": {"temperature": 0.7}}
    # Remove the auto-generated _thinking_level mock
    del model._thinking_level

    expected = "gemini-2.5-flash-preview-05-20"
    assert get_model_name(model) == expected


def test_get_model_name_no_model_kwargs():
    """Test that models without model_kwargs are unaffected."""
    model = Mock()
    model.model_name = "gemini-2.5-flash-preview-05-20"
    model.model = ""
    model.model_kwargs = None
    # Remove the auto-generated _thinking_level mock
    del model._thinking_level

    expected = "gemini-2.5-flash-preview-05-20"
    assert get_model_name(model) == expected


def test_get_model_name_empty_model_kwargs():
    """Test that models with empty model_kwargs are unaffected."""
    model = Mock()
    model.model_name = "gemini-2.5-flash-preview-05-20"
    model.model = ""
    model.model_kwargs = {}
    # Remove the auto-generated _thinking_level mock
    del model._thinking_level

    expected = "gemini-2.5-flash-preview-05-20"
    assert get_model_name(model) == expected


def test_get_model_name_google_thinking_custom_attribute():
    """Test model name with custom _thinking_level attribute (our preferred approach)."""
    model = Mock()
    model.model_name = "gemini-2.5-flash-preview-05-20"
    model.model = ""
    model.model_kwargs = None
    model._thinking_level = "MEDIUM"

    expected = "gemini-2.5-flash-preview-05-20-thinking-MEDIUM"
    assert get_model_name(model) == expected


def test_get_model_name_no_thinking_level_attribute():
    """Test that models without _thinking_level attribute fall back to model_kwargs."""
    model = Mock()
    model.model_name = "gemini-2.5-flash-preview-05-20"
    model.model = ""
    model.model_kwargs = {
        "generation_config": {
            "thinking_config": {"thinking_budget": GoogleThinkingLevel.HIGH.value}
        }
    }
    # Simulate no _thinking_level attribute
    delattr(model, "_thinking_level") if hasattr(model, "_thinking_level") else None

    expected = "gemini-2.5-flash-preview-05-20-thinking-HIGH"
    assert get_model_name(model) == expected


def test_get_model_name_integration_with_real_models():
    """Integration test using actual get_model function to ensure real models work correctly."""
    # Test regular Google model
    google_model = get_model(google_flash=True)
    google_name = get_model_name(google_model)
    assert "gemini-2.5-flash-preview-05-20" in google_name
    assert "thinking" not in google_name

    # Test Google thinking models - these should now have _thinking_level attribute
    google_think_low = get_model(google_think_low=True)
    think_low_name = get_model_name(google_think_low)
    assert "thinking-LOW" in think_low_name
    assert "gemini-2.5-flash-preview-05-20-thinking-LOW" == think_low_name

    google_think_medium = get_model(google_think_medium=True)
    think_medium_name = get_model_name(google_think_medium)
    assert "thinking-MEDIUM" in think_medium_name
    assert "gemini-2.5-flash-preview-05-20-thinking-MEDIUM" == think_medium_name

    google_think_high = get_model(google_think_high=True)
    think_high_name = get_model_name(google_think_high)
    assert "thinking-HIGH" in think_high_name
    assert "gemini-2.5-flash-preview-05-20-thinking-HIGH" == think_high_name
