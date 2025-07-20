from ell_helper import get_ell_model


def test_get_ell_model_kimi():
    """Test that get_ell_model returns correct Kimi model name."""
    kimi_model_name = get_ell_model(kimi=True)
    assert kimi_model_name == "moonshotai/kimi-k2-instruct"


def test_get_ell_model_default_priority():
    """Test that default behavior still selects OpenAI when no flags are provided."""
    # When no flags are provided, should default to OpenAI
    default_model = get_ell_model()
    assert default_model == "gpt-4.1-2025-04-14"


def test_get_ell_model_kimi_vs_other_models():
    """Test that Kimi is correctly selected when specified."""
    # Test against other models
    claude_model = get_ell_model(claude=True)
    assert claude_model == "claude-3-7-sonnet-20250219"

    llama_model = get_ell_model(llama=True)
    assert llama_model == "meta-llama/llama-4-maverick-17b-128e-instruct"

    kimi_model = get_ell_model(kimi=True)
    assert kimi_model == "moonshotai/kimi-k2-instruct"


def test_get_ell_model_exclusive_selection():
    """Test that only one model can be selected at a time."""
    from unittest import mock

    # Mock the exit function to prevent actual exit during test
    with mock.patch("ell_helper.exit") as mock_exit:
        get_ell_model(kimi=True, claude=True)
        mock_exit.assert_called_once_with(1)


def test_get_ell_model_kimi_defaults_false():
    """Test that kimi defaults to False, maintaining OpenAI as fallback."""
    # This ensures kimi=False by default doesn't interfere with other model selection
    openai_model = get_ell_model(openai=True)
    assert openai_model == "gpt-4.1-2025-04-14"

    # And that no models selected defaults to OpenAI, not Kimi
    default_model = get_ell_model()
    assert default_model == "gpt-4.1-2025-04-14"
    assert default_model != "moonshotai/kimi-k2-instruct"
