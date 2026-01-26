import pytest
from pathlib import Path
from unittest.mock import patch, MagicMock
import subprocess


def test_mlx_models_data_structure():
    """MLX_MODELS should have required fields for each model"""
    # Import here to avoid import errors if mlx-audio not installed
    import sys
    sys.path.insert(0, str(Path(__file__).parent.parent))

    # Import the module-level constant
    from tts import MLX_MODELS

    assert len(MLX_MODELS) > 0, "Should have at least one model"

    required_fields = {"name", "path", "size", "released", "features"}
    for model in MLX_MODELS:
        missing = required_fields - set(model.keys())
        assert not missing, f"Model {model.get('name', 'unknown')} missing fields: {missing}"

        # Validate field types
        assert isinstance(model["name"], str) and len(model["name"]) > 0
        assert isinstance(model["path"], str) and "/" in model["path"]
        assert isinstance(model["size"], str)
        assert isinstance(model["released"], str) and len(model["released"]) == 7  # YYYY-MM format
        assert isinstance(model["features"], str)


def test_mlx_models_sorted_by_date():
    """MLX_MODELS should be sorted by release date, newest first"""
    import sys
    sys.path.insert(0, str(Path(__file__).parent.parent))
    from tts import MLX_MODELS

    dates = [m["released"] for m in MLX_MODELS]
    assert dates == sorted(dates, reverse=True), "Models should be sorted by date, newest first"


def test_list_audio_devices_parsing():
    """_list_audio_devices should parse ffmpeg output correctly"""
    import sys
    sys.path.insert(0, str(Path(__file__).parent.parent))
    from tts import _list_audio_devices

    mock_stderr = """
[AVFoundation indev @ 0x123] AVFoundation video devices:
[AVFoundation indev @ 0x123] [0] FaceTime HD Camera
[AVFoundation indev @ 0x123] AVFoundation audio devices:
[AVFoundation indev @ 0x123] [0] MacBook Air Microphone
[AVFoundation indev @ 0x123] [1] External Mic
"""

    with patch("subprocess.run") as mock_run:
        mock_run.return_value = MagicMock(stderr=mock_stderr, returncode=1)
        devices = _list_audio_devices()

    assert len(devices) == 2
    assert devices[0] == (0, "MacBook Air Microphone")
    assert devices[1] == (1, "External Mic")


def test_list_audio_devices_empty():
    """_list_audio_devices should return empty list when no devices"""
    import sys
    sys.path.insert(0, str(Path(__file__).parent.parent))
    from tts import _list_audio_devices

    with patch("subprocess.run") as mock_run:
        mock_run.return_value = MagicMock(stderr="No devices found", returncode=1)
        devices = _list_audio_devices()

    assert devices == []


def test_mlx_clone_missing_voice_file(tmp_path):
    """mlx_clone should fail gracefully when voice file doesn't exist"""
    from typer.testing import CliRunner
    import sys
    sys.path.insert(0, str(Path(__file__).parent.parent))
    from tts import app

    runner = CliRunner()
    fake_voice = tmp_path / "nonexistent.wav"

    result = runner.invoke(app, ["mlx-clone", "--voice", str(fake_voice)])

    assert result.exit_code == 1
    assert "not found" in result.output.lower()


def test_voice_design_model_has_instruct():
    """VoiceDesign models should have instruct field"""
    import sys
    sys.path.insert(0, str(Path(__file__).parent.parent))
    from tts import MLX_MODELS

    voice_design_models = [m for m in MLX_MODELS if "VoiceDesign" in m["path"]]

    for model in voice_design_models:
        assert "instruct" in model, f"VoiceDesign model {model['name']} should have instruct field"
        assert isinstance(model["instruct"], str) and len(model["instruct"]) > 0


def test_get_mlx_output_file_with_suffix(tmp_path):
    """_get_mlx_output_file should return _000.wav file when it exists"""
    import sys
    sys.path.insert(0, str(Path(__file__).parent.parent))
    from tts import _get_mlx_output_file

    # Create base path and the _000 file
    base = tmp_path / "test.wav"
    actual = tmp_path / "test_000.wav"
    actual.write_text("audio data")

    result = _get_mlx_output_file(base)
    assert result == actual


def test_get_mlx_output_file_without_suffix(tmp_path):
    """_get_mlx_output_file should return base path when _000.wav doesn't exist"""
    import sys
    sys.path.insert(0, str(Path(__file__).parent.parent))
    from tts import _get_mlx_output_file

    base = tmp_path / "test.wav"
    # Don't create the _000 file

    result = _get_mlx_output_file(base)
    assert result == base
