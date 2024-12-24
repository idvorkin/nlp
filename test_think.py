import pytest
from think import (
    get_latest_github_commit_url,
    sanitize_filename,
    get_categories_and_description,
    CategoryInfo,
)


def test_sanitize_filename():
    # Test basic sanitization
    assert sanitize_filename("hello.txt") == "hello.txt"
    assert sanitize_filename("hello/world.txt") == "hello_world.txt"
    assert sanitize_filename('test:*?"<>|.txt') == "test_______.txt"
    assert sanitize_filename("hello world.txt") == "hello world.txt"
    assert sanitize_filename("") == ""
    assert sanitize_filename(" spaces ") == "spaces"


def test_get_categories_and_description():
    # Test default categories
    default_info = get_categories_and_description(
        core_problems=False, writer=False, interests=False
    )
    assert isinstance(default_info, CategoryInfo)
    assert default_info.description == "default questions"
    assert "Summary" in default_info.categories
    assert "Most Novel Ideas" in default_info.categories

    # Test core problems
    core_info = get_categories_and_description(
        core_problems=True, writer=False, interests=False
    )
    assert core_info.description == "core problems"
    assert "What's the real problem you are trying to solve?" in core_info.categories

    # Test writer
    writer_info = get_categories_and_description(
        core_problems=False, writer=True, interests=False
    )
    assert writer_info.description == "writer questions"
    assert any("audience" in cat.lower() for cat in writer_info.categories)

    # Test interests
    interests_info = get_categories_and_description(
        core_problems=False, writer=False, interests=True
    )
    assert interests_info.description == "interests questions"
    assert "Summary" in interests_info.categories


def test_github_url_format():
    url = get_latest_github_commit_url()
    assert url.startswith("https://github.com/idvorkin/nlp/blob/")
    assert url.endswith("/think.py")
    # Should contain a 40-character SHA
    parts = url.split("/")
    sha = parts[-2]
    assert len(sha) == 40 or sha == "main"  # main is the fallback


@pytest.mark.asyncio
async def test_analysis_body_generation():
    # This would be an integration test - we might want to mock the LLM calls
    pass
