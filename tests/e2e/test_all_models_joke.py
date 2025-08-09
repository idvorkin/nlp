"""
End-to-end test that runs a joke prompt through all available models in parallel.
This is a slow test that actually calls the LLM APIs.
"""

import pytest
import asyncio
import time
from typing import Tuple, Optional
from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages import HumanMessage
import sys
import os

# Add parent directory to path to import our modules
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(__file__))))

import langchain_helper


def get_model_safe(model_name: str, **kwargs) -> Optional[BaseChatModel]:
    """
    Safely get a model, returning None if API keys are missing.
    """
    try:
        return langchain_helper.get_model(**{model_name: True})
    except Exception as e:
        if "api_key" in str(e).lower() or "api key" in str(e).lower():
            print(f"Skipping {model_name}: Missing API key")
            return None
        raise


async def run_joke_on_model(
    model: BaseChatModel, prompt: str
) -> Tuple[str, str, float]:
    """
    Run the joke prompt on a single model and return the model name, response, and timing.
    """
    model_name = langchain_helper.get_model_name(model)
    start_time = time.time()

    try:
        # Use async invoke if available, otherwise fallback to sync
        if hasattr(model, "ainvoke"):
            response = await model.ainvoke([HumanMessage(content=prompt)])
        else:
            # Run sync method in thread pool for models without async support
            response = await asyncio.get_event_loop().run_in_executor(
                None, model.invoke, [HumanMessage(content=prompt)]
            )

        elapsed_time = time.time() - start_time
        return model_name, response.content, elapsed_time
    except Exception as e:
        elapsed_time = time.time() - start_time
        return model_name, f"Error: {str(e)}", elapsed_time


@pytest.mark.asyncio
@pytest.mark.slow
async def test_all_models_joke_parallel():
    """
    Test that runs a joke prompt through all available models in parallel.
    This test is marked as slow since it makes actual API calls.
    """
    prompt = "Tell me a joke about a manic depressed programmer"

    # Get all available models (skip those without API keys)
    model_configs = [
        ("openai", {}),
        ("openai_mini", {}),
        ("google", {}),
        ("google_flash", {}),
        ("claude", {}),
        ("llama", {}),
        ("deepseek", {}),
        ("kimi", {}),
        ("gpt_oss", {}),
        # Skip o4_mini as it might have different rate limits
        # Skip google_think variants to keep test faster
    ]

    models = []
    for model_name, config in model_configs:
        model = get_model_safe(model_name, **config)
        if model:
            models.append(model)

    if not models:
        pytest.skip("No models available (missing API keys)")

    print(f"\n\nRunning joke prompt on {len(models)} models in parallel...")
    print(f"Prompt: {prompt}\n")

    # Run all models in parallel
    tasks = [run_joke_on_model(model, prompt) for model in models]
    results = await asyncio.gather(*tasks, return_exceptions=True)

    # Process and display results
    successful_responses = []
    failed_responses = []
    timings = []

    for result in results:
        if isinstance(result, Exception):
            failed_responses.append(("Exception", str(result), 0.0))
        else:
            model_name, response, elapsed_time = result
            timings.append((model_name, elapsed_time))

            if "Error:" in response:
                failed_responses.append((model_name, response, elapsed_time))
            else:
                successful_responses.append((model_name, response, elapsed_time))
                print(f"\n{'=' * 60}")
                print(f"Model: {model_name}")
                print(f"Response time: {elapsed_time:.2f} seconds")
                print(f"Response: {response[:500]}...")  # Truncate long responses

    # Print timing summary
    print(f"\n{'=' * 60}")
    print("TIMING SUMMARY:")
    timings.sort(key=lambda x: x[1])  # Sort by response time
    for model_name, elapsed_time in timings:
        status = (
            "✓" if any(m == model_name for m, _, _ in successful_responses) else "✗"
        )
        print(f"  {status} {model_name}: {elapsed_time:.2f}s")

    # Print overall summary
    print(f"\n{'=' * 60}")
    print("OVERALL SUMMARY:")
    print(f"Successful responses: {len(successful_responses)}/{len(models)}")
    if timings:
        avg_time = sum(t for _, t in timings) / len(timings)
        print(f"Average response time: {avg_time:.2f}s")
        print(f"Fastest: {timings[0][0]} ({timings[0][1]:.2f}s)")
        print(f"Slowest: {timings[-1][0]} ({timings[-1][1]:.2f}s)")

    if failed_responses:
        print("\nFailed responses:")
        for model_name, error, elapsed_time in failed_responses:
            print(f"  - {model_name} ({elapsed_time:.2f}s): {error[:100]}...")

    # Assert that at least some models succeeded
    assert len(successful_responses) > 0, "No models returned successful responses"

    # Optionally assert that most models succeeded (allowing for some API failures)
    success_rate = len(successful_responses) / len(models)
    assert success_rate >= 0.5, (
        f"Too many models failed: {success_rate:.1%} success rate"
    )


@pytest.mark.asyncio
@pytest.mark.slow
async def test_specific_models_joke():
    """
    Test specific high-priority models with the joke prompt.
    This is a smaller test focusing on key models.
    """
    prompt = "Tell me a joke about a manic depressed programmer"

    # Test just the main models (skip those without API keys)
    model_configs = [
        ("openai", {}),
        ("claude", {}),
        ("google", {}),
    ]

    models = []
    for model_name, config in model_configs:
        model = get_model_safe(model_name, **config)
        if model:
            models.append((model_name, model))

    if not models:
        pytest.skip("No models available (missing API keys)")

    print(f"\n\nTesting {len(models)} primary models...")
    print(f"Prompt: {prompt}\n")

    tasks = [run_joke_on_model(model, prompt) for model_name, model in models]
    results = await asyncio.gather(*tasks)

    # Display results with timing
    timings = []
    successful = 0

    for (model_name, _), (returned_name, response, elapsed_time) in zip(
        models, results
    ):
        timings.append((model_name, elapsed_time))
        print(f"\n{'=' * 60}")
        print(f"Model: {model_name}")
        print(f"Response time: {elapsed_time:.2f} seconds")

        if not response.startswith("Error:"):
            print("Status: ✓ Success")
            print(f"Response: {response[:300]}...")
            successful += 1
        else:
            print("Status: ✗ Failed")
            print(f"Error: {response}")

    # Print timing summary
    print(f"\n{'=' * 60}")
    print("TIMING SUMMARY:")
    timings.sort(key=lambda x: x[1])  # Sort by response time
    for model_name, elapsed_time in timings:
        print(f"  {model_name}: {elapsed_time:.2f}s")

    # Print overall summary
    print(f"\n{'=' * 60}")
    print("RESULTS:")
    print(f"{successful}/{len(models)} models responded successfully!")
    if timings:
        avg_time = sum(t for _, t in timings) / len(timings)
        print(f"Average response time: {avg_time:.2f}s")
        print(f"Fastest: {timings[0][0]} ({timings[0][1]:.2f}s)")
        print(f"Slowest: {timings[-1][0]} ({timings[-1][1]:.2f}s)")


if __name__ == "__main__":
    # Allow running directly with: python test_all_models_joke.py
    asyncio.run(test_all_models_joke_parallel())
