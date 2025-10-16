from pathlib import Path
import subprocess
import os

# Suppress gRPC warnings from Google API (ALTS credentials, fork handlers)
os.environ.setdefault("GRPC_VERBOSITY", "ERROR")
os.environ.setdefault("GLOG_minloglevel", "2")

from langchain_core.language_models.chat_models import (
    BaseChatModel,
)
from langchain_core.messages import AIMessage
from typing import List, Any
import openai_wrapper
from icecream import ic
from types import FrameType
from typing import Callable, TypeVar
from datetime import datetime, timedelta
import asyncio
from enum import Enum


class GoogleThinkingLevel(Enum):
    """Enum for Google thinking levels with corresponding token budgets"""

    NONE = 0  # No thinking - fastest and cheapest
    LOW = 1024  # Light reasoning for simple tasks
    MEDIUM = 8192  # Moderate reasoning for complex tasks
    HIGH = 24576  # Maximum reasoning for very complex tasks


class OpenAIResponsesWrapper(BaseChatModel):
    """Wrapper to use OpenAI Responses API (v1/responses) with LangChain chat interface

    The Responses API is OpenAI's new stateful API that preserves reasoning state
    across turns, specifically designed for GPT-5 models including GPT-5-Codex.
    """

    model: str = "gpt-5-codex"
    model_name: str = "gpt-5-codex"  # For compatibility with get_model_name
    temperature: float = 0.7
    max_tokens: int = 4096

    @property
    def _llm_type(self) -> str:
        return "openai-responses"

    def _generate(
        self, messages: List[Any], stop=None, run_manager=None, **kwargs
    ) -> Any:
        """Convert chat messages to Responses API format and generate"""
        # Convert messages to single prompt
        prompt = self._messages_to_prompt(messages)

        # Get OpenAI client
        client = openai_wrapper.get_client()
        if not client:
            raise ValueError("OpenAI client not available")

        # Make Responses API call (v1/responses - new stateful API for GPT-5)
        # Note: GPT-5-Codex doesn't support temperature parameter
        response = client.responses.create(
            model=self.model,
            input=prompt,  # Responses API uses 'input' instead of 'prompt'
            max_output_tokens=self.max_tokens,  # Responses API uses 'max_output_tokens'
            **kwargs,
        )

        # Return in expected format
        # Responses API returns 'output_text' instead of 'choices[0].text'
        return self._create_chat_result(response.output_text)

    def _messages_to_prompt(self, messages) -> str:
        """Convert LangChain messages to a single prompt string"""
        prompt_parts = []

        for msg in messages:
            if hasattr(msg, "content"):
                if msg.__class__.__name__ == "SystemMessage":
                    prompt_parts.append(f"Instructions:\n{msg.content}")
                elif msg.__class__.__name__ == "HumanMessage":
                    prompt_parts.append(f"Input:\n{msg.content}")
                elif msg.__class__.__name__ == "AIMessage":
                    prompt_parts.append(f"Response:\n{msg.content}")

        # Add explicit instruction for response
        prompt_parts.append("\nGenerate a commit message based on the above diff:")

        return "\n\n".join(prompt_parts)

    def _create_chat_result(self, text: str):
        """Create a chat result from text response"""
        from langchain_core.outputs import ChatGeneration, ChatResult

        return ChatResult(generations=[ChatGeneration(message=AIMessage(content=text))])


def get_embeddings_model():
    """
    Returns an embeddings model for use with the FAISS vector store.
    Currently using OpenAI's embeddings model.
    """
    from langchain_openai import OpenAIEmbeddings

    return OpenAIEmbeddings(model="text-embedding-3-large")


def get_model_name(model: BaseChatModel):
    # if model has model_name, return that
    model_name = ""
    if hasattr(model, "model_name") and model.model_name != "":  # type: ignore
        model_name = model.model_name  # type: ignore
    elif hasattr(model, "model") and model.model != "":  # type: ignore
        model_name = model.model  # type: ignore
    else:
        model_name = str(model)

    # Remove "models/" prefix if present
    if model_name.startswith("models/"):
        model_name = model_name[7:]  # Skip "models/"

    # Check for custom _thinking_level attribute (added by our get_model function)
    if hasattr(model, "_thinking_level") and model._thinking_level:  # type: ignore
        thinking_level = model._thinking_level  # type: ignore
        model_name = f"{model_name}-thinking-{thinking_level}"
        return model_name

    # Check if this is a Google model with thinking configuration in model_kwargs
    if (
        hasattr(model, "model_kwargs")
        and model.model_kwargs is not None
        and model.model_kwargs
    ):  # type: ignore
        try:
            model_kwargs = model.model_kwargs  # type: ignore
            generation_config = model_kwargs.get("generation_config", {})
            thinking_config = generation_config.get("thinking_config", {})
            thinking_budget = thinking_config.get("thinking_budget")

            if thinking_budget is not None and isinstance(
                thinking_budget, (int, float)
            ):
                # Map thinking budget to level name using the enum
                thinking_level_map = {
                    GoogleThinkingLevel.LOW.value: "LOW",
                    GoogleThinkingLevel.MEDIUM.value: "MEDIUM",
                    GoogleThinkingLevel.HIGH.value: "HIGH",
                }
                thinking_level = thinking_level_map.get(
                    thinking_budget, f"CUSTOM-{thinking_budget}"
                )
                model_name = f"{model_name}-thinking-{thinking_level}"
        except (AttributeError, TypeError):
            # Handle cases where model_kwargs doesn't behave like a dict (e.g., Mock objects)
            pass

    return model_name


def get_models(
    openai: bool = False,
    google: bool = False,
    claude: bool = False,
    haiku: bool = False,
    llama: bool = False,
    google_think: bool = False,
    o4_mini: bool = False,
    google_flash: bool = False,
    structured: bool = False,
    openai_mini: bool = False,
    google_think_low: bool = False,
    google_think_medium: bool = False,
    google_think_high: bool = False,
    kimi: bool = False,
    gpt_oss: bool = False,
    gpt5_codex: bool = False,  # NEW
    grok4_fast: bool = False,  # NEW
) -> List[BaseChatModel]:
    ret = []

    if google:
        ret.append(get_model(google=True))

    if google_flash:
        ret.append(get_model(google_flash=True))

    if google_think:
        ret.append(get_model(google_think=True))

    if google_think_low:
        ret.append(get_model(google_think_low=True))

    if google_think_medium:
        ret.append(get_model(google_think_medium=True))

    if google_think_high:
        ret.append(get_model(google_think_high=True))

    if claude:
        ret.append(get_model(claude=True))

    if haiku:
        ret.append(get_model(haiku=True))

    if llama:
        ret.append(get_model(llama=True))

    if o4_mini:
        ret.append(get_model(o4_mini=True))

    if openai:
        ret.append(get_model(openai=True))

    if openai_mini:
        ret.append(get_model(openai_mini=True))

    if kimi:
        ret.append(get_model(kimi=True))

    if gpt_oss:
        ret.append(get_model(gpt_oss=True))

    if gpt5_codex:
        ret.append(get_model(gpt5_codex=True))

    if grok4_fast:
        ret.append(get_model(grok4_fast=True))

    return ret


def get_model(
    openai: bool = False,
    google: bool = False,
    claude: bool = False,
    haiku: bool = False,
    llama: bool = False,
    google_think: bool = False,
    o4_mini: bool = False,
    google_flash: bool = False,
    structured: bool = False,
    openai_mini: bool = False,
    google_think_low: bool = False,
    google_think_medium: bool = False,
    google_think_high: bool = False,
    kimi: bool = False,
    gpt_oss: bool = False,
    gpt5_codex: bool = False,  # NEW
    grok4_fast: bool = False,  # NEW
) -> BaseChatModel:
    """
    See changes in diff
    """
    # if more then one is true, exit and fail
    count_true = sum(
        [
            openai,
            google,
            claude,
            haiku,
            llama,
            google_think,
            o4_mini,
            google_flash,
            google_think_low,
            google_think_medium,
            google_think_high,
            kimi,
            gpt_oss,
            openai_mini,
            gpt5_codex,  # NEW
            grok4_fast,  # NEW
        ]
    )
    if count_true > 1:
        print("Only one model can be selected")
        exit(1)
    if count_true == 0:
        # default to claude (fast and good quality)
        claude = True

    if google:
        from langchain_google_genai import ChatGoogleGenerativeAI

        api_key = os.getenv("GOOGLE_API_KEY", "DUMMY_KEY")
        model = ChatGoogleGenerativeAI(model="gemini-2.5-pro", google_api_key=api_key)
    elif google_flash:
        from langchain_google_genai import ChatGoogleGenerativeAI

        api_key = os.getenv("GOOGLE_API_KEY", "DUMMY_KEY")
        model = ChatGoogleGenerativeAI(model="gemini-2.5-flash", google_api_key=api_key)
    elif google_think:
        from langchain_google_genai import ChatGoogleGenerativeAI

        api_key = os.getenv("GOOGLE_API_KEY", "DUMMY_KEY")
        model = ChatGoogleGenerativeAI(
            model="gemini-2.5-flash",
            model_kwargs={
                "generation_config": {
                    "thinking_config": {
                        "thinking_budget": GoogleThinkingLevel.LOW.value
                    }
                }
            },
            google_api_key=api_key,
        )
        # Add custom attribute to track thinking level
        model._thinking_level = "LOW"  # type: ignore
    elif google_think_low:
        from langchain_google_genai import ChatGoogleGenerativeAI

        api_key = os.getenv("GOOGLE_API_KEY", "DUMMY_KEY")
        model = ChatGoogleGenerativeAI(
            model="gemini-2.5-flash",
            model_kwargs={
                "generation_config": {
                    "thinking_config": {
                        "thinking_budget": GoogleThinkingLevel.LOW.value
                    }
                }
            },
            google_api_key=api_key,
        )
        # Add custom attribute to track thinking level
        model._thinking_level = "LOW"  # type: ignore
    elif google_think_medium:
        from langchain_google_genai import ChatGoogleGenerativeAI

        api_key = os.getenv("GOOGLE_API_KEY", "DUMMY_KEY")
        model = ChatGoogleGenerativeAI(
            model="gemini-2.5-flash",
            model_kwargs={
                "generation_config": {
                    "thinking_config": {
                        "thinking_budget": GoogleThinkingLevel.MEDIUM.value
                    }
                }
            },
            google_api_key=api_key,
        )
        # Add custom attribute to track thinking level
        model._thinking_level = "MEDIUM"  # type: ignore
    elif google_think_high:
        from langchain_google_genai import ChatGoogleGenerativeAI

        api_key = os.getenv("GOOGLE_API_KEY", "DUMMY_KEY")
        model = ChatGoogleGenerativeAI(
            model="gemini-2.5-flash",
            model_kwargs={
                "generation_config": {
                    "thinking_config": {
                        "thinking_budget": GoogleThinkingLevel.HIGH.value
                    }
                }
            },
            google_api_key=api_key,
        )
        # Add custom attribute to track thinking level
        model._thinking_level = "HIGH"  # type: ignore
    elif claude:
        from langchain_anthropic import ChatAnthropic

        model = ChatAnthropic(
            model_name="claude-sonnet-4-5-20250929",
            max_tokens=64000,
            model_kwargs={"format": "json"} if structured else {},
        )
    elif haiku:
        from langchain_anthropic import ChatAnthropic

        model = ChatAnthropic(
            model_name="claude-haiku-4-5",
            max_tokens=64000,
            model_kwargs={"format": "json"} if structured else {},
        )
    elif llama:
        from langchain_groq import ChatGroq

        model = ChatGroq(model_name="meta-llama/llama-4-maverick-17b-128e-instruct")
    elif kimi:
        from langchain_groq import ChatGroq

        model = ChatGroq(model_name="moonshotai/kimi-k2-instruct-0905")
    elif gpt_oss:
        from langchain_groq import ChatGroq

        model = ChatGroq(model_name="openai/gpt-oss-120b")
    elif o4_mini:
        from langchain_openai.chat_models import ChatOpenAI

        model = ChatOpenAI(model="o4-mini-2025-04-16", model_kwargs={})
    elif openai_mini:
        from langchain_openai.chat_models import ChatOpenAI

        model = ChatOpenAI(model="gpt-5-mini", model_kwargs={})
    elif gpt5_codex:
        # Use custom wrapper for Responses API
        model = OpenAIResponsesWrapper(model="gpt-5-codex")
    elif grok4_fast:
        from langchain_xai import ChatXAI

        model = ChatXAI(
            model="grok-4-fast",
            temperature=0,
            max_tokens=2_000_000,  # Set to grok-4-fast's large context window
        )
    else:
        from langchain_openai.chat_models import ChatOpenAI

        model = ChatOpenAI(model=openai_wrapper.gpt4.name)

    return model


def tracer_project_name():
    import inspect
    from pathlib import Path
    import socket

    # get the first caller name that is not in langchain_helper
    def app_frame(stack) -> FrameType:
        for frame in stack:
            if frame.filename != __file__:
                return frame
        # if can't find  anything  use my parent
        return stack[1]

    caller_frame = app_frame(inspect.stack())
    caller_function = caller_frame.function  # type:ignore
    caller_filename = Path(inspect.getfile(caller_frame.frame)).name  # type:ignore

    hostname = socket.gethostname()  # Get the hostname

    return f"{caller_filename}:{caller_function}[{hostname}]"


def langsmith_trace_if_requested(trace: bool, the_call):
    if trace:
        return langsmith_trace(the_call)
    else:
        the_call()
        return


T = TypeVar("T")


async def async_run_on_llms(
    lcel_func: Callable[[BaseChatModel], T], llms
) -> List[[T, BaseChatModel, timedelta]]:  # type: ignore
    async def timed_lcel_task(lcel_func, llm):
        start_time = datetime.now()
        result = await (lcel_func(llm)).ainvoke({})
        end_time = datetime.now()
        time_delta = end_time - start_time
        return result, llm, time_delta

    tasks = [timed_lcel_task(lcel_func, llm) for llm in llms]
    return [result for result in await asyncio.gather(*tasks)]


def langsmith_trace(the_call):
    from langchain_core.tracers.context import tracing_v2_enabled
    from langchain.callbacks.tracers.langchain import wait_for_all_tracers

    trace_name = tracer_project_name()
    with tracing_v2_enabled(project_name=trace_name) as tracer:
        ic("Using Langsmith:", trace_name)
        the_call()
        ic(tracer.get_run_url())
    wait_for_all_tracers()


def to_gist_multiple(paths: List[Path], description: str = ""):
    # Convert all paths to absolute paths and pass them as arguments
    cmd = ["gh", "gist", "create"]
    if description:
        cmd.extend(["-d", description])
    cmd.extend([str(path.absolute()) for path in paths])

    gist = subprocess.run(
        cmd,
        check=True,
        stdout=subprocess.PIPE,
        text=True,
    )
    ic(gist)
    ic(gist.stdout.strip())
    subprocess.run(["open", gist.stdout.strip()])


def to_gist(path: Path):
    gist = subprocess.run(
        ["gh", "gist", "create", str(path.absolute())],
        check=True,
        stdout=subprocess.PIPE,
        text=True,
    )
    ic(gist)
    ic(gist.stdout.strip())
    subprocess.run(["open", gist.stdout.strip()])
