from pathlib import Path
import subprocess
from langchain_core.language_models.chat_models import (
    BaseChatModel,
)
import openai_wrapper
from icecream import ic
from types import FrameType
from typing import Callable, List, TypeVar
from datetime import datetime, timedelta
import asyncio

def get_embeddings_model():
    """
    Returns an embeddings model for use with Chroma DB.
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

    return model_name


def get_models(
    openai: bool = False,
    google: bool = False,
    claude: bool = False,
    llama: bool = False,
    google_think: bool = False,
    deepseek: bool = False,
    o4_mini: bool = False,
    google_flash: bool = False,
    structured: bool = False,
    openai_mini: bool = False,
) -> List[BaseChatModel]:
    ret = []

    if google:
        ret.append(get_model(google=True))

    if google_flash:
        ret.append(get_model(google_flash=True))

    if google_think:
        ret.append(get_model(google_think=True))

    if claude:
        ret.append(get_model(claude=True))

    if llama:
        ret.append(get_model(llama=True))

    if deepseek:
        ret.append(get_model(deepseek=True))

    if o4_mini:
        ret.append(get_model(o4_mini=True))

    if openai:
        ret.append(get_model(openai=True))

    if openai_mini:
        ret.append(get_model(openai_mini=True))

    return ret


def get_model(
    openai: bool = False,
    google: bool = False,
    claude: bool = False,
    llama: bool = False,
    google_think: bool = False,
    deepseek: bool = False,
    o4_mini: bool = False,
    google_flash: bool = False,
    structured: bool = False,
    openai_mini: bool = False,
) -> BaseChatModel:
    """
    See changes in diff
    """
    # if more then one is true, exit and fail
    count_true = sum(
        [openai, google, claude, llama, google_think, deepseek, o4_mini, google_flash]
    )
    if count_true > 1:
        print("Only one model can be selected")
        exit(1)
    if count_true == 0:
        # default to openai
        openai = True

    if google:
        from langchain_google_genai import ChatGoogleGenerativeAI

        model = ChatGoogleGenerativeAI(model="gemini-2.5-pro-exp-03-25")
    elif google_flash:
        from langchain_google_genai import ChatGoogleGenerativeAI

        model = ChatGoogleGenerativeAI(model="gemini-2.5-flash-preview-04-17")
    elif google_think:
        from langchain_google_genai import ChatGoogleGenerativeAI

        model = ChatGoogleGenerativeAI(model="gemini-2.0-flash-thinking-exp-01-21")
    elif claude:
        from langchain_anthropic import ChatAnthropic

        model = ChatAnthropic(
            model_name="claude-3-7-sonnet-20250219",
            model_kwargs={"format": "json"} if structured else {},
        )
    elif llama:
        from langchain_groq import ChatGroq

        model = ChatGroq(model_name="meta-llama/llama-4-maverick-17b-128e-instruct")
    elif deepseek:
        from langchain_groq import ChatGroq

        model = ChatGroq(model_name="deepseek-r1-distill-llama-70b")
    elif o4_mini:
        from langchain_openai.chat_models import ChatOpenAI

        model = ChatOpenAI(model="o4-mini-2025-04-16", model_kwargs={})
    elif openai_mini:
        from langchain_openai.chat_models import ChatOpenAI

        model = ChatOpenAI(model="gpt-4.1-mini", model_kwargs={})
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
