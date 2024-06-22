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


def get_model_name(model: BaseChatModel):
    # if model has model_name, return that
    if hasattr(model, "model_name") and model.model_name != "":  # type: ignore
        return model.model_name  # type: ignore
    if hasattr(model, "model") and model.model != "":  # type: ignore
        return model.model  # type: ignore
    else:
        return str(model)


def get_models(
    openai: bool = False,
    google: bool = False,
    claude: bool = False,
    llama: bool = False,
) -> List[BaseChatModel]:
    ret = []

    if google:
        from langchain_google_genai import ChatGoogleGenerativeAI

        model = ChatGoogleGenerativeAI(model="gemini-1.5-pro-latest")
        ret.append(model)

    if claude:
        from langchain_anthropic import ChatAnthropic

        model = ChatAnthropic(model_name="claude-3-5-sonnet-20240620")
        ret.append(model)

    if llama:
        from langchain_groq import ChatGroq

        model = ChatGroq(model_name="llama3-70b-8192")
        ret.append(model)

    if openai:
        from langchain_openai.chat_models import ChatOpenAI

        model = ChatOpenAI(model=openai_wrapper.gpt4.name)
        ret.append(model)

    return ret


def get_model(
    openai: bool = False,
    google: bool = False,
    claude: bool = False,
    llama: bool = False,
) -> BaseChatModel:
    """
    See changes in diff
    """
    # if more then one is true, exit and fail
    count_true = sum([openai, google, claude, llama])
    if count_true > 1:
        print("Only one model can be selected")
        exit(1)
    if count_true == 1:
        # default to openai
        openai = True

    if google:
        from langchain_google_genai import ChatGoogleGenerativeAI

        model = ChatGoogleGenerativeAI(model="gemini-1.5-pro-latest")
    elif claude:
        from langchain_anthropic import ChatAnthropic

        model = ChatAnthropic(model_name="claude-3-opus-20240229")
    elif llama:
        from langchain_groq import ChatGroq

        model = ChatGroq(model_name="llama3-70b-8192")
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


def get_text_from_path_or_stdin(path):
    import sys
    import requests
    import html2text

    if not path:  # read from stdin
        return "".join(sys.stdin.readlines())
    # check if path is URL
    if path.startswith("http"):
        request = requests.get(path)
        out = html2text.html2text(request.text)
        return out
    if path:
        # try to open the file, using pathlib
        return Path(path).read_text()
    # read stdin
    return str(sys.stdin.readlines())
