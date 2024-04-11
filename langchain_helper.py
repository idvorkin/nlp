from langchain_core.language_models.chat_models import (
    BaseChatModel,
)
import openai_wrapper
from icecream import ic


def get_model_name(model: BaseChatModel):
    # if model has model_name, return that
    if hasattr(model, "model_name") and model.model_name != "":
        return model.model_name
    if hasattr(model, "model") and model.model != "":
        return model.model
    else:
        return str(model)


def get_model(
    openai: bool = False, google: bool = False, claude: bool = False
) -> BaseChatModel:
    """
    See changes in diff
    """
    # if more then one is true, exit and fail
    count_true = sum([openai, google, claude])
    if count_true > 1:
        print("Only one model can be selected")
        exit(1)
    if count_true == 1:
        # default to openai
        openai = True

    if google:
        from langchain_google_genai import ChatGoogleGenerativeAI

        model = ChatGoogleGenerativeAI(model="gemini-1.5.-pro-latest")
    elif claude:
        from langchain_anthropic import ChatAnthropic

        model = ChatAnthropic(model="claude-3-opus-20240229")
    else:
        from langchain_openai.chat_models import ChatOpenAI

        model = ChatOpenAI(model=openai_wrapper.gpt4.name)

    return model


def tracer_project_name():
    import inspect
    from pathlib import Path
    import socket

    caller_frame = inspect.stack()[1]
    caller_function = caller_frame.function
    caller_filename = Path(inspect.getfile(caller_frame.frame)).name  # type:ignore
    hostname = socket.gethostname()  # Get the hostname

    return f"{caller_filename}:{caller_function}[{hostname}]"


def langsmith_trace_if_requested(trace: bool, the_call):
    if trace:
        return langsmith_trace(the_call)
    else:
        return the_call


def langsmith_trace(the_call):
    from langchain_core.tracers.context import tracing_v2_enabled
    from langchain.callbacks.tracers.langchain import wait_for_all_tracers

    trace_name = tracer_project_name()
    with tracing_v2_enabled(project_name=trace_name) as tracer:
        ic("Using Langsmith:", trace_name)
        the_call()
        ic(tracer.get_run_url())
    wait_for_all_tracers()
