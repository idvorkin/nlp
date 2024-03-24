import json
import os
import time

import tiktoken
from icecream import ic
from openai import OpenAI
from pydantic import BaseModel
from tenacity import retry, stop_after_attempt, wait_random_exponential
from pathlib import Path


def setup_secret():
    secret_file = Path.home() / "gits/igor2/secretBox.json"
    SECRETS = json.loads(secret_file.read_text())
    os.environ["OPENAI_API_KEY"] = SECRETS["openai"]


def setup_gpt():
    PASSWORD = "replaced_from_secret_box"
    with open(os.path.expanduser("~/gits/igor2/secretBox.json")) as json_data:
        SECRETS = json.load(json_data)
        PASSWORD = SECRETS["openai"]

    return OpenAI(api_key=PASSWORD)


client = setup_gpt()


class CompletionModel(BaseModel):
    max_input_only_tokens: int
    max_output_tokens: int
    name: str


gpt4 = CompletionModel(
    max_input_only_tokens=100 * 1000,
    max_output_tokens=4 * 1000,
    name="gpt-4-0125-preview",
)
gpt35 = CompletionModel(
    max_input_only_tokens=12 * 1000,
    max_output_tokens=4 * 1000,
    name="gpt-3.5-turbo-0125",
)


def tracer_project_name():
    import inspect
    from pathlib import Path

    ic([s.function for s in inspect.stack()])
    caller_function = inspect.stack()[1].function

    def get_current_file_name():
        return Path(inspect.getfile(inspect.currentframe())).name  # type:ignore

    return f"{get_current_file_name()}:{caller_function}"


def get_model_type(u4: bool) -> CompletionModel:
    if u4:
        return gpt4
    else:
        return gpt35


text_model_gpt_4 = "gpt-4-0125-preview"
gpt_4_tokens = 100000
gpt_4_input_tokens = 100 * 1000
gpt_4_output_tokens = 100 * 1000
text_model_gpt35 = "gpt-3.5-turbo-1106"
gpt_3_5_tokens = 16000
code_model_best = "code-davinci-003"


def model_to_max_tokens(model):
    model_to_tokens = {text_model_gpt_4: gpt_4_tokens, text_model_gpt35: gpt_3_5_tokens}
    return model_to_tokens[model]


def get_model(u4):
    model = ""
    if u4:
        model = text_model_gpt_4
    else:
        model = text_model_gpt35
    return model


def get_remaining_output_tokens(model: CompletionModel, prompt: str):
    # For symetric models, max_input_only_tokens= 0 and max_output_tokens  = the full context window
    # For asymmetrics models, max_output_tokens = full context_window - max_input_only_tokens

    input_tokens = num_tokens_from_string(prompt, "cl100k_base")
    # If you only used input_context only tokens, don't remove anything f+ 100
    output_tokens_consumed = max((input_tokens - model.max_input_only_tokens), 0)
    return model.max_output_tokens - output_tokens_consumed


def choose_model(u4, tokens=0):
    model = "SPECIFY_MODEL"
    if u4:
        model = text_model_gpt_4
    else:
        model = text_model_gpt35

    is_token_count_the_default = tokens == 0  # TBD if we can do it without hardcoding.
    if is_token_count_the_default:
        tokens = model_to_max_tokens(model)

    return model, tokens


def remaining_response_tokens(model, system_prompt, user_prompt):
    tokens = model_to_max_tokens(model)
    input_tokens = (
        num_tokens_from_string(user_prompt + system_prompt, "cl100k_base") + 100
    )  # too lazy to count the messages stuf
    output_tokens = tokens - input_tokens
    return output_tokens


def num_tokens_from_string(string: str, encoding_name: str = "cl100k_base") -> int:
    """Returns the number of tokens in a text string."""
    encoding = tiktoken.get_encoding(encoding_name)
    num_tokens = len(encoding.encode(string))
    num_tokens = num_tokens + 1  # for newline
    return num_tokens


def ask_gpt(
    prompt_to_gpt="Make a rhyme about Dr. Seuss forgetting to pass a default paramater",
    tokens: int = 0,
    u4=False,
    debug=False,
):
    return ask_gpt_n(prompt_to_gpt, tokens=tokens, u4=u4, debug=debug, n=1)[0]


@retry(
    wait=wait_random_exponential(min=1, max=60),
    stop=stop_after_attempt(3),
)
def ask_gpt_n(
    prompt_to_gpt="Make a rhyme about Dr. Seuss forgetting to pass a default paramater",
    tokens: int = 0,
    u4=False,
    debug=False,
    n=1,
):
    text_model_best, tokens = choose_model(u4)
    messages = [
        {"role": "system", "content": "You are a really good improv coach."},
        {"role": "user", "content": prompt_to_gpt},
    ]

    model = get_model_type(u4)
    output_tokens = get_remaining_output_tokens(model, prompt_to_gpt)
    text_model_best = model.name

    if debug:
        ic(text_model_best)
        ic(tokens)
        ic(output_tokens)

    start = time.time()
    responses = n
    response_contents = ["" for _ in range(responses)]
    for chunk in client.chat.completions.create(  # type: Ignore
        model=text_model_best,
        messages=messages,  # type: ignore
        max_tokens=output_tokens,
        n=responses,
        temperature=0.7,
        stream=True,
    ):
        if "choices" not in chunk:
            continue

        for elem in chunk["choices"]:  # type: ignore
            delta = elem["delta"]
            delta_content = delta.get("content", "")
            response_contents[elem["index"]] += delta_content
    if debug:
        out = f"All chunks took: {int((time.time() - start)*1000)} ms"
        ic(out)

    # hard code to only return first response
    return response_contents


def openai_func(cls):
    return {
        "type": "function",
        "function": {"name": cls.__name__, "parameters": cls.model_json_schema()},
    }


def tool_choice(fn):
    r = {"type": "function", "function": {"name": fn["function"]["name"]}}
    ic(r)
    return r
