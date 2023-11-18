import os
import json
from openai import OpenAI

client = OpenAI(api_key=PASSWORD)
import tiktoken
from icecream import ic
import time
from tenacity import (
    retry,
    stop_after_attempt,
    wait_random_exponential,
    retry_if_exception_type,
)

from pydantic import BaseModel


class CompletionModel(BaseModel):
    max_input_only_tokens:int
    max_output_tokens:int
    name:str


gpt4 = CompletionModel(max_input_only_tokens=100*1000, max_output_tokens=4*1000, name="gpt-4-1106-preview")
gpt35 = CompletionModel(max_input_only_tokens=12*1000, max_output_tokens=4*1000, name="gpt-3.5-turbo-1106")

def get_model_type(u4:bool)->CompletionModel:
    if u4:
        return gpt4
    else:
        return gpt35

text_model_gpt_4 = "gpt-4-1106-preview"
gpt_4_tokens = 100000
gpt_4_input_tokens = 100*1000
gpt_4_output_tokens = 100*1000
text_model_gpt35 = "gpt-3.5-turbo-1106"
gpt_3_5_tokens = 16000
code_model_best = "code-davinci-003"


def model_to_max_tokens(model):
    model_to_tokens = {text_model_gpt_4: gpt_4_tokens, text_model_gpt35: gpt_3_5_tokens}
    return model_to_tokens[model]


def get_model(u4):
    model =""
    if u4:
        model = text_model_gpt_4
    else:
        model = text_model_gpt35
    return model

def get_remaining_output_tokens(model:CompletionModel,prompt:str):

    # For symetric models, max_input_only_tokens= 0 and max_output_tokens  = the full context window
    # For asymmetrics models, max_output_tokens = full context_window - max_input_only_tokens

    input_tokens = num_tokens_from_string(prompt, "cl100k_base")
    # If you only used input_context only tokens, don't remove anything f+ 100
    output_tokens_consumed =  max((input_tokens - model.max_input_only_tokens),0)
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


def setup_gpt():
    PASSWORD = "replaced_from_secret_box"
    with open(os.path.expanduser("~/gits/igor2/secretBox.json")) as json_data:
        SECRETS = json.load(json_data)
        PASSWORD = SECRETS["openai"]
    
    return openai


def num_tokens_from_string(string: str, encoding_name: str = "") -> int:
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
    retry=(
    ),
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

    input_tokens = num_tokens_from_string(prompt_to_gpt, "cl100k_base") + 100
    output_tokens = tokens - input_tokens

    if debug:
        ic(text_model_best)
        ic(tokens)
        ic(input_tokens)
        ic(output_tokens)

    start = time.time()
    responses = n
    response_contents = ["" for x in range(responses)]
    for chunk in client.chat.completions.create(model=text_model_best,
    messages=messages,
    max_tokens=output_tokens,
    n=responses,
    temperature=0.7,
    stream=True):
        if not "choices" in chunk:
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
