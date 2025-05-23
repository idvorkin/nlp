import os
import time

from icecream import ic
from pydantic import BaseModel
from tenacity import retry, stop_after_attempt, wait_random_exponential
from pathlib import Path

# Global variable to hold the client, instantiated lazily
_client = None


def setup_secret():
    # Only check if OPENAI_API_KEY is already set in environment
    # No file loading - secrets should be in environment variables
    pass


def setup_gpt():
    from openai import OpenAI

    setup_secret()

    # Check if we have an API key in environment
    api_key = os.environ.get("OPENAI_API_KEY")
    if not api_key:
        # Return None if no API key is available
        # Functions that need the client will need to handle this case
        return None

    return OpenAI(api_key=api_key)


def get_client():
    """Get the OpenAI client, creating it lazily if needed."""
    global _client
    if _client is None:
        _client = setup_gpt()
    return _client


class CompletionModel(BaseModel):
    max_input_only_tokens: int
    max_output_tokens: int
    name: str


gpt4 = CompletionModel(
    max_input_only_tokens=100 * 1000,
    max_output_tokens=16 * 1000,
    name="gpt-4.1-2025-04-14",
)
gpt35 = CompletionModel(
    max_input_only_tokens=12 * 1000,
    max_output_tokens=4 * 1000,
    name="gpt-3.5-turbo-0125",
)


def get_model_type(u4: bool) -> CompletionModel:
    if u4:
        return gpt4
    else:
        return gpt35


text_model_gpt_4 = "gpt-4.1-2025-04-14"
gpt_4_tokens = 100000
gpt_4_input_tokens = 100 * 1000
gpt_4_output_tokens = 100 * 1000
text_model_gpt35 = "gpt-3.5-turbo-1106"
gpt_3_5_tokens = 16000


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
    import tiktoken

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
    client = get_client()
    if client is None:
        raise ValueError(
            "OpenAI API key not found in environment variables. Please set OPENAI_API_KEY."
        )

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
        out = f"All chunks took: {int((time.time() - start) * 1000)} ms"
        ic(out)

    # hard code to only return first response
    return response_contents


def get_ell_model(
    openai: bool = False,
    openai_cheap: bool = False,
    google: bool = False,
    claude: bool = False,
    llama: bool = False,
) -> str:
    """
    See changes in diff
    """
    # if more then one is true, exit and fail
    count_true = sum([openai, google, claude, llama, openai_cheap])
    if count_true > 1:
        print("Only one model can be selected")
        exit(1)
    if count_true == 0:
        # default to openai
        openai = True

    if google:
        raise NotImplementedError("google")
    elif claude:
        return "claude-3-7-sonnet-20250219"
    elif llama:
        return "llama-3.2-90b-vision-preview"
    elif openai_cheap:
        return "gpt-4o-mini"
    else:
        return gpt4.name


def openai_func(cls):
    return {
        "type": "function",
        "function": {"name": cls.__name__, "parameters": cls.model_json_schema()},
    }


def tool_choice(fn):
    r = {"type": "function", "function": {"name": fn["function"]["name"]}}
    ic(r)
    return r


def get_youtube_transcript(url):
    from yt_dlp import YoutubeDL

    ic("Downloading captions from youtube")

    # random tmp file name for transcript
    transcript_file_base = f"transcript_{int(time.time())}"
    transcript_file_name = f"{transcript_file_base}.en.vtt"
    transcript_file_template = f"{transcript_file_base}.%(ext)s"

    ydl_opts = {
        "skip_download": True,  # We only want to download the transcript, not the video
        "writesubtitles": True,  # Download subtitles
        "subtitleslangs": [
            "en"
        ],  # Specify the language of the subtitles, e.g., 'en' for English
        "subtitlesformat": "vtt",  # Format of the subtitles
        "writeautomaticsub": True,
        "outtmpl": transcript_file_template,  # Output file name
        "quiet": True,  # Suppress output to stdout
        "no_warnings": True,  # Suppress warnings
    }

    with YoutubeDL(ydl_opts) as ydl:
        info = ydl.extract_info(
            url, download=False
        )  # Extract video info without downloading the video
        subtitles = info.get("subtitles", {})
        automatic_captions = info.get("automatic_captions", {})

        if "en" in subtitles or "en" in automatic_captions:
            ydl.download([url])  # Download the transcript
        else:
            raise ValueError("No English subtitles found.")

        subtitle_file = transcript_file_name
        with open(subtitle_file, "r", encoding="utf-8") as f:
            transcript = f.read()

        # erase the transcript file after reading
        Path(transcript_file_name).unlink(missing_ok=True)

        # # Clean up the transcript
        # import re
        # transcript = re.sub(r'WEBVTT\n\n', '', transcript)
        # transcript = re.sub(r'\d{2}:\d{2}:\d{2}\.\d{3} --> \d{2}:\d{2}:\d{2}\.\d{3}\n', '', transcript)
        # transcript = re.sub(r'\n\n', ' ', transcript)

        return transcript.strip()


def get_text_from_path_or_stdin(path):
    import sys
    from pathlib import Path

    if not path:  # read from stdin
        return "".join(sys.stdin.readlines())
    if (
        path.startswith("https://youtu.be")
        or path.startswith("https://youtube.com")
        or path.startswith("https://www.youtube.com")
    ):
        return get_youtube_transcript(path)

    if path.startswith("http"):
        import requests
        import html2text

        headers = {
            "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/58.0.3029.110 Safari/537.3"
        }
        request = requests.get(path, headers=headers)
        return html2text.html2text(request.text)
    if path:
        return Path(path).read_text()
    return str(sys.stdin.readlines())
