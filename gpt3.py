#!python3

import os
import openai
from icecream import ic
import typer
import sys
from rich import print as rich_print
from rich.console import Console
import rich
import re
import tiktoken
import time
from openai_wrapper import choose_model, setup_gpt, ask_gpt
import pudb
from typing_extensions import Annotated

import signal

console = Console()

# By default, when you hit C-C in a pipe, the pipe is stopped
# with this, pipe continues
def keep_pipe_alive_on_control_c(sig, frame):
    sys.stdout.write(
        "\nInterrupted with Control+C, but I'm still writing to stdout...\n"
    )
    sys.exit(0)


# Register the signal handler for SIGINT
signal.signal(signal.SIGINT, keep_pipe_alive_on_control_c)

original_print = print
is_from_console = False


def bold_console(s):
    if is_from_console:
        return f"[bold]{s}[/bold]"
    else:
        return s


# Load your API key from an environment variable or secret management service


gpt_model = setup_gpt()
app = typer.Typer()

# Todo consider converting to a class
class SimpleNamespace:
    def __init__(self, **kwargs):
        self.__dict__.update(kwargs)


# Shared command line arguments
# https://jacobian.org/til/common-arguments-with-typer/
@app.callback()
def load_options(
    ctx: typer.Context,
    u4: Annotated[bool, typer.Option] = typer.Option(False),
):
    ctx.obj = SimpleNamespace(u4=u4)


def process_shared_app_options(ctx: typer.Context):
    return ctx


# GPT performs poorly with trailing spaces (wow this function was writting by gpt)
def remove_trailing_spaces(str):
    return re.sub(r"\s+$", "", str)


def prep_for_fzf(s):
    # remove starting new lines
    while s.startswith("\n"):
        s = s[1:]
    s = re.sub(r"\n$", "", s)
    s = s.replace("\n", ";")
    return s


@app.command(help="Count the tokens passed via stdin")
def tokens():
    user_text = remove_trailing_spaces("".join(sys.stdin.readlines()))
    tokens = num_tokens_from_string(user_text, "cl100k_base")
    print(tokens)


@app.command()
def stdin(
    ctx: typer.Context,
    tokens: int = typer.Option(0),
    responses: int = typer.Option(1),
    to_fzf: bool = typer.Option(False),
    debug: bool = typer.Option(False),
    prompt: str = typer.Option("*"),
    stream: bool = typer.Option(True),
    u4: bool = typer.Option(False),
):
    process_shared_app_options(ctx)
    user_text = remove_trailing_spaces("".join(sys.stdin.readlines()))
    gpt_start_with = ""
    prompt_to_gpt = prompt.replace("*", user_text)

    base_query_from_dict(locals())


@app.command()
def compress(
    ctx: typer.Context,
    tokens: int = typer.Option(0),
    responses: int = typer.Option(1),
    to_fzf: bool = typer.Option(False),
    debug: bool = typer.Option(False),
    stream: bool = typer.Option(True),
    u4: bool = typer.Option(True),
):
    process_shared_app_options(ctx)
    user_text = remove_trailing_spaces("".join(sys.stdin.readlines()))
    # Find this prompt from the web
    gpt_start_with = ""
    prompt_to_gpt = (
        """Compress the text following the colon in a way that it uses 30 percent less tokens such that you (GPT-4) can reconstruct the intention of the human who wrote text as close as possible to the original intention. This is for yourself. It does not need to be human readable or understandable. Abuse of language mixing, abbreviations, symbols (unicode and emoji), or any other encodings or internal representations is all permissible, as long as it, if pasted in a new inference cycle, will yield near-identical results as the original text (skip links and formatting):

    """
        + user_text
    )

    base_query_from_dict(locals())


@app.command()
def uncompress(
    tokens: int = typer.Option(0),
    responses: int = typer.Option(1),
    to_fzf: bool = typer.Option(False),
    debug: bool = typer.Option(False),
    stream: bool = typer.Option(True),
    u4: bool = typer.Option(True),
):
    user_text = remove_trailing_spaces("".join(sys.stdin.readlines()))
    # Find this prompt from the web
    gpt_start_with = ""
    prompt_to_gpt = (
        """ You (GPT-4) Compressed the  following text in a way that it uses 30 percent less tokens,  reconstruct the original text without emojis for human consumption :

    """
        + user_text
    )

    base_query_from_dict(locals())


@app.command()
def joke(
    ctx: typer.Context,
    tokens: int = typer.Option(0),
    responses: int = typer.Option(1),
    to_fzf: bool = typer.Option(False),
    debug: bool = False,
):
    process_shared_app_options(ctx)
    user_text = remove_trailing_spaces("".join(sys.stdin.readlines()))
    gpt_start_with = ""
    prompt_to_gpt = f"I am a very funny comedian and after I read the following text:\n---\n {user_text}\n---\n After that I wrote these 5 jokes about it:\n 1."

    base_query(tokens, responses, debug, to_fzf, prompt_to_gpt, gpt_start_with)


@app.command()
def group(
    ctx: typer.Context,
    tokens: int = typer.Option(0),
    responses: int = typer.Option(1),
    to_fzf: bool = typer.Option(False),
    debug: bool = typer.Option(False),
    prompt: str = typer.Option("*"),
    stream: bool = typer.Option(True),
    u4: bool = typer.Option(True),
):
    process_shared_app_options(ctx)
    user_text = remove_trailing_spaces("".join(sys.stdin.readlines()))
    prompt_to_gpt = f"""Group the following:
-----
{user_text}

"""
    base_query_from_dict(locals())


def patient_facts():
    return """
* Kiro is a co-worker
* Zach, born in 2010 is son
* Amelia, born in 2014 is daughter
* Tori is wife
* Physical Habits is the same as physical health and exercisies
* Bubbles are a joy activity
* Turkish Getups (TGU) is about physical habits
* Swings refers to Kettle Bell Swings
* Treadmills are about physical health
* 750words is journalling
* I work as an engineering manager (EM) in a tech company
* A refresher is a synonym for going to the gym
"""


@app.command()
def life_group(
    ctx: typer.Context,
    tokens: int = typer.Option(0),
    responses: int = typer.Option(1),
    to_fzf: bool = typer.Option(False),
    debug: bool = typer.Option(False),
    prompt: str = typer.Option("*"),
    stream: bool = typer.Option(True),
    u4: bool = typer.Option(True),
):
    process_shared_app_options(ctx)
    user_text = remove_trailing_spaces("".join(sys.stdin.readlines()))
    prompt_to_gpt = f"""

You will be grouping elements. I'll provide group categories, facts, and then group the items.


* Try not to put items into multiple categories;
* Use markdown for category titles.
* Use facts to help you understand the grouping, not to be included into the output

# Group categories

<!-- prettier-ignore-start -->
<!-- vim-markdown-toc GFM -->

- [Work]
    - [Individual Contributor/Tech]
    - [Manager]
- [Friends]
- [Family]
    - [Tori](#tori)
    - [Zach](#zach)
    - [Amelia](#amelia)
- [Magic]
    - [Performing](#performing)
    - [Practice](#practice)
    - [General Magic](#general-magic)
- [Tech Guru]
    - [Blogging](#blogging)
    - [Programming](#programming)
- [Identity Health]
    - [Biking](#biking)
    - [Ballooning](#ballooning)
    - [Joy Activities](#joy-activities)
- [Motivation]
- [Emotional Health]
    - [Meditation](#meditation)
    - [750 words](#750-words)
    - [Avoid Procrastination](#avoid-procrastination)
- [Physical Health]
    - [Exercise](#exercise)
    - [Diet](#diet)
    - [Sleep](#sleep)
- [Inner Peace]
    - [General](#general)
    - [Work](#work)
    - [Family](#family)
- [Things not mentioned above]

<!-- vim-markdown-toc GFM -->
<!-- prettier-ignore-end -->

# Facts

{patient_facts()}

# Items

{user_text}
"""

    base_query_from_dict(locals())


@app.command()
def tldr(
    ctx: typer.Context,
    tokens: int = typer.Option(0),
    debug: bool = typer.Option(False),
    responses: int = typer.Option(1),
    to_fzf: bool = typer.Option(False),
    u4: bool = typer.Option(False),
):
    user_text = remove_trailing_spaces("".join(sys.stdin.readlines()))
    system_prompt = """You are an advanced AI that can summarize text so it's easy for high school students to understnad
    Your task is to creat a TL;DR from the text below. Use the following format:

#### TL;DR
#### Key Takeaways (point form)
#### Journaling Prompts (point form)
    """
    process_shared_app_options(ctx)
    gpt_start_with = ""
    prompt = f"""{user_text}"""
    prompt_to_gpt = remove_trailing_spaces(prompt)
    base_query_from_dict(locals())


def num_tokens_from_string(string: str, encoding_name: str = "") -> int:
    """Returns the number of tokens in a text string."""
    encoding = tiktoken.get_encoding(encoding_name)
    num_tokens = len(encoding.encode(string))
    return num_tokens


def base_query_from_dict(kwargs):
    a = kwargs
    return base_query(
        tokens=a["tokens"],
        responses=a["responses"],
        debug=a["debug"],
        to_fzf=a["to_fzf"],
        prompt_to_gpt=a["prompt_to_gpt"],
        system_prompt=a.get("system_prompt", "You are a helpful assistant."),
        gpt_response_start=a.get("gpt_response_start", ""),
        stream=a.get("stream", a["responses"] == 1 or not a["to_fzf"]),
        u4=a.get("u4", False),
    )


def query_no_print(
    prompt_to_gpt="Make a rhyme about Dr. Seuss forgetting to pass a default paramater",
    tokens: int = 0,
    u4=True,
    debug=False,
):
    text_model_best, tokens = choose_model(u4)
    messages = [
        {"role": "system", "content": "You are a helpful assistant."},
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
    responses = 1
    response_contents = ["" for x in range(responses)]
    for chunk in openai.ChatCompletion.create(
        model=text_model_best,
        messages=messages,
        max_tokens=output_tokens,
        n=responses,
        temperature=0.7,
        stream=True,
    ):
        if not "choices" in chunk:
            continue

        for elem in chunk["choices"]:  # type: ignore

            delta = elem["delta"]
            delta_content = delta.get("content", "")
            response_contents[elem["index"]] += delta_content
    if debug:
        out = f"All chunks took: {int((time.time() - start)*1000)} ms"
        ic(out)
    return response_contents


def base_query(
    tokens: int = 300,
    responses: int = 1,
    debug: bool = False,
    to_fzf: bool = False,
    prompt_to_gpt="replace_prompt",
    gpt_response_start="gpt_response_start",
    system_prompt="You are a helpful assistant.",
    stream=False,
    u4=False,
):
    text_model_best, tokens = choose_model(u4, tokens)

    # encoding = tiktoken.get_encoding("cl100k_base")
    # encoding = tiktoken.encoding_for_model("gpt-3.5-turbo")

    # Define the messages for the chat
    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": prompt_to_gpt},
    ]

    input_tokens = (
        num_tokens_from_string(prompt_to_gpt, "cl100k_base") + 100
    )  # too lazy to count the messages stuf
    output_tokens = tokens - input_tokens

    if debug:
        ic(system_prompt)
        # ic(prompt_to_gpt)
        ic(text_model_best)
        ic(tokens)
        ic(input_tokens)
        ic(output_tokens)
        ic(stream)

    start = time.time()
    response_contents = ["" for x in range(responses)]
    first_chunk = True
    for chunk in openai.ChatCompletion.create(
        model=text_model_best,
        messages=messages,
        max_tokens=output_tokens,
        n=responses,
        temperature=0.7,
        stream=True,
    ):
        if not "choices" in chunk:
            continue

        for elem in chunk["choices"]:  # type: ignore

            if first_chunk:
                if debug:
                    out = f"First Chunk took: {int((time.time() - start)*1000)} ms"
                    ic(out)
                first_chunk = False
            delta = elem["delta"]
            delta_content = delta.get("content", "")
            response_contents[elem["index"]] += delta_content

            if stream:
                # when streaming output, since it's interleaved, only output the first stream
                sys.stdout.write(delta_content)
                sys.stdout.flush()

    if stream:
        if debug:
            print()
            out = f"All chunks took: {int((time.time() - start)*1000)} ms"
            ic(out)
        return

    for i, content in enumerate(response_contents):
        if to_fzf:
            # ; is newline
            base = f"**{gpt_response_start}**" if len(gpt_response_start) > 0 else ""
            text = f"{base} {prep_for_fzf(content)}"
            print(text)
        else:
            text = ""
            base = gpt_response_start
            if len(gpt_response_start) > 0:
                base += " "
                text = f"{gpt_response_start} {content}"
            else:
                text = content

            if responses > 1:
                print(f"--- **{i}** ---")

            print(text)

    # Add a trailing output to make vim's life easier.
    if responses > 1:
        print(f"--- **{9}** ---")

    if debug:
        out = f"All chunks took: {int((time.time() - start)*1000)} ms"
        ic(out)


@app.command()
def mood(
    ctx: typer.Context,
    tokens: int = typer.Option(0),
    responses: int = typer.Option(1),
    debug: bool = False,
    to_fzf: bool = typer.Option(False),
    u4: bool = typer.Option(True),
):
    process_shared_app_options(ctx)
    text_model_best, tokens = choose_model(u4, tokens)

    user_text = remove_trailing_spaces("".join(sys.stdin.readlines()))
    system_prompt = f""" You are an expert psychologist who writes reports
    after reading patient's journal entries

You task it to write a report based on the passed in journal entry.

The reports include the entry date in the summary,
and a 1-10 scale rating of anxiety, depression, and mania.

Summary:
- Includes entry date
- Provides rating of depression anxiety and mania

Depression Rating:
- Uses a 1-10 scale
- 0 represents mild depression
- 10 represents hypomania
- Provides justification for rating

Anxiety Rating:
- Uses a 1-10 scale
- 10 signifies high anxiety
- Provides justification for rating

Mania Rating:
- Uses a 1-10 scale
- 10 signifies mania
- 5 signifies hypomania
- Provides justification for rating


# Here are some facts to help you assess
{patient_facts()}

# Report
<!-- prettier-ignore-start -->
<!-- vim-markdown-toc GFM -->

- [Summary, assessing mania, depression, and anxiety]
- [Summary of patient experience]
- [Patient Recommendations]
- [Journal prompts to support cognitive reframes, with concrete examples]
- [People and relationships, using names when possible]

<!-- prettier-ignore-end -->
<!-- vim-markdown-toc GFM -->
"""

    prompt_to_gpt = user_text
    base_query_from_dict(locals())


@app.command()
def anygram(
    ctx: typer.Context,
    tokens: int = typer.Option(0),
    responses: int = typer.Option(1),
    debug: bool = False,
    to_fzf: bool = typer.Option(False),
):
    process_shared_app_options(ctx)
    user_text = remove_trailing_spaces("".join(sys.stdin.readlines()))
    gpt_start_with = "\nFrom this I conclude the authors type and the following 5 points about how the patient feels, and 3 recommendations to the author\n"
    prompt_to_gpt = f"I am a enniagram expert and read the following journal entry:\n {user_text}\n {gpt_start_with} "
    base_query(tokens, responses, debug, to_fzf, prompt_to_gpt, gpt_start_with)


@app.command()
def summary(
    ctx: typer.Context,
    tokens: int = typer.Option(3),
    responses: int = typer.Option(1),
    debug: bool = False,
    to_fzf: bool = typer.Option(False),
):
    process_shared_app_options(ctx)
    user_text = remove_trailing_spaces("".join(sys.stdin.readlines()))
    gpt_start_with = ""
    prompt_to_gpt = f"Create an interesting opening paragraph for the following text, if it's possible, make the last sentance a joke or include an alliteration: \n\n {user_text}\n {gpt_start_with} "
    base_query(tokens, responses, debug, to_fzf, prompt_to_gpt, gpt_start_with)


@app.command()
def commit_message(
    ctx: typer.Context,
    tokens: int = typer.Option(0),
    responses: int = typer.Option(1),
    to_fzf: bool = typer.Option(False),
    debug: bool = typer.Option(False),
    stream: bool = typer.Option(False),
    u4: bool = typer.Option(True),
):
    process_shared_app_options(ctx)
    text_model_best, tokens = choose_model(u4)
    user_text = remove_trailing_spaces("".join(sys.stdin.readlines()))
    system_prompt = """
    You are an AI trained to write effective and concise Git commit messages.
    You have been presented with a diff from a recent system change.
    Your task is  to write a meaningful commit message .
    Start with a 1 line summary,  and then have the body of the commit message be point form for all the changes
    If it seems like there are multiple disjoint commits, then you can write multiple commit messages

    """
    prompt_to_gpt = f"""{user_text}"""
    base_query_from_dict(locals())


@app.command()
def headline(
    ctx: typer.Context,
    tokens: int = typer.Option(0),
    responses: int = typer.Option(1),
    debug: bool = False,
    to_fzf: bool = typer.Option(False),
):
    process_shared_app_options(ctx)
    user_text = remove_trailing_spaces("".join(sys.stdin.readlines()))
    gpt_start_with = ""
    prompt_to_gpt = f"Create an attention grabbing headline and then a reason why you should care of the following post:\n {user_text}\n {gpt_start_with} "
    base_query(tokens, responses, debug, to_fzf, prompt_to_gpt, gpt_start_with)


@app.command()
def protagonist(
    ctx: typer.Context,
    tokens: int = typer.Option(0),
    responses: int = typer.Option(1),
    debug: bool = False,
    to_fzf: bool = typer.Option(False),
):
    process_shared_app_options(ctx)
    user_text = remove_trailing_spaces("".join(sys.stdin.readlines()))
    gpt_start_with = "The protagonist"
    prompt_to_gpt = f"Summarize the following text:\n {user_text}\n {gpt_start_with} "
    base_query(tokens, responses, debug, to_fzf, prompt_to_gpt, gpt_start_with)


@app.command()
def poem(
    ctx: typer.Context,
    tokens: int = typer.Option(0),
    responses: int = typer.Option(1),
    debug: bool = False,
    to_fzf: bool = typer.Option(False),
):
    process_shared_app_options(ctx)
    user_text = remove_trailing_spaces("".join(sys.stdin.readlines()))
    gpt_start_with = ""
    prompt_to_gpt = f"Rewrite the following text in the form as a rhyming couplet poem by Dr. Seuss with at least 5 couplets:\n {user_text}\n {gpt_start_with} "
    base_query(tokens, responses, debug, to_fzf, prompt_to_gpt, gpt_start_with)


@app.command()
def study(
    ctx: typer.Context,
    points: int = typer.Option(5),
    tokens: int = typer.Option(0),
    debug: bool = False,
    responses: int = typer.Option(1),
    to_fzf: bool = typer.Option(False),
):
    process_shared_app_options(ctx)
    user_text = remove_trailing_spaces("".join(sys.stdin.readlines()))
    gpt_start_with = ""
    prompt_to_gpt = (
        f"""What are {points}  key points I should know when studying {user_text}?"""
    )
    gpt_start_with = ""
    base_query(tokens, responses, debug, to_fzf, prompt_to_gpt, gpt_start_with)


@app.command()
def eli5(
    ctx: typer.Context,
    tokens: int = typer.Option(0),
    debug: bool = False,
    responses: int = typer.Option(1),
    to_fzf: bool = typer.Option(False),
):
    process_shared_app_options(ctx)
    user_text = remove_trailing_spaces("".join(sys.stdin.readlines()))
    gpt_start_with = ""
    prompt = f"""Summarize this for a second-grade sudent: {user_text}"""
    prompt_to_gpt = remove_trailing_spaces(prompt)
    base_query(tokens, responses, debug, to_fzf, prompt_to_gpt, gpt_start_with)


@app.command()
def book(
    ctx: typer.Context,
    tokens: int = typer.Option(0),
    responses: int = typer.Option(1),
    to_fzf: bool = typer.Option(False),
    debug: bool = typer.Option(False),
    stream: bool = typer.Option(True),
    u4: bool = typer.Option(False),
):
    process_shared_app_options(ctx)
    text_model_best, tokens = choose_model(u4, tokens)

    user_text = remove_trailing_spaces("".join(sys.stdin.readlines()))
    gpt_start_with = ""
    prompt = f"""

    Write a book on the following topic: {user_text}.

    Write it in the style of the heath brothers, with an acronym, and chapter for each letter

    Use markdown in writing the books. And have chapter titles be an h2 in markdown.
    target {tokens} tokens for your response.

    Before the book starts write a paragraph summarzing the key take aways from the book
    Then write a detailed outline

    Then as you write each chapter, focus on
    The top 5 theories, with a few sentances about them, and how they are relevant.
    The top 5 take aways, with a few setances about them
    Then include
    5 exercise to try yourself
    5 journalling prompts to self reflect on how you're doing
    End with a conclusion

    """
    prompt_to_gpt = remove_trailing_spaces(prompt)
    base_query_from_dict(locals())


def get_embedding(text, model="text-embedding-ada-002"):
    text = text.replace("\n", " ")
    embedding_response = openai.Embedding.create(input=[text], model=model)
    return embedding_response["data"][0]["embedding"]  # type: ignore


@app.command()
def embed():
    text = "".join(sys.stdin.readlines())
    z = get_embedding(text)
    print(z)


def split_string(input_string):
    # TODO update the to use the tokenizer
    chunk_size = 1000 * 5 * 4
    for i in range(0, len(input_string), chunk_size):
        yield input_string[i : i + chunk_size]


@app.command()
def captions(
    ctx: typer.Context,
    tokens: int = typer.Option(0),
    debug: bool = typer.Option(1),
    responses: int = typer.Option(1),
    to_fzf: bool = typer.Option(False),
    u4: bool = typer.Option(False),
):
    """
    Convert captions files into readable text. You can get them via
    youtube-dl --write-sub --sub-format srt --skip-download
    """

    original_user_text = remove_trailing_spaces("".join(sys.stdin.readlines()))
    system_prompt = """
You  are a super smart AI, who understands captions formats, and also english grammar and spelling

You will be given captions as input, you should output the text that can be read by a human like a book

Be sure to include punctuation, correct errors and include paragraphs when they make sense.

Even if the person is talking continuously, still make sentances and paragraphs

A sentence should not be more then 30 words, and a paragraph should not be more then 10 sentences.


    """

    # Hymn lets do some chunking here.
    # Weird doing chunks here.
    # TODO: Put this into a yield  function
    for user_text in split_string(original_user_text):
        gpt_start_with = ""
        prompt = f"""{user_text}"""
        ic(len(prompt))
        prompt_to_gpt = remove_trailing_spaces(prompt)
        base_query_from_dict(locals())
        print("----")


@app.command()
def fix(
    ctx: typer.Context,
    tokens: int = typer.Option(0),
    debug: bool = typer.Option(1),
    responses: int = typer.Option(1),
    to_fzf: bool = typer.Option(False),
    u4: bool = typer.Option(False),
):
    process_shared_app_options(ctx)
    user_text = remove_trailing_spaces("".join(sys.stdin.readlines()))
    system_prompt = """You are an advanced AI with superior spelling correction abilities.
Your task is to correct any spelling errors you encounter in the text provided below.
The text is markdown, don't change the markdown formatting.
    """
    gpt_start_with = ""
    prompt = f"""{user_text}"""
    prompt_to_gpt = remove_trailing_spaces(prompt)
    base_query_from_dict(locals())


@app.command()
def debug():
    ic(print)
    ic(rich_print)
    ic(original_print)
    c = rich.get_console()
    ic(c.width)
    ic(is_from_console)
    print(
        "long line -brb:w aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa"
    )


@app.command()
def paginate(
    file_path: str,
    u4: bool = typer.Option(True),
    debug: bool = typer.Option(1),
):
    file_path = os.path.expanduser(file_path)
    text = open(file_path).read()
    paginate_internal(text, u4, debug)


def paginate_internal(
    text: str,
    u4: bool,
    debug: bool,
):

    transcript = text
    prompt = f""" You are a superb editor.

    "Given an inline text file of a speech that's had all newline characters removed, following the prompt, please identify where the new paragraphs should begin and split the text into paragraphs accordingly. The text should be intelligently divided.  Please ensure that the output text is at least as long as the input text, meaning that no part of the input text should be omitted in the output. Please output the newly formatted text with the appropriate paragraphs."

Inline File:

{transcript}

"""
    # TOOD: If the paginated transcript is shoreter, probably a bug

    paginated_transcript = ask_gpt(prompt, u4=u4, debug=debug)
    print(paginated_transcript)
    fudge_factor = 100
    is_cleanup_fishy = len(paginated_transcript) + fudge_factor < len(transcript)
    if is_cleanup_fishy:
        print("## ++ **Fishy cleanup, showing original**")
        print(f"Paginated: {len(paginated_transcript)}, Original: {len(transcript)}")
        print(transcript)
        print("## -- **Fishy cleanup, showing original*")
        return


@app.command()
def transcribe(
    ctx: typer.Context,
    file_path: str,
    cleanup: bool = typer.Option(True),
    split_input: bool = typer.Option(False),
    u4: bool = typer.Option(True),
    debug: bool = typer.Option(1),
):

    """
    Transcribe an audio file using openai's API

    If your file is too big, split it up w/ffmpeg then run the for loop

    ffmpeg -i input.mp3 -f segment -segment_time 600 -c copy output%03d.mp3

    for file in $(ls output*.mp3); do gpt transcribe $file & > $file.txt; done

    """
    if split_input:
        raise NotImplementedError("clean input and split output not implemented")

    file_path = os.path.expanduser(file_path)
    transcript = "None"
    with open(file_path, "rb") as audio_file:
        transcript = openai.Audio.transcribe(
            file=audio_file, model="whisper-1", response_format="text", language="en"
        )

    if not cleanup:
        print(transcript)
        return

    ic(transcript)
    paginate_internal(str(transcript), u4, debug)


def configure_width_for_rich():
    global is_from_console
    # need to think more, as CLI vs vim will be different
    c = rich.get_console()
    is_from_console = c.width != 80
    if is_from_console:
        print = rich_print  # NOQA
    else:
        print = original_print  # NOQA


if __name__ == "__main__":
    configure_width_for_rich()
    app()
