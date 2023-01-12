#!python3

import os
import openai
import json
from icecream import ic
import typer
import sys
from rich import print as rich_print
import rich
import re

original_print = print
is_from_console = False

text_model_best = "text-davinci-003"
code_model_best = "code-davinci-003"


def bold_console(s):
    if is_from_console:
        return f"[bold]{s}[/bold]"
    else:
        return s


# Load your API key from an environment variable or secret management service


def setup_gpt():
    PASSWORD = "replaced_from_secret_box"
    with open(os.path.expanduser("~/gits/igor2/secretBox.json")) as json_data:
        SECRETS = json.load(json_data)
        PASSWORD = SECRETS["openai"]
    openai.api_key = PASSWORD
    return openai


gpt3 = setup_gpt()
app = typer.Typer()


# GPT performs poorly with trailing spaces (wow this function was writting by gpt)
def remove_trailing_spaces(str):
    return re.sub(r"\s+$", "", str)


@app.command()
def py(tokens: int = typer.Option(50)):
    prompt = "\n".join(sys.stdin.readlines())
    response = gpt3.Completion.create(
        engine=code_model_best,
        temperature=0.5,
        prompt=remove_trailing_spaces(prompt),
        max_tokens=tokens,
    )
    response_text = response.choices[0].text
    print(f"{prompt}\n{response_text}")


def prep_for_fzf(s):
    # remove starting new lines
    while s.startswith("\n"):
        s = s[1:]
    s = re.sub(r"\n$", "", s)
    s = s.replace("\n", ";")
    return s


@app.command()
def stdin(
    tokens: int = typer.Option(900),
    responses: int = typer.Option(1),
    to_fzf: bool = typer.Option(False),
    debug: bool = False,
    prompt: str = typer.Option("*"),
):
    user_text = remove_trailing_spaces("".join(sys.stdin.readlines()))
    gpt_start_with = ""
    prompt_to_gpt = prompt.replace("*", user_text)

    base_query(tokens, responses, debug, to_fzf, prompt_to_gpt, gpt_start_with)


@app.command()
def joke(
    tokens: int = typer.Option(400),
    responses: int = typer.Option(1),
    to_fzf: bool = typer.Option(False),
    debug: bool = False,
):
    user_text = remove_trailing_spaces("".join(sys.stdin.readlines()))
    gpt_start_with = ""
    prompt_to_gpt = f"I am a very funny comedian and after I read the following text:\n---\n {user_text}\n---\n After that I wrote these 5 jokes about it:\n 1."

    base_query(tokens, responses, debug, to_fzf, prompt_to_gpt, gpt_start_with)


@app.command()
def group(
    tokens: int = typer.Option(400),
    responses: int = typer.Option(1),
    to_fzf: bool = typer.Option(False),
    debug: bool = False,
):
    user_text = remove_trailing_spaces("".join(sys.stdin.readlines()))
    gpt_start_with = "1."
    prompt_to_gpt = f"Group the following:\n---\n {user_text}\n---"

    base_query(tokens, responses, debug, to_fzf, prompt_to_gpt, gpt_start_with)


@app.command()
def tldr(
    tokens: int = typer.Option(300),
    responses: int = typer.Option(1),
    debug: bool = False,
    to_fzf: bool = typer.Option(False),
):
    prompt = "".join(sys.stdin.readlines())
    prompt_to_gpt = remove_trailing_spaces(prompt) + "\ntl;dr:"
    response = openai.Completion.create(
        engine=text_model_best,
        n=responses,
        prompt=prompt_to_gpt,
        max_tokens=tokens,
    )
    if debug:
        ic(prompt_to_gpt)
        print(prompt_to_gpt)

    for c in response.choices:
        if to_fzf:
            # ; is newline
            text = ";**tl,dr:* " + prep_for_fzf(c.text)
        else:
            text = f"\n**tl,dr:** {c.text}"
        print(text)


def base_query(
    tokens: int = typer.Option(300),
    responses: int = typer.Option(1),
    debug: bool = False,
    to_fzf: bool = typer.Option(False),
    prompt_to_gpt="replace_prompt",
    gpt_response_start="gpt_response_start",
):
    response = openai.Completion.create(
        engine=text_model_best,
        n=responses,
        prompt=prompt_to_gpt,
        max_tokens=tokens,
        temperature=1,
    )
    if debug:
        ic(prompt_to_gpt)
        print(prompt_to_gpt)

    for c in response.choices:
        if to_fzf:
            # ; is newline
            base = f"**{gpt_response_start}**" if len(gpt_response_start) > 0 else ""
            text = f"{base} {prep_for_fzf(c.text)}"
            print(text)
        else:
            base = gpt_response_start
            if len(gpt_response_start) > 0:
                base += " "
            text = f"{gpt_response_start} {c.text}"
            print(text)
            if len(response.choices) > 1:
                print("----")


@app.command()
def mood(
    tokens: int = typer.Option(900),
    responses: int = typer.Option(1),
    debug: bool = False,
    to_fzf: bool = typer.Option(False),
):
    user_text = remove_trailing_spaces("".join(sys.stdin.readlines()))
    gpt_start_with = """"""
    prompt_to_gpt = f""" I am a psychologist who writes reports after reading patient's journal entries

The reports include the entry date in the summary, and a 1-10 scale rating of depression to mania where 0 is depressed and 10 is hypo manic with my reasoning for the rating and
my assessment of their level of anxiety and my reasoning for that rating. After the summary the remaining sections are in point form only

# Journal entry
        {user_text}

# Report
<!-- prettier-ignore-start -->
<!-- vim-markdown-toc GFM -->

- [Summary](#summary)
- [Patient Recommendations - 5 point form](#patient-recommendations---5-point-form)
- [Recommended Cognitive Reframes - 5 point form](#recommended-cognitive-reframes---5-point-form)
- [Relationships and People mentioned - 10 point form](#relationships-and-people-mentioned---10-point-form)
- [Patient Experience - 10 point form](#patient-experience---10-point-form)

<!-- prettier-ignore-end -->
<!-- vim-markdown-toc GFM -->
"""
    base_query(tokens, responses, debug, to_fzf, prompt_to_gpt, gpt_start_with)


@app.command()
def anygram(
    tokens: int = typer.Option(300),
    responses: int = typer.Option(1),
    debug: bool = False,
    to_fzf: bool = typer.Option(False),
):
    user_text = remove_trailing_spaces("".join(sys.stdin.readlines()))
    gpt_start_with = "\nFrom this I conclude the authors type and the following 5 points about how the patient feels, and 3 recommendations to the author\n"
    prompt_to_gpt = f"I am a enniagram expert and read the following journal entry:\n {user_text}\n {gpt_start_with} "
    base_query(tokens, responses, debug, to_fzf, prompt_to_gpt, gpt_start_with)


@app.command()
def summary(
    tokens: int = typer.Option(300),
    responses: int = typer.Option(1),
    debug: bool = False,
    to_fzf: bool = typer.Option(False),
):
    user_text = remove_trailing_spaces("".join(sys.stdin.readlines()))
    gpt_start_with = ""
    prompt_to_gpt = f"Create an interesting opening paragraph for the following text, if it's possible, make the last sentance a joke or include an alliteration: \n\n {user_text}\n {gpt_start_with} "
    base_query(tokens, responses, debug, to_fzf, prompt_to_gpt, gpt_start_with)


@app.command()
def commit_message(
    tokens: int = typer.Option(300),
    responses: int = typer.Option(1),
    debug: bool = False,
    to_fzf: bool = typer.Option(False),
):
    user_text = remove_trailing_spaces("".join(sys.stdin.readlines()))
    gpt_start_with = ""
    prompt_to_gpt = f"Write a commit message for the following diff, with a headline and then a paragraph of more details:\n {user_text}\n {gpt_start_with} "
    base_query(tokens, responses, debug, to_fzf, prompt_to_gpt, gpt_start_with)


@app.command()
def headline(
    tokens: int = typer.Option(300),
    responses: int = typer.Option(1),
    debug: bool = False,
    to_fzf: bool = typer.Option(False),
):
    user_text = remove_trailing_spaces("".join(sys.stdin.readlines()))
    gpt_start_with = ""
    prompt_to_gpt = f"Create an attention grabbing headline and then a reason why you should care of the following post:\n {user_text}\n {gpt_start_with} "
    base_query(tokens, responses, debug, to_fzf, prompt_to_gpt, gpt_start_with)


@app.command()
def protagonist(
    tokens: int = typer.Option(300),
    responses: int = typer.Option(1),
    debug: bool = False,
    to_fzf: bool = typer.Option(False),
):
    user_text = remove_trailing_spaces("".join(sys.stdin.readlines()))
    gpt_start_with = "The protagonist"
    prompt_to_gpt = f"Summarize the following text:\n {user_text}\n {gpt_start_with} "
    base_query(tokens, responses, debug, to_fzf, prompt_to_gpt, gpt_start_with)


@app.command()
def poem(
    tokens: int = typer.Option(600),
    responses: int = typer.Option(1),
    debug: bool = False,
    to_fzf: bool = typer.Option(False),
):
    user_text = remove_trailing_spaces("".join(sys.stdin.readlines()))
    gpt_start_with = ""
    prompt_to_gpt = f"Rewrite the following text in the form as a rhyming couplet poem by Dr. Seuss with at least 5 couplets:\n {user_text}\n {gpt_start_with} "
    base_query(tokens, responses, debug, to_fzf, prompt_to_gpt, gpt_start_with)


@app.command()
def answer(tokens: int = typer.Option(50), responses: int = typer.Option(4)):
    prompt = "".join(sys.stdin.readlines())
    # clean input
    is_markdown = prompt.startswith("**")
    prompt = prompt.removeprefix("Q:")
    prompt = prompt.removeprefix("**Q:**")
    prompt = prompt.strip()
    prompt_in = """I am a highly intelligent question answering bot. If you ask me a question that is rooted in truth, I will give you the answer. If you ask me a question that is nonsense, trickery, or has no clear answer, I will respond with "Unknown".
Q: What is human life expectancy in the United States?
A: Human life expectancy in the United States is 78 years.

Q: Who was president of the United States in 1955?
A: Dwight D. Eisenhower was president of the United States in 1955.

Q: Which party did he belong to?
A: He belonged to the Republican Party.

Q: What is the square root of banana?
A: Unknown

Q: How does a telescope work?
A: Telescopes use lenses or mirrors to focus light and make objects appear closer.

Q: Where were the 1992 Olympics held?
A: The 1992 Olympics were held in Barcelona, Spain.

Q: How many squigs are in a bonk?
A: Unknown

Q:
    """
    response = openai.Completion.create(
        temperature=0.3,
        engine=text_model_best,
        n=responses,
        prompt=remove_trailing_spaces(prompt_in),
        max_tokens=tokens,
        stop=["\n\n"],
    )
    if is_markdown:
        print(f"**Q:** {prompt}")
    else:
        print(f"Q: {prompt}")

    for c in response.choices:
        print(f"**A:**{c.text}")


@app.command()
def study(
    points: int = typer.Option(5),
    tokens: int = typer.Option(200),
    debug: bool = False,
    responses: int = typer.Option(1),
    to_fzf: bool = typer.Option(False),
):
    user_text = remove_trailing_spaces("".join(sys.stdin.readlines()))
    gpt_start_with = ""
    prompt_to_gpt = (
        f"""What are {points}  key points I should know when studying {user_text}?"""
    )
    gpt_start_with = ""
    base_query(tokens, responses, debug, to_fzf, prompt_to_gpt, gpt_start_with)


@app.command()
def eli5(
    tokens: int = typer.Option(200),
    debug: bool = False,
    responses: int = typer.Option(1),
    to_fzf: bool = typer.Option(False),
):
    user_text = remove_trailing_spaces("".join(sys.stdin.readlines()))
    gpt_start_with = ""
    prompt = f"""Summarize this for a second-grade sudent: {user_text}"""
    prompt_to_gpt = remove_trailing_spaces(prompt)
    base_query(tokens, responses, debug, to_fzf, prompt_to_gpt, gpt_start_with)


@app.command()
def complex(tokens: int = typer.Option(200), debug: bool = False):
    prompt_input = "".join(sys.stdin.readlines())
    prompt = f"""The human asks the AI to make a simple description more complicated. The AI responds by writing a verbose response full of jargon to explain a simple concept.

Human: explain how to use a drinking fountain.

AI: To activate the mechanized dihydrogen monoxide dispensary service station, use the flanges to provide a downward force to the simple circular activation console and continue to apply force until the pneumatic system begins operating as intended, which can be surmised at a time when the dihydrogen monoxide travels in a parabolic arc which at is greatest exceeds three inches from the rounded excess liquid receptacle. Once hydration is completed, cease applying force to the console and regard whether or not the dispensary has ceased expelling contents as expected.

Human: explain how to play with a dog

AI: Being a typical domesticated animal, having been genetically selected for controllability, trainability, and general anthropomorphic features and functions, the typical dog possesses highly specialized and developed feeding behavior. The typical feeding behavior involves routine sustenance intake which can be observed by monitoring changes in animal size. Routine sustenance intake involves active and passive feeding habits, and one can gauge animal health by observing the relative levels of hydration. During training, reward conditioned reinforcement is applied. This involves the practice of rewarding targeted behaviors through the application of reinforcers, with the provision that the targeted behavior is observed. Relevant to the discussion at hand is the typical active and passive behavior exhibited by a typical dog.

Human: {prompt_input}
AI:"""
    response = gpt3.Completion.create(
        engine=text_model_best,
        temperature=0.7,
        prompt=remove_trailing_spaces(prompt),
        max_tokens=tokens,
        top_p=1,
        frequency_penalty=0.2,
        presence_penalty=0,
        stop=["\n\n\n"],
    )
    response_text = response.choices[0].text
    if debug:
        print(prompt)
    print(response_text)


def do_complete(prompt, max_tokens):
    response = openai.Completion.create(
        engine=text_model_best,
        prompt=remove_trailing_spaces(prompt),
        max_tokens=max_tokens,
    )

    # ic(response)
    # ic(response.choices[0].text)
    return response.choices[0].text


@app.command()
def complete(prompt: str, tokens: int = typer.Option(50)):
    response_text = do_complete(prompt, tokens)
    print(f"[bold]{prompt}[/bold] {response_text}")


def get_embedding(text, model="text-embedding-ada-002"):
    text = text.replace("\n", " ")
    return openai.Embedding.create(input=[text], model=model)["data"][0]["embedding"]


@app.command()
def embed():
    text = "".join(sys.stdin.readlines())
    z = get_embedding(text)
    print(z)


@app.command()
def fix(
    tokens: int = typer.Option(300),
    responses: int = typer.Option(1),
    debug: bool = False,
    to_fzf: bool = typer.Option(False),
):
    prompt_input = "".join(sys.stdin.readlines())
    response = openai.Edit.create(
        engine="text-davinci-edit-001",
        input=prompt_input,
        instruction="Fix all the spelling and grammer mistakes in this text ",
        temperature=0,
        top_p=1,
    )
    text = response.choices[0].text

    # remove a trailing newline
    if text[-1] == "\n":
        text = text[:-1]

    if to_fzf:
        # ; is newline
        print(prep_for_fzf(text))
    else:
        print(text)


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
