#!python3


import json
import os
import pickle
import sys
from pathlib import Path
from typing import Any, Callable, List

import pandas as pd
import pudb
import typer
from icecream import ic
from langchain.agents import AgentType, initialize_agent, load_tools
from langchain.chat_loaders.imessage import IMessageChatLoader
from langchain.chat_models import ChatOpenAI
from langchain.llms import OpenAI
from langchain.output_parsers.openai_functions import (
    JsonKeyOutputFunctionsParser,
    OutputFunctionsParser,
)
from langchain.prompts import ChatPromptTemplate
from langchain.schema import (
    Generation,
    HumanMessage,
    OutputParserException,
    SystemMessage,
)
from loguru import logger
from pydantic import BaseModel
from rich import print
from rich.console import Console
from typing_extensions import Annotated

import openai_wrapper

console = Console()
app = typer.Typer()


class JsonOutputFunctionsParser2(OutputFunctionsParser):
    """Parse an output as the Json object."""

    def parse_result(self, result: List[Generation]) -> Any:
        function_call_info = super().parse_result(result)
        if self.args_only:
            try:
                # Waiting for this to merge upstream
                return json.loads(function_call_info, strict=False)
            except (json.JSONDecodeError, TypeError) as exc:
                raise OutputParserException(
                    f"Could not parse function call data: {exc}"
                )
        function_call_info["arguments"] = json.loads(function_call_info["arguments"])
        return function_call_info


# Todo consider converting to a class
class SimpleNamespace:
    def __init__(self, **kwargs):
        self.__dict__.update(kwargs)


# Shared command line arguments
# https://jacobian.org/til/common-arguments-with-typer/
@app.callback()
def load_options(
    ctx: typer.Context,
    attach: Annotated[bool, typer.Option(help="Attach to existing process")] = False,
):
    ctx.obj = SimpleNamespace(attach=attach)


def process_shared_app_options(ctx: typer.Context):
    if ctx.obj.attach:
        pudb.set_trace()


llm = OpenAI(temperature=0.9)
chat = ChatOpenAI(temperature=0)
chat_model = chat


@logger.catch()
def app_wrap_loguru():
    app()


# Google search setup
# https://github.com/hwchase17/langchain/blob/d0c7f7c317ee595a421b19aa6d94672c96d7f42e/langchain/utilities/google_search.py#L9


@app.command()
def financial_agent(stock: str):
    # tools = load_tools(["serpapi", "llm-math"], llm=llm)
    tools = load_tools(["bing-search"], llm=llm)
    # braveSearch = BraveSearch()
    # tools += [braveSearch]
    agent = initialize_agent(
        tools=tools, llm=chat, agent=AgentType.ZERO_SHOT_REACT_DESCRIPTION, verbose=True
    )

    agent.run(
        f"""
              What's the price outlook for : {stock}?, What are the top 3 reasons for the stock to go up?, What are the top 3 reasons for the stock to go down?

The output should be of the form:

Stock: XXX
Price: XXX
Price 1 year ago: XXX
Price 1 year ahead: XXX
Price goes up because:
- Point 1
- Point 2
- Point 3
Price goes down because:
- Point 1
- Point 2
- Point 3
"""
    )


class DialogueAgent:
    def __init__(
        self,
        name: str,
        system_message: SystemMessage,
        model: ChatOpenAI,
    ) -> None:
        self.name = name
        self.system_message = system_message
        self.model = model
        self.prefix = f"{self.name}: "
        self.reset()

    def reset(self):
        self.message_history = ["Here is the conversation so far."]

    def send(self) -> str:
        """
        Applies the chatmodel to the message history
        and returns the message string
        """
        # NOTE: Not sure why doing it merged?
        # I guess this saves tokens (??)
        message = self.model(
            [
                self.system_message,
                HumanMessage(content="\n".join(self.message_history + [self.prefix])),
            ]
        )
        return message.content

    def receive(self, name: str, message: str) -> None:
        """
        Concatenates {message} spoken by {name} into message history
        """
        self.message_history.append(f"{name}: {message}")


class DialogueSimulator:
    def __init__(
        self,
        agents: List[DialogueAgent],
        selection_function: Callable[[int, List[DialogueAgent]], int],
    ) -> None:
        self.agents = agents
        self._step = 0
        self.select_next_speaker = selection_function

    def reset(self):
        for agent in self.agents:
            agent.reset()

    def inject(self, name: str, message: str):
        """
        Initiates the conversation with a {message} from {name}
        """
        for agent in self.agents:
            agent.receive(name, message)

        # increment time
        self._step += 1

    def step(self) -> tuple[str, str]:
        # 1. choose the next speaker
        speaker_idx = self.select_next_speaker(self._step, self.agents)
        speaker = self.agents[speaker_idx]

        # 2. next speaker sends message
        message = speaker.send()

        # 3. everyone receives message
        for receiver in self.agents:
            receiver.receive(speaker.name, message)

        # 4. increment time
        self._step += 1

        return speaker.name, message


@app.command()
def dnd(protagonist_name="Donald Trump", quest="Find all the social security spending"):
    storyteller_name = "Dungeon Master"
    ic(quest)
    ic(protagonist_name)

    word_limit = 50  # word limit for task brainstorming

    game_description = f"""Here is the topic for a Dungeons & Dragons game: {quest}.
    There is one player in this game: the protagonist, {protagonist_name}.
    The story is narrated by the storyteller, {storyteller_name}."""

    def make_player_description(role: str, name: str):
        player_descriptor_system_message = SystemMessage(
            content="You can add detail to the description of a Dungeons & Dragons player."
        )

        make_player_description_prompt = [
            player_descriptor_system_message,
            HumanMessage(
                content=f"""{game_description}
                Please reply with a creative description of the {role}, {name}, in {word_limit} words or less.
                Speak directly to {name}.
                Do not add anything else."""
            ),
        ]

        return ChatOpenAI(temperature=1.0)(make_player_description_prompt).content

    protagonist_description = make_player_description("protagonist", protagonist_name)
    storyteller_description = make_player_description("story teller", storyteller_name)

    ic(protagonist_description)
    ic(storyteller_description)

    # TODO, I should be able to remove duplication between system messages
    # Notice how I'm instructing the stop message to speak to the next agent.
    # That's an example of moving content from the AI to the Code
    # Anything that you want to be simple/deterministic keep as much as possible in the code
    # So intead of having end w/it's your turn next. Put tha tin the dialgo simulator

    protagonist_system_message = SystemMessage(
        content=(
            f"""{game_description}
                Never forget you are the protagonist, {protagonist_name}, and I am the storyteller, {storyteller_name}.
                Your character description is as follows: {protagonist_description}.
                You will propose actions you plan to take and I will explain what happens when you take those actions.
                Speak in the first person from the perspective of {protagonist_name}.
                For describing your own body movements, wrap your description in '*'.
                Do not change roles!
                Do not speak from the perspective of {storyteller_name}.
                Do not forget to finish speaking by saying, 'It is your turn, {storyteller_name}.'
                Do not add anything else.
                Remember you are the protagonist, {protagonist_name}.
                Stop speaking the moment you finish speaking from your perspective.
                """
        )
    )

    storyteller_system_message = SystemMessage(
        content=(
            f"""{game_description}
            Never forget you are the storyteller, {storyteller_name}, and I am the protagonist, {protagonist_name}.
            Your character description is as follows: {storyteller_description}.
            I will propose actions I plan to take and you will explain what happens when I take those actions.
            Speak in the first person from the perspective of {storyteller_name}.
            For describing your own body movements, wrap your description in '*'.
            Do not change roles!
            Do not speak from the perspective of {protagonist_name}.
            Do not forget to finish speaking by saying, 'It is your turn, {protagonist_name}.'
            Do not add anything else.
            Remember you are the storyteller, {storyteller_name}.
            Stop speaking the moment you finish speaking from your perspective.
            """
        )
    )

    make_detailed_quest_prompt = [
        SystemMessage(content="You can make a task more specific."),
        HumanMessage(
            content=f"""{game_description}

            You are the storyteller, {storyteller_name}.
            Please make the quest more specific. Be creative and imaginative.
            Please reply with the specified quest in {word_limit} words or less.
            Speak directly to the protagonist {protagonist_name}.
            Do not add anything else."""
        ),
    ]
    specified_quest = ChatOpenAI(temperature=1.0)(make_detailed_quest_prompt).content
    ic(f"Original quest:\n{quest}\n")
    ic(f"Detailed quest:\n{specified_quest}\n")

    protagonist = DialogueAgent(
        name=protagonist_name,
        system_message=protagonist_system_message,
        model=ChatOpenAI(temperature=0.2),
    )

    storyteller = DialogueAgent(
        name=storyteller_name,
        system_message=storyteller_system_message,
        model=ChatOpenAI(temperature=0.2),
    )

    def select_next_speaker(step: int, agents: List[DialogueAgent]) -> int:
        idx = step % len(agents)
        return idx

    max_iters = 6
    n = 0

    name_cache = []

    def colorize_name(name):
        colors = ["red", "yellow", "blue", "yellow", "magenta", "cyan"]
        if name not in name_cache:
            name_cache.append(name)
        # get index of name in name_cache
        name_idx = name_cache.index(name)
        color = colors[name_idx % len(colors)]
        return f"[{color}]{name}[/{color}]"

    simulator = DialogueSimulator(
        agents=[storyteller, protagonist], selection_function=select_next_speaker
    )
    simulator.reset()
    simulator.inject(storyteller_name, specified_quest)
    print(f"{colorize_name(storyteller_name)}: {specified_quest}")
    print("\n")

    while n < max_iters:
        name, message = simulator.step()
        print(f"{colorize_name(name)}: {message}")
        print("\n")
        n += 1


def load_cached_prompt(prompt_name):
    from langchain import hub

    prompt_cache = os.path.expanduser("~/tmp/pickle/prompts")
    # if prompt_cache directory doesn't exist, create it
    if not os.path.exists(prompt_cache):
        os.makedirs(prompt_cache)
    prompt_maker_filename = f"{prompt_name.replace('/','_')}.pickle"
    prompt_maker_path = os.path.join(prompt_cache, prompt_maker_filename)

    if not os.path.exists(prompt_maker_path):
        prompt_maker_template = hub.pull(prompt_name)
        with open(prompt_maker_path, "wb") as f:
            pickle.dump(prompt_maker_template, f)
    else:
        with open(prompt_maker_path, "rb") as f:
            prompt_maker_template = pickle.load(f)

    return prompt_maker_template


@app.command()
def great_prompt(prompt):
    prompt_maker_template = load_cached_prompt("hardkothari/prompt-maker")
    model = ChatOpenAI(temperature=0.9)
    chain = prompt_maker_template | model
    result = chain.invoke({"lazy_prompt": prompt, "task": prompt})
    print(result.content)


@app.command()
def summarize():
    prompt_maker_template = load_cached_prompt("langchain-ai/chain-of-density:ba34ae10")
    user_text = "".join(sys.stdin.readlines())
    model = ChatOpenAI(temperature=0.9, model="gpt-4")
    chain = prompt_maker_template | model
    result = chain.invoke({"ARTICLE": user_text})
    print(result.content)


@app.command()
def m2df():
    df = im2df()
    # pickle the dataframe
    ic("++ pickle df")
    df.to_pickle("df_messages.pickle.zip")
    ic("-- pickle df")
    # make compatible with archive
    df.sort_values("date", inplace=True)
    df.date = df.date.dt.strftime("%Y-%m-%d %H:%M:%S")
    df.to_csv("big_dump.txt", index=False, sep="\t")


@app.command()
def scratch():
    # I'm the opposite, I told my boss
    # I’m the opposite I told my boss I’m really struggling and getting depression medication

    df = im2df()
    ammon_from_me = df[(df.to_phone.str.contains("7091")) & (df.is_from_me)]
    ic(ammon_from_me)

    tori_from_me = df[(df.to_phone.str.contains("755")) & (df.is_from_me)]
    ic(tori_from_me)

    # df.to_csv("messages.csv", index=False)
    ic(df[df.is_from_me])


@app.command()
def fine_tune():
    df = im2df()
    df_ammon = df[(df.to_phone.str.contains("7091"))]
    create_fine_tune(df_ammon)


# date	text	is_from_me	to_phone


def make_message(role, content):
    return {"role": role, "content": content}


def write_jsonl(data_list: list, filename: Path) -> None:
    with open(filename, "w") as out:
        for ddict in data_list:
            messages = {"messages": ddict}
            jout = json.dumps(messages) + "\n"
            out.write(jout)


def create_fine_tune(df):
    ft_path = Path.home() / "tmp/fine_tune"
    system_prompt = "You are an imessage best friend converation simulator."
    system_message = make_message("system", system_prompt)

    # messages

    # messages =  [{role:, content}]

    # create a finetune file for every day

    traindata_set = []
    df["date_window"] = df.date.dt.strftime("%Y-%V")

    def to_message(row):
        role = "user" if row.is_from_me else "assistant"
        return make_message(role, row["text"])

    df["message"] = df.apply(to_message, axis=1)

    for date_window in df.date_window.unique():
        df_day = df[df.date_window == date_window]
        df_from_assistent = df_day[df_day.is_from_me == False]  # noqa - need this syntax for Pandas
        if df_from_assistent.empty:
            continue

        train_data = [system_message] + df_from_assistent.message.tolist()
        # count tokens
        if (
            tokens := openai_wrapper.num_tokens_from_string(json.dumps(train_data))
            > 15000
        ):
            ic(date_window, tokens)
            continue
        traindata_set.append(train_data)

    ratio = 20
    training = [t for i, t in enumerate(traindata_set) if i % ratio != 0]
    validation = [t for i, t in enumerate(traindata_set) if i % ratio == 0]
    write_jsonl(training, ft_path / "train.jsonl")
    write_jsonl(validation, ft_path / "validate.jsonl")


def im2df():
    # date	text	is_from_me	to_phone
    ic("start load")
    chats = pickle.load(open("raw_messages.pickle", "rb"))
    ic(f"done load {len(chats)}")

    output = []

    # transform to csv
    for c in chats:
        messages = c["messages"]
        if not len(messages):
            continue

        for m in messages:
            row = {
                "date": m.additional_kwargs["message_time_as_datetime"],
                "text": m.content,
                "is_from_me": m.additional_kwargs["is_from_me"],
                "to_phone": m.role,
            }
            output.append(row)

    ic(len(output))

    df = pd.DataFrame(output)
    return df


class GetHypotheticalQuestionsFromDoc(BaseModel):
    Questions: List[str]


@app.command()
def q_for_doc(questions: int = 10):
    get_questions = openai_wrapper.openai_func(GetHypotheticalQuestionsFromDoc)

    chain = (
        ChatPromptTemplate.from_template(
            "Generate a list of exactly {count} hypothetical questions that the below document could be used to answer:\n\n{doc}"
        )
        | ChatOpenAI(max_retries=0, model=openai_wrapper.gpt4.name).bind(
            functions=[get_questions], function_call={"name": get_questions["name"]}
        )
        | JsonKeyOutputFunctionsParser(key_name="Questions")
    )
    user_text = "".join(sys.stdin.readlines())
    r = chain.invoke({"doc": user_text, "count": questions})
    ic(r)


@app.command()
def debug():
    ic("debug")


@app.command()
def messages():
    chat_path = os.path.expanduser("~/imessage/chat.db")
    loader = IMessageChatLoader(path=chat_path)
    ic("loading messages")
    raw_messages = loader.load()
    ic("pickling")
    import pickle

    pickle.dump(raw_messages, open("raw_messages.pickle", "wb"))

    # Merge consecutive messages from the same sender into a single message
    # merged_messages = merge_chat_runs(raw_messages)
    for i, message in enumerate(raw_messages):
        ic(message)
        if i > 50:
            break


if __name__ == "__main__":
    ic("main")
    app_wrap_loguru()
