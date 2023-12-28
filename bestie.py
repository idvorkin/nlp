#!python3


import json
import os
import pickle
from pathlib import Path

import backoff
import numpy as np
import openai
import openai_wrapper
import pandas as pd
import typer
from icecream import ic
from langchain.chat_loaders.imessage import IMessageChatLoader
from langchain.chat_models import ChatOpenAI
from langchain.prompts import ChatPromptTemplate
from langchain.schema import SystemMessage
from loguru import logger
from rich import print
from rich.console import Console
from typing import Annotated
from pydantic import BaseModel


console = Console()
app = typer.Typer()


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
    df = im2df()
    ammon_from_me = df[(df.to_phone.str.contains("7091")) & (df.is_from_me)]
    ic(ammon_from_me)

    tori_from_me = df[(df.to_phone.str.contains("755")) & (df.is_from_me)]
    ic(tori_from_me)

    # df.to_csv("messages.csv", index=False)
    ic(df[df.is_from_me])


@app.command()
def fine_tune(number: str = "2255233"):
    df = im2df()
    df_convo = df[(df.to_phone.str.contains(number))]
    create_fine_tune(df_convo)


def make_message(role, content):
    return {"role": role, "content": content}


def write_jsonl(data_list: list, filename: Path) -> None:
    with open(filename, "w") as out:
        for ddict in data_list:
            messages = {"messages": ddict}
            jout = json.dumps(messages) + "\n"
            out.write(jout)


def create_fine_tune(df):
    ft_path = Path.home() / "tmp/fine-tune"
    system_prompt = "You are an imessage best friend converation simulator."
    system_message = make_message("system", system_prompt)

    # messages

    # messages =  [{role:, content}]

    # I think model is getting confused as too much knowledge about us, and what's been happenign has evolved.
    # So probably need some RAG to help with this.
    # df = df[df.date.dt.year > 2020]
    run_name = "ray_2d"
    df["group"] = 1e3 * df.date.dt.year + np.floor(df.date.dt.day_of_year / 2)
    # invert is_from_me if you want to train for Igor.
    # df.is_from_me = ~df.is_from_me

    # images are uffc - remove those
    # make ''' ascii to be more pleasant to look at
    df.text = df.text.apply(
        lambda t: t.replace("\ufffc", "").replace("\u2019", "'").strip()
    )
    df = df[df.text.str.len() > 0]

    def to_message(row):
        role = "user" if row.is_from_me else "assistant"
        return make_message(role, row["text"])

    df["message"] = df.apply(to_message, axis=1)

    traindata_set = []
    ic(len(df.group.unique()))
    for group in df.group.unique():
        df_group = df[df.group == group]

        # Not an interesting group if all from the same person
        if len(df_group.is_from_me.unique()) == 1:
            continue

        train_data = [system_message] + df_group.message.tolist()
        # count tokens
        if (
            tokens := openai_wrapper.num_tokens_from_string(json.dumps(train_data))
        ) > 15000:
            ic(group, tokens)
            continue
        traindata_set.append(train_data)

    ratio = 20
    training = [t for i, t in enumerate(traindata_set) if i % ratio != 0]
    validation = [t for i, t in enumerate(traindata_set) if i % ratio == 0]
    write_jsonl(training, ft_path / f"train.{run_name}.jsonl")
    write_jsonl(validation, ft_path / f"validate.{run_name}.jsonl")

    ic(len(training))
    flagged = 0
    for i, t in enumerate(training):
        output = moderate(json.dumps(t))
        if i % 100 == 0:
            ic(t)
            ic(i, flagged)
        if output.flagged:
            ic(i, output)
            flagged += 1

    ic(flagged)


@backoff.on_exception(backoff.expo, openai.RateLimitError)
def moderate(text):
    client = openai.OpenAI()
    response = client.moderations.create(input=text)
    return response.results[0]


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


models = {
    "r+1d": "ft:gpt-3.5-turbo-1106:idvorkinteam::8Z4f8RhL",
    "2021+1d": "ft:gpt-3.5-turbo-1106:idvorkinteam::8YkPgWs2",
    "i-2021+1d": "ft:gpt-3.5-turbo-1106:idvorkinteam::8Z3GDyd0",
    "2015+1d": "ft:gpt-3.5-turbo-1106:idvorkinteam::8YgPRpMB",
    "2021+3d": "ft:gpt-3.5-turbo-1106:idvorkinteam::8Yz10hf9",
    "2021+2d": "ft:gpt-3.5-turbo-1106:idvorkinteam::8Yys2osB",
}
models_list = "\n".join(models.keys())

@app.command()
def goal_helper(
    model_name: Annotated[
        str, typer.Option(help=f"Model any of: {models_list}")
    ] = "2021+3d",
):
    from langchain.memory import ChatMessageHistory

    system_prompt_base = "You are an imessage best friend converation simulator."
    custom_instructions = """
        * When you answer use atleast 6 words, or ask a question
        * Keep the conversation going if I anwer with the letter x
    * Remind me of things that are important to me (from the eulogy I hope to live):

        Dealer of smiles and wonder
        Mostly car free spirit
        Disciple of the 7 habits of highly effective people
        Fit fellow
        Emotionally healthy human
        Husband to Tori - his life long partner
        Technologist
        Professional
        Family man
        Father to Amelia - an incredible girl
        Father to Zach - a wonderful boy

        """
    system_prompt = f"{system_prompt_base}\n {custom_instructions}"
    memory = ChatMessageHistory()
    memory.add_message(SystemMessage(content=system_prompt))
    model = ChatOpenAI(model=models[model_name])
    ic(model_name)
    ic(custom_instructions)

    while True:
        user_input = input(">")
        if user_input == "debug":
            ic(model_name)
            ic(custom_instructions)
        memory.add_user_message(message=user_input)
        prompt = ChatPromptTemplate.from_messages(memory.messages)
        chain = prompt | model
        result = chain.invoke({})
        ai_output = str(result.content)
        memory.add_ai_message(ai_output)
        print(f"[yellow]{ai_output}")

class Memory(BaseModel):
    # long term knowledge
    # mid term conversations
    # short term conversations
    messages: list = []
    system_messages: list = []
    facts: list = []
    facts = []

    def add_user_message(self, content):
        self.messages.append(("user", content))

    def add_ai_message(self, content):
        self.messages.append(("assistant", content))

    def add_system_message(self, content):
        self.system_messages.append(("system", content))

    def add_facts(self, facts):
        self.system_messages.append(("facts", facts))
        pass

    def to_messages(self):
        return self.system_messages + self.messages


@app.command()
def convo(
    model_name: Annotated[
        str, typer.Option(help=f"Model any of: {models_list}")
    ] = "i-2021+1d",
):
    from langchain.memory import ChatMessageHistory

    system_prompt_base = "You are an imessage best friend converation simulator."
    custom_instructions = """
        * When you answer use atleast 6 words, or ask a question
        * Keep the conversation going if I anwer with the letter x
        * You are feeling supportive and positive
        """
    system_prompt = f"{system_prompt_base}\n {custom_instructions}"
    memory = ChatMessageHistory()
    memory.add_message(SystemMessage(content=system_prompt))
    model = ChatOpenAI(model=models[model_name])
    ic(model_name)
    ic(custom_instructions)

    while True:
        user_input = input(">")
        if user_input == "debug":
            ic(model_name)
            ic(custom_instructions)
        memory.add_user_message(message=user_input)
        prompt = ChatPromptTemplate.from_messages(memory.messages)
        chain = prompt | model
        result = chain.invoke({})
        ai_output = str(result.content)
        memory.add_ai_message(ai_output)
        print(f"[yellow]{ai_output}")


@app.command()
def a_i_convo(
    start: str = "Just woke up, bored",
    model_name: Annotated[
        str, typer.Option(help=f"Model any of: {models_list}")
    ] = "2021+3d",
    rounds: int = 10,
):
    from langchain.memory import ChatMessageHistory

    system_prompt_base = "You are an imessage best friend converation simulator."
    custom_instructions = """
        * When you answer use atleast 6 words, or ask a question
        * Keep the conversation going if I anwer with the letter x
        """
    system_prompt = f"{system_prompt_base}\n {custom_instructions}"

    bestie_memory = ChatMessageHistory()
    bestie_memory.add_message(SystemMessage(content=system_prompt))
    bestie_model = ChatOpenAI(model=models[model_name])

    igor_memory = ChatMessageHistory()
    igor_memory.add_message(SystemMessage(content=system_prompt))
    igor_memory.add_ai_message(start)
    igor_model = ChatOpenAI(model=models["i-2021+1d"])

    ic(model_name)
    ic(custom_instructions)

    print("[pink]First message is Igor supplied, the rest is an AI loop")
    for i in range(rounds):
        user_input = str(igor_memory.messages[-1].content)
        print(f"[green]{i}:{user_input}")
        bestie_memory.add_user_message(message=user_input)

        prompt = ChatPromptTemplate.from_messages(bestie_memory.messages)
        bestie_output = str((prompt | bestie_model).invoke({}).content)

        print(f"[yellow]{i}:{bestie_output}")
        bestie_memory.add_ai_message(bestie_output)
        igor_memory.add_user_message(bestie_output)

        # add something from igor model
        prompt = ChatPromptTemplate.from_messages(igor_memory.messages)
        igor_output = str((prompt | igor_model).invoke({}).content)
        igor_memory.add_ai_message(igor_output)


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


@logger.catch()
def app_wrap_loguru():
    app()


if __name__ == "__main__":
    ic("main")
    app_wrap_loguru()
