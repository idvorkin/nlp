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
    ft_path = Path.home() / "tmp/fine-tune"
    system_prompt = "You are an imessage best friend converation simulator."
    system_message = make_message("system", system_prompt)

    # messages

    # messages =  [{role:, content}]

    # I think model is getting confused as too much knowledge about us, and what's been happenign has evolved.
    # So probably need some RAG to help with this.
    df = df[df.date.dt.year > 2020]

    run_name = "2020_up_1d"
    df["group"] = 1e3 * df.date.dt.year + np.floor(df.date.dt.day_of_year / 1)
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
    for i, t in enumerate(training[:100]):
        output = moderate(json.dumps(t))
        if output.flagged:
            ic(i, output)


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


@app.command()
def bestie():
    from langchain.memory import ChatMessageHistory

    system_prompt_base = "You are an imessage best friend converation simulator."
    custom_instructions = "When you answer use atleast 6 words, or ask a question"
    system_prompt = f"{system_prompt_base}\n {custom_instructions}"
    memory = ChatMessageHistory()
    memory.add_message(SystemMessage(content=system_prompt))
    models = {
        "2021+1d": "ft:gpt-3.5-turbo-1106:idvorkinteam::8YkPgWs2",
        "2015+1d": "ft:gpt-3.5-turbo-1106:idvorkinteam::8YgPRpMB",
    }
    model_name = "2021+1d"
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
