#!python3


import json
import os
import pickle
import sys
from pathlib import Path
from typing import List

import backoff
import numpy as np
import openai
import openai_wrapper
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
)
from langchain.prompts import ChatPromptTemplate
from langchain.schema import SystemMessage
from loguru import logger
from pydantic import BaseModel
from rich import print
from rich.console import Console

console = Console()
app = typer.Typer()


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
    ft_path = Path.home() / "tmp/fine-tune"
    system_prompt = "You are an imessage best friend converation simulator."
    system_message = make_message("system", system_prompt)

    # messages

    # messages =  [{role:, content}]

    # I think model is getting confused as too much knowledge about us, and what's been happenign has evolved.
    # So probably need some RAG to help with this.
    df = df[df.date.dt.year > 2020]

    run_name = "2020_up_4d"
    df["group"] = 1e3 * df.date.dt.year + np.floor(df.date.dt.day_of_year / 4)
    # df["group"] = df.date.dt.strftime("%Y-%m-%d")

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
        df_from_assistent = df_group[df_group.is_from_me == False]  # noqa - need this syntax for Pandas
        if df_from_assistent.empty:
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


if __name__ == "__main__":
    ic("main")
    app_wrap_loguru()
