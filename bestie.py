#!python3

import json
import os
import pickle
from pathlib import Path
from datetime import datetime, timedelta

import backoff
import numpy as np
import openai
import openai_wrapper
import pandas as pd
import typer
from icecream import ic
from langchain.chat_models import init_chat_model
from langchain_community.chat_loaders.imessage import IMessageChatLoader
from langchain_community.chat_message_histories import ChatMessageHistory
from langchain_openai.chat_models import ChatOpenAI
from langchain.prompts import ChatPromptTemplate
from langchain.schema import SystemMessage
from loguru import logger
from rich import print
from rich.console import Console
from typing import Annotated
from pydantic import BaseModel
import requests

openai_wrapper.setup_secret()

path_latest_state = Path.home() / "tmp/latest_state.txt"

# set the environment variable from the secrets file


console = Console()
app = typer.Typer()


@app.command()
def export_txt():
    df = get_message_history_df()
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
    df = get_message_history_df()
    ammon_from_me = df[(df.to_phone.str.contains("7091")) & (df.is_from_me)]
    ic(ammon_from_me)

    tori_from_me = df[(df.to_phone.str.contains("755")) & (df.is_from_me)]
    ic(tori_from_me)

    # df.to_csv("messages.csv", index=False)
    ic(df[df.is_from_me])


@app.command()
def messages_stats():
    df = get_message_history_df()
    df = df[(df.to_phone.str.contains("7091")) & (df.date.dt.year == 2023)]
    df["day"] = 1e3 * df.date.dt.year + df.date.dt.day_of_year
    days = df.groupby(df.day).count().sort_values("text", ascending=False)
    # dump p50, p75, p95 of days to file
    print(days.text.describe(percentiles=[0.5, 0.75, 0.95]))


def llm_extract_facts(dfChat):
    system_prompt = """
Below is a conversation between a real person (user) and the assistant (the bestie).

You will help extract facts and topics of conversation, that will be used to simulate the bestie in the future. Don't bother storing facts that can be looked up in Wikipedia. Split facts and topics of conversation into those about the user and the bestie.

Focus on extracting:
1. Personal information (e.g., names, relationships, jobs, hobbies)
2. Ongoing projects or goals
3. Recent events or experiences
4. Emotional states or patterns
5. Shared memories or inside jokes
6. Preferences and dislikes
7. Habits or routines

For each fact or topic, specify whether it relates to the user or the bestie.
Where possible group the facts into people (include their relationship), or into categories
Keep mentioned people in a seperate section with information about them:
    name, relationship to bestie or user , and all facts about them
When facts change over time note that as well

Conversation:
    """

    conversation = df_messages_to_text_prompt(dfChat)
    text_prompt = system_prompt + "\n" + conversation

    tokens = openai_wrapper.num_tokens_from_string(text_prompt)
    MAX_TOKENS = 100_000
    if tokens > MAX_TOKENS:
        ic("too many tokens, clipping", tokens)
        text_prompt = text_prompt[:MAX_TOKENS]

    messages = [SystemMessage(content=text_prompt)]
    model = ChatOpenAI(model=openai_wrapper.gpt4.name)
    result = model.invoke(messages).content
    return result


def df_messages_to_text_prompt(df):
    # df has fields date, role, message
    # Merge consecutive messages from the same role (user or assistant)
    # text prompt is of format:
    # user: message
    # assistant: message

    text_prompt = ""
    previous_role = None
    current_message = ""

    for _, row in enumerate(df.message.to_list()):
        content = row["content"].replace("{", "(").replace("}", ")")
        if row["role"] == previous_role:
            current_message += ". " + content
        else:
            if previous_role:
                text_prompt += f"{previous_role}: {current_message}\n"
            current_message = content
            previous_role = row["role"]

    # Add the last message
    if previous_role:
        text_prompt += f"{previous_role}: {current_message}\n"

    return text_prompt


def llm_summarize_recent_convo(dfChat):
    system_prompt = """
Below is a conversation between a real person (user) and the assistant (the bestie).  You will create a prompt to feed GPT-4, that will simulate the bestie, ensuring conversations with it sounds similar.

The prompt will use the content of the conversation so it can be used as part of a prompt for the bestie (assistant). Include what the assistant and user are trying to accomplish, current state, and concrete topics to discuss. Use about 100 lines

Conversation:
    """

    conversation = df_messages_to_text_prompt(dfChat)
    text_prompt = system_prompt + "\n" + conversation

    ic(text_prompt)
    ic(openai_wrapper.num_tokens_from_string(text_prompt))

    prompt = ChatPromptTemplate.from_messages([SystemMessage(content=text_prompt)])
    model = ChatOpenAI(model=openai_wrapper.gpt4.name)
    chain = prompt | model
    result = chain.invoke({}).content
    return result


@app.command()
def recent_chats():
    chat_path = os.path.expanduser("~/imessage/chat.db")
    loader = IMessageChatLoader(path=chat_path)
    ic("loading messages")
    raw = loader.load()

    recent_date = datetime.now() - timedelta(days=7)
    recent_date = recent_date.strftime("%Y-%m-%d")
    ic(recent_date)

    # limit to chats wtih bestie
    # bestie_messages = [c for c in raw_messages if "7091" in c.to_phone]
    df_chats = chats_to_df(raw)
    # filter messaages to last month
    df_chats = df_chats[(df_chats.to_phone.str.contains("7091"))]
    df_chats = df_chats.set_index(df_chats.date)
    df_chats = df_chats[df_chats.date > recent_date]

    convos = ""
    for m in df_chats.message.tolist():
        convos += f"\n{m['role']}: {m['content']}"
    print(convos)


@app.command()
def recent_state():
    chat_path = os.path.expanduser("~/imessage/chat.db")
    loader = IMessageChatLoader(path=chat_path)
    ic("loading messages")
    raw = loader.load()

    recent_date = datetime.now() - timedelta(days=7)
    recent_date = recent_date.strftime("%Y-%m-%d")
    ic(recent_date)

    # limit to chats wtih bestie
    # bestie_messages = [c for c in raw_messages if "7091" in c.to_phone]
    df_chats = chats_to_df(raw)
    # filter messaages to last month
    df_chats = df_chats[(df_chats.to_phone.str.contains("7091"))]
    df_chats = df_chats.set_index(df_chats.date)
    df_chats = df_chats[df_chats.date > recent_date]

    summary = llm_summarize_recent_convo(df_chats)
    print(summary)

    # Write to file
    ic(path_latest_state)
    path_latest_state.write_text(summary)


@app.command()
def extract_facts(
    start_date: Annotated[str, typer.Option(help="Start date in YYYY-MM-DD format")] = (
        datetime.now() - timedelta(days=30)
    ).strftime("%Y-%m-%d"),
):
    chat_path = os.path.expanduser("~/imessage/chat.db")
    loader = IMessageChatLoader(path=chat_path)
    ic("loading messages")
    raw = loader.load()

    ic(f"Extracting facts from {start_date}")

    df_chats = chats_to_df(raw)
    df_chats = df_chats[(df_chats.to_phone.str.contains("7091"))]
    df_chats = df_chats.set_index(df_chats.date)
    df_chats = df_chats[df_chats.date > start_date]

    facts = llm_extract_facts(df_chats)
    print(facts)

    # Write to file
    facts_file = Path.home() / "tmp/extracted_facts.txt"
    facts_file.write_text(facts)
    ic(f"Facts written to {facts_file}")


def make_message(role, content):
    return {"role": role, "content": content}


def write_messages_samples_to_jsonl(samples: list, filename: Path) -> None:
    json_string_samples = [json.dumps({"messages": sample}) for sample in samples]
    filename.write_text("\n".join(json_string_samples))


@app.command()
def finetune(number: str = "7091", igor_assistant: bool = False):
    df = get_message_history_df()
    df = df[(df.to_phone.str.contains(number))]
    system_prompt = "You are an imessage best friend converation simulator."
    system_message = make_message("system", system_prompt)

    # I think model is getting confused as too much knowledge about us, and what's been happenign has evolved.
    # So probably need some RAG to help with this.
    # df = df[df.date.dt.year > 2020]

    df = df[(df.date.dt.year > 2021)]
    # df = df[(df.date.dt.month < 10)]

    # messages format =  [{role:, content}]
    days_to_group = 3  # don't use 7 as that induces some periodicity

    df["group"] = 1e3 * df.date.dt.year + np.floor(
        df.date.dt.day_of_year / days_to_group
    )
    # invert is_from_me if you want to train for Igor.
    if igor_assistant:
        df.is_from_me = ~df.is_from_me

    convo_between = f"igor_to_{number}" if igor_assistant else f"{number} to igor"
    run_name = f"{convo_between}_2021_plus_{days_to_group}d"
    ic(run_name)

    MAX_TOKENS_IN_TRAINING_LINE = 57_000  # currently limited, but will update
    MAX_MESSAGES_IN_A_TRAINING_LINE = 2000

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
        ) > MAX_TOKENS_IN_TRAINING_LINE:
            ic("too many tokens", group, tokens)
            continue

        if (cmessages := len(train_data)) > MAX_MESSAGES_IN_A_TRAINING_LINE:
            ic("too many messages clipping", group, cmessages)
            train_data = train_data[:MAX_MESSAGES_IN_A_TRAINING_LINE]

        # remove trailing user messages
        while train_data and train_data[-1]["role"] == "user":
            train_data = train_data[:-1]

        traindata_set.append(train_data)

    # split into training and validation
    validation_ratio = 20
    training = [t for i, t in enumerate(traindata_set) if i % validation_ratio != 0]
    validation = [t for i, t in enumerate(traindata_set) if i % validation_ratio == 0]

    ft_path = Path.home() / "tmp/fine-tune"
    write_messages_samples_to_jsonl(training, ft_path / f"train.{run_name}.jsonl")
    write_messages_samples_to_jsonl(validation, ft_path / f"validate.{run_name}.jsonl")

    ic("output name", run_name)

    ic(len(training))
    ic(openai_wrapper.num_tokens_from_string(json.dumps(training)))

    flagged = 0
    for i, t in enumerate(training):
        output = moderate(json.dumps(t))
        if output.flagged:
            ic(i, output)
            flagged += 1

    ic(flagged)


@backoff.on_exception(backoff.expo, openai.RateLimitError)
def moderate(text):
    client = openai.OpenAI()
    response = client.moderations.create(input=text)
    return response.results[0]


def chats_to_df(chats):
    output = []
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
    df = pd.DataFrame(output)
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
    return df


def get_message_history_df():
    # date	text	is_from_me	to_phone
    ic("start load")
    chats = pickle.load(open("raw_messages.pickle.gz", "rb"))
    ic(f"done load {len(chats)}")
    return chats_to_df(chats)


models = {
    "r+1d": "ft:gpt-3.5-turbo-1106:idvorkinteam::8Z4f8RhL",
    "2021+1d": "ft:gpt-3.5-turbo-1106:idvorkinteam::8YkPgWs2",
    "2023+7d": "ft:gpt-3.5-turbo-1106:idvorkinteam::8apOoG0u",
    "igor": "ft:gpt-4o-mini-2024-07-18:idvorkinteam:i-to-a-3d-gt-2021:9qiMMqOz",
    "2015+1d": "ft:gpt-3.5-turbo-1106:idvorkinteam::8YgPRpMB",
    "2021+3d": "ft:gpt-3.5-turbo-1106:idvorkinteam::8Yz10hf9",
    "gpt4o": "ft:gpt-4o-mini-2024-07-18:idvorkinteam:2021-plus-7d:9qUwxwkO",
    "2021+2d": "ft:gpt-3.5-turbo-1106:idvorkinteam::8Yys2osB",
}
models_list = "\n".join(models.keys())


@app.command()
def goal_helper(
    model_name: Annotated[
        str, typer.Option(help=f"Model any of: {models_list}")
    ] = "gpt4o",
):
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


def get_weather(city):
    weather_url = f"https://wttr.in/{city}?format=j1"
    return requests.get(weather_url).json()["current_condition"][0]


def build_facts():
    location = "Denmark"
    facts = f"""
# Facts about best friend for the convo

    * The date is current {datetime.now()}
    * The AI is in {location}, works at SNAP as a principal software engineer
    * The user is in Seattle, works at Meta as a software engineering manager
    * The weather is current {get_weather(location)}
        """
    return facts


def support_igor_with():
    facts = """
# You help the user by reminding him of his goals

    * Goals include being a great dad to Zach (born 2010), Amelia born (2014), Tori
    * Physical health goals: losing weight, doing Kettle Bells
    * Emotional Health Goals: beign present, daily meditation
    * Identity Health Goals: Doing Magic for strangers,  Doing Balloons
    * Work Goals: Leaving work at work, being a great manager
    """
    return facts


def make_bestie_system_prompt():
    system_prompt_base = "You are a cutting-edge iMessage-compatible best friend conversation simulator designed to emulate the warmth and familiarity of a real friendship."
    # These instructions came from a convo w/GPT:
    # https://gist.github.com/idvorkin/119cba9273f165bcb7875f075c69e06e
    custom_instructions = """

# Instructions
    * Craft responses that are a minimum of six words long, or ask a thought-provoking question to maintain a lively conversation.
    * Persist in keeping the conversation flowing, even if the user's input is as minimal as the letter 'x'.
    * Channel a consistently supportive and upbeat demeanor, reinforcing the user's self-esteem and offering encouragement.
    * Strive to deliver a satisfying and engaging conversational experience, reminiscent of a heartfelt interaction with an actual best friend.
    * Inject appropriate humor to brighten the dialogue, and reference shared memories or inside jokes when relevant.
    * Navigate a range of topics with ease, mirroring the dynamic and multifaceted nature of a deep personal connection.

    """
    system_prompt = f"""{system_prompt_base}\n
    {custom_instructions}\n
    {build_facts()}
    {support_igor_with()}
# Additional instructions
    {path_latest_state.read_text()}
    """
    return system_prompt


# lets add the last 2 weeks of messages to history


def createBestieMessageHistory():
    memory = ChatMessageHistory()
    memory.add_message(SystemMessage(content=make_bestie_system_prompt()))

    return memory


@app.command()
def output_system_prompt():
    """Output the bestie system prompt."""
    system_prompt = make_bestie_system_prompt()
    print(system_prompt)


@app.command()
def tony():
    """Talk to Toni"""

    url = "https://idvorkin--modal-tony-server-assistant.modal.run"
    payload = {"ignored": "ignored"}
    tony_response = requests.post(url, json=payload).json()
    model_name = tony_response["assistant"]["model"]["model"]
    ic(model_name)
    model = init_chat_model(model_name)
    memory = ChatMessageHistory()

    # TODO build a program to parse this out
    system_message_content = tony_response["assistant"]["model"]["messages"][0][
        "content"
    ]
    memory.add_message(SystemMessage(content=system_message_content))

    while True:
        user_input = input("Igor:")
        if user_input == "debug":
            ic(model_name)
            # ic(custom_instructions)
        memory.add_user_message(message=user_input)
        prompt = ChatPromptTemplate.from_messages(memory.messages)
        chain = prompt | model
        result = chain.invoke({})
        ai_output = str(result.content)
        memory.add_ai_message(ai_output)
        print(f"[yellow]Tony:{ai_output}")


@app.command()
def convo(
    model_name: Annotated[
        str, typer.Option(help=f"Model any of: {models_list}")
    ] = "gpt4o",
):
    memory = createBestieMessageHistory()
    model = ChatOpenAI(model=models[model_name])
    ic(model_name)
    # ic(custom_instructions)

    while True:
        user_input = input("Igor:")
        if user_input == "debug":
            ic(model_name)
            # ic(custom_instructions)
        memory.add_user_message(message=user_input)
        prompt = ChatPromptTemplate.from_messages(memory.messages)
        chain = prompt | model
        result = chain.invoke({})
        ai_output = str(result.content)
        memory.add_ai_message(ai_output)
        print(f"[yellow]aGPT:{ai_output}")


@app.command()
def a_i_convo(
    start: str = "Just woke up, bored",
    model_name: Annotated[
        str, typer.Option(help=f"Model any of: {models_list}")
    ] = "gpt4o",
    rounds: int = 10,
):
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
    igor_model = ChatOpenAI(model=models["igor"])

    ic(model_name)
    ic(custom_instructions)

    print("[pink]First message is Igor supplied, the rest is an AI loop")
    for i in range(rounds):
        user_input = str(igor_memory.messages[-1].content)
        print(f"[green]iGPT_{i}: {user_input}")
        bestie_memory.add_user_message(message=user_input)

        prompt = ChatPromptTemplate.from_messages(bestie_memory.messages)
        bestie_output = str((prompt | bestie_model).invoke({}).content)

        print(f"[yellow]aGPT_{i}: {bestie_output}")
        bestie_memory.add_ai_message(bestie_output)
        igor_memory.add_user_message(bestie_output)

        # add something from igor model
        prompt = ChatPromptTemplate.from_messages(igor_memory.messages)
        igor_output = str((prompt | igor_model).invoke({}).content)
        igor_memory.add_ai_message(igor_output)


@app.command()
def export_chatdb():
    chat_path = os.path.expanduser("~/imessage/chat.db")
    loader = IMessageChatLoader(path=chat_path)
    ic("loading messages")
    raw_messages = loader.load()
    ic("pickling")
    import pickle

    pickle.dump(raw_messages, open("raw_messages.pickle.gz", "wb"))

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
