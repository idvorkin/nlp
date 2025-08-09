#!python3


import json
from pydantic import BaseModel

import typer
from icecream import ic
import ell
import rich
import requests

app = typer.Typer(no_args_is_help=True)


""" Freddy is a PSC peer feedback bot to help senior engineers articulate high quality PSC feedback for e3-e5 engineers.

## Why

Precise, accurate, actionable, level appropriate PSC feedback is critical to assessing and growing engineers.
However, providing this kind of feedback can be hard and time consuming. Having a coach to help you articulate your feedback will be crucial.

Freddy (not his real name) was a great manager I had.

He always told me *"Don't waste your time over thinking your feedback. just brain dump your thoughts in a document then book a F2F."*
In the F2F, Freddy would ask open ended/socratic questions. This would help me narrow down and crispen my thinking, making it easy to help me turn it into actionable feedback. """


class InputMessage(BaseModel):
    role: str
    content: str


def process_and_flatten_json(input_string: str) -> list[dict]:
    """
    Processes a JSON string, flattens nested content, and dynamically extracts all fields.

    :param input_string: The input JSON string containing data.
    :return: A list of dictionaries with extracted and flattened fields.
    """

    def flatten_json_one_level(obj: dict) -> dict:
        """
        Flatten a JSON object to one level, removing parent keys and retaining lists.

        :param obj: The JSON object to flatten (dict or list).
        :return: A flattened dictionary.
        """
        flat_dict = {}
        for key, value in obj.items():
            if isinstance(value, dict):
                for sub_key, sub_value in value.items():
                    flat_dict[sub_key] = sub_value
            else:
                flat_dict[key] = value
        return flat_dict

    data = json.loads(input_string)
    # if data isn't a list, stick it in a list
    if not isinstance(data, list):
        data = [data]

    results = []

    for item in data:
        flattened_item = {}
        for key, value in item.items():
            if isinstance(value, str):
                try:
                    # Attempt to parse and flatten nested JSON strings
                    parsed_value = json.loads(value)
                    flattened_item.update(flatten_json_one_level(parsed_value))
                except json.JSONDecodeError:
                    # If the value is not JSON, keep it as-is
                    flattened_item[key] = value
            else:
                flattened_item[key] = value
        results.append(flattened_item)

    return results


def call_freddy_raw(messages: list[InputMessage]):
    # freddy lives on localhost:8011, with a fastapi app like
    # @app.post("/freddy")
    # def call_freddy(messages:list[InputMessage]):
    import requests

    url = "http://localhost:8011/freddy"
    # only include content and role from input messages
    messages_to_send = [{"content": m.content, "role": m.role} for m in messages]
    response = requests.post(url, json=messages_to_send)
    if response.status_code == 200:
        return response.json()
    else:
        raise Exception(
            f"Failed to call Freddy API: {response.status_code} - {response.text}"
        )


class FreddyChatBot:
    def __init__(self):
        self.freddy_messages: list[InputMessage] = []

    def call(self, content: str):
        self.freddy_messages.append(InputMessage(role="user", content=content))
        response = call_freddy_raw(self.freddy_messages)
        self.freddy_messages.append(
            InputMessage(role="assistant", content=response["content"])
        )
        return response


def ic_flat_response(response: dict):
    r = process_and_flatten_json(json.dumps(response))
    ic(r)


@app.command()
def generic_positive():
    freddy = FreddyChatBot()
    response = freddy.call("Bob is a great co-worker")
    ic_flat_response(response)
    response = freddy.call("Thanks to him the project was successful")
    ic_flat_response(response)


@app.command()
def generic_negative():
    freddy = FreddyChatBot()
    response = freddy.call("Bob is poopy")
    ic_flat_response(response)


@ell.complex(model="gpt-5", temperature=0.7)
def prompt_senior_engineer_bot(message_history: list[ell.Message]) -> list[ell.Message]:
    llm_system_message = """ You are going to roleplay Joe, a senior engineer at Meta who is not great at giving feedback. 
    You are giving feedback to Bob, an e3 engineer at Meta. Who you worked with
    on the ketchup project. 
    You  will make up what Bob did, and you will give feedback on it. 
    
When you speak:
    - Keep your response under 100 words
    
    """
    return [
        ell.system(llm_system_message),
    ] + message_history


class JoeChatBot:
    def __init__(self):
        self.joe_messages: list[ell.Message] = []

    def call(self, content: str):
        self.joe_messages.append(ell.Message(role="user", content=content))
        response = prompt_senior_engineer_bot(self.joe_messages)
        response_text = response.content[0].text
        self.joe_messages.append(ell.Message(role="assistant", content=response_text))
        return response_text


# NOTE, When you call freddy freddy is the AI, when you call Joe, Joe is the AI
# Make sure the message history is always in the right format
# Freddy needs to use InputMessage, Joe needs to use ell.Message


def chat_bot_loop(rounds: int = 5):
    joe_initial_message = "Bob is a great co-worker"
    joe_response = joe_initial_message
    joe = JoeChatBot()
    freddy = FreddyChatBot()
    for _ in range(rounds):
        freddy_response = freddy.call(joe_response)
        freddy_text = json.loads(freddy_response["content"])["response"]
        rich.print(f"[yellow]{freddy_text}[/yellow]")
        joe_response = joe.call(freddy_text)
        rich.print(f"[green]{joe_response}[/green]")
        freddy_response = freddy.call(joe_response)


@app.command()
def version():
    url = "http://localhost:8011/"
    response = requests.get(url)
    if response.status_code == 200:
        ic(response.json())
    else:
        print(f"Failed to call Freddy API: {response.status_code} - {response.text}")


@app.command()
def smoke_test():
    chat_bot_loop(5)
    # feedback_positive_generic()
    # feedback_negative_generic()


if __name__ == "__main__":
    app()
