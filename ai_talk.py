#!python3

import os
import json
from icecream import ic
import typer
from rich.console import Console
from rich import print
from typing import List
from pydantic import BaseModel
from loguru import logger
import pudb
from typing_extensions import Annotated

console = Console()
app = typer.Typer()
from langchain.llms import OpenAI
from langchain.chat_models import ChatOpenAI
from langchain.prompts.chat import (
    ChatPromptTemplate,
    SystemMessagePromptTemplate,
    HumanMessagePromptTemplate,
)
from typing import Any, Optional
from langchain.output_parsers.openai_functions import OutputFunctionsParser


from langchain.schema import (
    Generation,
    OutputParserException,
)


def openai_func(cls):
    return {"name": cls.__name__, "parameters": cls.model_json_schema()}


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
    attach: bool = Annotated[bool, typer.Option(prompt="Attach to existing process")],
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
def talk_1(ctx: typer.Context, topic: str = "software engineers", count: int = 2):
    """Tell me a joke"""
    process_shared_app_options(ctx)
    model = ChatOpenAI()
    prompt = ChatPromptTemplate.from_template("tell me {count} jokes about {topic}")
    chain = prompt | model
    response = chain.invoke({"topic": topic, "count": count})
    ic(response.content)


@app.command()
def talk_2(ctx: typer.Context, topic: str = "software engineers", count: int = 2):
    """Tell me a joke, but with structured output"""

    process_shared_app_options(ctx)
    print("FYI: Very handy to include reasoning")

    class Joke(BaseModel):
        setup: str
        punch_line: str
        reasoning_for_joke: str

    class Jokes(BaseModel):
        count: int
        jokes: List[Joke]

    get_joke = openai_func(Jokes)

    process_shared_app_options(ctx)
    model = ChatOpenAI()
    prompt = ChatPromptTemplate.from_template("tell me {count} jokes about {topic}")
    chain = (
        prompt
        | model.bind(
            functions=[get_joke]
            # function_call={"name": get_joke["name"]}, functions=[get_joke]
        )
        | JsonOutputFunctionsParser2()
    )

    # JsonKeyOutputFunctionsParser(key_name="jokes")

    response = chain.invoke({"topic": topic, "count": count})
    print(response)


@app.command()
def talk_3(ctx: typer.Context, n: int = 20234, count: int = 4):
    """Ask for the n-th prime"""
    process_shared_app_options(ctx)

    print("FYI: Like humans, models hallucinate ")
    prompt = ChatPromptTemplate.from_template(f"What is the {n}th prime")
    model = ChatOpenAI()
    chain = prompt | model

    for _ in range(count):
        response = chain.invoke({})
        ic(response)


@app.command()
def talk_4(ctx: typer.Context, n: int = 20234, count: int = 4):
    """Ask for the nth prime, but use tools"""
    process_shared_app_options(ctx)
    model = ChatOpenAI(
        model="gpt-4-0613"
    )  # code generation on gpt-3.5 isn't strong enough
    prompt = ChatPromptTemplate(
        messages=[
            SystemMessagePromptTemplate.from_template(
                "Write code to solve the users problem. the last line of the python  program should print the answer. Do not use sympy"
            ),
            HumanMessagePromptTemplate.from_template(f"What is the {n}th prime"),
        ]
    )

    class PythonExecutionEnvironment(BaseModel):
        valid_python: str
        code_explanation: str

    python_repl = {
        "name": "python_repl",
        "parameters": PythonExecutionEnvironment.model_json_schema(),
    }

    chain = prompt | model.bind(functions=[python_repl]) | JsonOutputFunctionsParser2()
    response = chain.invoke({})

    valid_python = response["valid_python"]
    print(valid_python)
    print("----")
    print(response["code_explanation"])
    print("----")
    input("Are you sure you want to run this code??")
    exec(valid_python)


@app.command()
def talk_5(ctx: typer.Context, topic: str = "software engineers", count: int = 2):
    """Tell me a joke, but with structured output"""
    process_shared_app_options(ctx)

    class Joke(BaseModel):
        setup: str
        punch_line: str
        reasoning_for_joke: str

    class Jokes(BaseModel):
        count: int
        jokes: List[Joke]

    class GetAudienceGender(BaseModel):
        pass

    model = ChatOpenAI()
    model = ChatOpenAI(model="gpt-4")
    model = model.bind(functions=[openai_func(Jokes), openai_func(GetAudienceGender)])

    prompt = ChatPromptTemplate(
        messages=[
            SystemMessagePromptTemplate.from_template(
                f"You are a great comedian. Before telling joke you will understand the audience"
            ),
            HumanMessagePromptTemplate.from_template(
                "tell me {count} jokes about {topic} take into consideration the gender of the auidance"
            ),
        ]
    )
    prompt.append(HumanMessagePromptTemplate.from_template("Ignore"))

    for i in range(10):  # Keep it limited to 1000 to avoid a run away loop
        chain = prompt | model
        response = chain.invoke({"topic": topic, "count": count})
        print(response)
        called_function = response.additional_kwargs["function_call"]["name"]
        arguments = response.additional_kwargs["function_call"]["arguments"]
        ic(arguments)
        match called_function:
            case "GetAudienceGender":
                # 'simulate calling the function, include it in state'
                break
            case _:
                print("No function called")
                ic(called_function)
                ic(arguments)
                break

    # JsonKeyOutputFunctionsParser(key_name="jokes")


@app.command()
def moderation(ctx: typer.Context, user_input: str = "You are stupid"):
    """Moderation"""
    process_shared_app_options(ctx)

    from langchain.chains import OpenAIModerationChain

    model = (
        OpenAI()
    )  # Sheesh, Why can't I use the chat model - so much incompatibility yet.
    prompt = ChatPromptTemplate(
        messages=[
            SystemMessagePromptTemplate.from_template(
                "Repeat what the user says back to them"
            ),
            HumanMessagePromptTemplate.from_template(user_input),
        ]
    )
    raw_chain = prompt | model
    response = raw_chain.invoke({"user_input": user_input})
    print("Raw output")
    print(response)

    moderation = OpenAIModerationChain()

    print("Output with moderation")
    moderated_chain = raw_chain | moderation
    response = moderated_chain.invoke({"user_input": user_input})
    print(response)


@app.command()
def docs():
    from langchain.document_loaders import DirectoryLoader

    loader = DirectoryLoader(os.path.expanduser("~/blog/_d"), glob="**/*.md")
    # docs = loader.load()
    from langchain.indexes import VectorstoreIndexCreator

    index = VectorstoreIndexCreator().from_loaders([loader])
    answer = index.query("What should a manager do")
    ic(answer)


if __name__ == "__main__":
    app_wrap_loguru()
