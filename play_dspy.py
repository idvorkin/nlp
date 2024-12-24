#!/usr/bin/env python3
from loguru import logger
import typer
from typing import Annotated
import dspy

# Create Typer app following conventions
app = typer.Typer()

@logger.catch()
def app_wrap_loguru():
    app()

@app.command()
def main(
    prompt: Annotated[str, typer.Argument(help="Prompt to test with DSPy")] = "What is machine learning?",
    model: Annotated[str, typer.Option(help="Model to use (default: gpt-3.5-turbo)")] = "gpt-3.5-turbo",
):
    """
    Play with DSPy functionality
    """
    # Initialize DSPy with the specified model
    lm = dspy.OpenAI(model=model)
    dspy.settings.configure(lm=lm)

    # Create a basic DSPy program
    class BasicQA(dspy.Module):
        def __init__(self):
            super().__init__()
            self.gen = dspy.ChainOfThought("question -> answer")

        def forward(self, question):
            return self.gen(question=question).answer

    # Run the program
    qa = BasicQA()
    result = qa(prompt)
    
    logger.info(f"Input prompt: {prompt}")
    logger.info(f"Response: {result}")

if __name__ == "__main__":
    app_wrap_loguru()
