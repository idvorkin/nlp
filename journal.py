#!python3

import openai
import typer
import ell
import rich
import os
from pathlib import Path
from icecream import ic

app = typer.Typer(no_args_is_help=True)

# Create custom client with Gemini endpoint
client = openai.Client(
    base_url="https://generativelanguage.googleapis.com/v1beta/openai/",
    api_key=os.environ["GOOGLE_API_KEY"],
)

model_to_use = "gemini-2.0-flash-exp"
# model_to_use = "gemini-1.5-pro-002"
# Register the model with your custom client
ell.config.register_model(model_to_use, client)


@ell.simple(model=model_to_use, temperature=0.7)
def prompt_hello():
    return "Hello world tell me a joke"


# Annoying, ell can't take a base64 input of a file, lets use gemini raw for that
gemini_prompt = """
You are an expert at transcribing handwritten text from PDFs. Your goal is to produce an accurate, well-formatted, and insightful transcription.

**Transcription Guidelines:**

*   **Accuracy:** Prioritize accuracy in transcribing the handwritten text.
*   **Error Correction:** When you see spelling errors, correct them. If you're low on confidence, correct them and insert [low confidence] after the word.
*   **Formatting:** Maintain the original formatting as closely as possible, including:
    *   Bullet points and numbering.
    *   Tables (use Markdown table format, e.g., `| Column 1 | Column 2 |`).
    *   Smart line wrapping at approximately 120 characters, merging multiple lines of the same paragraph into a single line.
*   **Uncertainty:**
    *   Indicate uncertain words with `[guess: word]`.
    *   Indicate illegible sections with `[illegible]`.
*   **Page Breaks:** Insert clear page breaks using the following format:

    ```
    ---
    Page: X of N
    ---
    ```

*   **Dates:** If a date is present on the first line of a page, include it in the page header (e.g., `--- Page 1 of 10 - 2024-12-20 ---`).
*   **Corrections:** Correct obvious spelling errors and expand abbreviations where you have high confidence.
*   **Headings and Lists:** Identify and preserve headings and lists, maintaining their hierarchical structure.
*   **Acronyms:** Use the following expansions:
    *   YAB = Yesterday Awesome Because
    *   TAB = Today Awesome Because
    *   P&P = Pullups and PIstols
    *   S&S = Simple and Sinister (Swings and TGUs)
    *   KB = Kettebells
    *   TGU = Turkish Get Up
    *   PSC = Perfromance Summary Cycle = Calibrations at Meta

**Analysis Section (at the end of the transcription):**

Provide a comprehensive analysis including:

*   **Summary:** A brief summary of the document's content.
*   **Key Insights:** Any significant observations or interpretations.
*   **Action Items:** A consolidated list of tasks and actions that need to be completed, including those marked with `[]` in the original text. Format these as a numbered list.
*   **Coalesced Lists:** Combine any recurring items or lists scattered throughout the document into unified lists. For example:

    ```
    YAB: blah
    other text
    YAB: Bloop

    Becomes:

    Yesterday Awesome Because:
    1. blah
    2. Bloop
    ```

*   **Expanded Acronyms:** List any acronyms you expanded during transcription, this helps with error checking.
*   **Proper Nouns:** List all proper nouns mentioned in the document. This helps with error checking.

**Example Table Formatting:**

```
| Column 1     | Column 2 | Column 3     |
|--------------|----------|--------------|
| A            | B        | C            |
| Hello World  | 123      | 456          |
```

By following these guidelines, you will provide a high-quality transcription and analysis of the handwritten document.
    """

# model_to_use = "gemini-1.5-flash-002"
# model_to_use = "gemini-1.5-pro-002"


def gemini_transcribe(pdf_path: str):
    import vertexai
    from vertexai.generative_models import GenerativeModel, Part

    pdf_data = Part.from_data(
        mime_type="application/pdf", data=Path(pdf_path).read_bytes()
    )

    generation_config = {
        "max_output_tokens": 8192,
        "temperature": 1,
        "top_p": 0.95,
    }
    vertexai.init(project="tranquil-hawk-325816", location="us-central1")
    model = GenerativeModel(model_name=model_to_use)
    chat = model.start_chat()
    ic("starting", model_to_use)
    response = chat.send_message(
        [gemini_prompt, pdf_data], generation_config=generation_config
    )
    ic(response.usage_metadata)
    return response.text


@app.command()
def transcribe(pdf: str = typer.Argument(..., help="Path to pdf file to transcribe")):
    """Transcribe handwritten text from an image file"""
    # Expand user path if needed
    full_path = os.path.expanduser(pdf)

    if not os.path.exists(full_path):
        rich.print(f"[red]Error: File not found: {full_path}")
        raise typer.Exit(1)

    # NOTE: I'm using gemini because I can pass the PDF inline without needing to manage filestorage
    # Though, perhaps that's not a big deal.
    response = gemini_transcribe(full_path)
    rich.print(response)


if __name__ == "__main__":
    app()
