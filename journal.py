#!python3

import openai
import typer
import ell
import rich
import os
import tempfile
import requests
from pathlib import Path
from icecream import ic
from urllib.parse import urlparse

app = typer.Typer(no_args_is_help=True)

# Create custom client with Gemini endpoint
client = openai.Client(
    base_url="https://generativelanguage.googleapis.com/v1beta/openai/",
    api_key=os.environ["GOOGLE_API_KEY"],
)

model_to_use = "gemini-2.0-pro-exp-02-05"
# model_to_use = "gemini-2.0-flash"
# model_to_use = "gemini-1.5-pro-002"
# Register the model with your custom client
ell.config.register_model(model_to_use, client)


@ell.simple(model=model_to_use, temperature=0.7)
def prompt_hello():
    return "Hello world tell me a joke"


# Annoying, ell can't take a base64 input of a file, lets use gemini raw for that
gemini_prompt = """
### **1. Context**
You are a **professional archivist and transcription specialist** with years of experience decoding handwritten text. Your expertise includes accurately transcribing challenging handwriting from PDF documents, preserving the original structure, and providing deeper analysis of the text's meaning.

---

### **2. Goal**
- **Primary Objective**: Deliver an **accurate**, **well-formatted**, and **insightful** transcription of a PDF containing handwritten text.
- **Secondary Objective**: Provide a concise analysis highlighting the document's key points, tasks, and any potential ambiguities.

---

### **3. Transcription Guidelines**
Adhere to these rules meticulously:

1. **Accuracy**
   - Prioritize precise transcription of every word.
   - If you spot a likely misspelling, correct it. In cases where you feel less confident, correct it and append `[low confidence]` afterward.

2. **Error Correction**
   - Correct evident spelling errors.
   - Expand abbreviations only when highly confident of the intended meaning.

3. **Formatting**
   - **Bullets & Numbering**: Preserve bullet points and use sequential numbering (1, 2, 3, etc.) for numbered lists.
   - **Tables**: Present tabular data in [Markdown table format](https://www.google.com/search?q=how+to+create+markdown+tables). For example:
     ```
     | Column 1     | Column 2 | Column 3     |
     |--------------|----------|--------------|
     | A            | B        | C            |
     | Hello World  | 123      | 456          |
     ```
   - **Line Wrapping**: Use ~120 characters per line. Merge multiple short lines from the same paragraph into one line.
   - **Tables Where Possible**: If a section of text naturally fits a tabular layout, convert it into a Markdown table.

4. **Uncertainty**
   - Mark any unclear or guessed words with `[guess: <word>]`.
   - Mark truly illegible sections as `[illegible]`.

5. **Page Breaks**
   - Insert clear page breaks using:
     ```
     ---
     Page: X of N
     ---
     ```
   - If a date appears on the first line of a new page, incorporate it in the page header. Example:
     ```
     --- Page 1 of 10 - 2024-12-20 ---
     ```

6. **Headings and Lists**
   - Preserve headings and subheadings wherever they appear.
   - Keep bullet and numbered lists intact, respecting the hierarchical structure.

7. **Acronyms**
   - Expand the following acronyms whenever they appear:
     - **YAB** → **Yesterday Awesome Because**
     - **TAB** → **Today Awesome Because**
     - **P&P** → **Pullups and Pistols**
     - **S&S** → **Simple and Sinister (Swings and TGUs)**
     - **KB** → **Kettlebells**
     - **TGU** → **Turkish Get Up**
     - **PSC** → **Performance Summary Cycle = Calibrations at Meta**
     - **PW** → **Psychic Weight**
   - Do not expand the following acronyms:
     - **CHOP** → Chat Oriented Programming
     - **CHOW** → Chat Oriented Writing
     - **CHOLI** → Chat Oriented Life Insights

---

### **4. Analysis Section (End of Transcription)**
After the transcription, provide a **comprehensive analysis**:

1. **Summary**
   - A short paragraph summarizing the overall content of the document.

2. **Key Insights**
   - Noteworthy observations or interpretations.
   - If any Psychic Weight (PW) items are mentioned, highlight these specifically as they represent mental burdens or tasks weighing on the mind.

3. **Action Items**
   - A numbered list of tasks or follow-ups—especially those denoted by `[]` in the original text.
   - When listing them, list them out with a ☐ if they need to be done, or a ☑ if completed

4. **Psychic Weight Items**
   - List all Psychic Weight (PW) items mentioned in the document
   - For each PW item, include:
     - The context it was mentioned in
     - Current status or resolution (if mentioned)
     - Any related action items or dependencies

5. **Coalesced Lists**
   - If certain items or lists (e.g., repeated YAB or TAB entries) appear multiple times, merge them into a single consolidated list.
   - Use sequential numbering (1, 2, 3, etc.) for all numbered lists.

6. **Expanded Acronyms**
   - List any acronyms you expanded in the transcription (for verification).

7. **Proper Nouns**
   - List any proper nouns identified in the document.

---

### **5. Final Output Format**
1. **Transcription** (following all formatting and accuracy guidelines).
2. **Analysis Section** (using the headings described above).

---

### **6. Example of Expected Transcription Snippet**
```
1. [Bullet Point]
   - Sub-bullet: Additional details

| Task          | Due Date       | Priority |
|---------------|----------------|----------|
| Draft Report  | 2024-12-20     | High     |
| Review Budget | [illegible]    | Low      |

[guess: uncertainWord]

---
Page: 2 of 5 - 2024-12-21
---

Heading Level 2
1. 1. 1.
```

**Analysis Section**
- **Summary**: A concise overview of the document.
- **Key Insights**: Observations…
- **Action Items**:
  1. Item from `[]` in the text
  2. Another item from `[]` in the text

…and so on.
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
def transcribe(
    pdf: str = typer.Argument(..., help="Path or URL to pdf file to transcribe"),
):
    """Transcribe handwritten text from a PDF file or URL"""

    # Check if input is a URL
    parsed = urlparse(pdf)
    is_url = bool(parsed.scheme and parsed.netloc)

    if is_url:
        # Create a temporary file
        with tempfile.NamedTemporaryFile(suffix=".pdf", delete=True) as tmp_file:
            # Download the file
            response = requests.get(pdf)
            response.raise_for_status()  # Raise exception for bad status codes

            # Write to temporary file
            tmp_file.write(response.content)
            tmp_file.flush()

            # Use the temporary file path
            full_path = tmp_file.name
            response = gemini_transcribe(full_path)
            # File will be automatically deleted when the with block exits
    else:
        # Handle local file as before
        full_path = os.path.expanduser(pdf)
        if not os.path.exists(full_path):
            rich.print(f"[red]Error: File not found: {full_path}")
            raise typer.Exit(1)
        response = gemini_transcribe(full_path)

    rich.print(response)


if __name__ == "__main__":
    app()
