#!python3

import typer
import rich
import os
import tempfile
import requests
from pathlib import Path
from icecream import ic
from urllib.parse import urlparse
import langchain_helper

app = typer.Typer(no_args_is_help=True)

# Use the correct Gemini model name directly
model_to_use = "gemini-2.5-pro-exp-03-25"


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
   - **Lists**: Pay special attention to YAB (Yesterday Awesome Because) and Grateful lists
     - Format each YAB entry on a new line with a bullet point
     - Format each Grateful entry on a new line with a bullet point
     - Preserve the exact wording of each entry
   - **Bullets & Numbering**: Preserve bullet points and use sequential numbering (1, 2, 3, etc.) for numbered lists.
   - **Tables**: Present tabular data in [Markdown table format].
   - **Line Wrapping**: Use ~120 characters per line. Merge multiple short lines from the same paragraph into one line.

4. **Uncertainty**
   - Mark any unclear or guessed words with `[guess: <word>]`.
   - Mark truly illegible sections as `[illegible]`.

5. **Page Breaks**
   - Check PAGE_BREAKS parameter at start of transcription
   - If PAGE_BREAKS is true:
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
   - If PAGE_BREAKS is false:
     - Do not include any page break markers
     - Keep content as continuous text
     - Still preserve paragraph breaks and section structure

6. **Headings and Lists**
   - Preserve headings and subheadings wherever they appear.
   - Keep bullet and numbered lists intact, respecting the hierarchical structure.

7. **Gratefulness List**
    - Sometimes a gratefulness line spans multiple lines.
    - The format is a shorhand to be expanded, user denotes fields with ;'s e.g.
        - Thing I'm grateful for; I'm grateful to god for this because; I'm grateful to this person; I'm grateful to them because; I'm Grateful to me because;
        - E.g.
            - It's Warm out; Makes it a sunny day; Mom; Giving birth to me; Enjoying it by going for a walk.
        - Sometimes the line will span multiple lines


7. **Acronyms**
   - Expand the following acronyms whenever they appear:
     - **YAB** → **Yesterday Awesome Because**
     - **TAB** → **Today Awesome Because**
     - **P&P** → **Pullups and Pistols**
     - **GTG** → **Grease the groove**
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
   - The summary should ignore the YAB, Grateful Analysis and Affirmations

2. **Key Insights**
   - Noteworthy observations or interpretations.
   - If any Psychic Weight (PW) items are mentioned, highlight these specifically.
   - The isights should ignore the YAB, Grateful Analysis and Affirmations

3. **YAB and Grateful Analysis**
   - List all YAB (Yesterday Awesome Because) entries in their original order
   - List all Grateful entries in their original order
   - Preserve the exact wording and sequence of each entry
   - Do not group or categorize the entries
   - Include any context or dates associated with the entries

4. **Action Items**
   - A numbered list of tasks or follow-ups—especially those denoted by `[]` in the original text.
   - When listing them, list them out with a ☐ if they need to be done, or a ☑ if completed

5. **Psychic Weight Items**
   - List all Psychic Weight (PW) items mentioned in the document
   - The isights should ignore the YAB, Grateful Analysis and Affirmations
   - For each PW item, include:
     - The context it was mentioned in
     - Current status or resolution (if mentioned)
     - Any related action items or dependencies

6. **Coalesced Lists**
   - If certain items or lists (e.g., repeated YAB or TAB entries) appear multiple times, merge them into a single consolidated list.
   - Use sequential numbering (1, 2, 3, etc.) for all numbered lists.

7. **Expanded Acronyms**
   - List any acronyms you expanded in the transcription (for verification).

8. **Proper Nouns**
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


def gemini_transcribe(pdf_path: str, page_breaks: bool = False):
    import google.generativeai as genai
    from google.generativeai.types import HarmCategory, HarmBlockThreshold

    try:
        # Configure the API key
        genai.configure(api_key=os.environ["GOOGLE_API_KEY"])

        # Read PDF file
        pdf_data = Path(pdf_path).read_bytes()

        # Add page_breaks parameter to prompt
        prompt = gemini_prompt + f"\nPAGE_BREAKS: {'true' if page_breaks else 'false'}"

        # Configure generation parameters
        generation_config = {
            "max_output_tokens": 8192,
            "temperature": 1,
            "top_p": 0.95,
        }

        # Initialize the model with safety filters disabled
        model = genai.GenerativeModel(
            model_name=model_to_use,
            generation_config=generation_config,
            safety_settings=None
        )

        # Create a multipart content with text and PDF
        contents = [
            {"text": prompt},
            {"inline_data": {"mime_type": "application/pdf", "data": pdf_data}}
        ]

        # Generate response
        ic("starting", model_to_use)
        response = model.generate_content(contents)

        # Log usage metadata if available
        if hasattr(response, 'usage_metadata'):
            ic(response.usage_metadata)

        return response.text
    except Exception as e:
        ic(f"Error during transcription: {str(e)}")
        raise

@app.command()
def transcribe(
    pdf: str = typer.Argument(..., help="Path or URL to pdf file to transcribe"),
    page_breaks: bool = typer.Option(
        False, "--page-breaks", help="Include page breaks in output"
    ),
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
            response = gemini_transcribe(full_path, page_breaks)
            # File will be automatically deleted when the with block exits
    else:
        # Handle local file as before
        full_path = os.path.expanduser(pdf)
        if not os.path.exists(full_path):
            rich.print(f"[red]Error: File not found: {full_path}")
            raise typer.Exit(1)
        response = gemini_transcribe(full_path, page_breaks)

    # Print to console
    print(response)

    # Also write to ~/tmp/journal.md
    output_path = Path.home() / "tmp" / "journal.md"
    output_path.write_text(response)


if __name__ == "__main__":
    app()
