#!/usr/bin/env -S uv run --script
# /// script
# dependencies = [
#   "typer",
#   "rich",
#   "requests",
#   "icecream",
#   "google-generativeai",
#   "pymupdf"
# ]
# ///

import typer
import rich
import os
import tempfile
import requests
from pathlib import Path
from icecream import ic
from urllib.parse import urlparse
import fitz  # PyMuPDF for PDF manipulation
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import List, Tuple

app = typer.Typer(no_args_is_help=True)

# Use the correct Gemini model name directly
model_to_use = "gemini-2.5-pro"


# Transcription prompt (without analysis)
transcription_prompt = """
### **1. Context**
You are a **professional archivist and transcription specialist** with years of experience decoding handwritten text. Your expertise includes accurately transcribing challenging handwriting from PDF documents, preserving the original structure, and providing deeper analysis of the text's meaning.

---

### **2. Goal**
- **Primary Objective**: Deliver an **accurate** and **well-formatted** transcription of a PDF containing handwritten text.

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

### **4. Final Output Format**
**Transcription only** (following all formatting and accuracy guidelines).

---

### **5. Example of Expected Transcription Snippet**
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
"""

# Analysis prompt (runs on complete transcription)
analysis_prompt = """
You are analyzing a transcribed journal document. Based on the complete transcription provided, create a comprehensive analysis.

### **Analysis Section Requirements**

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
   - List any acronyms that were expanded in the transcription (for verification).

8. **Proper Nouns**
   - List any proper nouns identified in the document.
"""


def split_pdf_into_chunks(pdf_path: str, pages_per_chunk: int = 10) -> List[bytes]:
    """Split a PDF into chunks of specified pages."""
    doc = fitz.open(pdf_path)
    chunks = []

    total_pages = len(doc)
    for start_page in range(0, total_pages, pages_per_chunk):
        end_page = min(start_page + pages_per_chunk, total_pages)

        # Create a new PDF with just these pages
        new_doc = fitz.open()
        for page_num in range(start_page, end_page):
            new_doc.insert_pdf(doc, from_page=page_num, to_page=page_num)

        # Convert to bytes
        pdf_bytes = new_doc.tobytes()
        chunks.append((start_page + 1, end_page, pdf_bytes))  # 1-indexed for display
        new_doc.close()

    doc.close()
    return chunks


def transcribe_chunk(
    chunk_data: Tuple[int, int, bytes], page_breaks: bool = False
) -> Tuple[int, int, str]:
    """Transcribe a single PDF chunk."""
    start_page, end_page, pdf_bytes = chunk_data
    import google.generativeai as genai

    try:
        # Configure the API key
        genai.configure(api_key=os.environ["GOOGLE_API_KEY"])

        # Add page_breaks parameter to prompt
        prompt = (
            transcription_prompt
            + f"\nPAGE_BREAKS: {'true' if page_breaks else 'false'}"
        )
        prompt += (
            f"\n\nNOTE: This is pages {start_page} to {end_page} of a larger document."
        )

        # Configure generation parameters
        generation_config = {
            "max_output_tokens": 8192,
            "temperature": 1,
            "top_p": 0.95,
        }

        # Initialize the model
        model = genai.GenerativeModel(
            model_name=model_to_use,
            generation_config=generation_config,
            safety_settings=None,
        )

        # Create content with PDF bytes
        contents = [
            {"text": prompt},
            {"inline_data": {"mime_type": "application/pdf", "data": pdf_bytes}},
        ]

        # Generate response
        rich.print(f"[yellow]Transcribing pages {start_page}-{end_page}...[/yellow]")
        response = model.generate_content(contents)

        return (start_page, end_page, response.text)
    except Exception as e:
        rich.print(
            f"[red]Error transcribing pages {start_page}-{end_page}: {str(e)}[/red]"
        )
        return (
            start_page,
            end_page,
            f"[Error transcribing pages {start_page}-{end_page}]",
        )


def analyze_transcription(transcription: str) -> str:
    """Run analysis on the complete transcription."""
    import google.generativeai as genai

    try:
        # Configure the API key
        genai.configure(api_key=os.environ["GOOGLE_API_KEY"])

        # Configure generation parameters
        generation_config = {
            "max_output_tokens": 8192,
            "temperature": 1,
            "top_p": 0.95,
        }

        # Initialize the model
        model = genai.GenerativeModel(
            model_name=model_to_use,
            generation_config=generation_config,
            safety_settings=None,
        )

        # Create content
        prompt = (
            analysis_prompt + "\n\n### Transcription to analyze:\n\n" + transcription
        )

        # Generate response
        rich.print("[yellow]Running analysis on complete transcription...[/yellow]")
        response = model.generate_content(prompt)

        return response.text
    except Exception as e:
        rich.print(f"[red]Error during analysis: {str(e)}[/red]")
        return f"[Error during analysis: {str(e)}]"


def gemini_transcribe(pdf_path: str, page_breaks: bool = False):
    """Transcribe a PDF file, splitting into chunks if larger than 10 pages."""
    import google.generativeai as genai

    try:
        # Check PDF page count
        doc = fitz.open(pdf_path)
        page_count = len(doc)
        doc.close()

        rich.print(f"[cyan]PDF has {page_count} pages[/cyan]")

        if page_count <= 10:
            # Small PDF, process as before
            rich.print("[green]Processing as single document...[/green]")

            # Configure the API key
            genai.configure(api_key=os.environ["GOOGLE_API_KEY"])

            # Read PDF file
            pdf_data = Path(pdf_path).read_bytes()

            # Add page_breaks parameter to prompt
            prompt = (
                transcription_prompt
                + f"\nPAGE_BREAKS: {'true' if page_breaks else 'false'}"
            )

            # Configure generation parameters
            generation_config = {
                "max_output_tokens": 8192,
                "temperature": 1,
                "top_p": 0.95,
            }

            # Initialize the model
            model = genai.GenerativeModel(
                model_name=model_to_use,
                generation_config=generation_config,
                safety_settings=None,
            )

            # Create content
            contents = [
                {"text": prompt},
                {"inline_data": {"mime_type": "application/pdf", "data": pdf_data}},
            ]

            # Generate response
            response = model.generate_content(contents)
            transcription = response.text

            # Run analysis
            analysis = analyze_transcription(transcription)

            return f"{transcription}\n\n---\n\n## Analysis\n\n{analysis}"

        else:
            # Large PDF, split and process in parallel
            rich.print(
                f"[green]Splitting into {(page_count + 9) // 10} chunks of 10 pages each...[/green]"
            )

            # Split PDF into chunks
            chunks = split_pdf_into_chunks(pdf_path, pages_per_chunk=10)

            # Process chunks in parallel
            transcriptions = []
            with ThreadPoolExecutor(max_workers=5) as executor:
                # Submit all chunks for processing
                future_to_chunk = {
                    executor.submit(transcribe_chunk, chunk, page_breaks): chunk
                    for chunk in chunks
                }

                # Collect results as they complete
                for future in as_completed(future_to_chunk):
                    start_page, end_page, transcription = future.result()
                    transcriptions.append((start_page, transcription))
                    rich.print(
                        f"[green]✓ Completed pages {start_page}-{end_page}[/green]"
                    )

            # Sort by page number and combine
            transcriptions.sort(key=lambda x: x[0])
            combined_transcription = "\n\n".join([t[1] for t in transcriptions])

            rich.print("[cyan]All chunks transcribed, running analysis...[/cyan]")

            # Run analysis on complete transcription
            analysis = analyze_transcription(combined_transcription)

            return f"{combined_transcription}\n\n---\n\n## Analysis\n\n{analysis}"

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
