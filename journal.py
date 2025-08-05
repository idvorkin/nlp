#!/usr/bin/env -S uv run --script
# /// script
# dependencies = [
#   "typer",
#   "rich",
#   "requests",
#   "icecream",
#   "google-generativeai",
#   "pymupdf",
#   "validators",
#   "typing-extensions"
# ]
# ///

"""
Journal PDF Transcription Tool

Transcribes handwritten journal PDFs using Google's Gemini AI with support for
parallel processing of large documents.
"""

import os
import tempfile
import logging
from pathlib import Path
from dataclasses import dataclass
from typing import List, Optional, Dict, Any
from concurrent.futures import ThreadPoolExecutor, as_completed
from contextlib import contextmanager
from urllib.parse import urlparse

import typer
import requests
import validators
import fitz  # PyMuPDF
import google.generativeai as genai
from rich.console import Console
from rich.logging import RichHandler
from typing_extensions import Annotated

# ===========================
# Configuration and Constants
# ===========================

DEFAULT_PAGES_PER_CHUNK = 10
DEFAULT_MAX_WORKERS = 5
DEFAULT_TIMEOUT = 600
DEFAULT_MAX_OUTPUT_TOKENS = (
    60000  # Conservative limit for Gemini 2.5 Pro (max is 65536)
)
DEFAULT_MODEL = "gemini-2.5-pro"
DEFAULT_OUTPUT_PATH = Path.home() / "tmp" / "journal.md"


@dataclass
class TranscriptionConfig:
    """Configuration for transcription service"""

    model_name: str = DEFAULT_MODEL
    max_output_tokens: int = DEFAULT_MAX_OUTPUT_TOKENS
    temperature: float = 1.0
    top_p: float = 0.95
    timeout_seconds: int = DEFAULT_TIMEOUT
    pages_per_chunk: int = DEFAULT_PAGES_PER_CHUNK
    max_workers: int = DEFAULT_MAX_WORKERS

    @classmethod
    def from_env(cls) -> "TranscriptionConfig":
        """Load configuration from environment variables"""
        return cls(
            model_name=os.environ.get("GEMINI_MODEL", DEFAULT_MODEL),
            timeout_seconds=int(os.environ.get("API_TIMEOUT", str(DEFAULT_TIMEOUT))),
            pages_per_chunk=int(
                os.environ.get("PAGES_PER_CHUNK", str(DEFAULT_PAGES_PER_CHUNK))
            ),
            max_workers=int(os.environ.get("MAX_WORKERS", str(DEFAULT_MAX_WORKERS))),
        )

    @property
    def generation_config(self) -> Dict[str, Any]:
        """Get generation config for Gemini API"""
        return {
            "max_output_tokens": self.max_output_tokens,
            "temperature": self.temperature,
            "top_p": self.top_p,
        }


# ===========================
# Custom Exceptions
# ===========================


class TranscriptionError(Exception):
    """Base exception for transcription errors"""


class PDFProcessingError(TranscriptionError):
    """Error processing PDF file"""


class APIError(TranscriptionError):
    """Error calling external API"""


class FileNotFoundError(TranscriptionError):
    """File not found or inaccessible"""


class InvalidURLError(TranscriptionError):
    """Invalid URL provided"""


# ===========================
# Logging Setup
# ===========================


def setup_logging(level: str = "INFO") -> logging.Logger:
    """Setup structured logging with Rich handler"""
    logging.basicConfig(
        level=level,
        format="%(message)s",
        datefmt="[%X]",
        handlers=[RichHandler(rich_tracebacks=True)],
    )
    return logging.getLogger("journal")


logger = setup_logging()
console = Console()

# ===========================
# Prompts
# ===========================

TRANSCRIPTION_PROMPT = """
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

ANALYSIS_PROMPT = """
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

# ===========================
# PDF Processing
# ===========================


@dataclass
class PDFChunk:
    """Represents a chunk of PDF pages"""

    start_page: int
    end_page: int
    data: bytes


class PDFProcessor:
    """Handles PDF file operations"""

    def __init__(self, config: TranscriptionConfig):
        self.config = config

    @contextmanager
    def open_pdf(self, pdf_path: str):
        """Context manager for PDF documents"""
        doc = None
        try:
            doc = fitz.open(pdf_path)
            yield doc
        except Exception as e:
            raise PDFProcessingError(f"Failed to open PDF: {e}")
        finally:
            if doc:
                doc.close()

    def get_page_count(self, pdf_path: str) -> int:
        """Get the number of pages in a PDF"""
        with self.open_pdf(pdf_path) as doc:
            return len(doc)

    def should_split(self, pdf_path: str) -> bool:
        """Determine if PDF should be split into chunks"""
        return self.get_page_count(pdf_path) > self.config.pages_per_chunk

    def split_into_chunks(self, pdf_path: str) -> List[PDFChunk]:
        """Split a PDF into chunks of specified pages"""
        chunks = []

        with self.open_pdf(pdf_path) as doc:
            total_pages = len(doc)

            for start_page in range(0, total_pages, self.config.pages_per_chunk):
                end_page = min(start_page + self.config.pages_per_chunk, total_pages)

                # Create a new PDF with just these pages
                new_doc = fitz.open()
                for page_num in range(start_page, end_page):
                    new_doc.insert_pdf(doc, from_page=page_num, to_page=page_num)

                # Convert to bytes
                pdf_bytes = new_doc.tobytes()
                chunks.append(
                    PDFChunk(
                        start_page=start_page + 1,  # 1-indexed for display
                        end_page=end_page,
                        data=pdf_bytes,
                    )
                )
                new_doc.close()

        return chunks

    def read_pdf_bytes(self, pdf_path: str) -> bytes:
        """Read entire PDF as bytes"""
        return Path(pdf_path).read_bytes()


# ===========================
# Gemini AI Integration
# ===========================


class GeminiTranscriber:
    """Handles transcription using Gemini AI"""

    def __init__(self, config: TranscriptionConfig):
        self.config = config
        self._configure_api()

    def _configure_api(self):
        """Configure Gemini API with API key"""
        api_key = os.environ.get("GOOGLE_API_KEY")
        if not api_key:
            raise APIError("GOOGLE_API_KEY environment variable is required")
        genai.configure(api_key=api_key)

    def _create_model(self) -> genai.GenerativeModel:
        """Create and configure Gemini model instance"""
        return genai.GenerativeModel(
            model_name=self.config.model_name,
            generation_config=self.config.generation_config,
            safety_settings=None,
        )

    def _generate_content(self, prompt: str, pdf_data: Optional[bytes] = None) -> str:
        """Common method for generating content with Gemini"""
        model = self._create_model()

        if pdf_data:
            contents = [
                {"text": prompt},
                {"inline_data": {"mime_type": "application/pdf", "data": pdf_data}},
            ]
        else:
            contents = prompt

        try:
            response = model.generate_content(
                contents, request_options={"timeout": self.config.timeout_seconds}
            )
            return response.text
        except Exception as e:
            raise APIError(f"Gemini API error: {e}")

    def transcribe_chunk(self, chunk: PDFChunk, page_breaks: bool = False) -> str:
        """Transcribe a single PDF chunk"""
        prompt = (
            TRANSCRIPTION_PROMPT
            + f"\nPAGE_BREAKS: {'true' if page_breaks else 'false'}"
        )
        prompt += f"\n\nNOTE: This is pages {chunk.start_page} to {chunk.end_page} of a larger document."

        console.print(
            f"[yellow]Transcribing pages {chunk.start_page}-{chunk.end_page}...[/yellow]"
        )

        try:
            return self._generate_content(prompt, chunk.data)
        except Exception as e:
            logger.error(
                f"Error transcribing pages {chunk.start_page}-{chunk.end_page}: {e}"
            )
            return f"[Error transcribing pages {chunk.start_page}-{chunk.end_page}]"

    def transcribe_full_pdf(self, pdf_data: bytes, page_breaks: bool = False) -> str:
        """Transcribe an entire PDF without splitting"""
        prompt = (
            TRANSCRIPTION_PROMPT
            + f"\nPAGE_BREAKS: {'true' if page_breaks else 'false'}"
        )
        return self._generate_content(prompt, pdf_data)

    def analyze_transcription(self, transcription: str) -> str:
        """Run analysis on the complete transcription"""
        prompt = (
            ANALYSIS_PROMPT + "\n\n### Transcription to analyze:\n\n" + transcription
        )

        console.print("[yellow]Running analysis on complete transcription...[/yellow]")

        try:
            return self._generate_content(prompt)
        except Exception as e:
            logger.error(f"Error during analysis: {e}")
            return f"[Error during analysis: {e}]"


# ===========================
# Main Transcription Service
# ===========================


class TranscriptionService:
    """Main service orchestrating PDF transcription"""

    def __init__(self, config: TranscriptionConfig):
        self.config = config
        self.pdf_processor = PDFProcessor(config)
        self.transcriber = GeminiTranscriber(config)

    def _calculate_optimal_workers(self, chunk_count: int) -> int:
        """Calculate optimal thread pool size based on workload"""
        max_workers = min(
            chunk_count,
            os.cpu_count() or 1,
            self.config.max_workers,  # API rate limiting consideration
        )
        return max(1, max_workers)

    def _process_chunks_parallel(
        self, chunks: List[PDFChunk], page_breaks: bool
    ) -> str:
        """Process PDF chunks in parallel"""
        transcriptions = []
        worker_count = self._calculate_optimal_workers(len(chunks))

        with ThreadPoolExecutor(max_workers=worker_count) as executor:
            # Submit all chunks for processing
            future_to_chunk = {
                executor.submit(
                    self.transcriber.transcribe_chunk, chunk, page_breaks
                ): chunk
                for chunk in chunks
            }

            # Collect results as they complete
            for future in as_completed(future_to_chunk):
                chunk = future_to_chunk[future]
                try:
                    transcription = future.result()
                    transcriptions.append((chunk.start_page, transcription))
                    console.print(
                        f"[green]✓ Completed pages {chunk.start_page}-{chunk.end_page}[/green]"
                    )
                except Exception as e:
                    logger.error(f"Failed to process chunk: {e}")
                    transcriptions.append(
                        (chunk.start_page, f"[Error processing chunk: {e}]")
                    )

        # Sort by page number and combine
        transcriptions.sort(key=lambda x: x[0])
        return "\n\n".join([t[1] for t in transcriptions])

    def transcribe_pdf(self, pdf_path: str, page_breaks: bool = False) -> str:
        """Main method to transcribe a PDF file"""
        # Check PDF page count
        page_count = self.pdf_processor.get_page_count(pdf_path)
        console.print(f"[cyan]PDF has {page_count} pages[/cyan]")

        if self.pdf_processor.should_split(pdf_path):
            # Large PDF, split and process in parallel
            console.print(
                f"[green]Splitting into chunks of {self.config.pages_per_chunk} pages each...[/green]"
            )

            chunks = self.pdf_processor.split_into_chunks(pdf_path)
            transcription = self._process_chunks_parallel(chunks, page_breaks)
        else:
            # Small PDF, process as single document
            console.print("[green]Processing as single document...[/green]")
            pdf_data = self.pdf_processor.read_pdf_bytes(pdf_path)
            transcription = self.transcriber.transcribe_full_pdf(pdf_data, page_breaks)

        # Run analysis on complete transcription
        console.print("[cyan]All pages transcribed, running analysis...[/cyan]")
        analysis = self.transcriber.analyze_transcription(transcription)

        return f"{transcription}\n\n---\n\n## Analysis\n\n{analysis}"


# ===========================
# URL Handling
# ===========================


def validate_url(url: str) -> None:
    """Validate that URL is safe and properly formatted"""
    if not validators.url(url):
        raise InvalidURLError(f"Invalid URL format: {url}")

    parsed = urlparse(url)
    if parsed.scheme not in ["http", "https"]:
        raise InvalidURLError(f"Only HTTP/HTTPS URLs allowed: {url}")


def download_pdf_from_url(url: str) -> str:
    """Download PDF from URL and return temporary file path"""
    validate_url(url)

    try:
        response = requests.get(url, timeout=30)
        response.raise_for_status()
    except requests.exceptions.HTTPError as e:
        if response.status_code == 404:
            raise FileNotFoundError(f"PDF not found at URL: {url}")
        elif response.status_code == 403:
            raise APIError(f"Access denied to PDF: {url}")
        else:
            raise APIError(f"HTTP error {response.status_code}: {e}")
    except requests.exceptions.RequestException as e:
        raise APIError(f"Network error downloading PDF: {e}")

    # Create temporary file
    with tempfile.NamedTemporaryFile(suffix=".pdf", delete=False) as tmp_file:
        tmp_file.write(response.content)
        return tmp_file.name


# ===========================
# Output Handling
# ===========================


def ensure_output_directory(output_path: Path) -> None:
    """Ensure output directory exists"""
    output_path.parent.mkdir(parents=True, exist_ok=True)


def write_transcription_output(
    content: str, output_path: Optional[Path] = None
) -> Path:
    """Write transcription to file and return the path used"""
    if output_path is None:
        output_path = DEFAULT_OUTPUT_PATH

    ensure_output_directory(output_path)
    output_path.write_text(content, encoding="utf-8")

    console.print(f"[green]✓ Transcription saved to: {output_path}[/green]")
    return output_path


# ===========================
# CLI Application
# ===========================

app = typer.Typer(
    help="Transcribe handwritten journal PDFs using Gemini AI",
    add_completion=False,
    no_args_is_help=True,
)


@app.command(help="Transcribe a PDF file or URL")
def transcribe(
    pdf: Annotated[str, typer.Argument(help="Path or URL to PDF file")],
    page_breaks: Annotated[
        bool,
        typer.Option(
            "--page-breaks", help="Include page breaks in transcription output"
        ),
    ] = False,
    output: Annotated[
        Optional[Path],
        typer.Option(
            "--output", "-o", help="Output file path (default: ~/tmp/journal.md)"
        ),
    ] = None,
    verbose: Annotated[
        bool, typer.Option("--verbose", "-v", help="Enable verbose logging")
    ] = False,
    pages_per_chunk: Annotated[
        Optional[int],
        typer.Option("--chunk-size", help="Pages per chunk for parallel processing"),
    ] = None,
):
    """Transcribe handwritten text from a PDF file or URL."""
    if verbose:
        setup_logging("DEBUG")

    # Load configuration
    config = TranscriptionConfig.from_env()
    if pages_per_chunk:
        config.pages_per_chunk = pages_per_chunk

    # Initialize service
    service = TranscriptionService(config)

    # Check if input is URL
    parsed = urlparse(pdf)
    is_url = bool(parsed.scheme and parsed.netloc)

    temp_file = None
    try:
        if is_url:
            logger.info(f"Downloading PDF from URL: {pdf}")
            pdf_path = download_pdf_from_url(pdf)
            temp_file = pdf_path
        else:
            # Handle local file
            pdf_path = os.path.expanduser(pdf)
            if not os.path.exists(pdf_path):
                raise FileNotFoundError(f"PDF file not found: {pdf_path}")

        # Transcribe PDF
        result = service.transcribe_pdf(pdf_path, page_breaks)

        # Write output
        write_transcription_output(result, output)

        # Also print to console
        console.print(result)

    except TranscriptionError as e:
        console.print(f"[red]Error: {e}[/red]")
        raise typer.Exit(1)
    except Exception as e:
        logger.exception("Unexpected error")
        console.print(f"[red]Unexpected error: {e}[/red]")
        raise typer.Exit(1)
    finally:
        # Clean up temporary file if created
        if temp_file and os.path.exists(temp_file):
            os.unlink(temp_file)


if __name__ == "__main__":
    app()
