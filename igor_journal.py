#!python3
import glob
import os
import re
import sys
from collections import defaultdict
from dataclasses import dataclass
from datetime import date, datetime, timedelta
from functools import lru_cache
from pathlib import Path
from typing import Annotated, Dict, Iterable, List
from icecream import ic

import typer
from rich.console import Console

console = Console()
app = typer.Typer()

# copied from pandas util (yuk)


@dataclass
class Measure_Helper:
    message: str
    start_time: datetime = datetime.now()

    def stop(self):
        print(f"-- [{(datetime.now() - self.start_time).seconds}s]: {self.message}")


def time_it(message):
    print(f"++ {message} ")
    return Measure_Helper(message, datetime.now())


# +
# This function is in the first block so you don't
# recreate it willy nilly, as it includes a cache.

# nltk.download("stopwords")

# Remove domain words that don't help analysis.
# Should be factored out
domain_stop_words = set(
    """
    yes yup Affirmations get that's Journal
    Deliberate Disciplined Daily
    Know Essential Provide Context
    First Understand Appreciate
    Hymn Monday
    Grateful
    ☐ ☑ K e tha Y X w
    for:2021 for:2020 for:2019
    """.lower().split()
)


# Load corpus of my daily ramblings
@dataclass
class Corpus:
    """
    A Corpus is the content combination of several journal entries.
    Journal entries can be  files/day, or data extracts from an XML archive.

    It handles tasks like cleaning up the text content
        * Fix Typos
        * Capitalize Proper Nouns

    Journal Entries can have semantic content based on the version
        * Grateful List
        * Inline Date
        * Daily Lists


    """

    all_content: str
    initial_words: List[str]
    words: List[str]
    journal: str = ""
    version: int = 0
    date_range: str = ""

    def __hash__(self):
        return hash(self.date_range)


def clean_string(line):
    def capitalizeProperNouns(s: str):
        properNouns = "zach ammon Tori amelia josh Ray javier Neha Amazon John".split()

        # NOTE I can Upper case Magic to make it a proper noun and see how it ranks!
        for noun in properNouns:
            noun = noun.lower()
            properNoun = noun[0].upper() + noun[1:]
            s = s.replace(" " + noun, " " + properNoun)
        return s

    # Grr- Typo replacer needs to be lexically smart - sigh
    typos = [
        ("waht", "what"),
        ('I"ll', "I'll"),
        ("that", "that"),
        ("taht", "that"),
        ("ti ", "it "),
        ("that'sa", "that's a"),
        ("Undersatnd", "Understand"),
        (" Ill ", " I'll "),
        ("Noitce", "Notice"),
        ("Whcih", "Which"),
        (" K ", " OK "),
        ("sTories", "stories"),
        ("htat", "that"),
        ("Getitng", "Getting"),
        ("Essenital", "Essential"),
        ("whcih", "which"),
        ("Apprecaite", "Appreciate"),
        (" nad ", " and "),
        (" adn ", " and "),
    ]

    def fixTypos(s: str):
        for typo in typos:
            s = s.replace(" " + typo[0], " " + typo[1])
        return s

    return capitalizeProperNouns(fixTypos(line))


class S50Export:
    # 750 words exports only has a body,  sometimes it has inline stats, consider stripping them.
    def __init__(self, path: Path):
        self.entries: Dict[date, List[str]] = dict()
        self.Load(path)

    def Load(self, path: Path):
        # entry starts with
        # Date: \s+ (\d{4}-\d{2}-\d{2})
        # skip meta data like ^Words:\s+\d+$
        # skip meta data like ^Minutes:\s+\d+$
        # WAKEUP:
        # Read the file

        entry_date: date = date.min
        entry_body = []
        if not path.exists():
            return

        f = path.open()
        for line in f:
            line = clean_string(line).strip("\n")
            is_end_of_entry = "-- ENTRY --" in line
            if is_end_of_entry and entry_date != date.min:
                # also do this when exiting the loop
                self.entries[entry_date] = entry_body
                entry_date = date.min
                entry_body = []
                continue
            is_date_line = line.startswith("Date:")
            if is_date_line:
                iso_date_pattern = r"\d{4}-\d{2}-\d{2}"
                match = re.search(iso_date_pattern, line)
                if match:
                    entry_date = date.fromisoformat(match.group())
                continue

            # meta data lines
            if line.startswith("Words:   "):
                continue
            if line.startswith("Minutes: "):
                continue

            entry_body.append(line)

        # finish the last entry if required
        if entry_date != date.min:
            self.entries[entry_date] = entry_body

        return


class JournalEntry:
    default_journal_section = "default_journal"

    def __init__(self, for_date: date):
        self.original: List[str] = []
        self.sections: Dict[str, List[str]] = {}
        self.sections_with_list: Dict[str, List[str]] = {}
        self.date: date = date(2099, 1, 1)
        self.journal: str = ""  # Just the cleaned journal entries
        self.init_from_date(for_date)

    def is_valid(self):
        contains_journal = (
            JournalEntry.default_journal_section in self.sections
            or "Journal" in self.sections
        )
        if not contains_journal:
            return False

        # Sometimes I create an empty entry, skip those
        body_content = "\n".join(self.body())
        is_super_short_entry = len(body_content) < 200
        return not is_super_short_entry

    def init_from_date(self, for_date: date):
        errors = []

        base_path = Path.home() / "gits/igor2"

        # check in new archive
        path = base_path / f"750words_new_archive/{for_date}.md"
        if path.exists():
            self.from_markdown_file(path)
            return

        errors.append(f"Not found new archive {path}")

        path = base_path / f"750words/{for_date}.md"

        if path.exists():
            self.from_markdown_file(path)
            return

        success = self.from_750words_export(for_date)
        if success:
            return

        errors.append("Not found in 750words export")

        raise FileNotFoundError(f"No file for that date {for_date}: {errors}")

    def __str__(self):
        out = ""
        out += f"Date:{self.date}\n"
        for _list in self.sections_with_list.items():
            out += f"List:{_list[0]}\n"
            for item in _list[1]:
                out += f"  * {item}\n"
        for section in self.sections.items():
            out += f"Section:{section[0]}\n"
            for line in section[1][:2]:
                out += f" {line[:80]}\n"

        return out

    def todo(self):
        # 2019-01 version has Journal section
        return self.sections_with_list["Day awesome if:"]

    def body(self):
        # 2019-01 version has Journal section
        if "Journal" in self.sections:
            return self.sections["Journal"]

        # Before that, return the body section
        return self.sections[JournalEntry.default_journal_section]

    def from_750words_export(self, for_date: date):
        # find year and month file - if not exists, exit
        path = (
            Path.home()
            / f"gits/igor2/750words_archive/750 Words-export-{for_date.year}-{for_date.month:02d}-01.txt"
        )
        if path.exists:
            archive = S50Export(path)
            if for_date in archive.entries:
                self.date = for_date
                self.sections[JournalEntry.default_journal_section] = archive.entries[
                    for_date
                ]
            return True

        return False

    # Consider starting with text so can handle XML entries
    def from_markdown_file(self, path: Path):
        output = []
        current_section = JournalEntry.default_journal_section
        sections = defaultdict(list)
        sections_as_list = defaultdict(list)
        _file = path.open()
        re_markdown_list_item = re.compile(r"^(\d+)\.")
        while line := _file.readline():
            is_blank_line = line.strip() == ""
            line = clean_string(line).strip("\n")

            if is_blank_line:
                continue

            is_date_line = line.startswith("750 words for:")
            if is_date_line:
                iso_date_pattern = r"\d{4}-\d{2}-\d{2}"
                match = re.search(iso_date_pattern, line)
                if match:
                    self.date = date.fromisoformat(match.group())
                continue

            is_section_line = line.startswith("##")
            if is_section_line:
                current_section = line.replace("## ", "")
                continue

            output.append(line)
            sections[current_section].append(line)

            is_list_item = re_markdown_list_item.match(line)
            if is_list_item:
                list_item = line.replace(is_list_item.group(0), "").strip()
                sections_as_list[current_section].append(list_item)
                continue

        _file.close()
        self.sections, self.sections_with_list = sections, sections_as_list


@lru_cache(maxsize=100)
def LoadCorpus(
    before=datetime.now().date() + timedelta(days=1), after=date(2011, 1, 1)
) -> Corpus:
    # TODO: Move this into the corpus class.

    # get nltk and corpus
    from nltk.corpus import stopwords

    # Hym consider memoizing this asweel..
    english_stop_words = set(stopwords.words("english"))
    all_stop_words = domain_stop_words | english_stop_words

    """
    ######################################################
    # Performance side-bar.
    ######################################################

    A] Below code results in all strings Loaded into memory for temporary,  then merged into a second string.
    aka Memory = O(2*file_conent) and CPU O(2*file_content)

    B] An alternative is to do += on a string results in a new memory allocation and copy.
    aka Memory = O(file_content) , CPU O(files*file_content)

    However, this stuff needs to be measured, as it's also a funtion of GC.
    Not in the GC versions there is no change in CPU
    Eg.

    For A] if GC happens after every "join", then were down to O(file_content).
    For B] if no GC, then it's still O(2*file_content)
    """

    # Make single string from all the file contents.
    bodies = [
        JournalEntry(entry).body()
        for entry in all_entries()
        if entry > after and entry < before
    ]

    # Flatten List of Lists
    all_lines = sum(bodies, [])

    cleaned_lines = [clean_string(line) for line in all_lines]
    single_clean_string = " ".join(cleaned_lines)

    # Clean out some punctuation (although does that mess up stemming later??)
    initial_words = single_clean_string.replace(",", " ").replace(".", " ").split()

    words = [word for word in initial_words if word.lower() not in all_stop_words]

    return Corpus(
        all_content=single_clean_string,
        initial_words=initial_words,
        words=words,
        date_range=f"Between {before} and {after}",
    )


@lru_cache(maxsize=1000)
def DocForCorpus(nlp, corpus: Corpus):
    ti = time_it(f"Building corpus {corpus.date_range} len:{len(corpus.all_content)} ")
    # We use all_file_content not initial_words because we want to keep punctuation.
    doc_all = nlp(corpus.all_content)

    # Remove domain specific stop words.
    doc = [token for token in doc_all if token.text.lower() not in domain_stop_words]
    ti.stop()

    return doc, doc_all


@app.command()
def todo(
    journal_for: str = typer.Argument(
        datetime.now().date(), help="Pass a date or int for days ago"
    ),
    close: Annotated[
        bool,
        typer.Option(
            help="Keep going back in days till you find the closest valid one"
        ),
    ] = False,
):
    date_journal_for = cli_date_to_entry_date(journal_for, close)
    entry = JournalEntry(date_journal_for)

    if not entry.is_valid():
        raise Exception(f"No Entry for {journal_for} ")

    if close:
        console.print(f"[blue]Using Date:{entry.date}[/blue]")

    for todo_line in entry.todo():
        # Safely remove these symbols
        todo_line = todo_line.replace("☑ ", "")
        todo_line = todo_line.replace("☐ ", "")
        print(todo_line)
    return


def cli_date_to_entry_date(journal_for, close) -> date:
    date_journal_for = datetime.now().date()
    if journal_for.isdigit():
        days_ago = int(journal_for)
        date_journal_for = datetime.now().date() - timedelta(days=days_ago)
    else:
        date_journal_for = date.fromisoformat(journal_for)

    if close:
        for i in range(1000):
            candidate = date_journal_for - timedelta(days=i)
            ic(candidate)
            entry = JournalEntry(candidate)
            if entry.is_valid():
                return candidate

    return date_journal_for


@app.command()
def body(
    journal_for: Annotated[
        str, typer.Argument(help="Pass a date or int for days ago")
    ] = str(datetime.now().date()),
    close: Annotated[
        bool,
        typer.Option(
            help="Keep going back in days till you find the closest valid one"
        ),
    ] = False,
    date_header: Annotated[
        bool, typer.Option(help="Always include the date header")
    ] = False,
    days: int = 1,
):
    date_journal_for = cli_date_to_entry_date(journal_for, close)
    entry = JournalEntry(date_journal_for)

    if not entry.is_valid():
        raise FileNotFoundError(f"No Entry for {date_journal_for} ")

    if close or date_header:
        console.print(f"[blue]Using Date:{entry.date} [/blue]")

    multi_day = days > 0
    for i in range(days):
        entry = JournalEntry(date_journal_for - timedelta(days=i))
        if not entry.is_valid():
            continue
        if multi_day:
            console.print(f"[blue] Journal For {entry.date}")
        for line in entry.body():
            print(line)


def entries_for_month(corpus_for: datetime) -> Iterable[date]:
    # Used to be, entries are stored in these exported files 1 per month

    old_export_path = (
        Path.home()
        / f"gits/igor2/750words_archive/750 Words-export-{corpus_for.year}-{corpus_for.month:02d}-01.txt"
    )

    if old_export_path.exists():
        archive = S50Export(old_export_path)
        for k, _ in archive.entries.items():
            yield k
        return

    # After that, moved to this new archive path, and then the working path

    possible_paths = [
        Path.home() / "gits/igor2" / p for p in ["750words_new_archive/", "750words/"]
    ]
    for path in possible_paths:
        pattern = path / f"{corpus_for.year}-{corpus_for.month:02d}-*"
        corpus_files = glob.glob(str(pattern))
        ic(corpus_files)
        for file_name in sorted(corpus_files):
            yield date.fromisoformat(file_name.split("/")[-1].replace(".md", ""))


@app.command()
def entries(
    for_month: datetime = typer.Argument(
        datetime.now(), help="Pass a date or int for days ago"
    ),
):
    for e in entries_for_month(for_month):
        print(e)


@app.command()
def s50_export():
    # test_journal_entry = JournalEntry(date(2012, 4, 8))
    print("hi")


def all_entries() -> Iterable[date]:
    curr = datetime(2012, 1, 1)
    date_final = datetime.now()
    while curr < date_final:
        curr += timedelta(1 * 30)
        yield from entries_for_month(curr)


@app.command()
def all():
    for x in all_entries():
        print(x)


@app.command()
def bodies(days: int = 7):
    for x in all_entries():
        print(JournalEntry(x).body())
    # print (len(list(all_entries())))


@app.command()
def sanity():
    # Load simple corpus for my journal
    corpus = LoadCorpus(datetime.now().date() - timedelta(days=180))  # NOQA  - see if it loads
    print(
        f"Initial words {len(corpus.initial_words)} remaining words {len(corpus.words)}"
    )
    # load using new archive
    new_archive_date = date(2021, 1, 8)
    test_journal_entry = JournalEntry(new_archive_date)
    print(new_archive_date, test_journal_entry.body()[0:2])
    # load using new archive
    old_archive_date = date(2012, 4, 8)
    test_journal_entry = JournalEntry(old_archive_date)
    print(old_archive_date, test_journal_entry.body()[0:2])


@app.command()
def files_with_word(word):
    # I need to put this in a DB so it's fast.
    directory = Path.home() / "gits/igor2/750words_new_archive"
    if not os.path.isdir(directory):
        print(f"Error: Directory '{directory}' not found.")
        sys.exit(1)

    word_count = []

    for root, _, files in os.walk(directory):
        for file in files:
            file_path = Path(root) / file
            with open(file_path, "r") as f:
                content = f.read()
                count = len(
                    re.findall(rf"\b{re.escape(word)}\b", content, re.IGNORECASE)
                )
            word_count.append((count, file_path))

    word_count.sort()

    for count, file_path in word_count:
        if count == 0:
            continue
        print(f"{count} {file_path}")


@app.command()
def build_embed():
    pass

    # Todo rebuild this using langchain and RAG


if __name__ == "__main__":
    app()
