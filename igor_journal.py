#!python3
from typing import Iterable, List, Dict
from dataclasses import dataclass, field
from loguru import logger

import glob
import os
from pathlib import Path

# get nltk and corpus
import nltk
from nltk.corpus import stopwords

from functools import lru_cache
from collections import defaultdict
from icecream import ic
import typer
from datetime import datetime, timedelta, date

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
                entry_date = date.fromisoformat(line.split(":")[1].strip())
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

    def init_from_date(self, for_date: date):

        errors = []

        base_path = Path(f"~/gits/igor2").expanduser()

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

        errors.append(f"Not found latest {path}")

        success = self.from_750words_export(for_date)
        if success:
            return

        errors.append(f"Not found in 750words export")

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

    def body(self):

        # 2019-01 version has Journal section
        if "Journal" in self.sections:
            return self.sections["Journal"]

        # Before that, return the body section
        return self.sections[JournalEntry.default_journal_section]

    def from_750words_export(self, for_date: date):
        # find year and month file - if not exists, exit
        path = Path(
            f"~/gits/igor2/750words_archive/750 Words-export-{for_date.year}-{for_date.month:02d}-01.txt"
        ).expanduser()
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
        while line := _file.readline():
            is_blank_line = line.strip() == ""
            line = clean_string(line).strip("\n")

            if is_blank_line:
                continue

            is_date_line = line.startswith("750 words for:20")
            if is_date_line:
                self.date = date.fromisoformat(line.split(":")[1])
                continue

            is_section_line = line.startswith("##")
            if is_section_line:
                current_section = line.replace("## ", "")
                continue

            output.append(line)
            sections[current_section].append(line)

            is_list_item = line.startswith("1.")
            if is_list_item:
                list_item = line.replace("1. ", "")
                sections_as_list[current_section].append(list_item)
                continue

        _file.close()
        self.sections, self.sections_with_list = sections, sections_as_list


@lru_cache(maxsize=100)
def LoadCorpus(
    before=datetime.now().date() + timedelta(days=1), after=date(2011, 1, 1)
) -> Corpus:
    # TODO: Move this into the corpus class.

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


def build_corpus_paths():
    def glob750_latest(year, month):
        assert month in range(1, 13)
        base750 = "~/gits/igor2/750words/"
        return f"{base750}/{year}-{month:02}-*.md"

    def glob750_new_archive(year, month):
        assert month in range(1, 13)
        base750 = "~/gits/igor2/750words_new_archive/"
        return f"{base750}/{year}-{month:02}-*.md"

    def glob750_old_archive(year, month):
        assert month in range(1, 13)
        base750archive = "~/gits/igor2/750words_archive/"
        return f"{base750archive}/750 Words-export-{year}-{month:02}-01.txt"

    def corpus_paths_months_for_year(year):
        return [glob750_old_archive(year, month) for month in range(1, 13)]

    # Corpus in "old archieve"  from 2012-2017.
    corpus_path_months = {
        year: corpus_paths_months_for_year(year) for year in range(2012, 2018)
    }

    # 2018 Changes from old archive to new_archieve.
    # 2018 Jan/Feb/October don't have enough data for analysis
    corpus_path_months[2018] = [
        glob750_old_archive(2018, month) for month in range(1, 9)
    ] + [glob750_new_archive(2018, month) for month in (9, 11, 12)]

    corpus_path_months[2019] = [
        glob750_new_archive(2019, month) for month in range(1, 13)
    ]
    corpus_path_months[2020] = [
        glob750_new_archive(2020, month) for month in range(1, 13)
    ]
    corpus_path_months[2021] = [
        glob750_new_archive(2021, month) for month in range(1, 13)
    ]
    corpus_path_months[2022] = [
        glob750_new_archive(2022, month) for month in range(1, 13)
    ]

    corpus_path_months_trailing = (
        [glob750_new_archive(2018, month) for month in (9, 11, 12)]
        + corpus_path_months[2019]
        + corpus_path_months[2020]
        + corpus_path_months[2021]
        + corpus_path_months[2022]
    )
    corpus_path_months_trailing
    return corpus_path_months, corpus_path_months_trailing


# Make today be the default date.


@app.command()
def body(journal_for: datetime = typer.Argument(datetime.now().date().isoformat())):

    # Use today as the default, this encourages me to write an article to make testing possible too

    entry = JournalEntry(journal_for.date())
    for l in entry.body():
        print(l)


def entries_for_month(corpus_for: datetime) -> Iterable[date]:
    corpus_path, corpus_path_months_trailing = build_corpus_paths()

    old_export_path = Path(
        f"~/gits/igor2/750words_archive/750 Words-export-{corpus_for.year}-{corpus_for.month:02d}-01.txt"
    ).expanduser()

    if old_export_path.exists():
        archive = S50Export(old_export_path)
        for k, v in archive.entries.items():
            yield k
        return

    # try new path
    the_path = os.path.expanduser(corpus_path[corpus_for.year][corpus_for.month - 1])
    corpus_files = glob.glob(the_path)
    for file_name in sorted(corpus_files):
        yield date.fromisoformat(file_name.split("/")[-1].replace(".md", ""))


@app.command()
def entries(corpus_for: datetime = typer.Argument("2021-01-08")):
    for e in entries_for_month(corpus_for):
        print(e)


@app.command()
def s50_export():
    test_journal_entry = JournalEntry(date(2012, 4, 8))
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
def all_body():
    for x in all_entries():
        print(JournalEntry(x).body())
    # print (len(list(all_entries())))


@app.command()
def sanity():

    # Load simple corpus for my journal

    corpus_path, corpus_path_months_trailing = build_corpus_paths()
    corpus = LoadCorpus(datetime.now().date() - timedelta(days=180))
    print(
        "Initial words {len(corpus.initial_words)} remaining words {len(corpus.words)}"
    )
    # load using new archive
    new_archive_date = date(2021, 1, 8)
    test_journal_entry = JournalEntry(new_archive_date)
    print(new_archive_date, test_journal_entry.body()[0:2])
    # load using new archive
    old_archive_date = date(2012, 4, 8)
    test_journal_entry = JournalEntry(old_archive_date)
    print(old_archive_date, test_journal_entry.body()[0:2])


@logger.catch
def app_with_loguru():
    app()


if __name__ == "__main__":
    app_with_loguru()
