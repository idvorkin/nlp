from typing import List
from dataclasses import dataclass

import glob
import os
from pathlib import Path

# get nltk and corpus
import nltk
from nltk.corpus import stopwords

from functools import lru_cache
import arrow

# copied from pandas util (yuk)


@dataclass
class Measure_Helper:
    message: str
    start_time: arrow.Arrow

    def stop(self):
        print(f"-- [{(arrow.now() - self.start_time).seconds}s]: {self.message}")


def time_it(message):
    print(f"++ {message}")
    return Measure_Helper(message, arrow.now())


# +
# This function is in the first block so you don't
# recreate it willy nilly, as it includes a cache.

nltk.download("stopwords")

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
@dataclass(frozen=True)
class Corpus:
    path: str
    all_content: str
    initial_words: List[str]
    words: List[str]

    def __hash__(self):
        return self.path.__hash__()

    def journal_only(self):
        return ""


@lru_cache(maxsize=100)
def LoadCorpus(corpus_path: str) -> Corpus:

    # Hym consider memoizing this asweel..
    english_stop_words = set(stopwords.words("english"))
    all_stop_words = domain_stop_words | english_stop_words

    corpus_path_expanded = os.path.expanduser(corpus_path)
    corpus_files = glob.glob(corpus_path_expanded)

    """
    ######################################################
    # Performance side-bar.
    ######################################################

    A] Below code results in all strings Loaded into memory for temporary,  then merged into a second string.
    aka Memory = O(2*file_conent) and CPU O(2*file_content)

    B] An alternative is to do += on a string results in a new memory allocation and copy.
    aka Memory = O(file_content) , CPU O(files*file_content)

    However, this stuff needs to be measured, as it's also a funtion of GC. Not in the GC versions there is no change in CPU
    Eg.

    For A] if GC happens after every "join", then were down to O(file_content).
    For B] if no GC, then it's still O(2*file_content)
    """

    # Make single string from all the file contents.
    list_file_content = [Path(file_name).read_text() for file_name in corpus_files]
    all_file_content = " ".join(list_file_content)

    # NOTE I can Upper case Magic to make it a proper noun and see how it ranks!

    properNouns = "zach ammon Tori amelia josh Ray javier Neha Amazon John".split()

    def capitalizeProperNouns(s: str):
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

    all_file_content = fixTypos(all_file_content)
    all_file_content = capitalizeProperNouns(all_file_content)

    # Clean out some punctuation (although does that mess up stemming later??)
    initial_words = all_file_content.replace(",", " ").replace(".", " ").split()

    words = [word for word in initial_words if word.lower() not in all_stop_words]
    return Corpus(
        path=corpus_path,
        all_content=all_file_content,
        initial_words=initial_words,
        words=words,
    )


@lru_cache(maxsize=100)
def DocForCorpus(nlp, corpus: Corpus):
    print(
        f"initial words {len(corpus.initial_words)} remaining words {len(corpus.words)}"
    )
    ti = time_it(
        f"Building corpus from {corpus.path} of len:{len(corpus.all_content)} "
    )
    # We use all_file_content not initial_words because we want to keep punctuation.
    doc_all = nlp(corpus.all_content)

    # Remove domain specific stop words.
    doc = [token for token in doc_all if token.text.lower() not in domain_stop_words]
    ti.stop()

    return doc


# -


# # Build corpus from my journal in igor2/750words

# +
# make function path for y/m


# Build up corpus reader.
# Current design, hard code knowledge of where everything is stored in forwards direction.
# Hard code insufficient data for analysis.
# Better model, read all files in paths and then do lookup.


# Hymn better model:
# A - lookup all files.
# B - Generate paths based on actual locations.


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
    glob750_old_archive(2018, month) for month in range(3, 8)
] + [glob750_new_archive(2018, month) for month in (9, 11, 12)]

corpus_path_months[2019] = [glob750_new_archive(2019, month) for month in range(1, 13)]
corpus_path_months[2020] = [glob750_new_archive(2020, month) for month in range(1, 13)]
corpus_path_months[2021] = [glob750_new_archive(2021, month) for month in range(1, 7)]

corpus_path_months_trailing = (
    [glob750_new_archive(2018, month) for month in (9, 11, 12)]
    + corpus_path_months[2019]
    + corpus_path_months[2020]
    + corpus_path_months[2021]
)

corpus_path_months_trailing

# ### Load simple corpus for my journal

corpus = LoadCorpus(corpus_path_months[2020][-1])
print(f"initial words {len(corpus.initial_words)} remaining words {len(corpus.words)}")
