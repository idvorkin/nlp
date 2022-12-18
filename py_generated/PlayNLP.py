# ---
# jupyter:
#   jupytext:
#     text_representation:
#       extension: .py
#       format_name: light
#       format_version: '1.5'
#       jupytext_version: 1.12.0
#   kernelspec:
#     display_name: Python 3 (ipykernel)
#     language: python
#     name: python3
# ---

# # Explore NLP against my journal entries
#
# This notebook allows me to play with NLP concepts using my personal journals.
# I've been writing personal journal entries ala 750 words a day for several years.

# +
"""
from sklearn.datasets import fetch_openml
from sklearn import datasets, svm, metrics
from pandas import DataFrame
import matplotlib as mpl
"""

from typing import List, Tuple
from dataclasses import dataclass
import pandas as pd

import matplotlib.pyplot as plt
import matplotlib

import glob
import os
from pathlib import Path

# get nltk and corpus
import nltk
from nltk.corpus import stopwords

# get scapy and corpus
import spacy
import time
from functools import lru_cache
from matplotlib import animation, rc
from IPython.display import HTML
from datetime import timedelta
import itertools
import importlib
import igor_journal

# python won't reload an already imported library
# since co-editting files in another directory, need to do a reload
importlib.reload(igor_journal)
from igor_journal import corpus_path_months
from icecream import ic
# -

print(igor_journal.test_journal_entry)

# +
# This function is in the first block so you don't
# recreate it willy nilly, as it includes a cache.

nltk.download("stopwords")


@lru_cache(maxsize=4)
def get_nlp_model(model: str):
    start_time = time.time()
    print(f"Loading Model {model}")
    nlp = spacy.load(model)  # python -m spacy download en_core_web_lg
    spacy.prefer_gpu()  # This will be cool if/when it happens.
    duration = time.time() - start_time
    print(f"Took: {int(duration)}")
    return nlp


# + [markdown] tags=[]
# # Build corpus from my journal in igor2/750words
# -

# make the plot wider
height_in_inches = 8
matplotlib.rc("figure", figsize=(2 * height_in_inches, height_in_inches))

# ### Load simple corpus for my journal

corpus = igor_journal.LoadCorpus(corpus_path_months[2021][-2])
print(f"initial words {len(corpus.initial_words)} remaining words {len(corpus.words)}")


# +
# Could use nltk frequency distribution plot, but better off building our own.
# fd = nltk.FreqDist(words)
# fd.plot(50, percents=True)
# Can also use scikit learn CountVectorizor

# +
# Same as NLTK FreqDist, except normalized, includes cumsum, and colors
def GraphWordDistribution(words, title="", skip=0, length=50, includeCDF=True) -> None:
    def GetPDFCDF(words):
        def ToPercent(x: float) -> float:
            return x * 100

        # NOTE: No point creating a full data frame when only using a single column.
        pdf = pd.Series(words).value_counts(normalize=True).apply(ToPercent)
        cdf = pdf.cumsum()
        return (pdf, cdf)

    def PlotOnAxis(series, ax, label: str, color: str):
        # RANT: Why is MPL so confusing? The OO interface vs the stateful interface, GRAH!!
        # The random non-obvious calls.
        # GRAH!!!

        ax.legend(label.split())
        ax.plot(series, color=color)

        # RANT: Why no YAxis.set_labal_params()? E.g.
        #                 ax.yaxis.set_label_params(label, color=color)
        ax.set_ylabel(label, color=color)
        ax.yaxis.set_tick_params(labelcolor=color)

        # technically all the X axis paramaters are duplicated since we "twinned the X paramater"
        ax.xticks = range(len(series))

        # RANT: rot can be set on plt.plot(), but not on axes.plot()
        ax.xaxis.set_tick_params(rotation=90)

    # NOTE: can make graph prettier with styles E.g.
    # with plt.style.context("ggplot"):
    fig, ax = plt.subplots(1)

    ax.set_title(title)
    ax.grid(True)

    # make pdf first axes, and cdf second axes.
    ax_pdf, ax_cdf = (ax, ax.twinx())
    color_pdf, color_cdf = ("green", "blue")
    pdf, cdf = GetPDFCDF(words)

    PlotOnAxis(pdf[skip : skip + length], ax_pdf, label="PDF*100", color=color_pdf)
    PlotOnAxis(cdf[skip : skip + length], ax_cdf, label="CDF*100", color=color_cdf)


GraphWordDistribution(corpus.words, title="Normalized Word Distribution")
# -

skip = 10
GraphWordDistribution(
    corpus.words, skip=skip, length=75, title=f"Distribution without top {skip} words"
)

# +
# wordcloud is non-deterministic, which is bizarre.
# from wordcloud import WordCloud
# wordcloud = WordCloud(max_font_size=50, max_words=100, background_color="white", stopwords=None).generate("".join(words))
# plt.imshow(wordcloud,  interpolation='bilinear')
# -

# # Play with POS tagging and lemmatisation

# +
# # !python -m spacy download en_core_web_lg
from igor_journal import Corpus, LoadCorpus, DocForCorpus

nlp = get_nlp_model("en_core_web_lg")
nlp.max_length = 100 * 1000 * 1000


def GetInterestingWords(pos: str, doc, corpus: Corpus):
    interesting_pos = pos
    interesting_pos_set = set(interesting_pos.split())
    interesting = [token for token in doc if token.pos_ in interesting_pos_set]
    interesting_words = [token.lemma_ for token in interesting]
    return interesting_words


def GraphPoSForDoc(pos: str, doc, corpus):
    GraphWordDistribution(
        GetInterestingWords(pos, doc, corpus=corpus),
        title=f"Distribution of {pos} on {corpus.path}",
        skip=0,
        length=20,
    )


def GraphScratchForCorpus(corpus_path: str, pos: str = "NOUN VERB ADJ ADV"):
    corpus = LoadCorpus(corpus_path)
    doc = DocForCorpus(nlp, corpus)
    GraphPoSForDoc(pos, doc, corpus)


def GetInterestingForCorpusPath(corpus_path: str, pos: str = "NOUN VERB ADJ ADV"):
    corpus = LoadCorpus(corpus_path)
    doc = DocForCorpus(nlp, corpus)
    return GetInterestingWords(pos, doc, corpus)


# -

# corpus_paths = corpus_paths_years
corpus_paths = corpus_path_months[2021]
print(corpus_paths)
for c in corpus_paths:
    GraphScratchForCorpus(c, pos="PROPN")

# # Debugging when stuff goes goofy.

_ = """
max_to_analyze = 15
interesting = [token for token in doc if token.tag_ == "NNS"]
for token in interesting[:max_to_analyze]:
    # print(token.text, token.lemma_, token.pos_, token.tag_, token.dep_, token.shape_, token.is_alpha, token.is_stop)
    print(token.text, token.lemma_, token.pos_, token.tag_, token.dep_)

# Parts of speech: https://spacy.io/usage/linguistic-features
GraphWordDistribution([token.pos_ for token in doc], title=f"POS Distribution on {corpus_path}")
# interesting = [ token for token in doc if token.pos_ != "PUNCT" and token.pos_ != "SYM" and len(token.text) > 3]
"""


# ### Visualizing the "Thought Distribution" over time.
# * A] Sentiment over time. Graph valence as line graph time series
#     (TBD: Use cloud service to analyze each file)
#
# * B] Graph a bar chart of Proper noun trending over time, have it update per corpus file.
#  * Build a data frame of word frequency "Proper Noun"x"Corpus"
#  * Graph update every second.

# +
def MakePDF(words, name):
    def ToPercent(x: float) -> float:
        return x * 100

    return pd.Series(words, name=name).value_counts(normalize=True).apply(ToPercent)


def PathToFriendlyTitle(path: str):
    path = path.split("/")[-1]
    if "export-" in path:
        return path.split("export-")[-1]
    else:
        return path


# +
# corpus_paths = corpus_path_months[2018]+corpus_path_months[2019]
# corpus_paths = corpus_path_months[2018] + corpus_path_months[2019]
corpus_paths = igor_journal.corpus_path_months_trailing[-24:]
top_words_to_skip, count_words = 0, 20
print(corpus_paths)
pdfs = [
    MakePDF(GetInterestingForCorpusPath(p, "PROPN"), PathToFriendlyTitle(p))
    for p in corpus_paths
]

# TODO: Why can't we use the join - gives an error.
# wordByTimespan = pd.DataFrame().join(pdfs, how="outer", sort=False)
wordByTimespan = pd.DataFrame()
for pdf in pdfs:
    wordByTimespan = wordByTimespan.join(pdf, how="outer")

# Sort by sum(word frequency) over all corpus
# I  suspect it'd be interesting to sort by TF*IDF because it'll make words that are present
# only in a few months get a boost.
wordByTimespan["word_frequency"] = wordByTimespan.sum(skipna=True, axis="columns")
wordByTimespan = wordByTimespan.sort_values("word_frequency", ascending=False)


# Remove total column
wordByTimespan = wordByTimespan.iloc[:, :-1]

print(f"skipping:{top_words_to_skip}, count:{count_words} ")

# wordByTimespan.iloc[:50, :].plot( kind="bar", subplots=False, legend=False, figsize=(15, 14), sharey=True )
wordByTimespan.iloc[top_words_to_skip : top_words_to_skip + count_words, :].T.plot(
    kind="bar", subplots=True, legend=False, figsize=(15, 9), sharey=True
)
# wordByTimespan.iloc[:13, :].T.plot( kind="bar", subplots=False, legend=True, figsize=(15, 14), sharey=True )

# +
top_words_to_skip, count_words = 5, 40
top_word_by_year = wordByTimespan.iloc[
    top_words_to_skip : top_words_to_skip + count_words, :
][::-1]
# top_word_by_year = wordByTimespan.iloc[:15,:][::-1] # the -1 on the end reverse the count

anim_fig_size = (16, 20)
fig = plt.figure()
ax = fig.add_subplot(1, 1, 1)
ax = top_word_by_year.iloc[:, 0].plot(
    title=f"Title Over Written", figsize=anim_fig_size, kind="barh"
)

animation.patches = ax.patches
loop_colors = itertools.cycle("bgrcmk")
animation.colors = list(itertools.islice(loop_colors, len(animation.patches)))


def animate(
    i,
):
    # OMG: That was impossible to find!!!
    # Turns out every time you call plot, more patches (bars) are added to graph.  You need to remove them, which is very non-obvious.
    # https://stackoverflow.com/questions/49791848/matplotlib-remove-all-patches-from-figure
    [p.remove() for p in reversed(animation.patches)]
    top_word_by_year.iloc[:, i].plot(
        title=f"Distribution {top_word_by_year.columns[i]}",
        kind="barh",
        color=animation.colors,
        xlim=(0, 10),
    )
    return (animation.patches,)


anim = animation.FuncAnimation(
    fig,
    animate,
    frames=len(top_word_by_year.columns),
    interval=timedelta(seconds=1).seconds * 1000,
    blit=False,
)
HTML(anim.to_html5_video())

# +
dmo = """
corpus_path = "~/gits/igor2/750words/2019-06-*md"
corpus = LoadCorpus(corpus_path)
doc = DocForCorpus(nlp, corpus)
for t in doc[400:600]:
print(f"{t} {t.lemma_} {t.pos_}")
"""
from spacy import displacy

displacy.render(nlp("Igor wonders if Ray is working too much"))
# -










