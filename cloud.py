#!python3

import typer
from loguru import logger
from icecream import ic
from wordcloud import WordCloud
import langchain_helper

app = typer.Typer(no_args_is_help=True)


@app.command()
def dump_cloud(path: str = "", top: int = 40):
    wc = WordCloud(min_word_length=3)
    raw_words = langchain_helper.get_text_from_path_or_stdin(path)
    # create dict of word frequencies
    words = wc.process_text(raw_words)
    # make tuples from words and frequencies
    word_freqs = list(words.items())
    # sort by frequency
    word_freqs.sort(key=lambda x: x[1], reverse=True)
    word_freqs = word_freqs[:top]
    # ic(word_freqs)

    # normalize the frequencies
    sum_freqs = sum([freq for word, freq in word_freqs])
    norm_freqs = [
        (word, round(100 * (freq / sum_freqs), 2)) for word, freq in word_freqs
    ]
    ic(norm_freqs)


@logger.catch()
def app_wrap_loguru():
    app()


if __name__ == "__main__":
    app_wrap_loguru()
