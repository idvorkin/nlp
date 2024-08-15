#!python3
import glob
import os
import re
from collections import defaultdict
from datetime import datetime, timedelta

import typer
from loguru import logger

app = typer.Typer(no_args_is_help=True)

# Open all md files
# read till hit line grateful
# skip all lines that don't start with d.
# copy all lines till hit  ## affirmations


def extractListItem(listItem):
    # find numbered lists
    matches = re.findall("\\d\\.\\s*(.*)", listItem)

    # only find the first - item
    matches += re.findall("^\\s*-\\s*(.*)", listItem)
    return matches


def isSectionStart(line, section):
    return re.match(f"^##.*{section}.*", line) is not None


def extractListInSection(f, section):
    fp = open(f)
    inSection = False
    for line in fp.readlines():
        if inSection:
            isSectionEnd = line.startswith("#")
            if isSectionEnd:
                return

        if isSectionStart(line, section):
            inSection = True

        if not inSection:
            continue

        yield from extractListItem(line)

    return


def extractListFromGlob(directory, section):
    files = [f for f in glob.glob(directory)]
    yield from extractListFromFiles(files, section)


def extractListFromFiles(files, section):
    for f in files:
        if not os.path.exists(f):
            continue
        yield from extractListInSection(f, section)


def makeCategoryMap():
    category_map_i = {}
    category_map_data = {
        "sleep": "up early;wake;woke;sleep;morning;bed",
        "magic": "magic;card;palm",
        "diet": "diet;eating;juice;juicing;weight",
        "exercise": "gym;exercise;ring;trainer;training;cardio;tristen;tristan;lynelle",
        "meditate": "meditate;meditation",
        "stress": "stress;anxiety;depression",
        "family": "family;zach;amelia;tori",
    }
    # todo figure out how to stem
    categories_flat = "psc;essential;appreciate;daily;offer;bike;interview".split(";")

    for category, words in category_map_data.items():
        category_map_i[category] = words.split(";")
    for c in categories_flat:
        category_map_i[c] = [c]

    # do some super stemming - probably more effiient way
    suffixes = "d;ed;s;ing".split(";")
    # print(suffixes)
    for c, words in category_map_i.items():
        words = words[:]  # force a copy
        # print (words)
        for w in words:
            if w == " " or w == "":
                continue
            for s in suffixes:
                # print (f"W:{w},s:{s}")
                with_stem = w + s
                # print(f"with_stem:{with_stem}")
                category_map_i[c] += [with_stem]
        # print(category_map_i[c])

    # print (category_map_i)
    return category_map_i


category_map = makeCategoryMap()
# print (category_map)
categories = category_map.keys()


def lineToCategory(line):
    # NLP tokenizing remove punctuation.
    punctuation = "/.,;'"
    for p in punctuation:
        line = line.replace(p, " ")
    words = line.lower().split()

    for c, words_in_category in category_map.items():
        for w in words:
            # print (f"C:{c},W:{w},L:{l}")
            if w in words_in_category:
                return c
    return None


def groupCategory(reasons_to_be_grateful):
    grateful_by_reason = defaultdict(list)

    for reason in reasons_to_be_grateful:
        if reason == "":
            continue

        category = lineToCategory(reason)
        grateful_by_reason[category] += [reason]

    l3 = sorted(grateful_by_reason.items(), key=lambda x: len(x[1]))
    return l3


def printCategory(grouped, markdown=False, text_only=False):
    def strip_if_text_only(s, text_only):
        if not text_only:
            return s
        return s.replace("1.", "").replace("☑", "").replace("☐", "").strip()

    for line in grouped:
        is_category = line[0] is not None
        category = line[0] if is_category else "general"

        if not markdown and not text_only:
            print(f"#### {category.capitalize()}")

        for m in line[1]:
            m = strip_if_text_only(m, text_only)
            if markdown:
                print(f"1. {m}")
            elif text_only:
                print(f"{m}")
            else:
                print(f"   - {m}")


# extractGratefulReason("a. hello world")
# m = list(extractListInSection("/home/idvorkin/gits/igor2/750words/2019-11-04.md", "Grateful"))
# print(m)
# r = dumpAll(os.path.expanduser("~/gits/igor2/750words/*md")
# all_reasons_to_be_grateful = extractGratefulFromGlob (os.path.expanduser("~/gits/igor2/750words_new_archive/*md"))


def dumpGlob(glob, thelist):
    all_reasons_to_be_grateful = extractListFromGlob(os.path.expanduser(glob), thelist)
    grouped = groupCategory(all_reasons_to_be_grateful)
    printCategory(grouped)


@app.command()
def grateful(
    days: int = typer.Argument(7),
    markdown: bool = typer.Option(False),
    text_only: bool = typer.Option(False),
):
    return dumpSectionDefaultDirectory(
        "Grateful", days=days, markdown=markdown, text_only=text_only
    )


@app.command()
def awesome(
    days: int = typer.Argument(7),
    markdown: bool = typer.Option(False),
    text_only: bool = typer.Option(False),
):
    return dumpSectionDefaultDirectory(
        "Yesterday", days=days, markdown=markdown, text_only=text_only
    )


@app.command()
def todo(
    days: int = typer.Argument(2),
    markdown: bool = typer.Option(False),
    text_only: bool = typer.Option(False),
):
    """Yesterday's Todos"""
    return dumpSectionDefaultDirectory(
        "if", days, day=True, markdown=markdown, text_only=text_only
    )


@app.command()
def week(weeks: int = typer.Argument(4), section: str = typer.Argument("Moments")):
    """Section of choice for count weeks"""
    return dumpSectionDefaultDirectory(section, weeks, day=False)


# section
def dumpSectionDefaultDirectory(
    section, days, day=True, markdown=False, text_only=False
):
    # assert section in   "Grateful Yesterday if".split()

    printHeader = markdown is False and text_only is False

    if printHeader:
        print(f"## ----- Section:{section}, days={days} ----- ")

    # Dump both archive and latest.
    listItem = []
    if day:
        files = [
            os.path.expanduser(
                f"~/gits/igor2/750words/{(datetime.now()-timedelta(days=d)).strftime('%Y-%m-%d')}.md"
            )
            for d in range(days)
        ]
        files += [
            os.path.expanduser(
                f"~/gits/igor2/750words_new_archive/{(datetime.now()-timedelta(days=d)).strftime('%Y-%m-%d')}.md"
            )
            for d in range(days)
        ]
        listItem = extractListFromFiles(files, section)
    else:
        # User requesting weeks.
        # Instead of figuring out sundays, just add 'em up.
        files = [
            os.path.expanduser(
                f"~/gits/igor2/week_report/{(datetime.now()-timedelta(days=d)).strftime('%Y-%m-%d')}.md"
            )
            for d in range(days * 8)
        ]
        # print (files)
        listItem = extractListFromFiles(files, section)

    grouped = groupCategory(listItem)
    printCategory(grouped, markdown, text_only)


@logger.catch
def app_with_loguru():
    app()


if __name__ == "__main__":
    app_with_loguru()
