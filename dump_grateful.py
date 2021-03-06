#!python3
from collections import defaultdict
import re
import glob
import os
import typer
from datetime import datetime, timedelta
from loguru import logger
from icecream import ic

app = typer.Typer()

# Open all md files
# read till hit line grateful
# skip all lines that don't start with d.
# copy all lines till hit  ## affirmations


def extractListItem(l):
    matches = re.findall("\\d\\.\\s*(.*)", l)
    matches += re.findall("-\\s*(.*)", l)
    return matches


def isSectionStart(l, section):
    return re.match(f"^##.*{section}.*", l) is not None


def extractListInSection(f, section):
    fp = open(f)
    inSection = False
    for l in fp.readlines():
        if inSection:
            isSectionEnd = l.startswith("#")
            if isSectionEnd:
                return

        if isSectionStart(l, section):
            inSection = True

        if not inSection:
            continue

        yield from extractListItem(l)

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
        "exercise": "gym;exercise;ring;trainer;training;cardio",
        "meditate": "meditate;meditation",
    }
    # todo figure out how to stem
    categories_flat = "anxiety;essential;appreciate;daily;zach;amelia;tori;offer;bike;meditate;interview".split(
        ";"
    )

    for (category, words) in category_map_data.items():
        category_map_i[category] = words.split(";")
    for c in categories_flat:
        category_map_i[c] = [c]

    # do some super stemming - probably more effiient way
    suffixes = "d;ed;s;ing".split(";")
    # print(suffixes)
    for (c, words) in category_map_i.items():
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


def lineToCategory(l):
    # NLP tokenizing remove punctuation.
    punctuation = "/.,;'"
    for p in punctuation:
        l = l.replace(p, " ")
    words = l.lower().split()

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
        return s.replace("1.", "").replace("???", "").replace("???", "").strip()

    for l in grouped:
        empty_group = l[0] == None
        if empty_group:
            for m in l[1]:
                m = strip_if_text_only(m, text_only)
                if markdown:
                    print(f"1. {m}")
                else:
                    print(f"{m}")

            continue

        if not markdown and not text_only:
            print(f"{l[0]}")

        for m in l[1]:
            m = strip_if_text_only(m, text_only)
            if markdown:
                print(f"1. {m}")
            elif text_only:
                print(f"{m}")
            else:
                print(f"   {m}")


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

    printHeader = markdown == False and text_only == False

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
