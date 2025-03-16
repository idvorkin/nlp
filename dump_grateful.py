#!python3
# -*- coding: utf-8 -*-
import glob
import os
import re
import asyncio
from collections import defaultdict
from datetime import datetime, timedelta
from typing import List, Tuple

import typer
from loguru import logger
from langchain_core.messages import HumanMessage, SystemMessage
import langchain_helper

app = typer.Typer(no_args_is_help=True)

# Open all md files
# read till hit line grateful
# skip all lines that don't start with d.
# copy all lines till hit  ## affirmations


def extractListItem(listItem):
    if not listItem or not isinstance(listItem, str):
        return []

    listItem = listItem.strip()
    if not listItem:
        return []

    matches = []

    # find items starting with a number followed by a period and whitespace (e.g., "1. item")
    numbered_matches = re.findall(r"^\d+\.\s*(.*?)$", listItem)
    matches.extend(m for m in numbered_matches if m)

    # find items starting with a dash and optional whitespace (e.g., "- item" or " - item")
    dash_matches = re.findall(r"^\s*[-\u2022*]\s*(.*?)$", listItem)
    matches.extend(m for m in dash_matches if m)

    return [m.strip() for m in matches if m.strip()]


def isSectionStart(line, section):
    return re.match("^##.*" + section + ".*", line) is not None


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


def get_system_prompt(predefined_categories):
    """Create a system prompt for categorizing items using Claude with predefined categories"""
    categories_list = ", ".join([f'"{cat}"' for cat in predefined_categories if cat])

    return f"""You are an expert at categorizing items into predefined categories.

Your task is to group the provided list of items into these specific categories: {categories_list}, and "general" for items that don't fit elsewhere.

Guidelines:
1. ONLY use the predefined categories listed above
2. Each item should be placed in exactly one category
3. Use "general" for items that don't clearly fit in any predefined category
4. Return ONLY a JSON object with this structure:
   {{
     "categories": {{
       "category_name_1": ["item1", "item2"],
       "category_name_2": ["item3", "item4"],
       ...
     }}
   }}
5. Keep the whole line don't split it

Do not include any explanations, notes, or additional text - ONLY return the JSON object.
"""


async def group_with_llm(items: List[str]) -> List[Tuple[str, List[str]]]:
    """Group items using Claude LLM with predefined categories"""
    if not items:
        return []

    # Filter empty items
    filtered_items = [item for item in items if item.strip()]
    if not filtered_items:
        return []

    # Get predefined categories from the category map
    global categories
    predefined_categories = list(categories)

    # Get Claude model
    llm = langchain_helper.get_model(claude=True)
    if not llm:
        logger.warning(
            "Claude model not available, falling back to manual categorization"
        )
        return groupCategory(items)

    # Create the messages with system prompt and items
    system_message = SystemMessage(content=get_system_prompt(predefined_categories))
    human_message = HumanMessage(
        content="\n".join([f"- {item}" for item in filtered_items])
    )

    try:
        # Call the LLM
        logger.info(
            f"Calling Claude to categorize {len(filtered_items)} items using predefined categories"
        )
        response = await llm.ainvoke([system_message, human_message])
        content = response.content
        logger.info("Received response from Claude")

        # Extract JSON from the response
        import json
        import re

        # Find JSON pattern in the response
        json_match = re.search(r"({[\s\S]*})", content)
        if json_match:
            json_str = json_match.group(1)
            try:
                result = json.loads(json_str)

                # Convert to the expected format
                categories = result.get("categories", {})
                if not categories:
                    logger.warning(
                        "No categories found in LLM response, falling back to manual categorization"
                    )
                    return groupCategory(filtered_items)

                # Ensure all categories are valid
                valid_categories = {}
                for category, items in categories.items():
                    # Convert category to lowercase for comparison
                    category_lower = category.lower()

                    # Check if category is in predefined categories or is "general"
                    if category_lower == "general" or category_lower in [
                        c.lower() for c in predefined_categories
                    ]:
                        valid_categories[category] = items
                    else:
                        # If not valid, put items in general category
                        if "general" not in valid_categories:
                            valid_categories["general"] = []
                        valid_categories["general"].extend(items)

                logger.info(
                    f"Successfully categorized items into {len(valid_categories)} predefined groups"
                )
                return sorted(
                    valid_categories.items(), key=lambda x: len(x[1]), reverse=True
                )
            except json.JSONDecodeError as e:
                logger.warning(f"Invalid JSON in LLM response: {e}")
                logger.warning(f"Response content: {content}")
                return groupCategory(filtered_items)
        else:
            logger.warning(
                "Could not extract JSON from LLM response, falling back to manual categorization"
            )
            logger.warning(f"Response content: {content}")
            return groupCategory(filtered_items)

    except Exception as e:
        logger.exception(f"Error using LLM for grouping: {e}")
        return groupCategory(filtered_items)


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

    # Filter out empty categories
    non_empty_grouped = [(category, items) for category, items in grouped if items]

    # Count total items for reporting
    total_items = sum(len(items) for _, items in non_empty_grouped)

    # Sort by category name if using LLM (already sorted by count if using manual categorization)
    sorted_grouped = sorted(non_empty_grouped, key=lambda x: (x[0] is None, x[0] or ""))

    for line in sorted_grouped:
        is_category = line[0] is not None
        category = line[0] if is_category else "general"
        items_count = len(line[1])

        if items_count > 0:  # Only print categories with items
            if not markdown and not text_only:
                print(f"#### {category.capitalize()} ({items_count})")

            for m in line[1]:
                m = strip_if_text_only(m, text_only)
                if markdown:
                    print(f"1. {m}")
                elif text_only:
                    print(f"{m}")
                else:
                    print(f"   - {m}")

    if not markdown and not text_only:
        print(f"\nTotal items: {total_items}")


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
    llm: bool = typer.Option(True, help="Use Gemini Flash to group items"),
):
    """List grateful items from journal entries"""
    return asyncio.run(
        dumpSectionDefaultDirectory(
            "Grateful", days=days, markdown=markdown, text_only=text_only, use_llm=llm
        )
    )


@app.command()
def awesome(
    days: int = typer.Argument(7),
    markdown: bool = typer.Option(False),
    text_only: bool = typer.Option(False),
    llm: bool = typer.Option(True, help="Use Gemini Flash to group items"),
):
    """List awesome things from journal entries"""
    return asyncio.run(
        dumpSectionDefaultDirectory(
            "Yesterday", days=days, markdown=markdown, text_only=text_only, use_llm=llm
        )
    )


@app.command()
def todo(
    days: int = typer.Argument(2),
    markdown: bool = typer.Option(False),
    text_only: bool = typer.Option(False),
    llm: bool = typer.Option(True, help="Use Gemini Flash to group items"),
):
    """Yesterday's Todos"""
    return asyncio.run(
        dumpSectionDefaultDirectory(
            "if", days, day=True, markdown=markdown, text_only=text_only, use_llm=llm
        )
    )


@app.command()
def week(
    weeks: int = typer.Argument(4),
    section: str = typer.Argument("Moments"),
    llm: bool = typer.Option(True, help="Use Gemini Flash to group items"),
):
    """Section of choice for count weeks"""
    return asyncio.run(
        dumpSectionDefaultDirectory(section, weeks, day=False, use_llm=llm)
    )


# section
async def dumpSectionDefaultDirectory(
    section, days, day=True, markdown=False, text_only=False, use_llm=True
):
    # assert section in   "Grateful Yesterday if".split()

    printHeader = markdown is False and text_only is False

    if printHeader:
        print(f"## ----- Section: {section}, days: {days} ----- ")
        if use_llm:
            print(
                "Using Claude with predefined categories for intelligent categorization"
            )
        else:
            print("Using manual categorization")

    # Dump both archive and latest.
    listItem = []
    if day:
        files = [
            os.path.expanduser(
                f"~/gits/igor2/750words/{(datetime.now() - timedelta(days=d)).strftime('%Y-%m-%d')}.md"
            )
            for d in range(days)
        ]
        files += [
            os.path.expanduser(
                f"~/gits/igor2/750words_new_archive/{(datetime.now() - timedelta(days=d)).strftime('%Y-%m-%d')}.md"
            )
            for d in range(days)
        ]
        listItem = list(extractListFromFiles(files, section))
    else:
        # User requesting weeks.
        # Instead of figuring out sundays, just add 'em up.
        files = [
            os.path.expanduser(
                f"~/gits/igor2/week_report/{(datetime.now() - timedelta(days=d)).strftime('%Y-%m-%d')}.md"
            )
            for d in range(days * 8)
        ]
        # print (files)
        listItem = list(extractListFromFiles(files, section))

    if use_llm:
        grouped = await group_with_llm(listItem)
    else:
        grouped = groupCategory(listItem)

    printCategory(grouped, markdown, text_only)


@logger.catch
def app_with_loguru():
    app()


if __name__ == "__main__":
    app_with_loguru()
