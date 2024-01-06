#!python3


from typing import Dict, Set, List
import typer

from icecream import ic
from pathlib import Path
import sys
import re

ic("hello")


def parse_glossary(file_path: Path):
    glossary = {}
    with open(file_path, "r") as file:
        content = file.read()
        # Split the content into chunks, each representing one glossary entry
        chunks = re.split(r"\n(?=## )", content)
        for chunk in chunks:
            # Extract the term and definition using regular expressions
            term_match = re.search(r"## (.+)", chunk)
            definition_match = re.search(r"\n\n(.+)", chunk, re.DOTALL)
            if term_match and definition_match:
                term = term_match.group(1).strip().lower()
                definition = definition_match.group(1).strip()
                glossary[term] = definition
    return glossary


def find_references(definition, glossary):
    potential_references = re.findall(r"\b[a-z]*\b", definition)
    ic(potential_references)
    references = [ref for ref in potential_references if ref in glossary]
    return references


def build_closure(terms: List[str], glossary: Dict[str, str]) -> Set[str]:
    closure = set()

    def recurse_closure(term: str, path: Set[str]):
        if term in path:
            return
        path.add(term)
        closure.add(term)
        references = find_references(glossary[term], glossary)
        for ref in references:
            recurse_closure(ref, path)
        path.remove(term)

    for term in terms:
        if term not in glossary:
            continue
        recurse_closure(term, set())

    return closure


# Define a function to print the list of definitions
def print_definitions(terms_closure: Set[str], glossary: Dict[str, str]):
    # sort the terms alphabetically
    sorted = list(terms_closure)
    sorted.sort()
    for term in sorted:
        print(f"#### {term}\n")
        print(f"{glossary[term]}\n")


app = typer.Typer()


@app.command()
def build():
    # Expecting the glossary file to be named 'glossary.txt'
    glossary_path = Path().home() / "tmp/whatilearnedsofar/glossary.md"
    glossary = parse_glossary(glossary_path)
    ic(glossary)

    closure = set()
    for line in sys.stdin:
        terms = line.strip().split()  # Split the line into individual terms
        closure = closure.union(build_closure(terms, glossary))
    print_definitions(closure, glossary)


if __name__ == "__main__":
    app()
