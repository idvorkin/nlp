#!python3

import typer
from icecream import ic
from loguru import logger
from rich.console import Console
from pathlib import Path
from AppKit import NSPasteboard, NSFilenamesPboardType, NSArray


console = Console()
app = typer.Typer()


@app.command()
def copy(file: Path):
    """Copy file to clipboard only on macos"""

    if not file.exists():
        ic("File does not exist")
        return
    pb = NSPasteboard.generalPasteboard()
    pb.clearContents()
    file_path = str(file.resolve())
    ic(file_path)
    array = NSArray.arrayWithObject_(file_path)
    pb.declareTypes_owner_([NSFilenamesPboardType], None)
    pb.setPropertyList_forType_(array, NSFilenamesPboardType)
    pb.writeObjects_(array)
    ic(pb.types())


def dump():
    pb = NSPasteboard.generalPasteboard()
    # Loop through all types and print out their corresponding content
    types = pb.types()
    for pboard_type in types:
        content = pb.stringForType_(pboard_type)
        ic(f"Content for type {pboard_type}:", content)
    types = pb.types()
    ic("Available types:", types)


@logger.catch()
def app_wrap_loguru():
    app()


if __name__ == "__main__":
    ic("main")
    app_wrap_loguru()
