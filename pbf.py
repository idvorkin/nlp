#!python3

import typer
from icecream import ic
from loguru import logger
from rich.console import Console
from pathlib import Path
from AppKit import NSPasteboard, NSFilenamesPboardType
import objc


console = Console()
app = typer.Typer()


@app.command()
def copy_to_clipboard(file: Path):
    """Copy file to clipboard only on macos"""

    if not file.exists():
        ic("File does not exist")
        return
    # copy file to clipboard
    # a = NSArray.arrayWithObject_("hello world")
    # pb.writeObjects_(a)
    pb = NSPasteboard.generalPasteboard()
    pb.clearContents()
    file_path = str(file.resolve())
    ic(file_path)
    array = objc.lookUpClass("NSArray").arrayWithObject_(file_path)
    pb.declareTypes_owner_([NSFilenamesPboardType], None)
    pb.setPropertyList_forType_(array, NSFilenamesPboardType)
    pb.writeObjects_(array)


@logger.catch()
def app_wrap_loguru():
    app()


if __name__ == "__main__":
    ic("main")
    app_wrap_loguru()
