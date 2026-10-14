#!/usr/bin/env -S uv run --script
# /// script
# requires-python = ">=3.13"
# dependencies = [
#     "typer",
#     "rich",
# ]
# ///
# ABOUTME: Exports an iMessage chat.db to a CSV backup, one row per message, in local time.
# ABOUTME: Reads SQLite directly and decodes text that Messages stores only in attributedBody.

import csv
import sqlite3
from datetime import datetime
from pathlib import Path

COLUMNS = [
    "date",
    "text",
    "is_from_me",
    "to_phone",
    "chat_name",
    "is_group",
    "attachment",
]

APPLE_EPOCH = 978307200  # 2001-01-01 UTC as a unix timestamp
GROUP_CHAT_STYLE = 43
OBJECT_REPLACEMENT_CHAR = (
    "￼"  # placeholder Messages puts in text where an attachment sits
)

MESSAGES_QUERY = """
SELECT m.ROWID, m.date, m.text, m.attributedBody, m.is_from_me,
       h.id AS handle, c.chat_identifier, c.display_name, c.style
FROM message m
LEFT JOIN handle h ON h.ROWID = m.handle_id
LEFT JOIN chat c ON c.ROWID = (
    SELECT min(cmj.chat_id) FROM chat_message_join cmj WHERE cmj.message_id = m.ROWID
)
ORDER BY m.date, m.ROWID
"""

ATTACHMENTS_QUERY = """
SELECT maj.message_id, a.transfer_name
FROM message_attachment_join maj
JOIN attachment a ON a.ROWID = maj.attachment_id
ORDER BY maj.message_id, a.ROWID
"""


def apple_time_to_local(apple_time: int) -> str:
    """Convert a chat.db date (nanoseconds since 2001, or seconds in old databases) to local time."""
    seconds = (
        apple_time // 1_000_000_000 if apple_time > 100_000_000_000 else apple_time
    )
    return datetime.fromtimestamp(seconds + APPLE_EPOCH).strftime("%Y-%m-%d %H:%M:%S")


def decode_attributed_body(body: bytes | None) -> str:
    """Pull the plain string out of a typedstream NSAttributedString archive."""
    if not body:
        return ""
    start = body.find(b"NSString")
    if start < 0:
        return ""
    marker = body.find(b"\x01+", start)
    if marker < 0:
        return ""
    pos = marker + 2
    length = body[pos]
    pos += 1
    if length == 0x81:
        length = int.from_bytes(body[pos : pos + 2], "little")
        pos += 2
    elif length == 0x82:
        length = int.from_bytes(body[pos : pos + 4], "little")
        pos += 4
    return body[pos : pos + length].decode("utf-8", errors="replace")


def read_attachments(db: sqlite3.Connection) -> dict[int, list[str]]:
    attachments: dict[int, list[str]] = {}
    for message_id, name in db.execute(ATTACHMENTS_QUERY):
        attachments.setdefault(message_id, []).append(name or "")
    return attachments


def read_messages(db_path) -> list[dict]:
    db = sqlite3.connect(f"file:{db_path}?mode=ro", uri=True)
    try:
        attachments = read_attachments(db)
        rows = []
        for (
            rowid,
            date,
            text,
            body,
            is_from_me,
            handle,
            chat_identifier,
            display_name,
            style,
        ) in db.execute(MESSAGES_QUERY):
            content = (
                (text or decode_attributed_body(body))
                .replace(OBJECT_REPLACEMENT_CHAR, "")
                .strip()
            )
            attachment = ";".join(attachments.get(rowid, []))
            if not content and not attachment:
                continue
            rows.append(
                {
                    "date": apple_time_to_local(date),
                    "text": content,
                    "is_from_me": int(bool(is_from_me)),
                    "to_phone": handle or chat_identifier or "",
                    "chat_name": display_name or chat_identifier or "",
                    "is_group": int(style == GROUP_CHAT_STYLE),
                    "attachment": attachment,
                }
            )
        return rows
    finally:
        db.close()


def export_csv(db_path, out_path) -> int:
    rows = read_messages(db_path)
    with Path(out_path).open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=COLUMNS)
        writer.writeheader()
        writer.writerows(rows)
    return len(rows)


def _build_app():
    """Wire the Typer app. Called only from __main__ so tests skip the typer import."""
    from typing import Annotated

    import typer
    from rich.console import Console

    app = typer.Typer(
        help="Back up iMessage history to CSV.",
        add_completion=False,
        no_args_is_help=True,
    )
    console = Console()

    @app.command(help="Export every message in chat.db to a CSV file")
    def export(
        db: Annotated[Path, typer.Option(help="chat.db to read")] = Path.home()
        / "imessage"
        / "chat.db",
        out: Annotated[Path, typer.Option(help="CSV file to write")] = Path.home()
        / "imessage"
        / "messages.csv",
    ):
        count = export_csv(db, out)
        console.print(f"Wrote [bold]{count:,}[/] messages to [blue]{out}[/]")

    @app.command(help="Show version")
    def version():
        console.print("imessage_export 1.0.0")

    return app


if __name__ == "__main__":
    _build_app()()
