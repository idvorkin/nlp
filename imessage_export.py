#!/usr/bin/env -S uv run --script
# /// script
# requires-python = ">=3.13"
# dependencies = [
#     "typer",
#     "rich",
# ]
# ///
# ABOUTME: Exports an iMessage chat.db to CSV or archive txt, one row per message, in local time.
# ABOUTME: Decodes text stored only in attributedBody and shows Contacts names for numbers and groups.

import csv
import re
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

ARCHIVE_COLUMNS = ["date", "text", "is_from_me", "to_phone", "chat"]

APPLE_EPOCH = 978307200  # 2001-01-01 UTC as a unix timestamp
GROUP_CHAT_STYLE = 43
OBJECT_REPLACEMENT_CHAR = (
    "￼"  # placeholder Messages puts in text where an attachment sits
)
# Imported contacts sometimes carry the phone label as a last name, e.g. "Ann Cell"
PHONE_LABEL_WORDS = {"cell", "mobile", "home", "work", "iphone", "main"}
CONTACTS_DIR = Path.home() / "Library" / "Application Support" / "AddressBook"

MESSAGES_QUERY = """
SELECT m.ROWID, m.date, m.text, m.attributedBody, m.is_from_me,
       h.id AS handle, c.ROWID, c.chat_identifier, c.display_name, c.style
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

PARTICIPANTS_QUERY = """
SELECT chj.chat_id, h.id
FROM chat_handle_join chj
JOIN handle h ON h.ROWID = chj.handle_id
ORDER BY chj.chat_id, h.ROWID
"""

CONTACT_HANDLES_QUERY = """
SELECT r.ZFIRSTNAME, r.ZLASTNAME, r.ZNICKNAME, r.ZORGANIZATION, p.ZFULLNUMBER
FROM ZABCDPHONENUMBER p JOIN ZABCDRECORD r ON r.Z_PK = p.ZOWNER
UNION ALL
SELECT r.ZFIRSTNAME, r.ZLASTNAME, r.ZNICKNAME, r.ZORGANIZATION, e.ZADDRESS
FROM ZABCDEMAILADDRESS e JOIN ZABCDRECORD r ON r.Z_PK = e.ZOWNER
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


def handle_key(handle: str) -> str:
    """Normalize a phone number or email so differently formatted copies match."""
    if "@" in handle:
        return handle.strip().lower()
    digits = re.sub(r"\D", "", handle)
    return digits[-10:]


def contact_display_name(first, last, nickname, organization) -> str:
    return (
        " ".join(part for part in (first, last) if part)
        or nickname
        or organization
        or ""
    )


def is_clean_name(name: str) -> bool:
    words = name.lower().split()
    return not re.search(r"\d", name) and not (words and words[-1] in PHONE_LABEL_WORDS)


def read_contacts(contact_dbs) -> dict[str, str]:
    """Map handle_key -> contact name, preferring clean names, then the most complete one."""
    candidates: dict[str, set[str]] = {}
    for path in contact_dbs:
        db = sqlite3.connect(f"file:{path}?mode=ro", uri=True)
        try:
            for first, last, nickname, organization, handle in db.execute(
                CONTACT_HANDLES_QUERY
            ):
                name = contact_display_name(first, last, nickname, organization)
                if name and handle and handle_key(handle):
                    candidates.setdefault(handle_key(handle), set()).add(name)
        finally:
            db.close()
    return {
        key: sorted(names, key=lambda n: (not is_clean_name(n), -len(n), n))[0]
        for key, names in candidates.items()
    }


def find_contact_dbs() -> list[Path]:
    return sorted(CONTACTS_DIR.glob("**/AddressBook-v22.abcddb"))


def read_participants(db: sqlite3.Connection) -> dict[int, list[str]]:
    participants: dict[int, list[str]] = {}
    for chat_id, handle in db.execute(PARTICIPANTS_QUERY):
        participants.setdefault(chat_id, []).append(handle)
    return participants


def read_attachments(db: sqlite3.Connection) -> dict[int, list[str]]:
    attachments: dict[int, list[str]] = {}
    for message_id, name in db.execute(ATTACHMENTS_QUERY):
        attachments.setdefault(message_id, []).append(name or "")
    return attachments


def read_messages(db_path, contacts: dict[str, str] | None = None) -> list[dict]:
    contacts = contacts or {}

    def name(handle: str) -> str:
        return contacts.get(handle_key(handle), handle)

    db = sqlite3.connect(f"file:{db_path}?mode=ro", uri=True)
    try:
        attachments = read_attachments(db)
        participants = read_participants(db)
        rows = []
        for (
            rowid,
            date,
            text,
            body,
            is_from_me,
            handle,
            chat_id,
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
            is_group = style == GROUP_CHAT_STYLE
            members = ", ".join(name(h) for h in participants.get(chat_id, []))
            chat_name = display_name or (
                members if is_group and members else name(chat_identifier or "")
            )
            # My messages carry no sender handle; group chat ids must not be looked up as numbers
            if handle:
                recipient = name(handle)
            elif is_group:
                recipient = chat_identifier or ""
            else:
                recipient = name(chat_identifier or "")
            rows.append(
                {
                    "date": apple_time_to_local(date),
                    "text": content,
                    "is_from_me": int(bool(is_from_me)),
                    "to_phone": recipient,
                    "chat_name": chat_name,
                    "is_group": int(is_group),
                    "attachment": attachment,
                }
            )
        return rows
    finally:
        db.close()


def export_csv(db_path, out_path, contacts: dict[str, str] | None = None) -> int:
    rows = read_messages(db_path, contacts)
    with Path(out_path).open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=COLUMNS)
        writer.writeheader()
        writer.writerows(rows)
    return len(rows)


def escape_archive_text(text: str) -> str:
    """Keep each message on one line, writing line breaks and tabs as literal \\n and \\t."""
    return (
        text.replace("\r\n", "\\n")
        .replace("\r", "\\n")
        .replace("\n", "\\n")
        .replace("\t", "\\t")
    )


def export_archive_txt(
    db_path, out_path, contacts: dict[str, str] | None = None
) -> int:
    """Write the tab-separated archive format: date, text, is_from_me, to_phone, chat."""
    rows = read_messages(db_path, contacts)
    with Path(out_path).open("w", newline="", encoding="utf-8") as f:
        f.write("\t".join(ARCHIVE_COLUMNS) + "\n")
        for row in rows:
            text = escape_archive_text(row["text"]) or OBJECT_REPLACEMENT_CHAR
            chat = escape_archive_text(row["chat_name"])
            f.write(
                f"{row['date']}\t{text}\t{row['is_from_me']}\t{row['to_phone']}\t{chat}\n"
            )
    return len(rows)


def _build_app():
    """Wire the Typer app. Called only from __main__ so tests skip the typer import."""
    from typing import Annotated

    import typer
    from rich.console import Console

    app = typer.Typer(
        help="Back up iMessage history to CSV or the big_dump.txt archive format.",
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
        count = export_csv(db, out, read_contacts(find_contact_dbs()))
        console.print(f"Wrote [bold]{count:,}[/] messages to [blue]{out}[/]")

    @app.command(
        help="Export every message in chat.db to the big_dump.txt archive format"
    )
    def export_archive(
        db: Annotated[Path, typer.Option(help="chat.db to read")] = Path.home()
        / "imessage"
        / "chat.db",
        out: Annotated[Path, typer.Option(help="txt file to write")] = Path.home()
        / "imessage"
        / "messages.txt",
    ):
        count = export_archive_txt(db, out, read_contacts(find_contact_dbs()))
        console.print(f"Wrote [bold]{count:,}[/] messages to [blue]{out}[/]")

    @app.command(help="Show version")
    def version():
        console.print("imessage_export 1.0.0")

    return app


if __name__ == "__main__":
    _build_app()()
