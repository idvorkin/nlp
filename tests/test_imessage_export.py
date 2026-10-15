# ABOUTME: Tests for imessage_export, run against a small real SQLite chat.db built per test.
# ABOUTME: Covers rows, local dates, attributedBody decoding, groups, attachments, contact names and archive txt.

import csv
import sqlite3
from datetime import datetime

import pytest

from imessage_export import (
    COLUMNS,
    apple_time_to_local,
    decode_attributed_body,
    export_archive_txt,
    export_csv,
    handle_key,
    merge_archive,
    read_archive_txt,
    read_contacts,
    read_messages,
)

APPLE_EPOCH = 978307200  # 2001-01-01 UTC as a unix timestamp

SCHEMA = """
CREATE TABLE handle (ROWID INTEGER PRIMARY KEY, id TEXT);
CREATE TABLE chat (ROWID INTEGER PRIMARY KEY, chat_identifier TEXT, display_name TEXT, style INTEGER);
CREATE TABLE message (ROWID INTEGER PRIMARY KEY, text TEXT, attributedBody BLOB,
                      date INTEGER, is_from_me INTEGER, handle_id INTEGER);
CREATE TABLE chat_message_join (chat_id INTEGER, message_id INTEGER);
CREATE TABLE chat_handle_join (chat_id INTEGER, handle_id INTEGER);
CREATE TABLE attachment (ROWID INTEGER PRIMARY KEY, transfer_name TEXT);
CREATE TABLE message_attachment_join (message_id INTEGER, attachment_id INTEGER);
"""

# chat.style values Messages uses for 1:1 and group chats
ONE_TO_ONE = 45
GROUP = 43


def attributed_body(text: str) -> bytes:
    """Build an attributedBody blob shaped like the typedstream archives Messages writes."""
    raw = text.encode("utf-8")
    if len(raw) < 0x80:
        length = bytes([len(raw)])
    else:
        length = b"\x81" + len(raw).to_bytes(2, "little")
    return (
        b"\x04\x0bstreamtyped\x81\xe8\x03\x84\x01@\x84\x84\x84\x12NSAttributedString\x00"
        b"\x84\x84\x08NSObject\x00\x85\x92\x84\x84\x84\x08NSString\x01\x94\x84\x01+"
        + length
        + raw
        + b"\x86\x84\x02iI\x01\x05\x92\x84\x84\x84\x0cNSDictionary\x00"
    )


def apple_ns(dt: datetime) -> int:
    return int((dt.timestamp() - APPLE_EPOCH) * 1_000_000_000)


@pytest.fixture
def chat_db(tmp_path):
    path = tmp_path / "chat.db"
    db = sqlite3.connect(path)
    db.executescript(SCHEMA)
    db.executemany(
        "INSERT INTO handle VALUES (?, ?)", [(1, "+12065550001"), (2, "+12065550002")]
    )
    db.executemany(
        "INSERT INTO chat VALUES (?, ?, ?, ?)",
        [(10, "+12065550001", "", ONE_TO_ONE), (20, "chat123", "Family", GROUP)],
    )
    # The group has two participants, which is what duplicates rows in a naive chat_handle_join query.
    db.executemany(
        "INSERT INTO chat_handle_join VALUES (?, ?)", [(10, 1), (20, 1), (20, 2)]
    )
    db.commit()
    db.close()
    return path


def add_message(
    path,
    rowid,
    chat_id,
    when,
    *,
    text=None,
    body=None,
    from_me=0,
    handle_id=0,
    attachments=(),
):
    db = sqlite3.connect(path)
    db.execute(
        "INSERT INTO message VALUES (?, ?, ?, ?, ?, ?)",
        (rowid, text, body, apple_ns(when), from_me, handle_id),
    )
    db.execute("INSERT INTO chat_message_join VALUES (?, ?)", (chat_id, rowid))
    for i, name in enumerate(attachments):
        attachment_id = rowid * 100 + i
        db.execute("INSERT INTO attachment VALUES (?, ?)", (attachment_id, name))
        db.execute(
            "INSERT INTO message_attachment_join VALUES (?, ?)", (rowid, attachment_id)
        )
    db.commit()
    db.close()


def test_columns_start_with_blog_format():
    assert COLUMNS[:4] == ["date", "text", "is_from_me", "to_phone"]
    assert COLUMNS[4:] == ["chat_name", "is_group", "attachment"]


def test_apple_time_is_local_time():
    when = datetime(2024, 1, 26, 8, 9, 8)
    assert apple_time_to_local(apple_ns(when)) == "2024-01-26 08:09:08"


def test_apple_time_accepts_old_seconds_format():
    when = datetime(2016, 3, 3, 21, 0, 5)
    assert (
        apple_time_to_local(int(when.timestamp() - APPLE_EPOCH))
        == "2016-03-03 21:00:05"
    )


def test_decode_short_attributed_body():
    assert decode_attributed_body(attributed_body("on my way")) == "on my way"


def test_decode_long_attributed_body():
    long_text = "héllo 👋 " * 40
    assert decode_attributed_body(attributed_body(long_text)) == long_text


def test_decode_attributed_body_without_string_is_empty():
    assert decode_attributed_body(b"\x04\x0bstreamtyped\x81\xe8\x03") == ""
    assert decode_attributed_body(None) == ""


def test_one_to_one_message(chat_db):
    add_message(chat_db, 1, 10, datetime(2024, 1, 26, 8, 9, 8), text="hi", handle_id=1)
    assert read_messages(chat_db) == [
        {
            "date": "2024-01-26 08:09:08",
            "text": "hi",
            "is_from_me": 0,
            "to_phone": "+12065550001",
            "chat_name": "+12065550001",
            "is_group": 0,
            "attachment": "",
        }
    ]


def test_text_falls_back_to_attributed_body(chat_db):
    add_message(
        chat_db,
        1,
        10,
        datetime(2024, 1, 1),
        body=attributed_body("only in body"),
        from_me=1,
        handle_id=1,
    )
    [row] = read_messages(chat_db)
    assert row["text"] == "only in body"
    assert row["is_from_me"] == 1


def test_group_message_appears_once(chat_db):
    add_message(chat_db, 1, 20, datetime(2024, 1, 1), text="hey all", handle_id=2)
    rows = read_messages(chat_db)
    assert len(rows) == 1
    assert rows[0]["to_phone"] == "+12065550002"
    assert rows[0]["chat_name"] == "Family"
    assert rows[0]["is_group"] == 1


def test_my_group_message_uses_chat_identifier(chat_db):
    add_message(
        chat_db, 1, 20, datetime(2024, 1, 1), text="from me", from_me=1, handle_id=0
    )
    [row] = read_messages(chat_db)
    assert row["to_phone"] == "chat123"


def test_attachment_only_message_is_kept(chat_db):
    add_message(
        chat_db,
        1,
        10,
        datetime(2024, 1, 1),
        text="￼",
        handle_id=1,
        attachments=["IMG_1.jpg", "IMG_2.heic"],
    )
    [row] = read_messages(chat_db)
    assert row["text"] == ""
    assert row["attachment"] == "IMG_1.jpg;IMG_2.heic"


def test_message_without_text_or_attachment_is_skipped(chat_db):
    add_message(chat_db, 1, 10, datetime(2024, 1, 1), handle_id=1)
    assert read_messages(chat_db) == []


def test_rows_are_sorted_by_date(chat_db):
    add_message(chat_db, 1, 10, datetime(2024, 1, 2), text="second", handle_id=1)
    add_message(chat_db, 2, 20, datetime(2024, 1, 1), text="first", handle_id=2)
    assert [r["text"] for r in read_messages(chat_db)] == ["first", "second"]


def test_export_csv_round_trips_multiline_text(chat_db, tmp_path):
    add_message(
        chat_db,
        1,
        10,
        datetime(2024, 1, 1),
        text='line one\nline "two", three',
        handle_id=1,
    )
    out = tmp_path / "messages.csv"
    assert export_csv(chat_db, out) == 1
    with out.open(newline="", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    assert list(rows[0].keys()) == COLUMNS
    assert rows[0]["text"] == 'line one\nline "two", three'


def test_export_archive_txt_matches_big_dump_format(chat_db, tmp_path):
    add_message(chat_db, 1, 10, datetime(2024, 1, 1, 9, 0, 0), text="hi", handle_id=1)
    add_message(
        chat_db,
        2,
        10,
        datetime(2024, 1, 1, 9, 0, 1),
        text="line one\r\nline\ttwo\nthree",
        from_me=1,
        handle_id=1,
    )
    out = tmp_path / "big_dump.txt"
    assert export_archive_txt(chat_db, out) == 2
    assert out.read_text(encoding="utf-8").split("\n") == [
        "date\ttext\tis_from_me\tto_phone\tchat",
        "2024-01-01 09:00:00\thi\t0\t+12065550001\t+12065550001",
        "2024-01-01 09:00:01\tline one\\nline\\ttwo\\nthree\t1\t+12065550001\t+12065550001",
        "",
    ]


def test_export_archive_txt_keeps_attachment_placeholder(chat_db, tmp_path):
    add_message(
        chat_db, 1, 10, datetime(2024, 1, 1), handle_id=1, attachments=["IMG_1.jpg"]
    )
    out = tmp_path / "big_dump.txt"
    export_archive_txt(chat_db, out)
    assert out.read_text(encoding="utf-8").split("\n")[1].split("\t")[1] == "\ufffc"


CONTACTS_SCHEMA = """
CREATE TABLE ZABCDRECORD (Z_PK INTEGER PRIMARY KEY, ZFIRSTNAME TEXT, ZLASTNAME TEXT,
                          ZNICKNAME TEXT, ZORGANIZATION TEXT);
CREATE TABLE ZABCDPHONENUMBER (ZOWNER INTEGER, ZFULLNUMBER TEXT);
CREATE TABLE ZABCDEMAILADDRESS (ZOWNER INTEGER, ZADDRESS TEXT);
"""


@pytest.fixture
def contacts_db(tmp_path):
    path = tmp_path / "AddressBook-v22.abcddb"
    db = sqlite3.connect(path)
    db.executescript(CONTACTS_SCHEMA)
    db.executemany(
        "INSERT INTO ZABCDRECORD VALUES (?, ?, ?, ?, ?)",
        [
            (1, "Ann", "Lee", None, None),
            (2, "Bob", None, None, None),
            (3, None, None, None, "Pizza Place"),
            (4, "Ann 2065550001", "Lee", None, None),
            (5, "Ann", "Cell", None, None),
        ],
    )
    db.executemany(
        "INSERT INTO ZABCDPHONENUMBER VALUES (?, ?)",
        [
            (1, "+1 (206) 555-0001"),
            (2, "206-555-0002"),
            (3, "5550003"),
            (4, "+12065550001"),
            (5, "1 (206) 555-0001"),
        ],
    )
    db.execute("INSERT INTO ZABCDEMAILADDRESS VALUES (2, 'Bob@Example.com')")
    db.commit()
    db.close()
    return path


def test_handle_key_matches_phone_formats():
    assert handle_key("+12065550001") == handle_key("(206) 555-0001") == "2065550001"
    assert handle_key("5550003") == "5550003"
    assert handle_key("Bob@Example.com") == "bob@example.com"


def test_read_contacts_maps_handles_to_names(contacts_db):
    contacts = read_contacts([contacts_db])
    assert contacts[handle_key("+12065550002")] == "Bob"
    assert contacts[handle_key("bob@example.com")] == "Bob"
    assert contacts[handle_key("5550003")] == "Pizza Place"


def test_read_contacts_prefers_most_complete_name_without_digits(contacts_db):
    assert read_contacts([contacts_db])[handle_key("+12065550001")] == "Ann Lee"


def test_one_to_one_message_uses_contact_names(chat_db, contacts_db):
    add_message(chat_db, 1, 10, datetime(2024, 1, 1), text="hi", handle_id=1)
    [row] = read_messages(chat_db, read_contacts([contacts_db]))
    assert row["to_phone"] == "Ann Lee"
    assert row["chat_name"] == "Ann Lee"


def test_unnamed_group_lists_participant_names(chat_db, contacts_db):
    db = sqlite3.connect(chat_db)
    db.execute("UPDATE chat SET display_name = '' WHERE ROWID = 20")
    db.commit()
    db.close()
    add_message(chat_db, 1, 20, datetime(2024, 1, 1), text="yo", from_me=1)
    [row] = read_messages(chat_db, read_contacts([contacts_db]))
    assert row["chat_name"] == "Ann Lee, Bob"
    assert row["to_phone"] == "chat123"


def test_unnamed_group_without_contacts_lists_handles(chat_db):
    db = sqlite3.connect(chat_db)
    db.execute("UPDATE chat SET display_name = NULL WHERE ROWID = 20")
    db.commit()
    db.close()
    add_message(chat_db, 1, 20, datetime(2024, 1, 1), text="yo", handle_id=2)
    [row] = read_messages(chat_db)
    assert row["chat_name"] == "+12065550001, +12065550002"


def test_my_one_to_one_message_uses_contact_name(chat_db, contacts_db):
    add_message(chat_db, 1, 10, datetime(2024, 1, 1), text="hi", from_me=1)
    [row] = read_messages(chat_db, read_contacts([contacts_db]))
    assert row["to_phone"] == "Ann Lee"


def test_group_chat_id_never_matches_a_contact(chat_db, contacts_db):
    db = sqlite3.connect(chat_db)
    db.execute("UPDATE chat SET chat_identifier = 'chat2065550001' WHERE ROWID = 20")
    db.commit()
    db.close()
    add_message(chat_db, 1, 20, datetime(2024, 1, 1), text="yo", from_me=1)
    [row] = read_messages(chat_db, read_contacts([contacts_db]))
    assert row["to_phone"] == "chat2065550001"


def write_lines(path, lines):
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return path


def test_read_archive_txt_four_columns(tmp_path):
    path = write_lines(
        tmp_path / "big_dump.txt",
        [
            "date\ttext\tis_from_me\tto_phone",
            "2016-01-01 10:00:00\thi\t1\t+12065550001",
        ],
    )
    assert read_archive_txt(path) == [
        {
            "date": "2016-01-01 10:00:00",
            "text": "hi",
            "is_from_me": "1",
            "to_phone": "+12065550001",
            "chat": "",
        }
    ]


def test_read_archive_txt_five_columns(tmp_path):
    path = write_lines(
        tmp_path / "big_dump.txt",
        [
            "date\ttext\tis_from_me\tto_phone\tchat",
            "2016-01-01 10:00:00\thi\\nthere\t0\tAnn Lee\tFamily",
        ],
    )
    [row] = read_archive_txt(path)
    assert row["text"] == "hi\\nthere"
    assert row["chat"] == "Family"


def test_read_archive_txt_repairs_year_missing_leading_digit(tmp_path):
    path = write_lines(
        tmp_path / "big_dump.txt",
        [
            "date\ttext\tis_from_me\tto_phone",
            "2022-12-06 05:05:27\tfirst\t1\t+12065550001",
            "022-12-06 05:10:10\tsecond\t0\t+12065550001",
        ],
    )
    assert [(r["date"], r["text"]) for r in read_archive_txt(path)] == [
        ("2022-12-06 05:05:27", "first"),
        ("2022-12-06 05:10:10", "second"),
    ]


def test_read_archive_txt_keeps_raw_line_breaks_inside_a_message(tmp_path):
    path = write_lines(
        tmp_path / "big_dump.txt",
        [
            "date\ttext\tis_from_me\tto_phone",
            "2016-01-01 10:00:00\tline one\r",
            "line two\t1\t+12065550001",
        ],
    )
    [row] = read_archive_txt(path)
    assert row["text"] == "line one\\nline two"
    assert row["to_phone"] == "+12065550001"


def archive_row(date, text, is_from_me="1", to_phone="+12065550001", chat=""):
    return {
        "date": date,
        "text": text,
        "is_from_me": is_from_me,
        "to_phone": to_phone,
        "chat": chat,
    }


def test_merge_archive_keeps_export_and_adds_only_messages_it_lacks():
    export = [
        archive_row("2016-01-01 10:00:00", "same", chat="Ann Lee"),
        archive_row("2016-01-01 10:00:01", "photo caption", chat="Ann Lee"),
        archive_row("2016-01-01 10:00:02", "ok", chat="Ann Lee"),
    ]
    archive = [
        archive_row("2015-12-31 09:00:00", "deleted since"),
        archive_row("2016-01-01 10:00:00", "same"),
        archive_row("2016-01-01 10:00:01", "\ufffcphoto caption"),
        archive_row("2016-01-01 10:00:02", "different message same second"),
        archive_row("2016-01-01 10:00:03", "\\n\ufffc"),
    ]
    merged = merge_archive(archive, export, contacts={"2065550001": "Ann Lee"})
    assert [(r["date"], r["text"], r["to_phone"]) for r in merged] == [
        ("2015-12-31 09:00:00", "deleted since", "Ann Lee"),
        ("2016-01-01 10:00:00", "same", "+12065550001"),
        ("2016-01-01 10:00:01", "photo caption", "+12065550001"),
        ("2016-01-01 10:00:02", "ok", "+12065550001"),
        ("2016-01-01 10:00:02", "different message same second", "Ann Lee"),
    ]


def test_export_archive_txt_merges_existing_archive(chat_db, contacts_db, tmp_path):
    add_message(chat_db, 1, 10, datetime(2024, 1, 1, 9, 0, 0), text="hi", handle_id=1)
    old = write_lines(
        tmp_path / "big_dump_full.txt",
        [
            "date\ttext\tis_from_me\tto_phone\tchat",
            "2020-05-05 12:00:00\tdeleted since\t0\tAnn Lee\tAnn Lee",
            "2024-01-01 09:00:00\thi\t0\tAnn Lee\tAnn Lee",
        ],
    )
    count = export_archive_txt(
        chat_db, old, read_contacts([contacts_db]), merge_with=old
    )
    assert count == 2
    assert old.read_text(encoding="utf-8").split("\n")[1:3] == [
        "2020-05-05 12:00:00\tdeleted since\t0\tAnn Lee\tAnn Lee",
        "2024-01-01 09:00:00\thi\t0\tAnn Lee\tAnn Lee",
    ]
