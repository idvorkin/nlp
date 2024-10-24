#!python3
# Remove line too long
# pep8: disable=E501

import sys
from typing import Dict
import numpy as np
import typer
from loguru import logger
import glob
import datetime
import re
import pandas as pd
from pydantic import BaseModel
from rich.console import Console
from rich.table import Table
from rich import print
from icecream import ic
from pathlib import Path


console = Console()
app = typer.Typer(no_args_is_help=True)


def df_to_table(
    pandas_dataframe: pd.DataFrame,
    show_index: bool = True,
    index_name: str | None = None,
) -> Table:
    """Convert a pandas.DataFrame obj into a rich.Table obj.

    Args:
        pandas_dataframe (DataFrame): A Pandas DataFrame to be converted to a rich Table.
        rich_table (Table): A rich Table that should be populated by the DataFrame values.
        show_index (bool): Add a column with a row count to the table. Defaults to True.
        index_name (str, optional): The column name to give to the index column. Defaults to None, showing no value.

    Returns:
        Table: The rich Table instance passed, populated with the DataFrame values."""

    rich_table = Table()
    if show_index:
        index_name = str(index_name) if index_name else ""
        rich_table.add_column(index_name)

    for column in pandas_dataframe.columns:
        rich_table.add_column(str(column))

    for index, value_list in enumerate(pandas_dataframe.values.tolist()):
        row = [str(index)] if show_index else []
        # set each row, but if a value is a date, format to just be a date
        for value in value_list:
            if isinstance(value, datetime.date):
                value = value.strftime("%Y-%m-%d")
            row.append(str(value))
        rich_table.add_row(*row)

    return rich_table


class Week(BaseModel):
    date: datetime.date = datetime.date.today()
    category_to_score: Dict[str, int | datetime.date] = {}

    def to_dict(self):
        copy = self.category_to_score.copy()
        copy["date"] = self.date
        return copy

    @classmethod
    def from_file(cls, fp):
        week = Week()
        for line in fp.readlines():
            line = line.strip()  # Clear trailing spaces
            find_weeks = re.findall("\\d\\d\\d\\d-\\d\\d-\\d\\d", line)
            isWeekLine = len(find_weeks) == 1
            if isWeekLine:
                the_week = find_weeks[0]
                week.date = datetime.date.fromisoformat(the_week)
                continue
            isCategory = len(re.findall("##.*(/5)", line)) != 0
            if not isCategory:
                continue

            line = line[3:]  # strip the ##
            # Example line
            # blah blah (X/5)
            category = line.split(" (")[0]

            # Do some category renames

            category = category.replace("Health", "")
            category = category.replace("Habits", "")
            category = category.replace("House and goods", "Stuff")
            category = category.replace("Mental quicksand", "Peace")
            category = category.replace("Inner Peace", "Peace")
            category = category.strip()

            score = line.split(" (")[1][0]
            if not score.isdigit():
                # implies score isn't filled in
                continue

            week.category_to_score[category] = int(score)

        return week


valid_week_glob = "*202*md"


def df_for_weeks():
    weeks = [Week.from_file(open(f)).to_dict() for f in glob.glob(valid_week_glob)]
    df = pd.DataFrame(weeks)
    ic(df.columns)
    cols_not_date = [c for c in df.columns if c != "date"]
    # Dump dates that where all columns but date are 0 or none, print them
    trouble_rows = df[
        (df[cols_not_date] == 0).all(axis=1) | (df[cols_not_date].isna().all(axis=1))
    ]
    trouble_rows["date"] = pd.to_datetime(trouble_rows["date"])
    # filter out any dates earlier then 2022
    trouble_rows = trouble_rows[trouble_rows["date"].dt.year > 2022]
    if not trouble_rows.empty:
        ic(trouble_rows["date"])

    # Filter out lines where all columns but date are 0 or none, print them

    df = df[
        (df[cols_not_date] != 0).any(axis=1) & (df[cols_not_date].notna().any(axis=1))
    ]

    df["date"] = pd.to_datetime(df["date"])
    return df


@app.command()
def df():
    df = df_for_weeks()
    print(df)


@app.command()
def debug_week():
    print("Not implemented yet")
    pass


@app.command()
def table(weeks: int = 30, transpose: bool = False):
    df = df_for_weeks()
    df = df.sort_values("date", ascending=False)
    df: pd.DataFrame = df[
        [
            "date",
            "Physical",
            "Emotional",
            "Peace",
            "Work",
            "Motivation",
            "Family",
            "Magic",
            "Identity",
            "Friends",
        ]
    ][:weeks]  # type:ignore

    if transpose:
        df = df.set_index("date").sort_index(ascending=False)
        df = df.T
        print(df_to_table(df, show_index=True))
    else:
        print(df_to_table(df, show_index=False))


@app.command()
def spark(weeks: int = 50, transpose: bool = False, latest_on_right: bool = True):
    from sparklines import sparklines

    df = df_for_weeks()
    df = df.sort_values("date", ascending=False)
    df: pd.DataFrame = df[
        [
            "date",
            "Physical",
            "Emotional",
            "Peace",
            "Work",
            "Motivation",
            "Family",
            "Magic",
            "Identity",
            "Friends",
        ]
    ][:weeks]  # type:ignore

    df = df.set_index("date").sort_index(ascending=False)
    # convert all flots to int

    # for each column in df, create a sparkline and print it
    rich_table = Table()
    rich_table.add_column("Category")
    for col in df.columns:
        clean = np.nan_to_num(df[col], nan=0).astype(int)
        spark = sparklines(clean, minimum=0, maximum=6)
        spark_str = "".join(spark)
        # reverse the string
        if latest_on_right:
            spark_str = spark_str[::-1]
        col = col.ljust(max([len(c) for c in df.columns]) + 1)
        print(f"{col}[blue]{spark_str}[/blue]")


@app.command()
def csv(transpose: bool = False, weeks: int = 10):
    df = df_for_weeks()
    base = df.set_index("date").sort_index()
    base = base[weeks * -1 :]
    if transpose:
        base.T.to_csv(sys.stdout)
    else:
        base.to_csv(sys.stdout)


@app.command()
def print_prompt():
    report_template_path = Path.home() / "gits/igor2/week_report/week_template.md"
    report_template = report_template_path.read_text()
    useful_prompt_info = f"""

    Let's write a report together summarizing my week. I'll start by giving you my journal entries.

    <instructions>
    When writing the report:

    * For every section,use bullet points, 3 to 6 each. Be specific use nouns and examples

    * Tech Exploration is Personal: Tech exploration, including coding, AI tools, and similar activities, is for your personal enjoyment and growth, not for creating joy for others. Make sure to frame this appropriately.

    * Tristan Only for Physical Health: Tristan, your trainer, should only be mentioned in the context of physical health and fitness. Avoid including him in other sections like friends unless it's directly related to training or fitness progress.

    * Joy Refers to Others' Joy: "Joy" activities refer to activities that create joy for others, not your own joy. Be mindful when discussing joy-related activities.


    * Categorize Work and Tech Correctly: Work-related tasks like date affinity for memories or Cyrus' support case belong in the Work section, not Tech Guru.
    </instructions>

    Report Template
    <template>
    {report_template}
    </template>

    """
    print(useful_prompt_info)


@logger.catch
def main():
    app()


if __name__ == "__main__":
    main()
