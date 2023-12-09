from icecream import ic
ic = print
import os
from typing import List, Dict
import glob
import pickle
tmp = os.path.expanduser("~/tmp")
from life import GetPychiatristReport

def load_all_reports():
    # load from pickle file
    return pickle.load(open(f"{tmp}/reports.pkl", "rb"))

def load_raw_reports():
    reports:List[life.GetPychiatristReport] = []
    report_path = os.path.expanduser("~/tmp/journal_report")
    # loop through the globbed json files
    for file in glob.glob(report_path + "/*.json"):
            try:
                text = open(file).read()
                report = life.GetPychiatristReport.model_validate_json(text)
                reports.append(report)
            except Exception as e:
                ic("failed to load file: ", file)
                ic (e)
    return reports


# pickle.dump(reports, open(f"{tmp}/reports.pkl", "wb"))
reports =  load_all_reports()
print(len(reports))

import pandas as pd

def report_to_people(r:GetPychiatristReport):
    row:Dict = {"date": r.Date}
    for p in r.PeopleInEntry:
        sentiment = p.Sentiment.lower()
        if sentiment in ["not mentioned", "unmentioned"]:
            continue
        if sentiment == "concerned":
            sentiment = "concern"
        row[p.Name.lower()] = sentiment

    return row

df = pd.DataFrame([report_to_people(r) for r in reports])
df = df.set_index("date")
df["tori"].value_counts()

df["clive"].value_counts()


def report_to_positive(r:GetPychiatristReport):
    row:Dict = {"date": r.Date}
    row["Positive"] = [c.reason for c in r.PostiveEmotionCause]
    return row

def report_to_negative(r:GetPychiatristReport):
    row:Dict = {"date": r.Date}
    row["Negative"] = [c.reason for c in r.NegativeEmotionCause]
    return row

def report_to_summary(r:GetPychiatristReport):
    row:Dict = {"date": r.Date}
    row["Summary"] = r.PointFormSummaryOfEntry
    return row

df = pd.DataFrame([report_to_summary(r) for r in reports]).set_index("date").sort_index().explode("Summary")
df
# output df.posiive to tmp/positive
# filter df to oonly yeras 2012-2013
df["2021-01-1":"2033/1/1"].to_csv(f"{tmp}/summary.csv")


