import pandas as pd
from transformers import AutoTokenizer, AutoModelForQuestionAnswering
from tqdm import tqdm # shows progress
from pathlib import Path
import torch


base = Path.cwd() / "data-merged" / "data" / "air-exercise-2" / "Part-3"
answers_path = base / "msmarco-fira-21.qrels.qa-answers.tsv"
tuples_path  = base / "msmarco-fira-21.qrels.qa-tuples.tsv"
retrieval_path = base / "msmarco-fira-21.qrels.retrieval.tsv"


"""
manual parsing because pandas.read_csv() does not work.
content needs to be cleaned and has an inconsistent number of columns.
"""

def parse_answers(answers_path: Path) -> pd.DataFrame:
    answers: pd.DataFrame = pd.DataFrame(columns=["queryid", "documentid", "relevance-grade", "text-selection"])
    answers_f = open(answers_path, "r")
    for line in tqdm(answers_f.readlines()):
        split_line = line.strip().split("\t")
        qid = split_line[0]
        docid = split_line[1]
        rel_grade = split_line[2]
        text_selection = split_line[3:]
        answers = answers.append({"queryid": qid, "documentid": docid, "relevance-grade": rel_grade, "text-selection": text_selection}, ignore_index=True)
    answers_f.close()
    return answers

answers: pd.DataFrame = parse_answers(answers_path)

# ... same for tuples and retrieval

# tokenizer = AutoTokenizer.from_pretrained("distilbert-base-uncased-distilled-squad")
# model = AutoModelForQuestionAnswering.from_pretrained("distilbert-base-uncased-distilled-squad")
