from pathlib import Path

import pandas as pd
import torch
from tqdm import tqdm  # shows progress
from transformers import AutoModelForQuestionAnswering, AutoTokenizer

base = Path.cwd() / "data-merged" / "data" / "air-exercise-2" / "Part-3"
answers_path = base / "msmarco-fira-21.qrels.qa-answers.tsv"
tuples_path = base / "msmarco-fira-21.qrels.qa-tuples.tsv"
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


def parse_tuples(tuples_path: Path) -> pd.DataFrame:
    tuples = pd.DataFrame(columns=["queryid", "documentid", "relevance-grade", "question", "context", "text-selection"])

    with open(tuples_path, "r") as tuples_f:
        for line_count, line in enumerate(tqdm(tuples_f.readlines()), 1):
            if line_count > 10:
                break
            split_line = line.strip().split("\t")
            qid = split_line[0]
            docid = split_line[1]
            rel_grade = split_line[2]
            question = split_line[3]
            context = split_line[4]
            text_selection = "\t".join(split_line[5:]).strip()
            tuples = tuples.append({"queryid": qid, "documentid": docid, "relevance-grade": rel_grade, "question": question, "context": context, "text-selection": text_selection}, ignore_index=True)

    return tuples


tuples = parse_tuples(tuples_path)
print("tuples parsed")


model_name = "distilbert-base-uncased-distilled-squad"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForQuestionAnswering.from_pretrained(model_name)
print("model downloaded")

inputs0 = tokenizer(tuples["question"][0], tuples["context"][0], return_tensors="pt")
print("input tokenized")
output0 = model(**inputs0)
print("model called successfully")


answer_start_idx = torch.argmax(output0.start_logits)
answer_end_idx = torch.argmax(output0.end_logits)

answer_tokens = inputs0.input_ids[0, answer_start_idx : answer_end_idx + 1]

answer = tokenizer.decode(answer_tokens)

print("ques: {}\nanswer: {}".format(tuples["question"][0], answer))
