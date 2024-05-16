"""
Use the transformers library to download a pre-trained extractive QA model from the HuggingFace model hub and run the extractive QA pipeline on the top-1 neural re-ranking result of the MSMARCO FIRA set as well as on the gold-label pairs of MSMARCO-FiRA-2021 (created in 2021).

- import a pre-trained extractive QA model from the huggingface model hub to use. Implement code to load the model, tokenize query passage pairs, and run inference, store results with the HuggingFace library -> The goal of extractive QA is to provide one or more text-spans that answers a given (query,passage) pair

- Evaluate both your top-1 (or more) MSMARCO passage results from the best re-ranking model using **msmarco-fira-21.qrels.qa-answers.tsv** (only evaluate the overlap of pairs that are in the result and the qrels) + the provided FiRA gold-label pairs **msmarco-fira-21.qrels.qa-tuples.tsv** using the provided qa evaluation methods in core_metrics.py with the MSMARCO-FiRA-2021 QA labels

data:

-   `msmarco-fira-21.qrels.qa-answers.tsv`: Used to evaluate the QA model.

    No header.

    Format: `queryid, documentid, relevance-grade, text-selection (multiple answers possible, split with tab)`

-   `msmarco-fira-21.qrels.qa-tuples.tsv`:

    No header.

    Format: `queryid, documentid, relevance-grade, query-text, document-text, text-selection (multiple answers possible, split with tab)`

-   `msmarco-fira-21.qrels.retrieval.tsv`:

    No header.

    Example: `135386 0 100163 3`

    Format: `queryid, 0, documentid, relevance-grade`

"""

import pandas as pd
from transformers import AutoTokenizer, AutoModelForQuestionAnswering
from tqdm import tqdm
from pathlib import Path
import torch


base = Path.cwd() / "data-merged" / "data" / "air-exercise-2" / "Part-3"
answers = pd.read_csv(base / "msmarco-fira-21.qrels.qa-answers.tsv", sep="\t", header=None, names=["queryid", "documentid", "relevance-grade", "text-selection"], converters={"text-selection": lambda x: x.split("\t")})
tuples: pd.DataFrame = pd.read_csv(base / "msmarco-fira-21.qrels.qa-tuples.tsv", sep="\t", header=None, names=["queryid", "documentid", "relevance-grade", "query-text", "document-text", "text-selection"], converters={"text-selection": lambda x: x.split("\t")})
retrieval: pd.DataFrame = pd.read_csv(base / "msmarco-fira-21.qrels.retrieval.tsv", sep="\t", header=None, names=["queryid", "Q0", "documentid", "relevance-grade"])

# tokenizer = AutoTokenizer.from_pretrained("distilbert-base-uncased-distilled-squad")
# model = AutoModelForQuestionAnswering.from_pretrained("distilbert-base-uncased-distilled-squad")


# def tokenize_inputs(query: str, passage: str):
#     inputs = tokenizer.encode_plus(
#         query,
#         passage,
#         add_special_tokens=True,
#         max_length=512,
#         padding='max_length',
#         truncation=True,
#         return_tensors="pt",
#     )
#     return inputs["input_ids"], inputs["attention_mask"]


# def predict_answer(input_ids, attention_mask):
#     outputs = model(input_ids=input_ids, attention_mask=attention_mask)
#     scores = outputs.start_logits + outputs.end_logits
#     all_answers = []
#     for score in scores[0]:
#         answer_start = torch.argmax(score).item()
#         answer_end = torch.argmax(torch.cat((score[:answer_start], score[answer_start+1:]))) + answer_start + 1
#         answer = tokenizer.convert_tokens_to_string(tokenizer.convert_ids_to_tokens(input_ids[0][answer_start:answer_end]))
#         all_answers.append(answer)
#     return all_answers

# # queries = [...]  # List of queries
# queries = retrieval["queryid"].unique()
# # passages = [...]  # Corresponding list of passages
# passages = retrieval["documentid"].unique()

# predictions = []
# for query, passage in zip(queries, passages):
#     input_ids, attention_mask = tokenize_inputs(query, passage)
#     answers = predict_answer(input_ids, attention_mask)
#     predictions.append({"query": query, "passage": passage, "answers": answers})

#     print(f"Query: {query}")
#     print(f"Passage: {passage}")
#     print(f"Answers: {answers}")
#     print()
