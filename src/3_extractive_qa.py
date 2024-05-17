"""
Use the transformers library to download a pre-trained extractive QA model from the HuggingFace model hub and run the extractive QA pipeline on the top-1 neural re-ranking result of the MSMARCO FIRA set as well as on the gold-label pairs of MSMARCO-FiRA-2021 (created in 2021).

- import a pre-trained extractive QA model from the huggingface model hub to use. Implement code to load the model, tokenize query passage pairs, and run inference, store results with the HuggingFace library -> The goal of extractive QA is to provide one or more text-spans that answers a given (query,passage) pair

- Evaluate both your top-1 (or more) MSMARCO passage results from the best re-ranking model using **msmarco-fira-21.qrels.qa-answers.tsv** (only evaluate the overlap of pairs that are in the result and the qrels) + the provided FiRA gold-label pairs **msmarco-fira-21.qrels.qa-tuples.tsv** using the provided qa evaluation methods in core_metrics.py with the MSMARCO-FiRA-2021 QA labels

data:

-   `msmarco-fira-21.qrels.qa-answers.tsv`: Used to evaluate the QA model.

    No header.

    Example: `810210	1029662	3		wide variety of conditions, such as hemorrhoids, diverticula, inflammatory bowel disease, rectal prolapse, colorectal cancer, rectal abscesses, intestinal infections, peptic ulcer, intestinal polyps, constipation or anal fissures`

    Format: `queryid, documentid, relevance-grade, text-selection (multiple answers possible, split with tab)`

-   `msmarco-fira-21.qrels.qa-tuples.tsv`:

    No header.

    Example: `480536	4049888	2	price of bath fitter for a tub	My parents needed to replace their tub in the 2nd bathroom of their manufactured home. They got a quote from Bath Fitter for $4800.00 less a senior citizen's discount of $500.00. So the bid was for $4300.00. My parents are retired and don't have a whole lot of money so they couldn't use Bath Fitter.		My parents needed to replace their tub in the 2nd bathroom of their manufactured home. They got a quote from Bath Fitter for $4800.00 less a senior citizen's discount of $500.00. So the bid was for $4300.00. My parents are retired and don't have a whole lot of money so they couldn't use Bath Fitter`

    Format: `queryid, documentid, relevance-grade, query-text, document-text, text-selection (multiple answers possible, split with tab)`

-   `msmarco-fira-21.qrels.retrieval.tsv`:

    No header.

    Example: `1007473 0 7251748 3`

    Format: `queryid, 0, documentid, relevance-grade`

"""

import pandas as pd
from transformers import AutoTokenizer, AutoModelForQuestionAnswering
from tqdm import tqdm # shows progress
from pathlib import Path
import torch


base = Path.cwd() / "data-merged" / "data" / "air-exercise-2" / "Part-3"
answers_path = base / "msmarco-fira-21.qrels.qa-answers.tsv"
tuples_path  = base / "msmarco-fira-21.qrels.qa-tuples.tsv"
retrieval_path = base / "msmarco-fira-21.qrels.retrieval.tsv"

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

# answers: pd.DataFrame = parse_answers(answers_path)




# tokenizer = AutoTokenizer.from_pretrained("distilbert-base-uncased-distilled-squad")
# model = AutoModelForQuestionAnswering.from_pretrained("distilbert-base-uncased-distilled-squad")


# def tokenize_inputs(query_text, passage_text):
#     inputs = tokenizer.encode_plus(
#         query_text,
#         passage_text,
#         add_special_tokens=True,
#         return_tensors="pt",
#         max_length=512,
#         truncation=True,
#         padding='max_length'
#     )
#     return {
#         "input_ids": inputs["input_ids"].flatten(),
#         "attention_mask": inputs["attention_mask"].flatten()
#     }


# def run_inference(tokenized_inputs):
#     outputs = model(**tokenized_inputs)
#     scores = outputs.start_logits + outputs.end_logits
#     all_answers = []
#     for i in range(len(scores)):
#         answer_start = torch.argmax(outputs.start_logits[i])
#         answer_end = torch.argmax(outputs.end_logits[i]) + 1
#         answer = tokenizer.convert_tokens_to_string(tokenizer.convert_ids_to_tokens(inputs["input_ids"][i][answer_start:answer_end]))
#         all_answers.append(answer)
#     return all_answers

# results_df = pd.DataFrame(columns=["queryid", "documentid", "relevance-grade", "predicted_answer"])


# for index, row in tuples.iterrows():

#     query_text = row['query-text']
#     document_text = row['document-text']
#     tokenized_inputs = tokenize_inputs(query_text, document_text)
#     predicted_answers = run_inference(tokenized_inputs)
#     # Assuming the first prediction is the correct one for simplicity
#     results_df = results_df.append({"queryid": row['queryid'], "documentid": row['documentid'], "relevance-grade": row['relevance-grade'], "predicted_answer": predicted_answers[0]}, ignore_index=True)

#     print(f"Query: {query_text}")
#     print(f"Document: {document_text}")
#     print(f"Predicted Answer: {predicted_answers[0]}")
#     print("---------------------------------------------------")


# # def tokenize_inputs(query_text, passage_text):
# #     inputs = tokenizer.encode_plus(
# #         query_text,
# #         passage_text,
# #         add_special_tokens=True,
# #         return_tensors="pt",
# #         max_length=512,
# #         truncation=True,
# #         padding='max_length'
# #     )
# #     return {
# #         "input_ids": inputs["input_ids"].flatten(),
# #         "attention_mask": inputs["attention_mask"].flatten()
# #     }


# # def run_inference(tokenized_inputs):
# #     outputs = model(**tokenized_inputs)
# #     scores = outputs.start_logits + outputs.end_logits
# #     all_answers = []
# #     for i in range(len(scores)):
# #         answer_start = torch.argmax(outputs.start_logits[i])
# #         answer_end = torch.argmax(outputs.end_logits[i]) + 1
# #         answer = tokenizer.convert_tokens_to_string(tokenizer.convert_ids_to_tokens(inputs["input_ids"][i][answer_start:answer_end]))
# #         all_answers.append(answer)
# #     return all_answers

# # results_df = pd.DataFrame(columns=["queryid", "documentid", "relevance-grade", "predicted_answer"])

# # for index, row in tuples.iterrows():
# #     query_text = row['query-text']
# #     document_text = row['document-text']
# #     tokenized_inputs = tokenize_inputs(query_text, document_text)
# #     predicted_answers = run_inference(tokenized_inputs)
# #     # Assuming the first prediction is the correct one for simplicity
# #     results_df = results_df.append({"queryid": row['queryid'], "documentid": row['documentid'], "relevance-grade": row['relevance-grade'], "predicted_answer": predicted_answers[0]}, ignore_index=True)

# #     print(f"Query: {query_text}")
# #     print(f"Document: {document_text}")
# #     print(f"Predicted Answer: {predicted_answers[0]}")
# #     print("---------------------------------------------------")

# # # Here, you would call the evaluation functions from core_metrics.py to evaluate the model's performance
# # # This step requires the implementation of the evaluation logic in core_metrics.py, which is not provided in the initial context.
