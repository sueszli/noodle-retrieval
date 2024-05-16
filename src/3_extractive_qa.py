"""
Use the transformers library to download a pre-trained extractive QA model from the HuggingFace model hub and run the extractive QA pipeline on the top-1 neural re-ranking result of the MSMARCO FIRA set as well as on the gold-label pairs of MSMARCO-FiRA-2021 (created in 2021).

- Select a pre-trained extractive QA model from the huggingface model hub to use
- Implement code  to load the model, tokenize query passage pairs, and run inference, store results with the HuggingFace library -> The goal of extractive QA is to provide one or more text-spans that answers a given (query,passage) pair

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
from transformers import pipeline
from tqdm import tqdm
from pathlib import Path


base = Path.cwd() / "data-merged" / "data" / "air-exercise-2" / "Part-3"
answers: pd.DataFrame = pd.read_csv(base / "msmarco-fira-21.qrels.qa-answers.tsv", sep="\t")
tuples: pd.DataFrame = pd.read_csv(base / "msmarco-fira-21.qrels.qa-tuples.tsv", sep="\t")
retrieval: pd.DataFrame = pd.read_csv(base / "msmarco-fira-21.qrels.retrieval.tsv", sep="\t")

# qa = pipeline("question-answering", model="distilbert-base-uncased-distilled-squad", tokenizer="distilbert-base-uncased")

# # evaluate top-1 re-ranking results
# retrieval = retrieval[retrieval["documentid"].isin(answers["documentid"].unique())]
# len_a = len(answers["documentid"].unique())
# len_r = len(retrieval["documentid"].unique())
# assert len_a == len_r

# # evaluate gold-label pairs
# tuples = tuples[tuples["documentid"].isin(answers["documentid"].unique())]
# len_a = len(answers["documentid"].unique())
# len_t = len(tuples["documentid"].unique())
# assert len_a == len_t

# # evaluate top-1 re-ranking results
# results = []
# for _, row in tqdm(retrieval.iterrows(), total=len(retrieval)):
#     query_id = row["queryid"]
#     doc_id = row["documentid"]
#     text = tuples[(tuples["queryid"] == query_id) & (tuples["documentid"] == doc_id)]["query-text"].values[0]
