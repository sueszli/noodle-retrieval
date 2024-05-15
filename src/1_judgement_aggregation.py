from pathlib import Path
import pandas as pd
from pandas.core.groupby.generic import DataFrameGroupBy
import numpy as np
import torch
from torch.nn.functional import cosine_similarity
from typing import List, Tuple


base_in = Path.cwd() / "data-merged" / "air-exercise-2" / "Part-1"
base_out = Path.cwd() / "data-merged"  # output of previous script

docs = pd.read_csv(base_out / "fira-22.documents.embeddings.tsv", sep="\t")
queries = pd.read_csv(base_out / "fira-22.queries.embeddings.tsv", sep="\t")
judgements: pd.DataFrame = pd.read_csv(base_in / "fira-22.judgements-anonymized.tsv", sep="\t")


def preprocess_docs(docs: pd.DataFrame) -> pd.DataFrame:
    # drop columns
    docs = docs[["doc_id", "doc_embedding"]]

    # remove documents without judgements
    docs = docs[docs["doc_id"].isin(judgements["documentId"].unique())]
    len_j = len(judgements["documentId"].unique())
    len_d = len(docs["doc_id"].unique())
    assert len_j == len_d

    return docs


def preprocess_queries(queries: pd.DataFrame) -> pd.DataFrame:
    # drop columns
    queries = queries[["query_id", "query_embedding"]]

    # remove queries without judgements
    queries = queries[queries["query_id"].isin(judgements["queryId"].unique())]
    len_j = len(judgements["queryId"].unique())
    len_q = len(queries["query_id"].unique())
    assert len_j == len_q

    return queries


def preprocess_judgements(judgements: pd.DataFrame) -> pd.DataFrame:
    prev_len = len(judgements)
    judgements = judgements.dropna().drop_duplicates()
    assert len(judgements) == prev_len

    # remove columns
    judgements = judgements[["relevanceLevel", "queryId", "documentId"]]

    # map votes to integers
    judgements["relevanceLevel"] = judgements["relevanceLevel"].map(
        {
            "0_NOT_RELEVANT": 0,
            "1_TOPIC_RELEVANT_DOES_NOT_ANSWER": 1,
            "2_GOOD_ANSWER": 2,
            "3_PERFECT_ANSWER": 3,
        }
    )
    return judgements


def get_cos_similarity(q_id: str, d_id: str) -> float:
    q_embedding: torch.tensor = torch.tensor([float(i) for i in queries[queries["query_id"] == q_id]["query_embedding"].values[0].strip("[]").split(", ")]).unsqueeze(0)  # type: ignore
    d_embedding: torch.tensor = torch.tensor([float(i) for i in docs[docs["doc_id"] == d_id]["doc_embedding"].values[0].strip("[]").split(", ")]).unsqueeze(0)  # type: ignore
    return cosine_similarity(q_embedding, d_embedding).item()


"""
merge multiple (query, document, relevance) tuples into one
"""
docs = preprocess_docs(docs)  # "doc_id", "doc_embedding"
queries = preprocess_queries(queries)  # "query_id", "query_embedding"
judgements = preprocess_judgements(judgements)  # "relevanceLevel", "queryId", "documentId"

# each query is unique and can have multiple documents
# each document is unique and can have multiple relevance levels

for _, q in queries.iterrows():
    q_judgements: pd.DataFrame = judgements[judgements["queryId"] == q["query_id"]]

    # sort documents in judgements by similarity to query
    docid_sim: List[Tuple[str, float]] = []
    for idx, j in q_judgements.iterrows():
        sim = get_cos_similarity(q["query_id"], j["documentId"])
        docid_sim.append((j["documentId"], sim))

    # algorithm 1: get sum of judgements weighted by similarity
    # total_sim = sum([similarity for _, similarity in docid_sim])
    # get_vote = lambda doc_id: judgements[(judgements["documentId"] == doc_id) & (judgements["queryId"] == q["query_id"])]["relevanceLevel"].values[0]
    # aggregated_judgement = np.sum([(sim / total_sim) * get_vote(doc_id) for doc_id, sim in docid_sim])
    # aggregated_judgement = round(aggregated_judgement)

    # algorithm 2: take the first relevant document
    docid_sim = sorted(docid_sim, key=lambda x: x[1], reverse=True)
    aggregated_judgement = 0
    for doc_id, similarity in docid_sim:
        print(doc_id, similarity)
        vote = judgements[(judgements["documentId"] == doc_id) & (judgements["queryId"] == q["query_id"])]["relevanceLevel"].values[0]
        if vote in [2, 3]:
            aggregated_judgement = vote
            break

    print(q["query_id"], "Q0", docid_sim[0][0], aggregated_judgement)

    # get result
    # rob_q_FBIS3-10909 "Q0" rob_FBIS3-10909 2
    # queryId, Q0, documentId, aggregated judgement
