from pathlib import Path
import pandas as pd
import numpy as np
import torch
from torch.nn.functional import cosine_similarity
from typing import List, Tuple


base_in = Path.cwd() / "data-merged" / "air-exercise-2" / "Part-1"
base_out = Path.cwd() / "data-merged"  # output of previous script

docs = pd.read_csv(base_out / "fira-22.documents.embeddings.tsv", sep="\t")
queries = pd.read_csv(base_out / "fira-22.queries.embeddings.tsv", sep="\t")
judgements: pd.DataFrame = pd.read_csv(base_in / "fira-22.judgements-anonymized.tsv", sep="\t")


def preprocess_judgements(judgements: pd.DataFrame) -> pd.DataFrame:
    prev_len = len(judgements)
    judgements = judgements.dropna().drop_duplicates()
    assert len(judgements) == prev_len

    # remove irrelevant columns
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


def preprocess_queries(queries: pd.DataFrame) -> pd.DataFrame:
    # remove queries without judgements
    queries = queries[queries["query_id"].isin(judgements["queryId"].unique())]

    len_j = len(judgements["queryId"].unique())
    len_q = len(queries["query_id"].unique())
    print(len_j, len_q)  # (4175, 4177)
    assert len_j == len_q

    return queries


"""
aggregate judgements: find a single judgement for each query
"""
judgements = preprocess_judgements(judgements)
queries = preprocess_queries(queries)


for _, q in queries.iterrows():
    q_judgements: pd.DataFrame = judgements[judgements["queryId"] == q["query_id"]]

    # sort documents in judgements by similarity to query
    docid_sim: List[Tuple[str, float]] = []
    for idx, j in q_judgements.iterrows():
        d: pd.DataFrame = docs[docs["doc_id"] == j["documentId"]]
        q_embedding: torch.tensor = torch.tensor([float(i) for i in q["query_embedding"].strip("[]").split(", ")]).unsqueeze(0)  # type: ignore
        d_embedding: torch.tensor = torch.tensor([float(i) for i in d["doc_embedding"].values[0].strip("[]").split(", ")]).unsqueeze(0)  # type: ignore
        similarity: float = cosine_similarity(q_embedding, d_embedding).item()
        docid_sim.append((j["documentId"], similarity))

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
