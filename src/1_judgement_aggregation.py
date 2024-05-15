import pandas as pd
import numpy as np
import torch
from torch.nn.functional import cosine_similarity
from pathlib import Path

"""
merge multiple (query, document, vote) tuples on the same (query, document) pair.

write the aggregated judgements to a file.
"""


base_in = Path.cwd() / "data-merged" / "data-merged" / "air-exercise-2" / "Part-1"
base_in_prev = Path.cwd() / "data-merged" / "data-merged"  # output of previous script
base_out = Path.cwd() / "output"

docs = pd.read_csv(base_in_prev / "fira-22.documents.embeddings.tsv", sep="\t")
queries = pd.read_csv(base_in_prev / "fira-22.queries.embeddings.tsv", sep="\t")
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
    judgements["relevanceLevel"] = judgements["relevanceLevel"].map({"0_NOT_RELEVANT": 0, "1_TOPIC_RELEVANT_DOES_NOT_ANSWER": 1, "2_GOOD_ANSWER": 2, "3_PERFECT_ANSWER": 3})
    return judgements


def get_cos_similarity(q_id: str, d_id: str) -> float:
    q_embedding: torch.tensor = torch.tensor([float(i) for i in queries[queries["query_id"] == q_id]["query_embedding"].values[0].strip("[]").split(", ")]).unsqueeze(0)  # type: ignore
    d_embedding: torch.tensor = torch.tensor([float(i) for i in docs[docs["doc_id"] == d_id]["doc_embedding"].values[0].strip("[]").split(", ")]).unsqueeze(0)  # type: ignore
    sim: float = cosine_similarity(q_embedding, d_embedding).item()
    assert 0 <= sim <= 1
    return sim


docs = preprocess_docs(docs)  # "doc_id", "doc_embedding"
queries = preprocess_queries(queries)  # "query_id", "query_embedding"
judgements = preprocess_judgements(judgements)  # "relevanceLevel", "queryId", "documentId"

if __name__ == "__main__":
    f = open(base_out / "fira-22.qrels.tsv", "w")
    total = len(queries)
    c = 0

    for _, q in queries.iterrows():
        q_id = q["query_id"]
        d_ids = judgements[judgements["queryId"] == q_id]["documentId"].unique()

        for doc_id in d_ids:
            votes = judgements[judgements["documentId"] == doc_id]["relevanceLevel"].values
            sim = get_cos_similarity(q_id, doc_id)

            # our additional vote
            sim_vote = 3 if sim >= 0.75 else 2 if sim >= 0.5 else 1 if sim >= 0.25 else 0
            votes = np.append(votes, sim_vote)

            # agg_vote = int(np.median(votes))
            agg_vote = np.round(np.mean(votes)).astype(int)
            # agg_vote = int(Counter(votes).most_common(1)[0][0])  # majority vote
            assert agg_vote in [0, 1, 2, 3]

            f.write(f"{q_id} Q0 {doc_id} {agg_vote}\n")
            f.flush()

        c += 1
        progress = c / total * 100
        print(f"\rprogress: {progress:.2f}%", end="\r")

    f.close()
