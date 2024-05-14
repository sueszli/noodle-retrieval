from pathlib import Path
import pandas as pd
from pandas.core.series import Series
import numpy as np
from transformers import AutoTokenizer, AutoModel
import torch
from torch.nn.functional import cosine_similarity
from typing import List, Tuple


base_in = Path.cwd() / "data-merged" / "air-exercise-2" / "Part-1"
base_out = Path.cwd() / "data-merged"  # output of previous script

docs = pd.read_csv(base_out / "fira-22.documents.embeddings.tsv", sep="\t")
queries = pd.read_csv(base_out / "fira-22.queries.embeddings.tsv", sep="\t")
judgements: pd.DataFrame = pd.read_csv(base_in / "fira-22.judgements-anonymized.tsv", sep="\t")


judgements = judgements[["relevanceLevel", "queryId", "documentId"]]

for _, q in queries.iterrows():
    q_judgements: pd.DataFrame = judgements[judgements["queryId"] == q["query_id"]]

    # sort documents in judgements by similarity to query
    sorted_docids: List[Tuple[str, float]] = []
    for _, j in q_judgements.iterrows():
        d: pd.DataFrame = docs[docs["doc_id"] == j["documentId"]]
        q_embedding: torch.tensor = torch.tensor([float(i) for i in q["query_embedding"].strip("[]").split(", ")]).unsqueeze(0)  # type: ignore
        d_embedding: torch.tensor = torch.tensor([float(i) for i in d["doc_embedding"].values[0].strip("[]").split(", ")]).unsqueeze(0)  # type: ignore
        similarity: float = cosine_similarity(q_embedding, d_embedding).item()
        sorted_docids.append((j["documentId"], similarity))
    sorted_docids.sort(key=lambda x: x[1], reverse=True)

    # iterate over votes and stop at the first good or perfect answer
    aggregated_judgement = "0_NOT_RELEVANT"
    for doc_id, similarity in sorted_docids:
        print(doc_id, similarity)
        vote = judgements[(judgements["documentId"] == doc_id) & (judgements["queryId"] == q["query_id"])]["relevanceLevel"].values[0]
        if vote in ["2_GOOD_ANSWER", "3_PERFECT_ANSWER"]:
            aggregated_judgement = vote
            break
