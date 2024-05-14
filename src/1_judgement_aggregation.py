from pathlib import Path
import pandas as pd
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


# def aggregate_judgements(query_id, judgements, documents, queries):
#     # Filter judgements for the given query
#     query_judgements = judgements[judgements["queryId"] == query_id]

#     # Get the query embedding
#     query_embedding = queries[queries["query_id"] == query_id]["query_embedding"].values[0]

#     # Calculate similarity scores
#     similarity_scores = []
#     for index, row in query_judgements.iterrows():
#         doc_id = row["documentId"]
#         doc_embedding = documents[documents["doc_id"] == doc_id]["doc_embedding"].values[0]
#         similarity = cosine_similarity(query_embedding, doc_embedding)
#         similarity_scores.append((doc_id, similarity[0][0]))

#     # Sort documents by similarity score
#     similarity_scores.sort(key=lambda x: x[1], reverse=True)

#     # Aggregate judgements based on similarity
#     aggregated_judgement = "0_NOT_RELEVANT"  # Default to not relevant
#     for doc_id, similarity in similarity_scores:
#         judgement = judgements[(judgements["documentId"] == doc_id) & (judgements["queryId"] == query_id)]["relevanceLevel"].values[0]
#         if judgement in ["2_GOOD_ANSWER", "3_PERFECT_ANSWER"]:
#             aggregated_judgement = judgement
#             break  # Stop at the first good or perfect answer

#     return aggregated_judgement


judgements = judgements[["relevanceLevel", "queryId", "documentId"]]

for _, row in queries.iterrows():
    query_id = row["query_id"]
    query_embedding = row["query_embedding"]

    query_judgements = judgements[judgements["queryId"] == query_id]
    similarity_scores = []
