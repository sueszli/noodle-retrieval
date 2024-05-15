from pathlib import Path
import pandas as pd
import numpy as np
from transformers import AutoTokenizer, AutoModel
import torch
from typing import List


base_in = Path.cwd() / "data-merged" / "data-merged" / "air-exercise-2" / "Part-1"
base_out = Path.cwd() / "data-merged" / "data-merged"

docs: pd.DataFrame = pd.read_csv(base_in / "fira-22.documents.tsv", sep="\t")
queries: pd.DataFrame = pd.read_csv(base_in / "fira-22.queries.tsv", sep="\t")

model_name = "bert-base-uncased"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModel.from_pretrained(model_name)


def gen_embedding(text: str) -> List[float]:
    tokens = tokenizer(text, return_tensors="pt", padding=True, truncation=True, max_length=512)
    embedding = None
    with torch.no_grad():
        output = model(**tokens)
        embedding = output.last_hidden_state.mean(dim=1)  # mean pooling
    return embedding.numpy()[0].tolist()


def gen_doc_embeddings():
    counter = 0
    len_docs = docs.shape[0]

    out_path = base_out / "fira-22.documents.embeddings.tsv"
    f = open(out_path, "w")
    f.write("doc_id\tdoc_text\tdoc_embedding\n")
    f.flush()

    for _, row in docs.iterrows():
        counter += 1
        progress_percent = counter / len_docs * 100
        print(f"progress: {progress_percent:.2f}%", end="\r")

        doc_id = row["doc_id"]
        doc_text = row["doc_text"]
        doc_embedding = gen_embedding(doc_text)

        f.write(f"{doc_id}\t{doc_text}\t{doc_embedding}\n")
        f.flush()

    f.close()


def gen_query_embeddings():
    counter = 0
    len_queries = queries.shape[0]

    out_path = base_out / "fira-22.queries.embeddings.tsv"
    f = open(out_path, "w")
    f.write("query_id\tquery_text\tquery_embedding\n")
    f.flush()

    for _, row in queries.iterrows():
        counter += 1
        progress_percent = counter / len_queries * 100
        print(f"progress: {progress_percent:.2f}%", end="\r")

        query_id = row["query_id"]
        query_text = row["query_text"]
        query_embedding = gen_embedding(query_text)

        f.write(f"{query_id}\t{query_text}\t{query_embedding}\n")
        f.flush()

    f.close()


if __name__ == "__main__":
    gen_query_embeddings()
    gen_doc_embeddings()
