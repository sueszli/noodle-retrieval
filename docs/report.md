Assignment: https://github.com/tuwien-information-retrieval/air-24-template/blob/main/assignment_2.md

Student 1 ID + Name: Yahya Jabary 11912007

Student 2 ID + Name:

Student 3 ID + Name:

Student 4 ID + Name:

# Part 1 - Test Collection Preparation

When you have multiple judgements (vote of relevance) for the same query-document pair, you need to aggregate them into a single judgement.

The most straightforward way to do this is to take the mode (majority vote) of the judgements.

The goal of the first part of this assignment is to implement this aggregation method and to evaluate it on the provided dataset.

## Data description

The origin of the data is unknown.

-   `fira-22-baseline-qrels.tsv`: baseline solution.

    Simple majority voting, by only looking at the judgement value, and a heuristic to take the higher grade if we have a tie.

    Format: `query_id, hardcoded-Q0, document_id, relevance-grade`

    Example: `rob_q_FBIS3-10909 Q0 rob_FBIS3-10909 2`

    The `Q0` is a placeholder for the rank of the document in the result list. it is not used in this assignment.

    This format is the output we are supposed to generate in the first part of the assignment.

-   `fira-22.documents.tsv`: document text data.

    Usage is optional.

    Format: `doc_id, doc_text`

-   `fira-22-queries.tsv`: query text data.

    Usage is optional.

    Format: `query_id, query_text`

-   `fira-22-judgements-anonymized.tsv`: raw judgement data.

    Format: `id, relevanceLevel, relevanceCharacterRanges, durationUsedToJudgeMs, judgedAtUnixTS, documentId, queryId, userId`

    -   `relevanceLevel`: judgement values - 4 distinct values (also called "4-grades")
        -   `0_NOT_RELEVANT` - 17%
        -   `1_TOPIC_RELEVANT_DOES_NOT_ANSWER` - 27%
        -   `2_GOOD_ANSWER` - 28%
        -   `3_PERFECT_ANSWER` - 27%
    -   `relevanceCharacterRanges`: not used - because 74% of the values are missing
    -   `durationUsedToJudgeMs`: not used
    -   `judgedAtUnixTS`: not used
    -   `documentId`: document id - 21190 distinct values
    -   `queryId`: query id - 4175 distinct values
    -   `userId`: not used

-   `fira-22.documents.embeddings.tsv`: document embeddings data.

    We extended the provided dataset by generating embeddings for each document and query using a pre-trained BERT model from the `transformers` library.

    Format: `doc_id, doc_text, doc_embedding`

-   `fira-22-queries.embeddings.tsv`: query embeddings data.

    Again, this is something we generated ourselves.

    Format: `query_id, query_text, query_embedding`

## Algorithm

We want to increase the quality of the baseline solution.

There are multiple ways to do this. Here are some ideas:

-   a) Content based:
    -   documents closer to the query in the latent / embedding space should be rated higher.
-   b) Voter based / User based:
    -   voters with a higher overall agreement rate across all queries should be taken more seriously (ie. by giving them more weight).
    -   voters with more voting experience should be taken more seriously.
    -   voters that take too long or too short to judge a document should be taken less seriously.

We decided to go with the content based approach.

**The hypothesis** / assumption is that we can use the embeddings to enhance the baseline solution.

**The algorithm** is as follows:

-   Load the data, generate embeddings for the documents and queries and add them to the data.
-   Remove all queries that never occur in the judgement data. This way we can turn the (query, document) pairs into our unique identifier.
-   For each unique (query, document) pair, in addition to the existing relevance grades, add a synthetic relevance grade based on the cosine similarity of the query and document embeddings.
-   Aggregate the relevance grades for each unique (query, document) pair by taking the median of all relevance grades (including the synthetic one).
-   Save the results in the format of the baseline solution.

## Meta judgement

Let's manually judge the quality of our aggregation.

A quick little sanity check before we start:

```bash
$ cat output/fira-22.qrels.tsv | wc -l
   24189

$ cat data-merged/data-merged/air-exercise-2/Part-1/fira-22.baseline-qrels.tsv | wc -l
   24189
```

We want to pick 5 random queries and manually decide how well we combined the existing relevance grades.

<br><br>

# Part 2 - Neural Re-Ranking

knrm model, see: https://github.com/sebastian-hofstaetter/matchmaker/blob/210b9da0c46ee6b672f59ffbf8603e0f75edb2b6/matchmaker/models/drmm.py#L37

<br><br>

# Part 3 - Extractive QA

<br><br>

# Bonus Points

pull request for reproducibility through docker: https://github.com/tuwien-information-retrieval/air-24-template-public/pull/1/commits

-   custom dockerfile for reproducability of the unmaintained AllenNLP library
-   little hack that chunks data and allows us to push multiple gigabytes of data to github without having to pay for LFS cloud storage: https://github.com/sueszli/github-lfs-bypass/blob/main/upload.sh
