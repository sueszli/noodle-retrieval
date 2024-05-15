Assignment: https://github.com/tuwien-information-retrieval/air-24-template/blob/main/assignment_2.md

Student 1 ID + Name: Yahya Jabary 11912007

Student 2 ID + Name:

Student 3 ID + Name:

Student 4 ID + Name:

# Part 1 - Test Collection Preparation

the first step is to merge / aggregate multiple (query x document, vote) judgement-tuples into a single one, should there be multiple votes for the same query-document pair. the most straightforward way to do this is to take the majority vote.

_data description:_

-   the origin of the data is unknown.

-   `fira-22-baseline-qrels.tsv`: baseline solution.

    simple majority voting, by only looking at the judgement value, and a heuristic to take the higher grade if we have a tie.

    format: `query_id, hardcoded-Q0, document_id, relevance-grade`

    example: `rob_q_FBIS3-10909 Q0 rob_FBIS3-10909 2`

    the `Q0` is a placeholder for the rank of the document in the result list. it is not used in this assignment.

    this format is the output we are supposed to generate in the first part of the assignment.

-   `fira-22.documents.tsv`: document text data.

    usage is optional.

    format: `doc_id, doc_text`

-   `fira-22-queries.tsv`: query text data.

    usage is optional.

    format: `query_id, query_text`

-   `fira-22-judgements-anonymized.tsv`: raw judgement data.

    format: `id, relevanceLevel, relevanceCharacterRanges, durationUsedToJudgeMs, judgedAtUnixTS, documentId, queryId, userId`

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

    we extended the provided dataset by generating embeddings for each document and query using a pre-trained BERT model from the `transformers` library.

    format: `doc_id, doc_text, doc_embedding`

-   `fira-22-queries.embeddings.tsv`: query embeddings data.

    again, this is something we generated ourselves.

    format: `query_id, query_text, query_embedding`

_hypotheses:_

-   the majority voting heuristic is not good enough.
-   content based: (our chosen method)
    -   documents closer to the query in the latent / embedding space should be rated higher.
-   voter based:
    -   voters with a higher overall agreement rate should have a higher weight in the majority voting.
    -   voters with more voting experience should have a higher weight in the majority voting.

_algorithm:_

_meta-judgement:_

-   TODO

a little sanity check:

```bash
noodle-retrieval git:main
❯ cat output/fira-22.qrels.tsv | wc -l
   24189

noodle-retrieval git:main
❯ cat data-merged/data-merged/air-exercise-2/Part-1/fira-22.baseline-qrels.tsv | wc -l
   24189
```

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
