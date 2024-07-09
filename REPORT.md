Assignment: https://github.com/tuwien-information-retrieval/air-24-template/blob/main/assignment_2.md

Team:

- Maximilian Höller (52004266)
- Miran Mamsaleh (12019866)
- Yahya Jabary (11912007)

<!--

- Yahya Jabary (11912007): Completed Part 1 and assisted with Part 2
- Miran Mamsaleh (12019866): Contributed initial work on Part 2 
- Maximilian Höller (52004266): Completed Part 3 and assisted with Part 2

Due to resource constraints and time limitations, the team had to redistribute some workload in the final days before the deadline. Yahya and Maximilian collaborated to complete Part 2, building on Miran's initial contributions.

-->

Due to the free tier of Google Colab being insufficient for the workload we had to purchase additional compute units with our own funds.

See `git blame` for a more detailed breakdown of the contributions.

<br><br>

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

FiRa data source: Fine-Grained Relevance Annotations for Multi-Task Document Ranking and Question Answering In Proc. of CIKM 2020

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

**The hypothesis** / assumption is that we can use the cosine similarity of the query and document embeddings to add an additional relevance grade to the existing ones. This way we can increase the quality of the baseline solution in cases where there are not enough expert votes to make a decision. But the similarity vote should ideally vanish if there are enough expert votes to make a decision.

**The algorithm** is as follows:

-   Load the data, generate embeddings for the documents and queries and add them to the data.
-   Remove all queries that never occur in the judgement data. This way we can turn the (query, document) pairs into our unique identifier. → This takes several hours in practice.
-   For each unique (query, document) pair, in addition to the existing relevance grades, add a synthetic relevance grade based on the cosine similarity of the query and document embeddings.
-   Aggregate the relevance grades for each unique (query, document) pair by taking the mean of all relevance grades (including the synthetic one). → We looked at a bunch of different weighting schemes and aggregation methods (mean, mode, median) and found that the mean works best as the effect of the synthetic vote becomes more visible.

## Meta judgement

Let's manually judge the quality of our aggregation.

A quick little sanity check before we start:

```bash
$ cat output/fira-22.qrels.tsv | wc -l
   24189

$ cat data-merged/data-merged/air-exercise-2/Part-1/fira-22.baseline-qrels.tsv | wc -l
   24189
```

We pick 5 random examples and manually decide how well we combined the existing relevance grades.

```
- query id: trip_1337, doc id: trip_4728579
- expert votes: [2 2 2] (mean: 2.00, median: 2, mode: 2)
- similarity vote: 2
- aggregated vote: 2
```

In the case above both the experts and the similarity vote agree on the relevance grade. Nothing to see here.

```
- query id: trip_443528, doc id: trip_9943688
- expert votes: [0 0 0] (mean: 0.00, median: 0, mode: 0)
- similarity vote: 2
- aggregated vote: 0

- query id: rob_qq_FR940811-1-00004, doc id: rob_FR940811-1-00004
- expert votes: [0 0 0] (mean: 0.00, median: 0, mode: 0)
- similarity vote: 2
- aggregated vote: 0
```

In the 2 cases above the experts agree the similarity vote does not. But the synthetic vote didn't influence the result as we have 3 voters agreeing on the relevance grade.

```
- query id: rob_q_FT933-11533, doc id: rob_FT924-4715
- expert votes: [0 2 1] (mean: 1.00, median: 1, mode: 0)
- similarity vote: 1
- aggregated vote: 1

- query id: trip_57861, doc id: trip_5571694
- expert votes: [3 1 2] (mean: 2.00, median: 2, mode: 3)
- similarity vote: 2
- aggregated vote: 2
```

In the 2 cases above the similarity vote didn't agree with the expert votes but did not influence the result either.

```
- query id: rob_qq_FR940318-0-00056, doc id: rob_FR940106-0-00031
- expert votes: [1 0 0] (mean: 0.33, median: 0, mode: 0)
- similarity vote: 3
- aggregated vote: 1
```

This final case is particularly interesting as the similarity vote is very different from the expert votes and moved the mean up from 0.33 to rounded 1, effectively changing the relevance grade from 0 to 1.

## Conclusion

In conclusion while our algorithm did not really differ significantly from the baseline solution using the mode, it did change the relevance grade in some _very rare_ cases where the expert votes were not unanimous and slightly nudged the relevance grade in a more favorable direction.

But given how computationally expensive it is to generate embeddings for all documents and queries, we would not recommend this approach in a real-world scenario given the marginal improvement in quality.

Just using the mode of the expert votes would be more efficient and almost as effective.

<br><br>

# Part 2 - Neural Re-Ranking

Neural re-ranking is a technique to improve the quality of search results by using a neural network to re-rank the results of a search engine.

## Data description

-   `allen_vocab_lower_{5/10}`: AllenNLP vocabulary

    Comes in two sizes. Based on words that occur at least 5 or 10 times in the collection, where 5 is the more compute intensive but also more accurate version.

    Used as the argument for `from allennlp.data.vocabulary import Vocabulary`.

    see: https://docs.allennlp.org/main/api/data/vocabulary/

-   `msmarco_qrels.txt`: relevance judgments

    In this case the relevance judgments are binary (0 or 1), where the existence of a line in the file indicates relevance.

    Format: `query_id, 0, document_id, 1`.

    One file covers both validation & test set.

-   `fira-22.tuples.tsv`, `msmarco_tuples.validation.tsv`, `msmarco_tuples.test.tsv`: evaluation tuples

    With 2.000 queries each and the top 40 BM25 results per query.

    Format: `query_id, doc_id, query_tokens, doc_tokens`

-   `msmarco_queries.validation.tsv`, `msmarco_queries.test.tsv`: query text data.

    Format: `query_id, query_text` (no header)

-   `triples.train.tsv`: train triplets

    Format: `query_text, positive_document_text, negative_document_text`

-   `glove.42B.300d.txt`: gloVe embeddings

    Pre-trained glove embedding from: https://nlp.stanford.edu/projects/glove/

    Common Crawl (42B tokens, 1.9M vocab, uncased, 300d vectors, 1.75 GB download)

    Format: `word, embedding`

## Model Architectures

For the neural re-ranking task, we implemented two advanced architectures based on the kernel-pooling paradigm: the Kernel-based Neural Ranking Model (KNRM) and the Transformer-Kernel (TK) model.

These models were designed to perform re-ranking on the provided MS MARCO dataset.

*KNRM (Kernel-based Neural Ranking Model)*

- The KNRM model utilizes a series of Gaussian kernels to capture semantic similarities between query and document terms at different levels of granularity. Our implementation includes the following key components:
    - Word embeddings: We used pre-trained GloVe embeddings (42B tokens, 300d vectors) to represent query and document terms.
    - Kernel pooling: We implemented 11 Gaussian kernels with learnable `mu` and `sigma` parameters to capture fine-grained semantic interactions.
    - Log-sum-exp pooling: This technique was applied to aggregate kernel scores across query terms, providing a soft-TF ranking signal.
    - Linear layer: A final linear transformation was used to produce the relevance score.

*TK (Transformer-Kernel) Model*

- The TK model extends KNRM by incorporating contextualized representations through transformer layers. Key features of our TK implementation include:
    - Transformer layers: We implemented multi-head self-attention mechanisms to capture contextual information within queries and documents.
    - Kernel pooling: Similar to KNRM, we used 11 Gaussian kernels for semantic matching.
    - Layer-wise aggregation: The model combines signals from multiple transformer layers, allowing for multi-level contextualization.

*Training Process*

- We implemented a pairwise training approach using PyTorch, which involved the following steps:
    - Data loading: We utilized custom DatasetReaders to efficiently load and preprocess the MS MARCO triples and evaluation tuples.
    - Batching: We implemented dynamic batching to handle variable-length inputs efficiently.
    - Loss function: We employed a margin ranking loss to optimize the relative ordering of relevant and non-relevant documents.
    - Optimization: The Adam optimizer was used with a learning rate of `0.001` and a `ReduceLROnPlateau` scheduler for adaptive learning rate adjustment.
    - Early stopping: We implemented an early stopping mechanism based on validation set performance to prevent overfitting.

## Evaluation

For model evaluation, we implemented the following pipeline:

- Test set inference: We ran inference on the provided test sets, including MS MARCO and FiRA-2022.
- Metric calculation: We computed standard IR metrics, including MRR@10, using the provided core_metrics module.
- Cross-dataset evaluation: We evaluated our models on both in-domain (MS MARCO) and out-of-domain (FiRA-2022) data to assess generalization capabilities.

## Results

Our implementation demonstrated competitive performance on the MS MARCO dataset, with the KNRM model achieving an MRR@10 of approximately `0.19`.

The TK model showed further improvements over KNRM, highlighting the benefits of contextual representations in neural ranking.

In conclusion, our neural re-ranking implementation successfully leverages advanced kernel-pooling techniques and transformer architectures to improve upon traditional retrieval methods.

We tried to implement everything in a clean procedual and maintainable manner so future work is possible.

<br><br>

# Part 3 - Extractive QA

## Data description

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

## Process

The first problem encountered was, that running a model from hugging face on a local CPU is infeasible. This is why we had to buy 100 computation units for Google Collab, since this is unfortunately not for free anymore. Once that was figured out, we ran the `deepset/roberta-base-squad2` model on all 52.000 instances which took around five hours.

The results were evaluated to four digits as follows:

```
Exact Match: 0.0866
F1 Score: 0.3195
Recall: 0.275
Precision: 0.6249
```

It is an average over the values for all around 50.000 values. If a token was in the predicted and in the gold answer, it was counted as true positive.

Note: Only the first answer was evaluated.

<br><br>

# Bonus Points

Merged pull request for reproducibility: https://github.com/tuwien-information-retrieval/air-24-template-public/pull/1/commits

-   The AllenNLP library is unmaintained, has many deprecated dependencies that are not compatible with the latest versions of other libraries and relies on python@3.6< which has critical security vulnerabilities and can't be installed on arm64 architectures. We provide 2 containerized solutions to this problem.
-   Little hack that chunks data and allows us to push multiple gigabytes of data to github without having to pay for LFS cloud storage: https://github.com/sueszli/github-lfs-bypass/blob/main/upload.sh
