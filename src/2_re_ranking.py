import logging
import sys
from pathlib import Path
from typing import Dict, List

import torch
import torch.nn as nn
from allennlp.common import Tqdm  # progress bar in loops
from allennlp.common import Params
from allennlp.common.checks import ConfigurationError
from allennlp.common.file_utils import cached_path
from allennlp.common.util import prepare_environment
from allennlp.data.dataset_readers.dataset_reader import DatasetReader
from allennlp.data.fields import MetadataField, TextField
from allennlp.data.instance import Instance
from allennlp.data.token_indexers import SingleIdTokenIndexer, TokenIndexer
from allennlp.data.tokenizers import Tokenizer
from allennlp.data.tokenizers.token import Token
from allennlp.data.vocabulary import Vocabulary
from allennlp.modules.text_field_embedders import BasicTextFieldEmbedder, TextFieldEmbedder
from allennlp.modules.token_embedders import Embedding
from blingfire import text_to_words
from overrides import overrides
from torch.optim import Adam, lr_scheduler

prepare_environment(Params({}))  # seed
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

from allennlp.common import Params
from allennlp.common.util import prepare_environment
from allennlp.data.dataloader import PyTorchDataLoader
from allennlp.nn.util import move_to_device


from typing import Dict, List
import logging

from overrides import overrides
from blingfire import *

from allennlp.common.checks import ConfigurationError
from allennlp.common.file_utils import cached_path
from allennlp.data.dataset_readers.dataset_reader import DatasetReader
from allennlp.data.fields import TextField, MetadataField
from allennlp.data.instance import Instance
from allennlp.data.tokenizers import Tokenizer
from allennlp.data.token_indexers import TokenIndexer, SingleIdTokenIndexer
from allennlp.data.tokenizers.token import Token

from typing import Dict, List
import logging

from overrides import overrides
from blingfire import *

from allennlp.common.checks import ConfigurationError
from allennlp.common.file_utils import cached_path
from allennlp.data.dataset_readers.dataset_reader import DatasetReader
from allennlp.data.fields import TextField, MetadataField
from allennlp.data.instance import Instance
from allennlp.data.tokenizers import Tokenizer
from allennlp.data.token_indexers import TokenIndexer, SingleIdTokenIndexer
from allennlp.data.tokenizers.token import Token

from typing import Dict, Iterator, List

import torch
import torch.nn as nn
from torch.nn.functional import cosine_similarity
from torch.autograd import Variable

from allennlp.modules.text_field_embedders import TextFieldEmbedder

from allennlp.common import Params, Tqdm
from allennlp.common.util import prepare_environment

prepare_environment(Params({}))  # sets the seeds to be fixed

from core_metrics import calculate_metrics_plain, unrolled_to_ranked_result, load_qrels  # , out_of_domain_eval

from torch.nn import MarginRankingLoss, Module
from torch.optim import Adam, lr_scheduler

from allennlp.data.vocabulary import Vocabulary
from allennlp.modules.token_embedders import Embedding
from allennlp.modules.text_field_embedders import BasicTextFieldEmbedder

from model_knrm import *
from model_tk import *

from allennlp.common import Params, Tqdm
from allennlp.common.util import prepare_environment
from allennlp.data.dataloader import PyTorchDataLoader

prepare_environment(Params({}))  # sets the seeds to be fixed

from allennlp.nn.util import move_to_device



# region utils


class EarlyStopper:
    def __init__(self, patience=5, min_delta=0):
        self.patience = patience
        self.min_delta = min_delta
        self.counter = 0
        self.min_metric = 0

    def early_stop(self, metric):
        if metric > self.min_metric + self.min_delta:
            self.min_metric = metric
            self.counter = 0
        else:
            self.counter += 1
            if self.counter >= self.patience:
                return True
        return False


def test_step(batch, model, device):
    # move batch to device and keep existing structure
    batch = move_to_device(batch, device)

    results = {}
    with torch.no_grad():
        output = model(batch["query_tokens"], batch["doc_tokens"]).tolist()
        for i in range(len(batch["query_id"])):
            query_id = batch["query_id"][i]
            doc_id = batch["doc_id"][i]
            score = output[i]
            results.setdefault(query_id, []).append((doc_id, score))
    return results


def triple_loader(path: str, vocab):
    _triple_reader = IrTripleDatasetReader(lazy=True, max_doc_length=200, max_query_length=30)
    _triple_reader = _triple_reader.read(path)
    _triple_reader.index_with(vocab)
    return PyTorchDataLoader(_triple_reader, batch_size=64)


def tuple_loader(path: str, vocab) -> PyTorchDataLoader:
    _tuple_reader = IrLabeledTupleDatasetReader(lazy=True, max_doc_length=200, max_query_length=30)
    _tuple_reader = _tuple_reader.read(path)
    _tuple_reader.index_with(vocab)
    return PyTorchDataLoader(_tuple_reader, batch_size=128)


# endregion utils


# region file-reader


class BlingFireTokenizer:
    def tokenize(self, sentence: str) -> List[Token]:
        return [Token(t) for t in text_to_words(sentence).split()]


class IrTripleDatasetReader(DatasetReader):
    """
    convert `triples.train.tsv` into 3 `TextField`s - which are a list of string tokens.
    """

    def __init__(self, tokenizer: Tokenizer = None, token_indexers: Dict[str, TokenIndexer] = None, source_add_start_token: bool = True, max_doc_length: int = -1, max_query_length: int = -1, lazy: bool = False) -> None:
        super().__init__(lazy)
        self._tokenizer = tokenizer or BlingFireTokenizer()
        self._token_indexers = token_indexers or {"tokens": SingleIdTokenIndexer(lowercase_tokens=True)}
        self._source_add_start_token = source_add_start_token
        self.max_doc_length = max_doc_length
        self.max_query_length = max_query_length

    @overrides
    def _read(self, file_path):
        # split line in 3 parts

        with open(cached_path(file_path), "r", encoding="utf8") as data_file:
            logger.debug("reading instances from lines in file at: %s", file_path)
            for line_num, line in enumerate(data_file):
                line = line.strip("\n")

                if not line:
                    continue

                line_parts = line.split("\t")
                if len(line_parts) != 3:
                    raise ConfigurationError("Invalid line format: %s (line number %d)" % (line, line_num + 1))
                query_sequence, doc_pos_sequence, doc_neg_sequence = line_parts
                yield self.text_to_instance(query_sequence, doc_pos_sequence, doc_neg_sequence)

    @overrides
    def text_to_instance(self, query_sequence: str, doc_pos_sequence: str, doc_neg_sequence: str) -> Instance:
        # tokenize each of 3 parts

        query_tokenized = self._tokenizer.tokenize(query_sequence)
        if self.max_query_length > -1:
            query_tokenized = query_tokenized[: self.max_query_length]

        query_field = TextField(query_tokenized, self._token_indexers)

        doc_pos_tokenized = self._tokenizer.tokenize(doc_pos_sequence)
        if self.max_doc_length > -1:
            doc_pos_tokenized = doc_pos_tokenized[: self.max_doc_length]

        doc_pos_field = TextField(doc_pos_tokenized, self._token_indexers)

        doc_neg_tokenized = self._tokenizer.tokenize(doc_neg_sequence)
        if self.max_doc_length > -1:
            doc_neg_tokenized = doc_neg_tokenized[: self.max_doc_length]

        doc_neg_field = TextField(doc_neg_tokenized, self._token_indexers)

        return Instance({"query_tokens": query_field, "doc_pos_tokens": doc_pos_field, "doc_neg_tokens": doc_neg_field})


class IrLabeledTupleDatasetReader(DatasetReader):
    """
    convert `msmarco_tuples.test.tsv` into 4 `TextField`s - which are a list of string tokens.
    """

    def __init__(self, tokenizer: Tokenizer = None, token_indexers: Dict[str, TokenIndexer] = None, source_add_start_token: bool = True, max_doc_length: int = -1, max_query_length: int = -1, lazy: bool = False) -> None:
        super().__init__(lazy)
        self._tokenizer = tokenizer or BlingFireTokenizer()
        self._token_indexers = token_indexers or {"tokens": SingleIdTokenIndexer(lowercase_tokens=True)}
        self._source_add_start_token = source_add_start_token
        self.max_doc_length = max_doc_length
        self.max_query_length = max_query_length

    @overrides
    def _read(self, file_path):
        # split line in 4 parts

        with open(cached_path(file_path), "r", encoding="utf8") as data_file:
            logger.debug("reading instances from lines in file at: %s", file_path)
            for line_num, line in enumerate(data_file):
                line = line.strip("\n")

                if not line:
                    continue

                line_parts = line.split("\t")
                if len(line_parts) != 4:
                    raise ConfigurationError("Invalid line format: %s (line number %d)" % (line, line_num + 1))
                query_id, doc_id, query_sequence, doc_sequence = line_parts
                yield self.text_to_instance(query_id, doc_id, query_sequence, doc_sequence)

    @overrides
    def text_to_instance(self, query_id: str, doc_id: str, query_sequence: str, doc_sequence: str) -> Instance:
        # tokenize each of 4 parts

        query_id_field = MetadataField(query_id)
        doc_id_field = MetadataField(doc_id)

        query_tokenized = self._tokenizer.tokenize(query_sequence)
        if self.max_query_length > -1:
            query_tokenized = query_tokenized[: self.max_query_length]

        query_field = TextField(query_tokenized, self._token_indexers)

        doc_tokenized = self._tokenizer.tokenize(doc_sequence)
        if self.max_doc_length > -1:
            doc_tokenized = doc_tokenized[: self.max_doc_length]

        doc_field = TextField(doc_tokenized, self._token_indexers)

        return Instance({"query_id": query_id_field, "doc_id": doc_id_field, "query_tokens": query_field, "doc_tokens": doc_field})


# endregion file-reader


# region models


class KNRM(nn.Module):
    def __init__(self, word_embeddings: TextFieldEmbedder, n_kernels: int):
        super(KNRM, self).__init__()
        self.word_embeddings = word_embeddings
        mu = torch.FloatTensor(self.kernel_mus(n_kernels)).view(1, 1, 1, n_kernels)
        sigma = torch.FloatTensor(self.kernel_sigmas(n_kernels)).view(1, 1, 1, n_kernels)
        self.register_buffer("mu", mu)
        self.register_buffer("sigma", sigma)
        self.mu = mu
        self.sigma = sigma
        self.lin = nn.Linear(n_kernels, 1, bias=True)
        torch.nn.init.uniform_(self.lin.weight, -0.001, 0.001)
        self.lin.bias.data.fill_(1)

    def forward(self, query: Dict[str, torch.Tensor], document: Dict[str, torch.Tensor]) -> torch.Tensor:
        query_pad_oov_mask = (query["tokens"]["tokens"] > 0).float()  # > 1 to also mask oov terms
        document_pad_oov_mask = (document["tokens"]["tokens"] > 0).float()
        kernel_mask = torch.bmm(query_pad_oov_mask.unsqueeze(-1), document_pad_oov_mask.unsqueeze(-1).transpose(1, 2))
        query_embeddings = self.word_embeddings(query)
        document_embeddings = self.word_embeddings(document)
        query_embeddings_normalized = query_embeddings / (query_embeddings.norm(p=2, dim=-1, keepdim=True) + 1e-13)
        document_embeddings_normalized = document_embeddings / (document_embeddings.norm(p=2, dim=-1, keepdim=True) + 1e-13)
        translation_matrix = torch.bmm(query_embeddings_normalized, document_embeddings_normalized.transpose(-1, -2)).unsqueeze(-1)
        kernel_results = torch.exp(-1 / 2 * torch.pow((translation_matrix - self.mu) / (self.sigma), 2))
        kernel_results = kernel_results * kernel_mask.unsqueeze(-1)
        per_kernel_query = torch.sum(kernel_results, 2)
        log_per_kernel_query = torch.log(torch.clamp(per_kernel_query, min=1e-10)) * 0.01
        log_per_kernel_query = log_per_kernel_query * query_pad_oov_mask.unsqueeze(-1)
        phi = torch.sum(log_per_kernel_query, 1)
        output = self.lin(phi)
        output = torch.squeeze(torch.tanh(output), 1)
        return output

    def kernel_mus(self, n_kernels: int):
        l_mu = [1.0]
        if n_kernels == 1:
            return l_mu

        bin_size = 2.0 / (n_kernels - 1)  # score range from [-1, 1]
        l_mu.append(1 - bin_size / 2)  # mu: middle of the bin
        for i in range(1, n_kernels - 1):
            l_mu.append(l_mu[i] - bin_size)
        return l_mu

    def kernel_sigmas(self, n_kernels: int):
        bin_size = 2.0 / (n_kernels - 1)
        l_sigma = [0.0001]  # for exact match. small variance -> exact match
        if n_kernels == 1:
            return l_sigma

        l_sigma += [0.5 * bin_size] * (n_kernels - 1)
        return l_sigma


class TK(nn.Module):
    """
    Paper: S. Hofstätter, M. Zlabinger, and A. Hanbury 2020. Interpretable & Time-Budget-Constrained Contextualization for Re-Ranking. In Proc. of ECAI
    """

    def __init__(self, word_embeddings: TextFieldEmbedder, n_kernels: int, n_layers: int, n_tf_dim: int, n_tf_heads: int):
        super(TK, self).__init__()

        self.word_embeddings = word_embeddings
        mu = torch.FloatTensor(self.kernel_mus(n_kernels)).view(1, 1, 1, n_kernels)
        sigma = torch.FloatTensor(self.kernel_sigmas(n_kernels)).view(1, 1, 1, n_kernels)
        self.register_buffer("mu", mu)
        self.register_buffer("sigma", sigma)

    def forward(self, query: Dict[str, torch.Tensor], document: Dict[str, torch.Tensor]) -> torch.Tensor:
        query_pad_oov_mask = (query["tokens"]["tokens"] > 0).float()  # > 1 to also mask oov terms
        document_pad_oov_mask = (document["tokens"]["tokens"] > 0).float()
        query_embeddings = self.word_embeddings(query)
        document_embeddings = self.word_embeddings(document)
        output = torch.rand(1, 1)
        return output

    def kernel_mus(self, n_kernels: int):
        l_mu = [1.0]
        if n_kernels == 1:
            return l_mu
        bin_size = 2.0 / (n_kernels - 1)  # score range from [-1, 1]
        l_mu.append(1 - bin_size / 2)  # mu: middle of the bin
        for i in range(1, n_kernels - 1):
            l_mu.append(l_mu[i] - bin_size)
        return l_mu

    def kernel_sigmas(self, n_kernels: int):
        bin_size = 2.0 / (n_kernels - 1)
        l_sigma = [0.0001]  # for exact match. small variance -> exact match
        if n_kernels == 1:
            return l_sigma
        l_sigma += [0.5 * bin_size] * (n_kernels - 1)
        return l_sigma


# endregion models


class EarlyStopper:
    def __init__(self, patience=5, min_delta=0):
        self.patience = patience
        self.min_delta = min_delta
        self.counter = 0
        self.min_metric = 0

    def early_stop(self, metric):
        if metric > self.min_metric + self.min_delta:
            self.min_metric = metric
            self.counter = 0
        else:
            self.counter += 1
            if self.counter >= self.patience:
                return True
        return False


def test_step(batch, model, device):
    batch = move_to_device(batch, device)
    results = {}
    with torch.no_grad():
        output = model(batch["query_tokens"], batch["doc_tokens"]).tolist()
        for i in range(len(batch["query_id"])):
            query_id = batch["query_id"][i]
            doc_id = batch["doc_id"][i]
            score = output[i]
            results.setdefault(query_id, []).append((doc_id, score))
    return results


def triple_loader(path: str, vocab):
    _triple_reader = IrTripleDatasetReader(lazy=True, max_doc_length=200, max_query_length=30)
    _triple_reader = _triple_reader.read(path)
    _triple_reader.index_with(vocab)
    return PyTorchDataLoader(_triple_reader, batch_size=64)


def tuple_loader(path: str, vocab) -> PyTorchDataLoader:
    _tuple_reader = IrLabeledTupleDatasetReader(lazy=True, max_doc_length=200, max_query_length=30)
    _tuple_reader = _tuple_reader.read(path)
    _tuple_reader.index_with(vocab)
    return PyTorchDataLoader(_tuple_reader, batch_size=128)


base_in = Path.cwd() / "data-merged" / "data" / "air-exercise-2"
base_out = Path.cwd() / "output"

config = {
    # settings
    "mode": "train",  # 'train', 'eval'
    "model": "knrm",  # 'knrm', 'tk'
    "eval_data": "baseline",  # 'baseline', 'ds', 'reranking'
    "epochs": 10,  # int [1;]
    "eval_step_size": 50,  # int [1;] - only relevant in training phase, steps after which the model should be evaluated
    # train paths
    "vocab_directory": base_in / "Part-2" / "allen_vocab_lower_10",
    "pre_trained_embedding": base_in / "Part-2" / "glove.42B.300d.txt",
    "train_data": base_in / "Part-2" / "triples.train.tsv",
    "validation_data": base_in / "Part-2" / "msmarco_tuples.validation.tsv",
    "eval": base_in / "Part-2" / "msmarco_qrels.txt",
    # eval paths
    "reranking": {
        "input": base_in / "Part-2" / "msmarco_tuples.test.tsv",
        "eval": base_in / "Part-2" / "msmarco_qrels.txt",
    },
    "baseline": {
        "input": base_in / "Part-2" / "fira-22.tuples.tsv",
        "eval": base_in / "Part-1" / "fira-22.baseline-qrels.tsv",
    },
    "ds": {
        "input": base_in / "Part-2" / "fira-22.tuples.tsv",
        "eval": base_out / "fira-22.qrels.tsv",
    },
}
model_export_path = base_out / f"{config['model']}_model.pth"
results_export_path = base_out / f"{config['model']}_model_{config['eval_data']}.txt"

assert Path(config["vocab_directory"]).exists()
assert Path(config["pre_trained_embedding"]).exists()
assert Path(config["train_data"]).exists()
assert Path(config["validation_data"]).exists()
assert Path(config["eval"]).exists()
assert Path(config["reranking"]["input"]).exists()
assert Path(config["reranking"]["eval"]).exists()
assert Path(config["baseline"]["input"]).exists()
assert Path(config["baseline"]["eval"]).exists()
assert Path(config["ds"]["input"]).exists()
assert Path(config["ds"]["eval"]).exists()

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")  # no mps in torch==1.6.0
print(f"device: {device}")

# get words that occur at least 10 times
vocab = Vocabulary.from_files(config["vocab_directory"])

# map words to (pre-trained) embeddings
tokens_embedder = Embedding(vocab=vocab, pretrained_file=str(config["pre_trained_embedding"]), embedding_dim=300, trainable=True, padding_index=0)
word_embedder = BasicTextFieldEmbedder({"tokens": tokens_embedder})

if config["model"] == "knrm":
    model = KNRM(word_embedder, n_kernels=11).to(device)
elif config["model"] == "tk":
    model = TK(word_embedder, n_kernels=11, n_layers=2, n_tf_dim=300, n_tf_heads=10).to(device)

criterion = nn.MarginRankingLoss(margin=1, reduction="mean").to(device)
optimizer = Adam(model.parameters(), lr=0.001)
lr_reducer = lr_scheduler.ReduceLROnPlateau(optimizer, patience=3, factor=0.5)

print("model", config["model"], "total parameters:", sum(p.numel() for p in model.parameters() if p.requires_grad))
print("network:", model)


"""
train
"""

if config["mode"] != "train":
    train_loader = triple_loader(config["train_data"], vocab)
    val_loader = tuple_loader(config["validation_data"], vocab)
    qrels = load_qrels(config["eval"])

    optimizer = Adam(model.parameters(), lr=1e-4, eps=1e-5)
    criterion = MarginRankingLoss(margin=1, reduction='mean').to(device)

    print('Model', config["model"], 'total parameters:', sum(p.numel() for p in model.parameters() if p.requires_grad))
    print('Network:', model)

    training_results = []
    best_score = 0

    for epoch in range(config["epochs"]):
        metricEarlyStopper = EarlyStopper()
        losses = []

        for i, batch in enumerate(Tqdm.tqdm(train_loader)):
            model.train()
            optimizer.zero_grad()

            batch = move_to_device(batch, device)
            current_batch_size = batch['query_tokens']['tokens']['tokens'].shape[0]
            target = torch.ones(current_batch_size, requires_grad=True).to(device)

            target_relevant_doc = model.forward(batch['query_tokens'], batch['doc_pos_tokens'])
            target_unrelevant_doc = model.forward(batch['query_tokens'], batch['doc_neg_tokens'])

            loss = criterion(target_relevant_doc, target_unrelevant_doc, target)

            loss.backward()
            optimizer.step()
            losses.append(loss.item())

            if (i + 1) % config["eval_step_size"] == 0:
                model.eval()
                results = {}

                for batch in Tqdm.tqdm(val_loader):
                    result = test_step(batch, model, device)
                    for query_id, document_rank in result.items():
                        if query_id in results.keys():
                            results[query_id].extend(document_rank)
                        else:
                            results[query_id] = document_rank

                ranked_results = unrolled_to_ranked_result(results)
                metrics = calculate_metrics_plain(ranked_results, qrels)
                model.train()
                metric = metrics["MRR@10"]

                if metric > best_score:
                    best_score = metric
                    torch.save(model.state_dict(), config.get("model_export_path"))

                training_results.append(f"EPOCH: {epoch}\tBATCH: {i}\tLOSS: {loss:.3f}\t MRR@10: {metric}")

                if metricEarlyStopper.early_stop(metric):
                    print("Metric early stopping triggered, exiting epoch")
                    break

    with open(r'logs.txt', 'w+') as fp:
        for item in training_results:
            fp.write("%s\n" % item)
        print('Done')
    sys.exit()

"""
eval
"""

if config["mode"] == "eval":
    test_loader = tuple_loader(config["eval_data"]["input"], vocab)
    qrels = load_qrels(config["eval_data"]["eval"], config["eval_data"] == 'ds')

    model.load_state_dict(torch.load(config.get("model_export_path"), map_location=device))
    model.eval()
    results = {}

    for batch in Tqdm.tqdm(test_loader):
        result = test_step(batch, model, device)

        for query_id, document_rank in result.items():
            if query_id in results.keys():
                results[query_id].extend(document_rank)
            else:
                results[query_id] = document_rank

    ranked_results = unrolled_to_ranked_result(results)
    metrics = calculate_metrics_plain(ranked_results, qrels)
    metric = metrics["MRR@10"]

    with open(config.get("results_export_path"), "w+") as outfile:
        for metric_name, metric_value in metrics.items():
            outfile.write(f"{metric_name}  :  {metric_value}\n")
    print(f"Metric is {metric}")
    sys.exit()
