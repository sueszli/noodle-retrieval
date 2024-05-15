from allennlp.common import Params, Tqdm
from allennlp.common.util import prepare_environment
from allennlp.common.checks import ConfigurationError
from allennlp.common.file_utils import cached_path
from allennlp.data.dataloader import PyTorchDataLoader
from allennlp.data.vocabulary import Vocabulary
from allennlp.data.dataset_readers.dataset_reader import DatasetReader
from allennlp.data.fields import TextField, MetadataField
from allennlp.data.instance import Instance
from allennlp.data.tokenizers import Tokenizer
from allennlp.data.token_indexers import TokenIndexer, SingleIdTokenIndexer
from allennlp.data.tokenizers.token import Token
from allennlp.modules.token_embedders import Embedding
from allennlp.modules.text_field_embedders import BasicTextFieldEmbedder, TextFieldEmbedder

import torch
import torch.nn as nn
from torch.autograd import Variable

from overrides import overrides
from blingfire import text_to_words

from pathlib import Path
from typing import Dict, Iterator, List
import logging


prepare_environment(Params({}))  # seed
logger = logging.getLogger(__name__)  # pylint: disable=invalid-name


class BlingFireTokenizer:
    def tokenize(self, sentence: str) -> List[Token]:
        return [Token(t) for t in text_to_words(sentence).split()]


class IrTripleDatasetReader(DatasetReader):
    """
    Read a tsv file containing triple sequences, and create a dataset suitable for a neural IR model, or any model with a matching API.
    Expected format for each input line: <query_sequence_string>\t<pos_doc_sequence_string>\t<neg_doc_sequence_string>
    The output of ``read`` is a list of ``Instance`` s with the fields:
        query_tokens: ``TextField`` and
        doc_pos_tokens: ``TextField`` and
        doc_neg_tokens: ``TextField``
    """

    def __init__(
        self, tokenizer: Tokenizer = None, token_indexers: Dict[str, TokenIndexer] = None, source_add_start_token: bool = True, max_doc_length: int = -1, max_query_length: int = -1, lazy: bool = False
    ) -> None:
        super().__init__(lazy)
        self._tokenizer = tokenizer or BlingFireTokenizer()
        self._token_indexers = token_indexers or {"tokens": SingleIdTokenIndexer(lowercase_tokens=True)}
        self._source_add_start_token = source_add_start_token
        self.max_doc_length = max_doc_length
        self.max_query_length = max_query_length

    @overrides
    def _read(self, file_path):
        with open(cached_path(file_path), "r", encoding="utf8") as data_file:
            # logger.info("Reading instances from lines in file at: %s", file_path)
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
    def text_to_instance(self, query_sequence: str, doc_pos_sequence: str, doc_neg_sequence: str) -> Instance:  # type: ignore
        # pylint: disable=arguments-differ
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
    Read a tsv file containing labeled tuple sequences, and create a dataset suitable for a
    neural IR model, or any model with a matching API.

    The output of ``read`` is a list of ``Instance`` s with the fields:
        "query_id",
        "doc_id",
        "query_tokens",
        "doc_tokens"
    """

    def __init__(
        self, tokenizer: Tokenizer = None, token_indexers: Dict[str, TokenIndexer] = None, source_add_start_token: bool = True, max_doc_length: int = -1, max_query_length: int = -1, lazy: bool = False
    ) -> None:
        super().__init__(lazy)
        self._tokenizer = tokenizer or BlingFireTokenizer()
        self._token_indexers = token_indexers or {"tokens": SingleIdTokenIndexer(lowercase_tokens=True)}
        self._source_add_start_token = source_add_start_token
        self.max_doc_length = max_doc_length
        self.max_query_length = max_query_length

    @overrides
    def _read(self, file_path):
        with open(cached_path(file_path), "r", encoding="utf8") as data_file:
            # logger.info("Reading instances from lines in file at: %s", file_path)
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
    def text_to_instance(self, query_id: str, doc_id: str, query_sequence: str, doc_sequence: str) -> Instance:  # type: ignore
        # pylint: disable=arguments-differ

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


# --------------------------------------------------------------------------------------------------------------


class KNRM(nn.Module):
    """
    Paper: End-to-End Neural Ad-hoc Ranking with Kernel Pooling, Xiong et al., SIGIR'17
    """

    def __init__(self, word_embeddings: TextFieldEmbedder, n_kernels: int):

        super(KNRM, self).__init__()

        self.word_embeddings = word_embeddings

        # static - kernel size & magnitude variables
        mu = torch.FloatTensor(self.kernel_mus(n_kernels)).view(1, 1, 1, n_kernels)
        sigma = torch.FloatTensor(self.kernel_sigmas(n_kernels)).view(1, 1, 1, n_kernels)

        self.register_buffer("mu", mu)
        self.register_buffer("sigma", sigma)

        # todo

    def forward(self, query: Dict[str, torch.Tensor], document: Dict[str, torch.Tensor]) -> torch.Tensor:
        # pylint: disable=arguments-differ

        #
        # prepare embedding tensors & paddings masks
        # -------------------------------------------------------

        # shape: (batch, query_max)
        query_pad_oov_mask = (query["tokens"]["tokens"] > 0).float()  # > 1 to also mask oov terms
        # shape: (batch, doc_max)
        document_pad_oov_mask = (document["tokens"]["tokens"] > 0).float()

        # shape: (batch, query_max,emb_dim)
        query_embeddings = self.word_embeddings(query)
        # shape: (batch, document_max,emb_dim)
        document_embeddings = self.word_embeddings(document)

        # todo
        output = torch.zeros(1)
        return output

    def kernel_mus(self, n_kernels: int):
        """
        get the mu for each guassian kernel. Mu is the middle of each bin
        :param n_kernels: number of kernels (including exact match). first one is exact match
        :return: l_mu, a list of mu.
        """
        l_mu = [1.0]
        if n_kernels == 1:
            return l_mu

        bin_size = 2.0 / (n_kernels - 1)  # score range from [-1, 1]
        l_mu.append(1 - bin_size / 2)  # mu: middle of the bin
        for i in range(1, n_kernels - 1):
            l_mu.append(l_mu[i] - bin_size)
        return l_mu

    def kernel_sigmas(self, n_kernels: int):
        """
        get sigmas for each guassian kernel.
        :param n_kernels: number of kernels (including exactmath.)
        :param lamb:
        :param use_exact:
        :return: l_sigma, a list of simga
        """
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

        # static - kernel size & magnitude variables
        mu = torch.FloatTensor(self.kernel_mus(n_kernels)).view(1, 1, 1, n_kernels)
        sigma = torch.FloatTensor(self.kernel_sigmas(n_kernels)).view(1, 1, 1, n_kernels)

        self.register_buffer("mu", mu)
        self.register_buffer("sigma", sigma)

        # todo

    def forward(self, query: Dict[str, torch.Tensor], document: Dict[str, torch.Tensor]) -> torch.Tensor:
        # pylint: disable=arguments-differ

        #
        # prepare embedding tensors & paddings masks
        # -------------------------------------------------------

        # shape: (batch, query_max)
        query_pad_oov_mask = (query["tokens"]["tokens"] > 0).float()  # > 1 to also mask oov terms
        # shape: (batch, doc_max)
        document_pad_oov_mask = (document["tokens"]["tokens"] > 0).float()

        # shape: (batch, query_max,emb_dim)
        query_embeddings = self.word_embeddings(query)
        # shape: (batch, document_max,emb_dim)
        document_embeddings = self.word_embeddings(document)

        # todo
        output = torch.zeros(1)
        return output

    def kernel_mus(self, n_kernels: int):
        """
        get the mu for each guassian kernel. Mu is the middle of each bin
        :param n_kernels: number of kernels (including exact match). first one is exact match
        :return: l_mu, a list of mu.
        """
        l_mu = [1.0]
        if n_kernels == 1:
            return l_mu

        bin_size = 2.0 / (n_kernels - 1)  # score range from [-1, 1]
        l_mu.append(1 - bin_size / 2)  # mu: middle of the bin
        for i in range(1, n_kernels - 1):
            l_mu.append(l_mu[i] - bin_size)
        return l_mu

    def kernel_sigmas(self, n_kernels: int):
        """
        get sigmas for each guassian kernel.
        :param n_kernels: number of kernels (including exactmath.)
        :param lamb:
        :param use_exact:
        :return: l_sigma, a list of simga
        """
        bin_size = 2.0 / (n_kernels - 1)
        l_sigma = [0.0001]  # for exact match. small variance -> exact match
        if n_kernels == 1:
            return l_sigma

        l_sigma += [0.5 * bin_size] * (n_kernels - 1)
        return l_sigma


# --------------------------------------------------------------------------------------------------------------

base = Path.cwd() / "data-merged" / "data" / "air-exercise-2" / "Part-2"

config = {
    "vocab_directory": base / "allen_vocab_lower_10",
    "pre_trained_embedding": base / "glove.42B.300d.txt",
    "model": "knrm",
    "train_data": base / "triples.train.tsv",
    "validation_data": base / "msmarco_tuples.validation.tsv",
    "test_data": base / "msmarco_tuples.test.tsv",
}
assert Path(config["vocab_directory"]).exists()
assert Path(config["pre_trained_embedding"]).exists()
assert Path(config["train_data"]).exists()
assert Path(config["validation_data"]).exists()
assert Path(config["test_data"]).exists()


"""
load data, define model
"""

# get words that occur at least 10 times
vocab = Vocabulary.from_files(config["vocab_directory"])

# map words to (pre-trained) embeddings
tokens_embedder = Embedding(vocab=vocab, pretrained_file=str(config["pre_trained_embedding"]), embedding_dim=300, trainable=True, padding_index=0)
word_embedder = BasicTextFieldEmbedder({"tokens": tokens_embedder})

# define model
if config["model"] == "knrm":
    model = KNRM(word_embedder, n_kernels=11)
elif config["model"] == "tk":
    model = TK(word_embedder, n_kernels=11, n_layers=2, n_tf_dim=300, n_tf_heads=10)
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
print("Model", config["model"], "total parameters:", sum(p.numel() for p in model.parameters() if p.requires_grad))
print("Network:", model)


"""
train
"""

_triple_reader = IrTripleDatasetReader(lazy=True, max_doc_length=180, max_query_length=30)
_triple_reader = _triple_reader.read(config["train_data"])
_triple_reader.index_with(vocab)
loader = PyTorchDataLoader(_triple_reader, batch_size=32)
# for epoch in range(2):
#     for batch in Tqdm.tqdm(loader):
#         # TODO: train loop
#         pass


"""
eval
"""

# duplicate for validation inside train loop - but rename "loader",
# otherwise it will overwrite the original train iterator, which is instantiated outside the loop
_tuple_reader = IrLabeledTupleDatasetReader(lazy=True, max_doc_length=180, max_query_length=30)
_tuple_reader = _tuple_reader.read(config["test_data"])
_tuple_reader.index_with(vocab)
loader = PyTorchDataLoader(_tuple_reader, batch_size=128)

# for batch in Tqdm.tqdm(loader):
#     # todo test loop
#     # todo evaluation
#     pass
