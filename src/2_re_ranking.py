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
import torch
import torch.nn as nn
from torch.autograd import Variable
from allennlp.modules.matrix_attention.cosine_matrix_attention import CosineMatrixAttention
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
logger = logging.getLogger(__name__)


#region file-reader


class BlingFireTokenizer:
    def tokenize(self, sentence: str) -> List[Token]:
        return [Token(t) for t in text_to_words(sentence).split()]


class IrTripleDatasetReader(DatasetReader):
    """
    convert `triples.train.tsv` into 3 `TextField`s - which are a list of string tokens.
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


#endregion file-reader


#region models


class KNRM(nn.Module):
    """
    Paper: End-to-End Neural Ad-hoc Ranking with Kernel Pooling, Xiong et al., SIGIR'17

    see: https://github.com/sebastian-hofstaetter/matchmaker/blob/210b9da0c46ee6b672f59ffbf8603e0f75edb2b6/matchmaker/models/knrm.py
    """

    def __init__(self, word_embeddings: TextFieldEmbedder, n_kernels: int):

        super(KNRM, self).__init__()

        self.word_embeddings = word_embeddings

        # static list of mu and sigma values for the gaussian convolutional kernels
        mu = torch.FloatTensor(self.kernel_mus(n_kernels)).view(1, 1, 1, n_kernels)
        sigma = torch.FloatTensor(self.kernel_sigmas(n_kernels)).view(1, 1, 1, n_kernels)
        self.register_buffer("mu", mu) # prevents updates
        self.register_buffer("sigma", sigma)

        # this does not really do "attention" - just a plain cosine matrix calculation (without learnable weights)
        self.cosine_module = CosineMatrixAttention()

        # bias is set to True in original code (we found it to not help, how could it?)
        self.dense = nn.Linear(n_kernels, 1, bias=False)

        # init with small weights, otherwise the dense output is way to high for the tanh -> resulting in loss == 1 all the time
        torch.nn.init.uniform_(self.dense.weight, -0.014, 0.014)  # inits taken from matchzoo
        #self.dense.bias.data.fill_(0.0)


    def forward(self, query: Dict[str, torch.Tensor], document: Dict[str, torch.Tensor]) -> torch.Tensor:
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

    def github__init__(self,
                 n_kernels: int):

        super(KNRM, self).__init__()

        # static - kernel size & magnitude variables
        self.mu = Variable(torch.cuda.FloatTensor(self.kernel_mus(n_kernels)), requires_grad=False).view(1, 1, 1, n_kernels)
        self.sigma = Variable(torch.cuda.FloatTensor(self.kernel_sigmas(n_kernels)), requires_grad=False).view(1, 1, 1, n_kernels)

        # this does not really do "attention" - just a plain cosine matrix calculation (without learnable weights) 
        self.cosine_module = CosineMatrixAttention()

        # bias is set to True in original code (we found it to not help, how could it?)
        self.dense = nn.Linear(n_kernels, 1, bias=False)

        # init with small weights, otherwise the dense output is way to high for the tanh -> resulting in loss == 1 all the time
        torch.nn.init.uniform_(self.dense.weight, -0.014, 0.014)  # inits taken from matchzoo
        #self.dense.bias.data.fill_(0.0)


    def github_forward(self, query_embeddings: torch.Tensor, document_embeddings: torch.Tensor,
                query_pad_oov_mask: torch.Tensor, document_pad_oov_mask: torch.Tensor, 
                output_secondary_output: bool = False) -> torch.Tensor:
        # pylint: disable=arguments-differ

        #
        # prepare embedding tensors & paddings masks
        # -------------------------------------------------------

        query_by_doc_mask = torch.bmm(query_pad_oov_mask.unsqueeze(-1), document_pad_oov_mask.unsqueeze(-1).transpose(-1, -2))
        query_by_doc_mask_view = query_by_doc_mask.unsqueeze(-1)

        #
        # cosine matrix
        # -------------------------------------------------------

        # shape: (batch, query_max, doc_max)
        cosine_matrix = self.cosine_module.forward(query_embeddings, document_embeddings)
        cosine_matrix_masked = cosine_matrix * query_by_doc_mask
        cosine_matrix_extradim = cosine_matrix_masked.unsqueeze(-1)

        #
        # gaussian kernels & soft-TF
        #
        # first run through kernel, then sum on doc dim then sum on query dim
        # -------------------------------------------------------
        
        raw_kernel_results = torch.exp(- torch.pow(cosine_matrix_extradim - self.mu, 2) / (2 * torch.pow(self.sigma, 2)))
        kernel_results_masked = raw_kernel_results * query_by_doc_mask_view

        per_kernel_query = torch.sum(kernel_results_masked, 2)
        log_per_kernel_query = torch.log(torch.clamp(per_kernel_query, min=1e-10)) * 0.01
        log_per_kernel_query_masked = log_per_kernel_query * query_pad_oov_mask.unsqueeze(-1) # make sure we mask out padding values

        per_kernel = torch.sum(log_per_kernel_query_masked, 1) 

        ##
        ## "Learning to rank" layer - connects kernels with learned weights
        ## -------------------------------------------------------

        dense_out = self.dense(per_kernel)
        score = torch.squeeze(dense_out,1) #torch.tanh(dense_out), 1)

        if output_secondary_output:
            query_mean_vector = query_embeddings.sum(dim=1) / query_pad_oov_mask.sum(dim=1).unsqueeze(-1)
            return score, {"score":score,"per_kernel":per_kernel,"query_mean_vector":query_mean_vector,"cosine_matrix_masked":cosine_matrix_masked}
        else:
            return score

    def forward_representation(self, sequence_embeddings: torch.Tensor, sequence_mask: torch.Tensor) -> torch.Tensor:
        return sequence_embeddings * sequence_mask.unsqueeze(-1)

    def get_param_stats(self):
        return "KNRM: linear weight: "+str(self.dense.weight.data)

    def get_param_secondary(self):
        return {"kernel_weight":self.dense.weight}


    def kernel_mus(self, n_kernels: int):
        """
        mu for each guassian kernel. sits in the middle of each bin.
        :param n_kernels: number of kernels - where first one is the exact match.
        :return: l_mu, a list of mu.
        """
        l_mu = [1.0] # exact match
        if n_kernels == 1:
            return l_mu

        bin_size = 2.0 / (n_kernels - 1)  # score range from [-1, 1]
        l_mu.append(1 - bin_size / 2)  # mu: middle of the bin
        for i in range(1, n_kernels - 1):
            l_mu.append(l_mu[i] - bin_size)
        return l_mu

    def kernel_sigmas(self, n_kernels: int):
        """
        sigma for each guassian kernel.
        :param n_kernels: number of kernels - where first one is the exact match.
        :return: l_sigma, a list of sigma
        """
        bin_size = 2.0 / (n_kernels - 1)
        l_sigma = [0.0001]  # exact match (small variance)
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


#endregion models

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

device = torch.device("cuda" if torch.cuda.is_available() else "cpu") # no mps in torch==1.6.0
print(f"device: {device}")

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
print("model", config["model"], "total parameters:", sum(p.numel() for p in model.parameters() if p.requires_grad))
print("network:", model)


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
