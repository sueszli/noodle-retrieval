from typing import Dict, Iterator, List

import torch
import torch.nn as nn
from torch.autograd import Variable

from allennlp.modules.text_field_embedders import TextFieldEmbedder


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

        return output


    # def forward(self, query_embeddings: torch.Tensor, document_embeddings: torch.Tensor,
    #             query_pad_oov_mask: torch.Tensor, document_pad_oov_mask: torch.Tensor, 
    #             output_secondary_output: bool = False) -> torch.Tensor:
    #     # pylint: disable=arguments-differ

    #     #
    #     # prepare embedding tensors & paddings masks
    #     # -------------------------------------------------------

    #     query_by_doc_mask = torch.bmm(query_pad_oov_mask.unsqueeze(-1), document_pad_oov_mask.unsqueeze(-1).transpose(-1, -2))
    #     query_by_doc_mask_view = query_by_doc_mask.unsqueeze(-1)

    #     #
    #     # cosine matrix
    #     # -------------------------------------------------------

    #     # shape: (batch, query_max, doc_max)
    #     cosine_matrix = self.cosine_module.forward(query_embeddings, document_embeddings)
    #     cosine_matrix_masked = cosine_matrix * query_by_doc_mask
    #     cosine_matrix_extradim = cosine_matrix_masked.unsqueeze(-1)

    #     #
    #     # gaussian kernels & soft-TF
    #     #
    #     # first run through kernel, then sum on doc dim then sum on query dim
    #     # -------------------------------------------------------
        
    #     raw_kernel_results = torch.exp(- torch.pow(cosine_matrix_extradim - self.mu, 2) / (2 * torch.pow(self.sigma, 2)))
    #     kernel_results_masked = raw_kernel_results * query_by_doc_mask_view

    #     per_kernel_query = torch.sum(kernel_results_masked, 2)
    #     log_per_kernel_query = torch.log(torch.clamp(per_kernel_query, min=1e-10)) * 0.01
    #     log_per_kernel_query_masked = log_per_kernel_query * query_pad_oov_mask.unsqueeze(-1) # make sure we mask out padding values

    #     per_kernel = torch.sum(log_per_kernel_query_masked, 1) 

    #     ##
    #     ## "Learning to rank" layer - connects kernels with learned weights
    #     ## -------------------------------------------------------

    #     dense_out = self.dense(per_kernel)
    #     score = torch.squeeze(dense_out,1) #torch.tanh(dense_out), 1)

    #     if output_secondary_output:
    #         query_mean_vector = query_embeddings.sum(dim=1) / query_pad_oov_mask.sum(dim=1).unsqueeze(-1)
    #         return score, {"score":score,"per_kernel":per_kernel,"query_mean_vector":query_mean_vector,"cosine_matrix_masked":cosine_matrix_masked}
    #     else:
    #         return score

    # def forward_representation(self, sequence_embeddings: torch.Tensor, sequence_mask: torch.Tensor) -> torch.Tensor:
    #     return sequence_embeddings * sequence_mask.unsqueeze(-1)

    # def get_param_stats(self):
    #     return "KNRM: linear weight: "+str(self.dense.weight.data)

    # def get_param_secondary(self):
    #     return {"kernel_weight":self.dense.weight}


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
