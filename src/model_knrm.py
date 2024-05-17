from typing import Dict, Iterator, List

import torch
import torch.nn as nn
from torch.nn.functional import cosine_similarity
from torch.autograd import Variable

from allennlp.modules.text_field_embedders import TextFieldEmbedder


class KNRM(nn.Module):
    '''
    Paper: End-to-End Neural Ad-hoc Ranking with Kernel Pooling, Xiong et al., SIGIR'17
    '''

    def __init__(self,
                 word_embeddings: TextFieldEmbedder,
                 n_kernels: int):

        super(KNRM, self).__init__()

        self.word_embeddings = word_embeddings

        # static - kernel size & magnitude variables
        mu = torch.FloatTensor(self.kernel_mus(n_kernels)).view(1, 1, 1, n_kernels)
        sigma = torch.FloatTensor(self.kernel_sigmas(n_kernels)).view(1, 1, 1, n_kernels)

        self.register_buffer('mu', mu)
        self.register_buffer('sigma', sigma)
        self.mu = mu
        self.sigma = sigma
        #todo

        # defining the linear layer at the end of the training process to compute the targets
        # bias=true tells pytorch to learn the bias as well (w * X + b)
        self.lin = nn.Linear(n_kernels, 1, bias=True)
        torch.nn.init.uniform_(self.lin.weight, -0.001, 0.001)
        self.lin.bias.data.fill_(1)

    def forward(self, query: Dict[str, torch.Tensor], document: Dict[str, torch.Tensor]) -> torch.Tensor:
        # pylint: disable=arguments-differ

        #
        # prepare embedding tensors & paddings masks
        # -------------------------------------------------------

        # shape: (batch, query_max)
        query_pad_oov_mask = (query["tokens"]["tokens"] > 0).float() # > 1 to also mask oov terms
        # shape: (batch, doc_max)
        document_pad_oov_mask = (document["tokens"]["tokens"] > 0).float()

        # Computes the mask that the kernel_result is multiplied with.
        # This is needed because after the kernels, prior 0 values are now non-zero values.
        # These need to be masked out to not influence the output
        # shape: (batch_size, max_query, max_document)
        kernel_mask = torch.bmm(query_pad_oov_mask.unsqueeze(-1), document_pad_oov_mask.unsqueeze(-1).transpose(1, 2))

        # shape: (batch, query_max, emb_dim)
        query_embeddings = self.word_embeddings(query)
        # shape: (batch, document_max, emb_dim)
        document_embeddings = self.word_embeddings(document)

        # to compute the cosine similarity,
        # the embeddings of the query and the documents are normalized with the L2 norm
        query_embeddings_normalized = query_embeddings / (
            query_embeddings.norm(p=2, dim=-1, keepdim=True) + 1e-13
        )
        document_embeddings_normalized = document_embeddings / (
                document_embeddings.norm(p=2, dim=-1, keepdim=True) + 1e-13
        )
        
        # Calculate translation matrix (M): now normalized embedding are multiplied element wise and added together.
        # This is done via the matrix-multiplication if the normalized query and document embedding
        # shape (batch_size, query_max, document_max, 1): for each batch, each element in the matrix
        # represents the cosine-similarity between the i-th query word and the j-th document word
        # the unsqueeze at the end adds another dimension for the next computation.
        translation_matrix = torch.bmm(query_embeddings_normalized, document_embeddings_normalized.transpose(-1, -2)).unsqueeze(-1)

        # apply RBF Kernel:
        # now for each row, each kernel is applied to each column value
        # an example to further explain what happens: when accessing kernel_results[b, r, c, k], one would access the
        # result of the k-th kernel, applied to the c-th column of the r-th row of batch b.
        # shape of result: (batch_size, query_max, doc_max, kernel_size)
        kernel_results = torch.exp(-1/2 * torch.pow((translation_matrix - self.mu)/(self.sigma), 2))

        # Now the results of the kernel is masked. The kernel mask is expanded with an additional dimension.
        # The 4th dimension is either 0 or 1 which represents if the kernel results should be kept or not.
        # e.g.: max_query_length = 14 max_doc_length = 180 and actual_query_length=10 actual_doc_length=100
        # now the entry of the unsqueezed masked int 11th row and the 100th column would be actually be zero as there is
        # no 11th word in the query. Still, it the results of the kernel at said row and column could be !=0.
        # To prevent this influence on our output, the results are masked.
        kernel_results = kernel_results * kernel_mask.unsqueeze(-1)

        
        # Soft-TF features - Phi
        # Now, the kernel results of every column for every row are summed together, yielding a trensor of
        # shape (batch_size, query_max, kernel_size)
        per_kernel_query = torch.sum(kernel_results, 2)

        # not allow too small of a values as log explodes in the negatives
        # log_per_kernel_query = torch.log(torch.clamp(per_kernel_query, min=1e-5) + 1) * 0.01

        log_per_kernel_query = torch.log(torch.clamp(per_kernel_query, min=1e-10)) * 0.01

        # the result of the log of the kernel queries is again masked, as the null values in the per_kernel_query tensor
        # are clamped and replaced with 1e-10, which yields negative values.
        # Those values are again masked from the result.
        log_per_kernel_query = log_per_kernel_query * query_pad_oov_mask.unsqueeze(-1)

        # sum up kernels. Phi now is the k-dimensional feature vector described in the paper
        phi = torch.sum(log_per_kernel_query, 1) 

        # calculate ranking score via tanh(w * Phi + b)
        # where w * Phi + b is done by nn.Linear()
        output = self.lin(phi)

        output = torch.squeeze(torch.tanh(output), 1)
        # output = torch.tanh(output)
        # output = torch.squeeze(output, 1)
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
