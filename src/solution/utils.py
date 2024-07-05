from allennlp.common import Params, Tqdm
from allennlp.common.util import prepare_environment
from allennlp.data.dataloader import PyTorchDataLoader

prepare_environment(Params({}))  # sets the seeds to be fixed

from allennlp.nn.util import move_to_device

from data_loading import *
from model_tk import *


class EarlyStopper:
    def __init__(self, patience=5, min_delta=0):
        self.patience = patience
        self.min_delta = min_delta
        self.counter = 0
        self.min_metric= 0

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
        output = model(batch['query_tokens'], batch['doc_tokens']).tolist()
        for i in range(len(batch['query_id'])):
            query_id = batch['query_id'][i]
            doc_id = batch['doc_id'][i]
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