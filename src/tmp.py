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


# change paths to your data directory
config = {
    "vocab_directory": "../Part-2/allen_vocab_lower_10",
    "pre_trained_embedding": "../Part-2/glove.42B.300d.txt",
    "train_data": "../Part-2/triples.train.tsv",
    "validation_data": "../Part-2/msmarco_tuples.validation.tsv",
    "eval": "../Part-2/msmarco_qrels.txt",
    "reranking": {"input": "../Part-2/msmarco_tuples.test.tsv", "eval": "../Part-2/msmarco_qrels.txt", "suffix": "reranking.txt"},
    "baseline": {"input": "../Part-2/fira-22.tuples.tsv", "eval": "../Part-1/fira-22.baseline-qrels.tsv", "suffix": "baseline.txt"},
    "ds": {"input": "../Part-2/fira-22.tuples.tsv", "eval": "../result_ds.csv", "suffix": "ds.txt"},
    "model_export_path": "../Part-2/<tmp>_model.pth",
    "results_export_path": "../Part-2/<tmp>_model_<sffx>",
}

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

mode = "eval"  # Change to eval/train depending on the desired mode
model = "tk"  # Change to knrm/tk depending on the desired model
evaluation_data = "baseline"  # Change between baseline/ds/reranking to evaluate different data sets

eval_step_size = 50  # After how many batched the model should be evaluated. Only relevant in training phase.
num_epochs = 10  # Number of epochs the models should train

vocab = Vocabulary.from_files(config["vocab_directory"])

config["results_export_path"] = config.get("results_export_path").replace("<tmp>", model).replace("<sffx>", config[evaluation_data]["suffix"])
config["model_export_path"] = config.get("model_export_path").replace("<tmp>", model)

tokens_embedder = Embedding(vocab=vocab, pretrained_file=config["pre_trained_embedding"], embedding_dim=300, trainable=True, padding_index=0)

word_embedder = BasicTextFieldEmbedder({"tokens": tokens_embedder})

# recommended default params for the models (but you may change them if you want)
if model == "knrm":
    model = KNRM(word_embedder, n_kernels=11).to(device)
elif model == "tk":
    model = TK(word_embedder, n_kernels=11, n_layers=2, n_tf_dim=300, n_tf_heads=15, n_ff_dim=100, max_doc_size=200).to(device)

# Model evaluation
if mode == "eval":
    test_loader = tuple_loader(config[evaluation_data]["input"], vocab)
    qrels = load_qrels(config[evaluation_data]["eval"], evaluation_data == "ds")

    model.load_state_dict(torch.load(config.get("model_export_path"), map_location=device))
    model.eval()
    results = {}

    print("Starting to read test data...")
    for batch in Tqdm.tqdm(test_loader):
        result = test_step(batch, model, device)

        for query_id, document_rank in result.items():
            if query_id in results.keys():
                results[query_id].extend(document_rank)
            else:
                results[query_id] = document_rank

    ranked_results = unrolled_to_ranked_result(results)

    # if evaluation_data in ["ds", "baseline"]:
    # metrics_ood = out_of_domain_eval(results, qrels)

    metrics = calculate_metrics_plain(ranked_results, qrels)
    metric = metrics["MRR@10"]

    with open(config.get("results_export_path"), "w+") as outfile:
        for metric in metrics.keys():
            outfile.write(f"{metric}  :  {metrics.get(metric)}\n")
    print(f"Metric is {metric}")
    sys.exit()

else:
    # Training of model

    # load training and evaluation data
    train_loader = triple_loader(config["train_data"], vocab)
    val_loader = tuple_loader(config["validation_data"], vocab)
    qrels = load_qrels(config["eval"])

    # initialize AdamW optimizer
    optimizer = Adam(model.parameters(), lr=1e-4, eps=1e-5)

    # Defining the loss function
    criterion = MarginRankingLoss(margin=1, reduction="mean").to(device)
    # lr_reducer = lr_scheduler.ReduceLROnPlateau(optimizer, patience=3, factor=0.5)
    print("Model", model, "total parameters:", sum(p.numel() for p in model.parameters() if p.requires_grad))
    print("Network:", model)

    # train
    training_results = []
    best_score = 0

    for epoch in range(num_epochs):
        metricEarlyStopper = EarlyStopper()
        losses = []

        # Looping through the training data
        for i, batch in enumerate(Tqdm.tqdm(train_loader)):
            # setting model to train mode.
            model.train()
            optimizer.zero_grad()

            batch = move_to_device(batch, device)

            # target is always 1 because we want to rank the first input (target_relevant_doc) higher
            current_batch_size = batch["query_tokens"]["tokens"]["tokens"].shape[0]
            target = torch.ones(current_batch_size, requires_grad=True).to(device)

            # forward: get output of model for relevant and un-relevant documents
            target_relevant_doc = model.forward(batch["query_tokens"], batch["doc_pos_tokens"])
            target_unrelevant_doc = model.forward(batch["query_tokens"], batch["doc_neg_tokens"])

            loss = criterion(target_relevant_doc, target_unrelevant_doc, target)

            loss.backward()
            print(f"EPOCH: {epoch}\tBATCH: {i}\tLOSS: {loss:.3f}")

            optimizer.step()
            losses.append(loss.item())

            # Validation
            if (i + 1) % eval_step_size == 0:
                model.eval()
                results = {}
                print("starting to read validation data")

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
                print(f"metric is {metric}")

                # saving best model we have seen so far
                if metric > best_score:
                    best_score = metric
                    torch.save(model.state_dict(), config.get("model_export_path"))

                training_results.append(f"EPOCH: {epoch}\tBATCH: {i}\tLOSS: {loss:.3f}\t MRR@10: {metric}")

                # lr_reducer.step(metric)
                if metricEarlyStopper.early_stop(metric):
                    print("Metric early stopping triggered, exiting epoch")
                    break

    # Export logs of epoch, iteration, loss and MRR metric
    with open(r"logs.txt", "w+") as fp:
        for item in training_results:
            # write each item on a new line
            fp.write("%s\n" % item)
        print("Done")
