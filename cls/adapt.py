# %%
from datasets import IMDB
from data import Corpus
from model import ClsModel
from loss import CELossWithMetric, UDALossWithMetric, FixMatchLossWithMetric
from utils.framework import Framework
import random
import torch.utils.data as data
import torch.optim as optim


if __name__ == "__main__":
    # Define models
    random.seed(0)
    model = ClsModel(max_length=512)
    optimizer = optim.AdamW(model.parameters(), lr=3e-5)
    loss_fn = CELossWithMetric()
    with_uda = True
    train_mode = True
    # Framework
    if train_mode:
        train_data = IMDB("datasets/aclImdb/train/", dataset_size=25)
        # train_corpus = Corpus(train_data, augmentation={"N": 10, "SYNONYM": 0.3, "BERT": 0.3}, return_type="uni")
        train_loader = data.DataLoader(train_data, batch_size=4, shuffle=True)
        
        val_data = IMDB("datasets/aclImdb/test/", dataset_size=1000)
        val_loader = data.DataLoader(val_data, batch_size=8, shuffle=True)

        if with_uda:
            uda_data = IMDB("datasets/aclImdb/train/", dataset_size=2500)
            uda_corpus = Corpus(uda_data, augmentation={"N": 1, "SYNONYM": 0.5})
            uda_loader = data.DataLoader(uda_corpus, batch_size=4, shuffle=True)
        else:
            uda_loader = None

        framework = Framework(
            model=model,
            data_pipeline_mapper=["input_list", None],
            optimizer=optimizer,
            scheduler=None,
            save_ckpt="imdb_bert_fxm_25.pt",
            # load_ckpt = "imdb_bert_uda_25.pt",
        )
        framework.train(
            criterion=loss_fn,
            data_loader=train_loader,
            num_iters=10000,
            val_loader=val_loader,
            val_iters=100,
            uda_loader=uda_loader,
            # uda_criterion=FixMatchLossWithMetric(threshold=0.8),
            uda_criterion=UDALossWithMetric(train_iters=10000)
        )
    else:
        test_data = IMDB("datasets/aclImdb/test/")
        test_loader = data.DataLoader(test_data, batch_size=8, shuffle=False)

        framework = Framework(
            model=model,
            data_pipeline_mapper=["input_list", None],
            optimizer=optimizer,
            scheduler=None,
            load_ckpt="imdb_bert_uda_25.pt",
        )
        stats = framework.test(
            criterion=loss_fn,
            data_loader=test_loader,
            num_iters=1e9,
        )
        print(stats)