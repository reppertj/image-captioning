import os
from unittest.mock import MagicMock

import torch
import wandb
from project.captioners import CaptioningRNN
from project.datasets import CombinedDataModule
from project.metrics import CorpusBleu
from pytest import mark
from pytorch_lightning import Trainer, seed_everything

"""
In real life, we would have many more tests. An overfitting test, however, is a great
way to sanity check an end-to-end machine learning training pipeline. We should be
able to overfit a single training set.
"""


@mark.slow
def test_captioner_overfit():
    seed_everything(42)

    wandb.run = MagicMock()
    wandb.run.__lt__.return_value = True

    datamodule = CombinedDataModule(
        flickr_csv=os.path.join("tests", "test_data", "test_flickr.csv"),
        flickr_dir=os.path.join("tests", "test_data", "test_flickr_images"),
        batch_size=4,
        val_size=4,
        test_size=4,
        transform="normalize",
        target_transform="tokenize",
        dev_set=12,
        num_workers=0,
    )

    datamodule.setup()

    config = CaptioningRNN.default_config()

    config["rnn_dropout"] = False
    config["label_smoothing_epsilon"] = 0.0
    config["inference_beam_width"] = 1
    config["inference_beam_alpha"] = 0.0

    model = CaptioningRNN(datamodule, config)

    trainer = Trainer(
        max_epochs=150,
        num_sanity_val_steps=0,
        log_every_n_steps=250,
        check_val_every_n_epoch=250,
    )
    trainer.fit(model)

    assert trainer.logged_metrics["train_loss"] < 0.1

    batch = next(iter(model.datamodule.train_dataloader()))
    with torch.no_grad():
        model.eval()
        preds = model(batch)

    bleu = CorpusBleu(model.tokenizer)

    bleu_score = bleu(preds.squeeze(), batch["captions"])
    assert bleu_score > 0.99
