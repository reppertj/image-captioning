import os
from project.metrics import CorpusBleu

from torch.utils import data
from project.datasets import CombinedDataModule
from project.captioners import CaptioningRNN
from pytest import mark
from unittest.mock import MagicMock
import torch
from pytorch_lightning import Trainer, seed_everything
import pytorch_lightning as pl
import wandb


"""
In real life, we would have many more tests. An overfitting test, however, is a great
way to sanity check an end-to-end machine learning training pipeline. We should be
able to overfit a single training set.  To simulate the full program as much as possible, this requires the presence of a "mini" version of the Flickr30K and COCO datasets.
"""

def test_captioner_overfit():
    seed_everything(42)
    
    wandb.run = MagicMock()
    wandb.run.__lt__.return_value = True
    wandb_logger = pl.loggers.WandbLogger(project='project')
    wandb_logger._experiment = MagicMock()
    wandb_logger._experiment.__call__ = MagicMock(return_value=True)

    datamodule = CombinedDataModule(
        flickr_csv=os.path.join('tests', 'test_data', 'test_flickr.csv'),
        flickr_dir=os.path.join('tests', 'test_data', 'test_flickr_images'),
        batch_size=4,
        val_size=4,
        test_size=4,
        transform='normalize',
        target_transform='tokenize',
        dev_set=12,
        num_workers=0,
    )
    
    model = CaptioningRNN(datamodule, rnn_type='attention', batch_size=4, num_rnns=1, rnn_dropout=False)
    
    trainer = Trainer(max_epochs=200, num_sanity_val_steps=0, log_every_n_steps=250, check_val_every_n_epoch=250, logger=wandb_logger)
    trainer.fit(model)

    batch = next(iter(model.datamodule.train_dataloader()))
    with torch.no_grad():
        model.eval()
        preds = model(batch)
        
    bleu = CorpusBleu(model.datamodule.tokenizer)
    
    bleu_score = bleu(preds.squeeze(), batch['captions'])
    print(bleu_score)
    assert bleu_score > 0.7
