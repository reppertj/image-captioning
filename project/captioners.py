import pytorch_lightning as pl
import torch
import torch.nn.functional as F
import wandb
from torch import nn

from project.datasets import tokens_to_ids, BOS, EOS, UNK, PAD
from project.decoders import GRU, LSTM, RNN, ParallelAttentionLSTM, ParallelFCScorer
from project.feature_extraction import ImageFeatureExtractor, WordEmbedder
from project.loss import multi_caption_smoothing_temporal_softmax_loss
from project.metrics import CorpusBleu
from project.utils import batch_beam_search, log_wandb_preds


class CaptioningRNN(pl.LightningModule):
    def __init__(
        self,
        datamodule,
        batch_size=64,
        wordvec_dim=768,
        hidden_size=512,
        wd_embedder_init="kaiming",
        image_encoder="mobilenetv2",
        encoder_init="kaiming",
        rnn_type="rnn",
        num_rnns=5,
        num_rnn_layers=2,
        rnn_nonlinearity="relu",
        rnn_init="kaiming",
        rnn_dropout=0.5,
        rnn_bidirectional=False,
        fc_init="kaiming",
        learning_rate=3e-4,
    ):
        """
        
        """
        super().__init__()

        self.batch_size = batch_size
        self.datamodule = datamodule
        self.datamodule.setup()
        self.val_bleu = CorpusBleu(self.datamodule.tokenizer)
        self.test_bleu = CorpusBleu(self.datamodule.tokenizer)

        if isinstance(image_encoder, str) and image_encoder not in {
            "resnet50",
            "resnet101",
            "resnet152",
            "mobilenetv2",
            "vgg16",
            "resnext50",
        }:
            raise ValueError(f"Encoder {image_encoder} not implemented")

        if rnn_type not in ("rnn", "gru", "lstm", "attention"):
            raise ValueError(f"RNN type {rnn_type} not implemented")

        self.rnn_type = rnn_type

        if self.rnn_type in ("rnn", "lstm", "gru"):
            self.image_extractor = ImageFeatureExtractor(
                encoder=image_encoder, projection_out=hidden_size,
            )
        elif self.rnn_type in ("attention"):
            self.image_extractor = ImageFeatureExtractor(
                encoder=image_encoder,
                projection_out=hidden_size,
                pooling=False,
                convolution_in="infer",
                projection_in=False,
            )
        if encoder_init:
            self.image_extractor.init_weights(encoder_init)

        self.word_embedder = WordEmbedder(wordvec_dim, datamodule.tokenizer)
        if wd_embedder_init:
            self.word_embedder.init_weights(wd_embedder_init)

        self.vocab_size = self.word_embedder.vocab_size
        self.wordvec_dim = self.word_embedder.wordvec_dim

        self._pad = datamodule.tokenizer.padding["pad_id"]
        self._start = tokens_to_ids(datamodule.tokenizer, [BOS])[BOS]
        self._end = tokens_to_ids(datamodule.tokenizer, [EOS])[EOS]

        self.ignore_index = self._pad

        self.num_rnn_layers = num_rnn_layers
        self.num_rnn_directions = 2 if rnn_bidirectional else 1
        self.rnn_dropout = rnn_dropout if rnn_dropout else 0

        self.learning_rate = learning_rate

        # RNN
        if rnn_type == "rnn":
            self.decoder = RNN(
                input_size=self.wordvec_dim,
                hidden_size=hidden_size,
                num_rnns=num_rnns,
                num_layers=num_rnn_layers,
                nonlinearity=rnn_nonlinearity,
                dropout=self.rnn_dropout,
                bidirectional=rnn_bidirectional,
            )
        elif rnn_type == "gru":
            self.decoder = GRU(
                input_size=self.wordvec_dim,
                hidden_size=hidden_size,
                num_rnns=num_rnns,
                num_layers=num_rnn_layers,
                nonlinearity=rnn_nonlinearity,
                dropout=self.rnn_dropout,
                bidirectional=rnn_bidirectional,
            )
        elif rnn_type == "lstm":
            self.decoder = LSTM(
                input_size=self.wordvec_dim,
                hidden_size=hidden_size,
                num_rnns=num_rnns,
                num_layers=num_rnn_layers,
                nonlinearity=rnn_nonlinearity,
                dropout=self.rnn_dropout,
                bidirectional=rnn_bidirectional,
            )
        if rnn_type == "attention":
            self.decoder = ParallelAttentionLSTM(
                input_size=self.wordvec_dim,
                hidden_size=hidden_size,
                num_features=hidden_size,
                num_heads=num_rnn_layers,
                dropout=self.rnn_dropout,
                num_rnns=num_rnns,
            )

        if rnn_init:
            self.decoder.init_weights(rnn_init)

        self.fc_scorer = ParallelFCScorer(num_rnns, hidden_size, self.vocab_size)
        if fc_init:
            self.fc_scorer.init_weights(fc_init)

    def train_dataloader(self):
        return self.datamodule.train_dataloader(self.batch_size)

    def val_dataloader(self):
        return self.datamodule.val_dataloader(self.batch_size)

    def test_dataloader(self):
        return self.datamodule.test_dataloader(self.batch_size)

    def forward(self, batch, n_captions=1, return_attn=False):
        """This is a pl.LightningModule, so `forward` is *not* called
        during training. We use this to define inference logic instead."""
        max_length = self.datamodule.max_caption_length
        if n_captions > self.decoder.num_rnns:
            raise ValueError("Cannot generate more captions than trained rnns")
        x = batch["image"]
        batch_size = x.shape[0]
        x = self.image_extractor.encoder(x)  # (N, cnn_out, K, K)
        if self.image_extractor.pooling:
            x = self.image_extractor.pooling(x)  # (N, cnn_out, 1, 1)
        x = F.normalize(x, p=2, dim=1)  # L2 normalize image features
        if self.image_extractor.projector:
            x = x.view(batch_size, -1)  # (N, cnn_out)
            x = self.image_extractor.projector(x)  # (N, hidden_size)
        if self.image_extractor.convolution:
            x = self.image_extractor.convolution(x)  # (N, hidden_size, pixels, pixels)

        captions = torch.empty(
            (batch_size, n_captions, max_length + 1), device=x.device, dtype=torch.long
        )

        y = torch.tensor([self._start] * batch_size, device=x.device).view(
            batch_size, -1
        )
        y = self.word_embedder(y)

        cn, states, features = None, None, None

        if self.rnn_type in ("rnn", "gru", "lstm", "attention"):
            # Build predictions network-by-network
            for i in range(captions.shape[1]):
                yn = y
                if self.rnn_type in ("rnn", "gru", "lstm"):
                    hn = x.unsqueeze(0).repeat(
                        self.num_rnn_layers * self.num_rnn_directions, 1, 1
                    )
                elif self.rnn_type == "attention":
                    hn, cn = None, None
                    features = x
                if self.rnn_type == "lstm":
                    states = (hn, torch.zeros_like(hn))
                captions[:, i, :] = batch_beam_search(
                    rnn_captioner=self,
                    yns=yn,
                    hns=hn,
                    cns=cn,
                    states=states,
                    features=features,
                    max_length=self.datamodule.max_caption_length,
                    which_rnn=i,
                    alpha=1.0,
                    beam_width=10,
                )

        if return_attn and self.rnn_type == "attention":
            return captions, attn_weights
        else:
            return captions

    def forward_step(self, batch, batch_idx):
        """Training-time loss for the RNN.
        images: (N, 3, 224, 224)
        captions: (N, C, T)
        """
        ### Ingest inputs, shapes ###
        x, y = batch["image"], batch["captions"]
        batch_size = y.shape[0]

        ### Image features to initial hidden state ###
        x = self.image_extractor.encoder(x)  # (N, cnn_out, K, K)
        if self.image_extractor.pooling:
            x = self.image_extractor.pooling(x)  # (N, cnn_out, 1, 1)
        x = F.normalize(x, p=2, dim=1)  # L2 normalize image features
        if self.image_extractor.projector:
            x = x.flatten(start_dim=1)  # (N, cnn_out)
            x = self.image_extractor.projector(x)  # (N, hidden_size)
        if self.image_extractor.convolution:
            x = self.image_extractor.convolution(x)  # (N, hidden_size, K, K)

        ### Offset captions for teacher forcing ###
        y_in, y_out = y[:, :, :-1], y[:, :, 1:]

        ### Get input caption features ###
        y_in = self.word_embedder(y_in)  # (N, C, T - 1, W)

        if self.rnn_type in ("rnn", "lstm", "gru"):
            x = x.unsqueeze(0).repeat(
                self.num_rnn_layers * self.num_rnn_directions, 1, 1
            )
        if self.rnn_type in ("rnn", "gru"):
            rnn_outs = self.decoder(y_in, x)
        elif self.rnn_type == "attention":
            rnn_outs = self.decoder(y_in, x)
        elif self.rnn_type == "lstm":
            c0 = torch.zeros_like(x)
            rnn_outs = self.decoder(y_in, (x, c0))
        scores = self.fc_scorer(rnn_outs)

        y_out = y_out[:, : self.decoder.num_rnns, :]

        loss = multi_caption_smoothing_temporal_softmax_loss(
            scores, y_out, ignore_index=self.ignore_index
        )
        return loss

    def training_step(self, batch, batch_idx):
        loss = self.forward_step(batch, batch_idx)
        self.log("train_loss", loss)
        return loss

    def validation_step(self, batch, batch_idx):
        loss = self.forward_step(batch, batch_idx)
        preds = self.forward(batch, n_captions=self.decoder.num_rnns)
        for i in range(self.decoder.num_rnns):
            self.val_bleu(preds[:, i, :], batch["captions"])
        if batch_idx % 100 == 0:
            #  Periodically log minibatch of predictions with their images
            images = batch["image"][:5]
            preds = preds[:5, :, :]
            ground_truth = batch["captions"][:5]
            captions = ground_truth[:, : preds.shape[1], :]
            examples = log_wandb_preds(
                self.datamodule.tokenizer, images, preds, captions
            )
            wandb.log({"val_examples": examples}, commit=False)
        self.log("val_loss", loss, on_step=True)
        self.log("val_bleu_score", self.val_bleu, on_step=False, on_epoch=True)

    def test_step(self, batch, batch_idx):
        loss = self.forward_step(batch, batch_idx)
        preds = self.forward(batch, n_captions=self.decoder.num_rnns)
        for i in range(self.decoder.num_rnns):
            self.test_bleu(preds[:, i, :], batch["captions"])
        self.log("test_loss", loss, on_step=True)
        self.log("test_bleu_score", self.test_bleu, on_step=False, on_epoch=True)

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=self.learning_rate)
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, patience=5, min_lr=1e-6
        )
        return {
            "optimizer": optimizer,
            "scheduler": scheduler,
            "monitor": "val_loss",
        }

