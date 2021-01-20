import torch
import wandb
from torch import nn
import torch.nn.functional as F
import pytorch_lightning as pl
from project.feature_extraction import ImageFeatureExtractor, WordEmbedder
from project.datasets import tokens_to_ids
from project.decoders import RNN, GRU, LSTM, ParallelAttentionLSTM, ParallelFCScorer
from project.loss import multi_caption_temporal_softmax_loss, temporal_softmax_loss, multi_caption_smoothing_temporal_softmax_loss
from project.utils import log_wandb_preds, get_new_candidates, batch_beam_search, PQCandidate, Candidate

BOS = '[CLS]'
EOS = '[SEP]'
UNK = '[UNK]'
PAD = '[PAD]'


class CaptioningRNN(pl.LightningModule):
    def __init__(
        self,
        datamodule,
        batch_size=64,
        wordvec_dim=768,
        hidden_size=512,
        wd_embedder_init='kaiming',
        image_encoder='mobilenetv2',
        encoder_init='kaiming',
        rnn_type='rnn',
        num_rnns=5,
        num_rnn_layers=2,
        rnn_nonlinearity='relu',
        rnn_init='kaiming',
        rnn_dropout=0.5,
        rnn_bidirectional=False,
        fc_init='kaiming',
        learning_rate=3e-4,
        ):
        """
        
        """
        super().__init__()

        self.batch_size = batch_size
        self.datamodule = datamodule
        self.datamodule.setup()


        if isinstance(image_encoder, str) and image_encoder not in {
            'resnet50',
            'resnet101',
            'resnet152',
            'mobilenetv2',
            'vgg16',
        }:
            raise ValueError(f'Encoder {image_encoder} not implemented')

        if rnn_type not in ('rnn', 'gru', 'lstm', 'attention'):
            raise ValueError(f'RNN type {rnn_type} not implemented')

        self.rnn_type = rnn_type

        if self.rnn_type in ('rnn', 'lstm', 'gru'):
            self.image_extractor = ImageFeatureExtractor(
                encoder=image_encoder,
                projection_out=hidden_size,
                )
        elif self.rnn_type in ('attention'):
            self.image_extractor = ImageFeatureExtractor(
                encoder=image_encoder,
                projection_out=hidden_size,
                pooling=False,
                convolution_in='infer',
                projection_in=False,
            )
        if encoder_init:
            self.image_extractor.init_weights(encoder_init)

        self.word_embedder = WordEmbedder(wordvec_dim, datamodule.tokenizer)
        if wd_embedder_init:
            self.word_embedder.init_weights(wd_embedder_init)

        self.vocab_size = self.word_embedder.vocab_size
        self.wordvec_dim = self.word_embedder.wordvec_dim

        self._pad = datamodule.tokenizer.padding['pad_id']
        self._start = tokens_to_ids(datamodule.tokenizer, [BOS])[BOS]
        self._end = tokens_to_ids(datamodule.tokenizer, [EOS])[EOS]

        self.ignore_index = self._pad

        self.num_rnn_layers = num_rnn_layers
        self.num_rnn_directions = 2 if rnn_bidirectional else 1
        self.rnn_dropout = rnn_dropout if rnn_dropout else 0

        self.learning_rate = learning_rate

        # RNN
        if rnn_type == 'rnn':
            self.decoder = RNN(
                input_size=self.wordvec_dim,
                hidden_size=hidden_size,
                num_rnns=num_rnns,                
                num_layers=num_rnn_layers,
                nonlinearity=rnn_nonlinearity,
                dropout=self.rnn_dropout,
                bidirectional=rnn_bidirectional,
            )
        elif rnn_type == 'gru':
            self.decoder = GRU(
                input_size=self.wordvec_dim,
                hidden_size=hidden_size,
                num_rnns=num_rnns,                
                num_layers=num_rnn_layers,
                nonlinearity=rnn_nonlinearity,
                dropout=self.rnn_dropout,
                bidirectional=rnn_bidirectional,
            )            
        elif rnn_type == 'lstm':
            self.decoder = LSTM(
                input_size=self.wordvec_dim,
                hidden_size=hidden_size,
                num_rnns=num_rnns,                
                num_layers=num_rnn_layers,
                nonlinearity=rnn_nonlinearity,
                dropout=self.rnn_dropout,
                bidirectional=rnn_bidirectional,
            )            
        if rnn_type == 'attention':
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
            raise ValueError('Cannot generate more captions than trained rnns')
        x = batch['image']
        batch_size = x.shape[0]
        x = self.image_extractor.encoder(x) # (N, cnn_out, K, K)
        if self.image_extractor.pooling:
            x = self.image_extractor.pooling(x) #  (N, cnn_out, 1, 1)
        x = F.normalize(x, p=2, dim=1)  # L2 normalize image features
        if self.image_extractor.projector:
            x = x.view(batch_size, -1)  # (N, cnn_out)
            x = self.image_extractor.projector(x)  # (N, hidden_size)
        if self.image_extractor.convolution:
            x = self.image_extractor.convolution(x)  # (N, hidden_size, pixels, pixels)
            pixels = x.shape[-1]

        captions = [
            self._pad * x.new(batch_size, max_length).fill_(1).long()
            ] * n_captions
        captions = torch.stack(captions, dim=1)
        captions_start = torch.empty((batch_size, n_captions, max_length + 1), device=x.device, dtype=torch.long)        
        
        y = torch.tensor([self._start] * batch_size, device=x.device).view(batch_size, -1)
        y = self.word_embedder(y)

        cn, states, features = None, None, None
        
        if self.rnn_type in ('rnn', 'gru', 'lstm', 'attention'):
            # Build predictions network-by-network and word-by-word
            for i in range(captions.shape[1]):
                # Outer loop: network (initialize hidden states here)
                if self.rnn_type in ('rnn', 'gru', 'lstm'):
                    hn = x.unsqueeze(0).repeat(self.num_rnn_layers * self.num_rnn_directions, 1, 1)
                elif self.rnn_type == 'attention':
                    hn, cn = None, None
                    yn = y
                    features = x
                if self.rnn_type == 'lstm':
                    states = (hn, torch.zeros_like(hn))
                captions_start[:, i, :] = batch_beam_search(
                    rnn_captioner=self,
                    yns=yn,
                    hns=hn,
                    cns=cn,
                    states=states,
                    features=features,
                    max_length=self.datamodule.max_caption_length,
                    which_rnn=i,
                    alpha=1.,
                    beam_width=2,
                )
                
                for t in range(max_length):
                    # Inner loop: sequence - feed hidden states back in recurrence relation
                    if self.rnn_type in ('rnn', 'gru'):
                        output, hn = getattr(self.decoder, "rnn%d" % i)(yn, hn)
                    elif self.rnn_type == 'lstm':
                        output, states = getattr(self.decoder, "rnn%d" % i)(yn, states)
                    elif self.rnn_type == 'attention':
                        output, hn, cn = getattr(self.decoder, "rnn%d" % i)(yn, x, hn, cn)
                    scores = getattr(self.fc_scorer, "fc%d" % i)(output)
                    _, idx = scores.max(dim=2)
                    top_win, idxs = scores.topk(5, dim=2)
                    yn = self.word_embedder(idx).view(batch_size, 1, -1)
                    captions[:, i, t] = idx.view(-1)
        
        if return_attn and self.rnn_type == 'attention':
            return captions, attn_weights
        else:
            return captions, captions_start

        # Recursive beam search?
    def beam_search(self, candidates, beam_width=10, step=None):
        if step is None:
            step = beam_width
        elif step == 0:
            return get_best(candidates, 10)
        accumulates = []
        for candidate in candidates:
            pass
        # Candidates: [words so far, 0 + value of how confident the network was for each word, nn state information]
        # def beam_search(candidates, step=10):
        #   if step == 0:
        #       return best 1 of candidates
        #   accumulates = []
        #   for candidate in candidates
        #        accumulates += (get 10 candidates)
        #   of accumulates, get 10 best -> candidates 
        #   beam_seach(candidates, step-1)

    def forward_step(self, batch, batch_idx):
        """Training-time loss for the RNN.
        images: (N, 3, 224, 224)
        captions: (N, C, T)
        """
        ### Ingest inputs, shapes ###
        x, y = batch['image'], batch['captions']
        batch_size = y.shape[0]

        ### Image features to initial hidden state ###
        x = self.image_extractor.encoder(x) # (N, cnn_out, K, K)
        if self.image_extractor.pooling:
            x = self.image_extractor.pooling(x) #  (N, cnn_out, 1, 1)
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

        if self.rnn_type in ('rnn', 'lstm', 'gru'):
            x = x.unsqueeze(0).repeat(self.num_rnn_layers * self.num_rnn_directions, 1, 1)
        if self.rnn_type in ('rnn', 'gru'):
            rnn_outs = self.decoder(y_in, x)
        elif self.rnn_type == 'attention':
            rnn_outs = self.decoder(y_in, x)
        elif self.rnn_type == 'lstm':
            c0 = torch.zeros_like(x)
            rnn_outs = self.decoder(y_in, (x, c0))
        scores = self.fc_scorer(rnn_outs)

        y_out = y_out[:, :self.decoder.num_rnns, :]

        loss = multi_caption_smoothing_temporal_softmax_loss(
            scores,
            y_out,
            ignore_index=self.ignore_index
            )
        return loss

    def training_step(self, batch, batch_idx):
        loss = self.forward_step(batch, batch_idx)
        self.log('train_loss', loss)
        return loss

    def validation_step(self, batch, batch_idx):
        loss = self.forward_step(batch, batch_idx)
        if batch_idx % 100 == 0:
            images = batch['image'][:5]
            ground_truth = batch['captions'][:5]
            batch = {'image': images, 'captions': ground_truth}
            preds = self.forward(batch)
            captions = ground_truth[:, :preds.shape[1], :]
            examples = log_wandb_preds(
                self.datamodule.tokenizer,
                images,
                preds,
                captions)
            wandb.log({
                "val_examples": examples,
                "val_loss": loss.cpu(),
            })
        self.log('val_loss', loss, on_step=True)

    def test_step(self, batch, batch_idx):
        loss = self.forward_step(batch, batch_idx)
        self.log('test_loss', loss, on_step=True)

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=self.learning_rate)
        return optimizer