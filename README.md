<div align="center">

### Image Captioning in Pytorch

![CI testing](https://github.com/reppertj/image-captioning/workflows/CI%20testing/badge.svg)

<style>@font-face {  font-family: 'Open Sans';  font-style: normal;  font-weight: 400;  src: local('Open Sans'), local('OpenSans'), url('data:font/woff2;base64,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') format('woff2');}@font-face {  font-family: 'Open Sans';  font-style: normal;  font-weight: 700;  src: local('Open Sans Bold'), local('OpenSans-Bold'), url('data:font/woff2;base64,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') format('woff2');}.rc-pill {  font-family: 'Open Sans', sans-serif;  border-radius: 4px;  display: inline-block;  position: relative;  overflow: hidden;  white-space: nowrap;  font-size: 11px;  cursor: pointer;}.rc-pill strong {  font-weight: bold;}.rc-pill a {  color: white;  text-shadow: rgba(1, 1, 1, 0.3) 0.75px 0.75px 0px;}.rc-pill div {  display: inline-block;  padding: 2px 5px 2px 5px;}.rc-pill .l {  background: #555;  background: -webkit-linear-gradient(#555, #484848);  background: linear-gradient(#555, #484848);}.rc-pill .r {  background: #61ae24;  background: -webkit-linear-gradient(#61ae24, #559920);  background: linear-gradient(#61ae24, #559920);}.rc-pill i.icon-svg svg {  height: 1em;  margin-right: 0.33em;  margin-bottom: -1px;  display: inline-block;}</style><div class="rc-pill"><a href="http://www.recurse.com" title="Made with love at the Recurse Center"><div class="l"><span>made at </span></div><div class="r"><i class="icon-svg"><svg viewBox="0 0 12 15"><rect x="0" y="0" width="12" height="10" fill="black"></rect><rect x="1" y="1" width="10" height="8" fill="white"></rect><rect x="2" y="2" width="8" height="6" fill="black"></rect><rect x="2" y="3" width="1" height="1" fill="#61ae24"></rect><rect x="4" y="3" width="1" height="1" fill="#61ae24"></rect><rect x="6" y="3" width="1" height="1" fill="#61ae24"></rect><rect x="3" y="5" width="2" height="1" fill="#61ae24"></rect><rect x="6" y="5" width="2" height="1" fill="#61ae24"></rect><rect x="4" y="9" width="4" height="3" fill="black"></rect><rect x="1" y="11" width="10" height="4" fill="black"></rect><rect x="0" y="12" width="12" height="3" fill="black"></rect><rect x="2" y="13" width="1" height="1" fill="white"></rect><rect x="3" y="12" width="1" height="1" fill="white"></rect><rect x="4" y="13" width="1" height="1" fill="white"></rect><rect x="5" y="12" width="1" height="1" fill="white"></rect><rect x="6" y="13" width="1" height="1" fill="white"></rect><rect x="7" y="12" width="1" height="1" fill="white"></rect><rect x="8" y="13" width="1" height="1" fill="white"></rect><rect x="9" y="12" width="1" height="1" fill="white"></rect></svg></i><span>Recurse Center</span></div></a></div>

</div>

## Description

This project is a framework for experimenting with decoder-encoder image captioning models using Pytorch and Pytorch Lightning. Although it is a toy implementation for educational purposes, it includes several elements not found in some other image captioning explainers, including customizable preprocessing and augmentation, label smoothing, and beam search.
## Notebooks

To demo some predictions and visualize the attention weights, check out this notebook on Google Colab.

- [Inference and attention]() [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/reppertj/image-captioning/blob/master/notebooks/inference-attention.ipynb)

The following notebooks illustrate how to train the model using Colab and resume training from a checkpoint.

- Training:
- Resuming from a checkpoint:

## Setup

First, install dependencies

```bash
# clone project   
git clone https://github.com/reppertj/image-captioning

# install project   
cd image-captioning
pip install -e .   
pip install -r requirements.txt

# project folder
cd project

# run training
python captioner.py    
```

## Training

Train on the COCO and or Flickr30k datasets:

- Flickr30k from [kaggle](https://www.kaggle.com/hsankesara/flickr-image-dataset)
- COCO [images](http://images.cocodataset.org/zips/train2014.zip) and [annotations](http://images.cocodataset.org/annotations/annotations_trainval2014.zip)

```python
from project.datasets import CombinedDataModule
from project.captioners import CaptioningRNN
from pytorch_lightning import Trainer

# data (train on coco, flickr30k, or both!)
dataset = CombinedDataModule(
    coco_json="coco_labels/annotations/captions_train2014.json",
    coco_dir="coco2014_train/train2014",
    flickr_csv="flickr30k_images/results.csv",
    flickr_dir="flickr30k_images/flickr30k_images",
)

# preprocess data
dataset.setup()

# model
model = CaptioningRNN(dataset)

# train
trainer = Trainer()
trainer.fit(model)

# test using the best model!
trainer.test()
```

## Configuration

You can configure the network and its training parameters via a configuration dictionary:

```python
config = {
    "max_length": 25,  # int - max caption length
    "batch_size": 64,  # int
    "wordvec_dim": 768,  # int - size of word embedding
    "hidden_size": 576,  # int - for attention, ensure hidden_size % num_rnn_layers == 0
    "wd_embedder_init": "xavier",  # or "kaiming"
    "image_encoder": "resnext50",  # or "resnet50", "resnet101", "resnet152", "mobilenetv2", "vgg16"
    "encoder_init": "xavier",  # or "kaiming"
    "rnn_type": "attention",  # or "rnn", "lstm", "gru"
    "num_rnns": 1,  # int - train up to 5 captioners in parallel
    "num_rnn_layers": 3,  # int - for attention, ensure hidden_size % num_rnn_layers == 0
    "rnn_nonlinearity": None,  # or "relu"
    "rnn_init": None,  # or "xavier", "kaiming"
    "rnn_dropout": 0.1,  # float or False
    "rnn_bidirectional": False,
    "fc_init": "xavier",  # or "kaiming", None
    "label_smoothing_epsilon": 0.05,  # float - 0. to turn off label smoothing
    "inference_beam_width": 15,  # int - 1 to turn off beam search
    "inference_beam_alpha": 1.0,  # float - higher numbers favor shorter captions
    "learning_rate": 3e-4,  # float; also configurable in trainer
    "optimizer": "adam",  # or "sgd"; also configurable in trainer
    "scheduler": "plateau",  # or None; also configurable in trainer
    "momentum": 0.9,  # also configurable in trainer
}

# Or, start with default config and change it as desired:
config = CaptioningRNN.default_config()
config["hidden_size"] = 256
config["num_rnn_layers"] = 2
config["inference_beam_alpha"] = 0.9
config["image_encoder"] = "mobilenetv2"

# Dataset options are in CombinedDataModule
dataset = CombinedDataModule(
    coco_json="coco_labels/annotations/captions_train2014.json",
    coco_dir="coco2014_train/train2014",
    transform="normalize",  # Or pass in a function
    target_transform="tokenize",
    vocab_size=1000,
)
dataset.setup()

model = CaptioningRNN(dataset, config)

trainer = Trainer()
trainer.fit(model)
```
