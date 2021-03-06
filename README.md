<div align="center">

### Image Captioning in Pytorch

![CI testing](https://github.com/reppertj/image-captioning/workflows/CI%20testing/badge.svg)

<a href='http://www.recurse.com' title='Made with love at the Recurse Center'><img src='https://cloud.githubusercontent.com/assets/2883345/11325206/336ea5f4-9150-11e5-9e90-d86ad31993d8.png' height='20px'/></a>
</div>

## Description

This project is a framework for experimenting with decoder-encoder image captioning models using Pytorch and Pytorch Lightning. Although it is a toy implementation for educational purposes, it includes several elements not found in some other image captioning explainers, including customizable preprocessing and augmentation, label smoothing, and configurable beam search.

![Visualize attention weights for other pictures with the notebook below](weights-visuals.png)
*A picture of the author*
## Notebooks

For a demo, check out the first notebook.

- [Inference and attention visualization using a pretrained model](https://colab.research.google.com/github/reppertj/image-captioning/blob/master/notebooks/visualizing_attention.ipynb) [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/reppertj/image-captioning/blob/master/notebooks/visualizing_attention.ipynb)
- [Training](https://colab.research.google.com/github/reppertj/image-captioning/blob/master/notebooks/image_captioning_begin_training.ipynb) [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/reppertj/image-captioning/blob/master/notebooks/image_captioning_begin_training.ipynb)
- [Resuming training from a checkpoint](https://colab.research.google.com/github/reppertj/image-captioning/blob/master/notebooks/image_captioning_resume_training.ipynb) [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/reppertj/image-captioning/blob/master/notebooks/image_captioning_resume_training.ipynb)

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

Configure the network and its training parameters via a configuration dictionary:

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
