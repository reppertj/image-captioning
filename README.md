<div align="center">

### Image Captioning in Pytorch

![CI testing](https://github.com/PyTorchLightning/deep-learning-project-template/workflows/CI%20testing/badge.svg?branch=master&event=push)

</div>

## Description

This is a *work in progress*! Expect things to change drastically.

This project is a framework for experimenting with decoder-encoder image captioning models using Pytorch and Pytorch Lightning. Although it is a toy implementation for educational purposes, it includes several elements not found in some other image captioning explainers, including customizable preprocessing and augmentation, label smoothing, and beam search. It also includes a notebook with two different methods for introspecting into the performance of the model, first through capturing the attention weights used in the multihead attention mechanism, and second by approximating the Shapley values by permuting feature inputs through the encoder. The second method is model agnostic, so it can be used with models that do not feature an attention mechanism.

## How to run

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

## Imports

This project is setup as a package, which means you can easily import any file into any other file like so:

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

# model
model = CaptioningRNN(dataset)

# train
trainer = Trainer()
trainer.fit(model)

# test using the best model!
trainer.test()
```

## Notebooks

In progress.