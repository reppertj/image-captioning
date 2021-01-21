from collections import OrderedDict

from torch import nn
from project.datasets import tokens_to_ids, vocab_size
import torchvision

BOS = '[CLS]'
EOS = '[SEP]'
UNK = '[UNK]'
PAD = '[PAD]'

models_ = {
    'resnet50': {
        'model': torchvision.models.resnet50,
        'features_out': 2048,
        'pooling': True,
        'remove_last': 2,
    },
    'resnet101': {
        'model': torchvision.models.resnet101,
        'features_out': 2048,
        'pooling': True,
        'remove_last': 2,
    },
    'resnet152': {
        'model': torchvision.models.resnet152,
        'features_out': 2048,
        'pooling': True,
        'remove_last': 2,
    },
    'mobilenetv2': {
        'model': torchvision.models.mobilenet_v2,
        'features_out': 1280,
        'pooling': True,
        'remove_last': 1,
    },  
    'vgg16': {
        'model': torchvision.models.vgg16,
        'features_out': 512,
        'pooling': True,
        'remove_last': 2,
    },
}

class ImageFeatureExtractor(nn.Module):
    def __init__(
        self,
        encoder='mobilenetv2',
        freeze_weights=True,
        remove_last='infer',
        pooling='infer',
        convolution_in=False,
        projection_in='infer',
        projection_out=128,
        ):
        """
        Instantiate a network to extract features from images. The class
        encapsulates up to four models:
            self.encoder: The encoder CNN (N is the batch size)
                (N, 3, H_in, W_in) -> (N, projection_in, H, W)
            self.pooling: An adaptive average pooling layer.
                (N, projection_in, H, W) -> (N, projection_in, 1, 1)
            self.convolution: A 1x1 kernel convolution layer
                (N, convolution_in, H, W) -> (N, projection_out, H, W)
            self.projector: A linear layer.
                (N, projection_in) -> (N, projection_out)
        
        You *cannot* call this object directly. Instead, call the submodels
        as appropriate. You will need to reshape as appropriate before the
        linear layer.

        You would typically use either pooling + projector to get the pooled
        CNN features (for a vanilla RNN or LSTM) or convolution to get the
        CNN activation map (for an attention LSTM).

        Args:
            encoder: name of pretrained encoder to download or instance of
                nn.Module.
            freeze_weights: turn off autograd on the encoder weights/put the
                encoder in evaluation mode
            remove_last: remove the final n layers from the encoder. If 'infer',
                decide based on the encoder architecture (only for named
                encoders). If False, do not remove layers.
            pooling: instantiate an average pooling layer. If 'infer',
                decide based on the encoder architecture (only for named
                encoders).
            convolution_in: Number of input channels for the convolution layer.
                If False, do not instantiate a convolution layer. If 'infer',
                 decide the number of input channels based on the encoder
                 architecture (only for named encoders).
            projection_in: 
                Number of input features for the linear layer and/or input
                channels for the convolution layer. If False, do not instantiate
                a linear layer. If 'infer', decide the number of input features
                based on the encoder architecture (only for named encoders).
            projection_out:
                Number of output features for the linear layer and/or output
                channels for the 1x1 convolution layer.
        """
        super().__init__()
        if isinstance(encoder, str):
            try:
                self.encoder = models_[encoder]['model'](pretrained=True)
            except ValueError:
                raise ValueError(f'Encoder {encoder} not supported')
        else:
            self.encoder = encoder
        if freeze_weights:        
            for param in self.encoder.parameters():
                param.requires_grad_(False)
            if hasattr(self.encoder, 'eval'):
                self.encoder.eval()
        if remove_last == 'infer':
            remove_last = models_[encoder]['remove_last']
        if remove_last:
            self.encoder = nn.Sequential(
                *list(self.encoder.children())[:(-1 * remove_last)]
            )
        if pooling == 'infer':
            pooling = models_[encoder]['pooling']
        if pooling:
            self.pooling = nn.AdaptiveAvgPool2d(1)
        else:
            self.pooling = None
        if projection_in == 'infer':
            projection_in = models_[encoder]['features_out']
        if projection_in:
            self.projector = nn.Sequential(OrderedDict([
                ('linear', nn.Linear(projection_in, projection_out)),
                 ('relu', nn.ReLU())
            ]))
            
        else:
            projection_in = None
            self.projector = None
        if convolution_in == 'infer':
            convolution_in = models_[encoder]['features_out']
            self.convolution = nn.Conv2d(convolution_in, projection_out, 1)
        else:
            self.convolution = None
        self.projection_in = projection_in
        self.projection_out = projection_out

    def init_weights(self, method='kaiming'):
        """ Initialize weights/biases in convolution and projector layers """
        if method not in {'kaiming', 'xavier'}:
            raise ValueError(f'Initialization method {method} not supported')
        
        if method == 'kaiming':
            if self.convolution:
                nn.init.kaiming_normal_(self.convolution.weight)
            if self.projector:
                nn.init.kaiming_normal_(self.projector.linear.weight)
        elif method == 'xavier':
            if self.convolution:
                nn.init.xavier_normal_(self.convolution.weight)
            elif self.projector:
                nn.init.xavier_normal_(self.projector.linear.weight)
        
        if self.convolution:
            nn.init.zeros_(self.convolution.bias)
        if self.projector:
            nn.init.zeros_(self.projector.linear.bias)

    def forward(self, image):
        raise NotImplementedError("Don't the extractor directly; use the "
        "submodels instead.")

class WordEmbedder(nn.Module):
    """
    
    """
    def __init__(self, wordvec_dim, tokenizer):
        super().__init__()
        self.wordvec_dim = wordvec_dim
        self.vocab_size = vocab_size(tokenizer)
        self._pad = tokens_to_ids(tokenizer, [PAD])[PAD]
        self.embedder = nn.Embedding(
            num_embeddings = self.vocab_size,
            embedding_dim = self.wordvec_dim,
            padding_idx = self._pad
        )
    
    def init_weights(self, method='kaiming'):
        if method not in {'kaiming', 'xavier'}:
            raise ValueError(f'Initialization method {method} not supported')
        
        if method == 'kaiming':
            nn.init.kaiming_normal_(self.embedder.weight)
        elif method == 'xavier':
            nn.init.xavier_normal_(self.embedder.weight)

    def forward(self, x):
        return self.embedder(x)
