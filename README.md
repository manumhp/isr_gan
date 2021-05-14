# SRGAN -PyTorch

A PyTorch implementation of SRGAN

Implmentation of [Photo-Realistic Single Image Super-Resolution Using a Generative Adversarial Network](https://arxiv.org/abs/1609.04802) using PyTorch



## Prerequisites

- [Pytorch](https://pytorch.org/get-started/locally/)


## Training



### Training SRRESNET
SRResNet will act as the base model for training SRGAN
```
python main.py

optional arguments:
  --batchSize BATCHSIZE   training batch size
  --lr LR                 learning rate
  --resume RESUME         contains checkpoint to resume
  --startEpoch STARTEPOCH starting epoch number
  --numEpochs NUMEPOCHS   number of epoches..
  --cuda                  Use cuda??

```



### Training SRGAN

```
python srgan_main.py

optional arguments:
  --batchSize BATCHSIZE   training batch size
  --lr LR                 learning rate
  --resumeG RESUMEG       contains checkpoint to resume generator
  --resumeD RESUMED       contains checkpoint to reume discriminator
  --startEpoch STARTEPOCH starting epoch number
  --numEpochs NUMEPOCHS   number of epoches..
  --cuda                  Use cuda??
  --premodel PREMODEL     model to initalize generator
```

## Running the tests


### Evaluate Single Image

Run this to evaluate the model for a single Image:

```
python eval.py

optional arguments:
  --cuda         use cuda?
  --model MODEL  path to saved model
  --image IMAGE  path to input image
```

### Evaluate Batch of Images
Use this to run tests on standard input set and to calculate the psnr scores: 
```
python evalfolder.py

optional arguments:
  --cuda           use cuda?
  --model MODEL    checkpoint path
  --source SOURCE  test set directory

```
