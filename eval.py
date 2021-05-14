import torch
from srresnet import G_Net
from torchvision import transforms
from PIL import Image
from torch.autograd import Variable
import argparse
import numpy as np
import matplotlib.pyplot as plt

parser = argparse.ArgumentParser()
parser.add_argument('--cuda',action='store_true',help='use cuda?')
parser.add_argument('--model',default="./checkpoint/model.pth",type=str,help='checkpoint path')
parser.add_argument('--image',default='baby_LR.png',type=str,help='read input image')
opt = parser.parse_args()

print '>>>Loading model'
model = G_Net()

print '>>>Loading saved model'
checkpoint = torch.load(opt.model)
print '>>>Epoch: {}'.format(checkpoint['epoch'])
model.load_state_dict(checkpoint['model'].state_dict())
if opt.cuda:
    model = model.cuda()

ToTensor = transforms.ToTensor()
input = Image.open(opt.image)
x,y = input.size
input = ToTensor(input)
input = input.view(-1,3,y,x)
input = Variable(input)
if opt.cuda:
    input = input.cuda()

output = model(input)
output = output.cpu()

#im_h = output.data[0].numpy().astype(np.float32)
#im_h = im_h * 255
#im_h[im_h<0] = 0
#im_h[im_h>255] = 255.
#im_h = im_h.transpose(1,2,0)
#print im_h.shape
#print im_h
#ax = plt.imshow(im_h.astype(np.uint8))
#plt.show()
output.data[0] = torch.clamp(output.data[0],min=0.0,max=1.0)
ToPILImage = transforms.ToPILImage()
output = ToPILImage(output.data[0])

output.save('sample.png','png')


