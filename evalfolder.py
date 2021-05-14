import torch
from srresnet import G_Net
from torchvision import transforms
from PIL import Image
from torch.autograd import Variable
import argparse
import glob
import os.path
import math
import numpy as np

parser = argparse.ArgumentParser()
parser.add_argument('--cuda',action='store_true',help='use cuda?')
parser.add_argument('--model',default="./checkpoint/model.pth",type=str,help='checkpoint path')
parser.add_argument('--source',default="Set5",type=str,help='test set directory')

opt = parser.parse_args()

def psnr_score(sr_image,hr_image):

    mse = np.mean((((sr_image*255) - (hr_image*255)).numpy().astype(np.uint8))**2)
    psnr = 20*math.log10(255) - 10*math.log10(mse)
    return psnr


print '>>>Loading model'
model = G_Net()

print '>>>Loading saved model'
checkpoint = torch.load(opt.model)
print '>>>Epoch: {}'.format(checkpoint['epoch'])
model.load_state_dict(checkpoint['model'].state_dict())
if opt.cuda:
    model = model.cuda()

ToTensor = transforms.ToTensor()
ToPILImage = transforms.ToPILImage()
psnr_list = []
if not os.path.exists('./output'):
    os.makedirs('./output')

for sample in glob.glob(opt.source +'/*'):
    hr_image = Image.open(sample)
    x,y = hr_image.size
    lr_image = hr_image.resize((x//4,y//4),Image.BICUBIC)
    x,y = lr_image.size
    hr_image = ToTensor(hr_image)
    input = ToTensor(lr_image)
    input = input.view(-1,3,y,x)
    input = Variable(input)
    if opt.cuda:
        input = input.cuda()

    output = model(input)
    output = output.cpu()

    output.data[0] = torch.clamp(output.data[0],min=0.0,max=1.0)
    psnr_list.append(psnr_score(output.data[0],hr_image))
    output = ToPILImage(output.data[0])
    output.save('./output/SR_'+sample,'png')
    print 'saved image {}'.format(sample)

print 'psnr score: {}'.format(sum(psnr_list)/len(psnr_list))

