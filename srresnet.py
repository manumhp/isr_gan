import torch
import torch.nn as nn

'''
The feature space should maintain their dimensionality throughout the network

hence for a stride of 1 the zero padding is calulated as:

zero padding = (kernel_size -1)/2
'''


class Residual_Block(nn.Module):
    def __init__(self):
	super(Residual_Block,self).__init__()

	self.conv1 = nn.Conv2d(in_channels=64,out_channels=64,kernel_size=3,stride=1,padding=1,bias=False)
	self.bn1 = nn.BatchNorm2d(64)
	self.relu = nn.LeakyReLU(0.2,inplace=True)
	self.conv2 = nn.Conv2d(in_channels=64,out_channels=64,kernel_size=3,stride=1,padding=1,bias=False)
	self.bn2 = nn.BatchNorm2d(64)

    def forward(self,x):
	identity_data = x
	output = self.relu(self.bn1(self.conv1(x)))
	output = self.bn2(self.conv2(output))
	output = torch.add(output,identity_data)
	return output




class G_Net(nn.Module):
    def __init__(self):
	super(G_Net,self).__init__()

	self.conv_input = nn.Conv2d(in_channels= 3,out_channels=64,kernel_size=9,stride=1,padding=4,bias=False)
	self.relu = nn.LeakyReLU(0.2,inplace=True)

	self.residual = self.make_rlayers(Residual_Block,16)

	self.conv_middle = nn.Conv2d(in_channels=64,out_channels=64,kernel_size=3,stride=1,padding=1,bias=False)
	self.bn_middle = nn.BatchNorm2d(64)

	self.upscale4x = nn.Sequential(
	    nn.Conv2d(in_channels=64,out_channels=256,kernel_size=3,stride=1,padding=1,bias=False),
	    nn.PixelShuffle(2),
	    nn.LeakyReLU(0.2,inplace=True),
	    nn.Conv2d(in_channels=64,out_channels=256,kernel_size=3,stride=1,padding=1,bias=False),
	    nn.PixelShuffle(2),
	    nn.LeakyReLU(0.2,inplace=True)
	)

	self.conv_output = nn.Conv2d(in_channels=64,out_channels=3,kernel_size=9,stride=1,padding=4,bias=False)

	#initalize weights to reach deep into the network

    def make_rlayers(self,block,num_layers):
	layers = []
	for _ in range(num_layers):
	    layers.append(block())

	return nn.Sequential(*layers)


    def forward(self,x):
	output = self.relu(self.conv_input(x))
	identity_data = output
	output = self.residual(output)
	output = self.bn_middle(self.conv_middle(output))
	output = torch.add(output,identity_data)
	output = self.upscale4x(output)
	output = self.conv_output(output)
	return output

class D_Net(nn.Module):
    def __init__(self):
	super(D_Net,self).__init__()

	self.features = nn.Sequential(
	    # 3 x 96 x 96
	    nn.Conv2d(in_channels=3,out_channels=64,kernel_size=3,stride=1,padding=1,bias=False),
	    nn.LeakyReLU(0.2,inplace=True),

	    # 64 x 96 x 96
	    nn.Conv2d(in_channels=64,out_channels=64,kernel_size=3,stride=2,padding=1,bias=False),
	    nn.BatchNorm2d(64),
	    nn.LeakyReLU(0.2,inplace=True),

	    #64 x 96 x 96
	    nn.Conv2d(in_channels=64,out_channels=128,kernel_size=3,stride=1,padding=1,bias=False),
	    nn.BatchNorm2d(128),
	    nn.LeakyReLU(0.2,inplace=True),

	    # 128 x 48 x 48
	    nn.Conv2d(in_channels=128,out_channels=128,kernel_size=3,stride=2,padding=1,bias=False),
	    nn.BatchNorm2d(128),
	    nn.LeakyReLU(0.2,inplace=True),

	    # 128 x 48 x 48
	    nn.Conv2d(in_channels=128,out_channels=256,kernel_size=3,stride=1,padding=1,bias=False),
	    nn.BatchNorm2d(256),
	    nn.LeakyReLU(0.2,inplace=True),

	    #256 x 24 x 24
	    nn.Conv2d(in_channels=256,out_channels=256,kernel_size=3,stride=2,padding=1,bias=False),
	    nn.BatchNorm2d(256),
	    nn.LeakyReLU(0.2,inplace=True),

	    #256 x 24 x 24
	    nn.Conv2d(in_channels=256,out_channels=512,kernel_size=3,stride=1,padding=1,bias=False),
	    nn.BatchNorm2d(512),
	    nn.LeakyReLU(0.2,inplace=True),

	    #512 x 12 x 12
	    nn.Conv2d(in_channels=512,out_channels=512,kernel_size=3,stride=2,padding=1,bias=False),
	    nn.BatchNorm2d(512),
	    nn.LeakyReLU(0.2,inplace=True),

        )

	self.fc1 = nn.Linear(512 * 6 * 6,1024)
	self.fc2 = nn.Linear(1024,1)
	self.Leaky = nn.LeakyReLU(0.2,inplace=True)
	self.sigmoid = nn.Sigmoid()


    def forward(self,x):
	out = self.features(x)

	out = out.view(out.size(0),-1)

	out = self.fc1(out)

	out = self.Leaky(out)

	out = self.fc2(out)

	out =  self.sigmoid(out)
	return out.view(-1,1).squeeze(1)
