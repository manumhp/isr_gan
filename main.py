import matplotlib
matplotlib.use('Agg')
import torch
from dataset import ImageFromFolder
import torch.backends.cudnn as cudnn
from srresnet import G_Net
import torch.nn as nn
from torch.autograd import Variable
from torch.utils.data import DataLoader
import torch.optim as optim
import argparse
import os.path
import matplotlib.pyplot as plt
#hyper parameters
parser = argparse.ArgumentParser(description="PyTorch implementation of SRGAN")
parser.add_argument('--batchSize',type=int,default=16,help='training batch size')
parser.add_argument('--lr',type=float,default=1e-4,help='learning rate')
parser.add_argument('--resume',default="",type=str,help='contains checkpoint to resume')
parser.add_argument('--startEpoch',default=1,type=int,help='starting epoch number')
parser.add_argument('--numEpochs',default=100,type=int,help="number of epoches..")
parser.add_argument('--cuda',action="store_true",help="Use cuda??")
def main():

    global opt,model

    opt = parser.parse_args()

    print (opt)

    loss_list = []

    if opt.cuda and not torch.cuda.is_available():
	raise Exception("NO GPU FOUND")


    cudnn.benchmark = True


    print '>>>Loading datasets'

    train_set = MNISTDataset()

    #train_set = ImageFromFolder()

    #train_set = DIV2K()

    print 'DATASET SIZE: {}'.format(len(train_set))

    training_data_loader = DataLoader(dataset=train_set,batch_size=opt.batchSize,shuffle=True,num_workers=0)

    print '>>>Building Model'
    model = G_Net()
    criterion = nn.MSELoss(size_average=False)


    if opt.cuda:
        print '>>Setting GPU'
        model = model.cuda()
        criterion = criterion.cuda()

    if opt.resume:
	if os.path.isfile(opt.resume):
	    print ">>>Loading Checkpoint '{}'".format(opt.resume)
	    checkpoint = torch.load(opt.resume)
	    opt.startEpoch = checkpoint['epoch'] + 1
	    model.load_state_dict(checkpoint['model'].state_dict())

	else:
	    print '>>>No model found at : ' + opt.resume


    print '>>Setting Optimizer'
    optimizer = optim.Adam(model.parameters(),opt.lr)

    for epoch in range(opt.startEpoch,opt.numEpochs+1):
	current_loss = train(training_data_loader,optimizer,model,criterion,epoch)
	save_checkpoint(model,epoch)
	loss_list.append(current_loss)
	save_plot(loss_list)


def adjust_learning_rate(optimizer,epoch):

    lr = opt.lr * ( 0.1 ** (epoch // 200))
    return lr

def train(training_data_loader,optimizer,model,criterion,epoch):

    lr = adjust_learning_rate(optimizer,epoch-1)

    for param_group in optimizer.param_groups:
	param_group["lr"] = lr

    print "epoch=", epoch,"lr=",optimizer.param_groups[0]["lr"]
    model.train()

    for iteration,batch in enumerate(training_data_loader,1):

        input,target = Variable(batch[0]),Variable(batch[1],requires_grad=False)

        if opt.cuda:
            input  = input.cuda()
	    target = target.cuda()

	output = model(input)
	loss = criterion(output,target)

	optimizer.zero_grad()

	loss.backward()

	optimizer.step()

	if iteration%10 == 0:
	    print '===>Epoch[{}]({}/{}): Loss: {:.5}'.format(epoch,iteration,len(training_data_loader),loss.data[0])
    return loss.data[0]

def save_checkpoint(model,epoch):
    model_out_path = '/output/' + 'model_epoch_{}.pth'.format(epoch)
    #model_out_path = '/output/' + 'model.pth'
    state = {'epoch':epoch,'model':model}
    if not os.path.exists('/output/'):
        os.makedirs('/output/')

    torch.save(state,model_out_path)

    print 'Checkpoint saved to: {}'.format(model_out_path)

def save_plot(loss_list):
    plt.ylabel('loss')
    plt.xlabel('epochs')
    plt.plot(loss_list)
    plt.savefig('/output/plot.png')


if __name__ == '__main__':
    main()
