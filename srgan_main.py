import matplotlib
matplotlib.use('Agg')
import torch
from dataset import ImageFromFolder
import torch.backends.cudnn as cudnn
from srresnet import *
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
parser.add_argument('--resumeG',default="",type=str,help='contains checkpoint to resume generator')
parser.add_argument('--resumeD',default='',type=str,help="contains checkpoint to reume discriminator")
parser.add_argument('--startEpoch',default=1,type=int,help='starting epoch number')
parser.add_argument('--numEpochs',default=2000,type=int,help="number of epoches..")
parser.add_argument('--cuda',action="store_true",help="Use cuda??")
parser.add_argument('--premodel',default ='./checkpoint/model.pth',type=str,help='model to initalize generator')

def main():

    global opt,model

    opt = parser.parse_args()

    print (opt)

    loss_list = []

    if opt.cuda and not torch.cuda.is_available():
	raise Exception("NO GPU FOUND")


    cudnn.benchmark = True


    print '>>>Loading datasets'

    #train_set = MNISTDataset()

    train_set = ImageFromFolder()

    #train_set = DIV2K()

    training_data_loader = DataLoader(dataset=train_set,batch_size=opt.batchSize,shuffle=True)

    print '>>>Building Model'
    model_G = G_Net()
    model_D = D_Net()
    gan_criterion = nn.BCELoss()
    criterion = nn.MSELoss(size_average=False)


    if opt.cuda:
        print '>>Setting GPU'
        model_G = model_G.cuda()
        model_D = model_D.cuda()
        gan_criterion = gan_criterion.cuda()
        criterion = criterion.cuda()

    print '>>>Loading Init Weights'
    if os.path.isfile(opt.premodel):
        print ">>>Loading Checkpoint '{}'".format(opt.premodel)
        checkpoint = torch.load(opt.premodel)
	print '>>EPOCH PRETRAINED: {}'.format(checkpoint['epoch'])
	model_G.load_state_dict(checkpoint['model'].state_dict())

    if opt.resumeG:
	    if os.path.isfile(opt.resumeG):
	        print ">>>Loading Checkpoint '{}'".format(opt.resumeG)
	        checkpoint = torch.load(opt.resumeG)
	        opt.startEpoch = checkpoint['epoch'] + 1
	        model_G.load_state_dict(checkpoint['model'].state_dict())

	    else:
	        print '>>>No model found at : ' + opt.resume

    if opt.resumeD:
	    if os.path.isfile(opt.resumeD):
        	print '>>>Loading Checkpoint "{}"'.format(opt.resumeD)
		checkpoint = torch.load(opt.resumeD)
		model_D.load_state_dict(checkpoint['model'].state_dict())


    print '>>Setting Optimizer'
    optimizer_G = optim.Adam(model_G.parameters(),opt.lr)
    optimizer_D = optim.Adam(model_D.parameters(),opt.lr)

    for epoch in range(opt.startEpoch,opt.numEpochs+1):
	    current_loss = train(training_data_loader,optimizer_D,optimizer_G,model_G,model_D,criterion,gan_criterion,epoch)
	    save_checkpoint(model_D,epoch,prefix='D')
	    save_checkpoint(model_G,epoch,prefix='G')
	    loss_list.append(current_loss)
	    save_plot(loss_list)


def adjust_learning_rate(epoch):

    lr = opt.lr * ( 0.1 ** (epoch // 1000))
    return lr

def train(training_data_loader,optimizer_D,optimizer_G,model_G,model_D,criterion,gan_criterion,epoch):

    lr = adjust_learning_rate(epoch-1)

    for param_group in optimizer_G.param_groups:
	    param_group["lr"] = lr
    for param_group in optimizer_D.param_groups:
      param_group["lr"] = lr

    print "epoch=", epoch,"lrG=",optimizer_G.param_groups[0]["lr"] ,"lrD ",optimizer_D.param_groups[0]["lr"]
    model_G.train()
    model_D.train()

    real_label = Variable(torch.ones(opt.batchSize))
    fake_label = Variable(torch.zeros(opt.batchSize))
    if opt.cuda:
      real_label = real_label.cuda()
      fake_label = fake_label.cuda()

    for iteration,batch in enumerate(training_data_loader,1):

        input,target = Variable(batch[0]),Variable(batch[1],requires_grad=False)

        if opt.cuda:
            input  = input.cuda()
	    target = target.cuda()

        optimizer_D.zero_grad()
        #train discriminator using real data
        target_dis_label = model_D(target)
        loss_D_real = gan_criterion(target_dis_label,real_label)
        loss_D_real.backward()
        #train discriminator using fake data
        target_dis_label = model_D(model_G(input).detach())
        loss_D_fake = gan_criterion(target_dis_label,fake_label)
        loss_D_fake.backward()

        optimizer_D.step()
        #train generator
        optimizer_G.zero_grad()

        fake_target = model_G(input)
        fake_eval = model_D(fake_target)
        loss_G = gan_criterion(fake_eval,real_label)
        loss_mse = criterion(fake_target,target)
        total_loss = loss_mse + 1e-3 * loss_G
        total_loss.backward()

        optimizer_G.step()

	if iteration%10 == 0:
	    print '===>Epoch[{}]({}/{}): Loss_G: {:.5} Loss_D: {:.5}'.format(epoch,iteration,len(training_data_loader),loss_D_real.data[0] + loss_D_fake.data[0],total_loss.data[0])
    return total_loss.data[0]

def save_checkpoint(model,epoch,postfix=''):
    #model_out_path = '/output/' + 'model_epoch_{}.pth'.format(epoch)
    model_out_path = './checkpoint/' + 'model_'+postfix+'.pth'
    state = {'epoch':epoch,'model':model}
    if not os.path.exists('checkpoint/'):
        os.makedirs('checkpoint/')

    torch.save(state,model_out_path)

    print 'Checkpoint saved to: {}'.format(model_out_path)

def save_plot(loss_list):
    plt.ylabel('loss')
    plt.xlabel('epochs')
    plt.plot(loss_list)
    plt.savefig('/output/plot.png')


if __name__ == '__main__':
    main()

