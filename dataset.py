from torch.utils.data import Dataset
from PIL import Image
import torchvision.datasets as visionset
from torchvision import transforms
import glob


class MNISTDataset(Dataset):

    """
     loads MNIST DATASET FROM TORCHVISION AND GENERATE LR/HR paris
     HR IMAGE SIZE : 28x28
     LR IMAGE SIZE : 7x7 (BY APPLYING BICUBIC DOWNSAMPLING FILTERS)

    """
    def __init__(self,root='./data',download=True):
	trans = transforms.Compose([transforms.ToTensor()])
	self.hr_images = visionset.MNIST(root=root,train=True,transform= trans,download=download)



    def __getitem__(self,idx):

	hr_image = self.hr_images[idx][0]

	ToPilFunction = transforms.ToPILImage()
	lr_image = ToPilFunction(hr_image)
	lr_image = lr_image.resize((7,7),Image.BICUBIC)
	lr_image = lr_image

	ToTensorFunction = transforms.ToTensor()
	lr_image = ToTensorFunction(lr_image)

	return lr_image,hr_image

    def __len__(self):
	return len(self.hr_images)


class ImageFromFolder(Dataset):

    def __init__(self,root='./train_data/291'):
	self.hr_image_set = self.PreProcessImages(root)

    def __getitem__(self,idx):
	hr_image = self.hr_image_set[idx]
	lr_image = hr_image.resize((24,24),Image.BICUBIC)
	ToTensorFn = transforms.ToTensor()
	lr_image,hr_image = ToTensorFn(lr_image),ToTensorFn(hr_image)

	return lr_image,hr_image

    def __len__(self):
	return len(self.hr_image_set)

    def PreProcessImages(self,root,crop_size=96,sample_rate=16):
	image_set_bmp = map(Image.open,glob.glob(root + '/*.bmp'))
	image_set_jpg = map(Image.open,glob.glob(root + '/*.jpg'))

	image_set = []
	image_set +=(image_set_bmp)
	image_set +=(image_set_jpg)

	new_image_set = []
	RandomCropFn = transforms.RandomCrop(crop_size)
	for image in image_set:
	    for _ in range(sample_rate):
 		sample =RandomCropFn(image)
		sample_flip = sample.transpose(0)
	        new_image_set.append(sample)
		new_image_set.append(sample_flip)
		for j in range(2,5):
		    new_image_set.append(sample.transpose(j))
		    new_image_set.append(sample_flip.transpose(j))

	return new_image_set


class DIV2K(Dataset):

    def __init__(self,root='./DIV2K_train_HR'):
	self.hr_image_set = self.PreProcessImages(root)

    def __getitem(self,idx):
	hr_image = self.hr_image_set[idx]
	lr_image = hr_image.resize((96,96),Image.BICUBIC)
	ToTensor = transforms.ToTensor()
	lr_image,hr_image = ToTensor(lr_image),ToTensor(hr_image)
	return lr_image,hr_image

    def __len__(self):
	return len(self.hr_image_set)

    def PreProcessImages(self,root,crop_size=96*4,sample_rate=1):
        image_set = map(Image.open,glob.glob(root+'/*.png'))
	new_image_set = []
	RandomCrop = transforms.RandomCrop(crop_size)
	for image in image_set:
	    new_image_set.append(RandomCrop(image))

	return new_image_set
