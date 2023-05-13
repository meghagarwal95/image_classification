# load Fashion MNIST dataset
import matplotlib.pyplot as plt
import numpy as np
import torch
import torchvision
import torchvision.transforms as transforms
import matplotlib.pyplot as plt

class Dataloader:
	def __init__(self, batch, height = 28, width = 28, channels = 1, path = None):
		self.nChannels = channels
		self.batch_size = batch
		self.size = (height, width)
		self.path = path
		self.load()

	# Helper function for inline image display
	def matplotlib_imshow(self, img, one_channel=False):
	    if one_channel:
	        img = img.mean(dim=0)
	    img = img / 2 + 0.5     # unnormalize
	    npimg = img.numpy()
	    if one_channel:
	        plt.imshow(npimg, cmap="Greys")
	    else:
	        plt.imshow(np.transpose(npimg, (1, 2, 0)))

	def load(self):
		# Create datasets for training & validation, download if necessary
		transform = self.transform()

		training_set = torchvision.datasets.FashionMNIST('./data', train=True, transform=transform, download=True)
		validation_set = torchvision.datasets.FashionMNIST('./data', train=False, transform=transform, download=True)

		# Create data loaders for our datasets; shuffle for training, not for validation
		self.training_loader = torch.utils.data.DataLoader(training_set, batch_size=self.batch_size, shuffle=True)
		self.validation_loader = torch.utils.data.DataLoader(validation_set, batch_size=self.batch_size, shuffle=False)

		# Class labels
		self.classes = ('T-shirt/top', 'Trouser', 'Pullover', 'Dress', 'Coat',
		        'Sandal', 'Shirt', 'Sneaker', 'Bag', 'Ankle Boot')

		# Report split sizes
		print('Training set has {} instances'.format(len(training_set)))
		print('Validation set has {} instances'.format(len(validation_set)))
		return

	
	def show_samples(self):
		'''
		Function to see one batch of samples from training dataset 
		in a grid
		'''

		dataiter = iter(self.training_loader)
		images, labels = next(dataiter)

		# Create a grid from the images and show them
		img_grid = torchvision.utils.make_grid(images)
		self.matplotlib_imshow(img_grid, one_channel=True)
		print('True labels are:')
		print('  '.join(classes[labels[j]] for j in range(self.batch_size)))


	def transform(self):
		'''
		Pre-processing of images 
		Converting to tensor and normalize all channels 
		with a mean and std of 0.5 across all channels
		'''
		transform = transforms.Compose(
		    [transforms.ToTensor(),
		    transforms.Normalize((0.5,), (0.5,))])
		return transform












