import torch
from model import GarmentClassifier
from dataloader import Dataloader
from datetime import datetime

class Trainer:
	def __init__(self, batch_size = 4, n_epochs = 5, learning_rate=0.001, momentum = 0.9):
		self.learning_rate = learning_rate
		self.n_epochs = n_epochs
		self.batchSize = batch_size
		self.momentum = momentum
		self.init_load()
	
	def lossFunction(self, outputs, labels):
    	#CE Loss function
		loss_fn = torch.nn.CrossEntropyLoss()

		return loss_fn(outputs, labels)

	def init_load(self):
		self.device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
		self.model = GarmentClassifier().to(self.device)
		self.dataLoader = Dataloader(self.batchSize)
		self.load_optimizer(self.learning_rate, self.momentum)

	def load_optimizer(self, learning_rate, momentum):
		#SGD for optimizer
		# Optimizers specified in the torch.optim package
		self.optimizer = torch.optim.SGD(self.model.parameters(), lr=self.learning_rate, momentum=self.momentum)
		return

	#Training Loop

	def train_one_epoch(self, epoch_index): #, tb_writer):
	    running_loss = 0.
	    last_loss = 0.

	    # Here, we use enumerate(training_loader) instead of
	    # iter(training_loader) so that we can track the batch
	    # index and do some intra-epoch reporting
	    for i, data in enumerate(self.dataLoader.training_loader):
	        # Every data instance is an input + label pair
	        inputs, labels = data[0].to(self.device), data[1].to(self.device)
	        
	        # Zero your gradients for every batch!
	        self.optimizer.zero_grad()

	        # Make predictions for this batch
	        outputs = self.model(inputs)

	        # Compute the loss and its gradients
	        loss = self.lossFunction(outputs, labels) #loss_fn(outputs, labels)
	        loss.backward()

	        # Adjust learning weights
	        self.optimizer.step()

	        # Gather data and report
	        running_loss += loss.item()
	        if i % 1000 == 999:
	            last_loss = running_loss / 1000 # loss per batch
	            print('  batch {} loss: {}'.format(i + 1, last_loss))
	            tb_x = epoch_index * len(self.dataLoader.training_loader) + i + 1
	            #tb_writer.add_scalar('Loss/train', last_loss, tb_x)
	            running_loss = 0.

	    return last_loss


	def run(self, n_epochs):
		
		epoch_number = 0

		EPOCHS = n_epochs

		best_vloss = 1_000_000.
		timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')

		for epoch in range(EPOCHS):
		    print('EPOCH {}:'.format(epoch_number + 1))

		    # Make sure gradient tracking is on, and do a pass over the data
		    self.model.train(True)
		    avg_loss = self.train_one_epoch(epoch_number) #, writer)

		    # We don't need gradients on to do reporting
		    self.model.train(False)

		    running_vloss = 0.0
		    for i, vdata in enumerate(self.dataLoader.validation_loader):
		        vinputs, vlabels = vdata[0].to(self.device), vdata[1].to(self.device)
		        voutputs = self.model(vinputs) #.to(device)
		        vloss = self.lossFunction(voutputs, vlabels)
		        running_vloss += vloss

		    avg_vloss = running_vloss / (i + 1)
		    print('LOSS train {} valid {}'.format(avg_loss, avg_vloss))

		    # Log the running loss averaged per batch
		    # for both training and validation
		    # writer.add_scalars('Training vs. Validation Loss',
		    #                 { 'Training' : avg_loss, 'Validation' : avg_vloss },
		    #                 epoch_number + 1)
		    # writer.flush()

		    # Track best performance, and save the model's state
		    if avg_vloss < best_vloss:
		        best_vloss = avg_vloss
		        model_path = 'models/model_{}_{}.pt'.format(timestamp, epoch_number)
		        torch.save(self.model.state_dict(), model_path)

		    epoch_number += 1
		
		return

