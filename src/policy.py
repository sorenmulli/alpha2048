import torch
import torch.nn as nn
from torch.autograd import Variable

cpu = torch.device("cpu")
gpu = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

class Policy(nn.Module):

	def __init__(self, layers, lr=1e-4, dropout=.6, device=gpu):

		self.device = device
		self.params = {
			"layers": layers
		}

		# Inherets from torch's neural netowrk class
		super().__init__()

		# Saves the number of layers as well as the dropout
		self.nlayers = len(layers)
		self.lr = lr
		self.dropout = dropout
		
		# Adds all the layers to the class, so they can be accessed directly from the class itself 
		for i in range(self.nlayers-1):
			setattr(self, "linear%s" % (i+1), nn.Linear(layers[i], layers[i+1]))

		# Defines loss function and optimizer
		self.loss_fn = nn.MSELoss(reduction="sum")
		self.optimizer = torch.optim.Adam(self.parameters(), lr=self.lr)

	def forward(self, x):

		'''
		Tager input-arrayet og indfører det trænede policy-netværk på det.
		'''
		
		# Builds the model by loading the layers from the class itself
		seq = list()
		for i in range(self.nlayers-2):
			seq.append(getattr(self, "linear%s" % (i+1)))
			# Adds dropout and ReLu from non-linearity
			seq.append(nn.Dropout(p=self.dropout))
			seq.append(nn.ReLU())
		
		# Adds the last layer and softmax
		seq.append(getattr(self, "linear%s" % (i+2)))
		seq.append(nn.Softmax(dim=-1))
		
		# Adds all layers and non-linearity into a sequential torch-nn-model
		model = nn.Sequential(*seq).to(self.device)

		# Executes a feed-forward using the model
		return model(x)
	
	def trainSL(self, x, y, n):
		
		datapoints = len(x)
		x, y = x.to(self.device), y.to(self.device)
		for i in range(n):
			y_pred = self.forward(x)
			loss = self.loss_fn(y_pred, y)
			print("Loss:", i, loss.item()/datapoints)
			self.optimizer.zero_grad()
			loss.backward()
			self.optimizer.step()
	
	def trainRL(self, policyhist, rewards, n=1):

		for i in range(n):
			loss = torch.sum(torch.mul(policyhist, Variable(rewards)).mul(-1), -1)
			self.optimizer.zero_grad()
			loss.backward()
			self.optimizer.step()
			# with torch.no_grad():
			# 	for p in self.parameters():
			# 		p -= self.lr * p.grad
			# 		p.grad.zero_()

	def save_me(self, path):
		
		open(path.replace("\\", "/"), "w+", encoding="utf-8").close()
		# Saves the weights, optimizer, parameters and the current batch
		state = {
			'state_dict': self.state_dict(),
			'optimizer': self.optimizer.state_dict(),
			'params': self.params
		}
		torch.save(state, path)

