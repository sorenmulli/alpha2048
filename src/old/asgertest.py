import torch
import matplotlib.pyplot as plt
import numpy as np

from math import erf

from game import Game2048

from time import sleep
device = torch.device("cpu")

# np.seterr(all="raise")

def sigmoid(z):

	return 1 / (1 + np.exp(-z))

class Network:

	def __init__(self, layers, eta=1e-4):

		self.layers = layers
		self.nlayers = len(layers)
		self.eta = eta

		self.weights = [torch.randn(self.layers[i+1], self.layers[i], requires_grad=True) for i in range(self.nlayers-1)]

		self.scoremeans = None
		self.maxtiles = None
	
	def train(self, nbatches, batchsize):

		self.scoremeans = np.empty(nbatches)
		self.maxtiles = np.empty(nbatches)

		for i in range(nbatches):
			self.weights, s, m = self.playbatch(i, batchsize)
			for w in self.weights:
				w.requires_grad = True
			self.scoremeans[i] = s
			self.maxtiles[i] = m
		
		return self.scoremeans, self.maxtiles
	
	def feedforward(self, x, weights):

		"""
		Kører en input-tensor x gennem netværket
		"""

		for i in range(self.nlayers-1):
			x = self.weights[i].mv(x).clamp(0)
		
		return x
	
	def playbatch(self, idx, ngames=100):

		newweights = [w.clone().detach() for w in self.weights]
		scores = []
		maxtiles = []

		for i in range(ngames):

			directions = []
			logps = []
			game = Game2048()
			# Spiller et spil
			for k in range(100):
				logp = self.feedforward(torch.Tensor(game.board.reshape(game.n**2)), newweights)
				while True:
					with torch.no_grad():
						direction = self.sampler(logp)
						change = game.play(direction)
						if change in (1, 2):
							break
				
				directions.append(direction)
				logps.append(logp)
				if change == 2:
					break
			
			reward = self.reward(game.score)

			for k in range(len(directions)):
				y = torch.zeros(game.n)
				y[directions[k]] = reward
				logps[k].backward(y)
			
			with torch.no_grad():
				for i in range(self.nlayers-1):
					newweights[i] += self.eta * reward * self.weights[i].grad / self.weights[i].grad.norm()
					self.weights[i].zero_()
			
			scores.append(game.score)
			maxtiles.append(game.board.max())
		
		return newweights, np.mean(scores), np.mean(maxtiles)

	@staticmethod
	def sampler(values):

		"""
		Udvælger indexværdi af values vægtet efter størrelse
		"""

		# values = -np.array(values.clone().detach())

		# Omregner til sandsynlighed
		values = 1 / (1 + torch.exp(-values))

		# Normaliserer
		values /= values.sum()

		return np.random.choice(np.arange(4), p=np.array(values.clone().detach()))



	
	@staticmethod
	def reward(score):
		
		return float(score)



def run(layers=[16, 16, 16, 4], nbatches=100, batchsize=10):

	N = Network(layers, 1e-0)
	s, m = N.train(nbatches, batchsize)

	plt.plot(s[1:])
	plt.show()




if __name__ == "__main__":

	run()


