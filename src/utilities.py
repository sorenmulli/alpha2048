import matplotlib.pyplot as plt
import numpy as np
import reward_functions as rf 

from policy import Policy

import torch
from torch.autograd import Variable

from os import chdir
from os.path import dirname, realpath

cpu = torch.device("cpu")
tiny_number = np.finfo(np.float32).eps.item()


def max_tile_distribution(maxtiles):
	# Returns a string containing the distribution of maxtiles
	# Finds unique maxtiles and calculates their percentages
	tiles, counts = np.unique(maxtiles, return_counts=True)
	percent_dist = 100 * counts / len(maxtiles)

	# Creates a string
	dist_string = ", ".join(["{0}: {1}%".format(int(tiles[i]), int(percent_dist[i])) for i in range(len(tiles))])

	return dist_string


def create_one_hot_repr(board):
	#Flattens the board to a vector
	board = np.ravel(board)
	
	hot_one = np.zeros((16, 16), dtype = np.uint8)
	
	for i, row in enumerate(hot_one):
		row[board == i] = 1
	
	return np.ravel(hot_one)


def rollingavg(values, n, startbatch=0):

	"""
	Returns a rolling average based on n elements to each side, sa well as corresponding x coordinates
	"""

	if n == 0:
		return values

	weightvec = np.concatenate((np.arange(1, n+1), np.arange(n-1, 0, -1)))
	values = np.concatenate((np.zeros(n), values, np.zeros(n)))
	weightedvalues = values.copy()
	
	for i in range(n, values.size-n):
		weightedvalues[i] = np.sum(values[i-n+1:i+n] * weightvec) / n**2
	
	x = np.arange(n, len(values)-3*n) + startbatch
	
	return x, weightedvalues[2*n:-2*n]


def evalplot(scores, maxtiles, agent, plotpath, with_show=True):

	n = len(scores)

	# Plots scores
	plt.subplot(211)
	plt.title("Agent: %s" % agent.displayname)
	_, bwidths, _ = plt.hist(scores, int(np.sqrt(n)))
	plt.ylabel("Antal spil (hver bin er %i bred)" % (bwidths[1]-bwidths[0]))
	plt.xlabel("Score")

	# Plots max tiles
	plt.subplot(212)
	plt.hist(np.log2(maxtiles), bins=np.arange(3, 12)+.5)
	plt.ylabel("Antal spil")
	plt.xlabel("log2(Maxtile)")

	plt.savefig(plotpath)
	if with_show:
		plt.show()
	else:
		plt.close()


def trainplot(network, rundata, n, savepath, with_show=True):

	startbatch = network.params["startbatch"]
	nbatches = network.params["nbatches"]
	xcoords = np.arange(nbatches) + startbatch
	xlims = {
		"left": startbatch - .05 * (nbatches - startbatch) - 0.5,
		"right": nbatches + .05 * (nbatches - startbatch) - 0.5
	}

	# Plot of runtime
	plt.subplot(311)
	# Title
	if type(network.params["rewarder"]) == rf.RewarderCombination:
		rewardlist=[r.__class__.__name__ for r in network.params["rewarder"].rewarders]
		plt.title("Rewarder = %s, %s" % (rewardlist, network.params["rewarder"].weights))
	else:
		plt.title("Rewarder = %s" % network.params["rewarder"].__class__.__name__)
	plt.xlim(**xlims)
	plt.ylabel("Køretid pr. batch [s]")
	plt.plot(xcoords, rundata["runtimes"], "o", label="Samlet køretid")
	plt.plot(*rollingavg(rundata["runtimes"], n, startbatch), label="Køretid, rg: n=%i" % n)
	plt.plot(rundata["playtimes"], "o", label="Spilletider")
	plt.plot(rundata["backproptimes"], "o", label="Backproptider")
	plt.legend(loc=2)
	plt.grid(True)
	
	# Plot of scores
	ax1 = plt.subplot(312)
	plt.xlim(**xlims)
	plt.ylabel("Gennemsnitsscore pr. batch")
	plt.plot(xcoords, rundata["scoremeans"], "o", label="Scores")
	plt.plot(*rollingavg(rundata["scoremeans"], n, startbatch), label="Score, rg: n=%i" % n)
	plt.legend(loc=2)
	plt.grid(True)
	ax2 = ax1.twinx()
	ax2.set_ylabel("Gennemsnitligt antal træk pr. spil")
	plt.plot(xcoords, rundata["nmovemeans"], "go", label="Gnsntl. træk pr. spil")
	plt.plot(*rollingavg(rundata["nmovemeans"], n, startbatch), "r-", label="Træk, rg: n=%i" % n)
	plt.legend(loc="center left")

	# Plot of rewards
	plt.subplot(313)
	plt.xlim(**xlims)
	plt.xlabel("Batchnummer")
	plt.ylabel("Gnstl. reward pr. batch")
	plt.plot(xcoords, rundata["rewardmeans"], "o", label="Gnstl. reward for batch")
	plt.plot(*rollingavg(rundata["rewardmeans"], n, startbatch), "-", label="Reward, rg: n=%i" % n)
	plt.legend(loc=2)
	plt.grid(True)
	
	plt.savefig(savepath)
	if with_show:
		plt.show()
	else:
		plt.close()


class Agent:
	def __init__(self, model=None):
		self.displayname = None
		self.loaded = None
		if type(model) is str:
			self.from_saved_model(model)
		elif isinstance(model, torch.nn.Module):
			self.from_policy(model)
		elif callable(model):
			self.from_algorithm(model)
	
	def from_policy(self, model, hot_one=True):
		
		self.policy = model
		self.policy.device = cpu
		self.displayname = "Prætrænet på %s" % model
		self.hot_one = hot_one
		self.loaded = "policy"
	
	def from_saved_model(self, modelpath, hot_one=True):

		state = torch.load(realpath(modelpath))
		self.policy = Policy(state["params"]["layers"], device=cpu)
		self.policy.load_state_dict(state["state_dict"])

		self.hot_one = hot_one
		self.displayname = state["params"]["rewarder"].__class__.__name__ if "rewarder" in state["params"].keys() else "Saved model"
		self.loaded = "policy"
		
	def from_algorithm(self, algorithm):

		self.count = 0
		self.algorithm = algorithm
		self.displayname = algorithm.__name__
		self.loaded = "algorithm"
		
	def make_a_move(self, game):
		
		if self.loaded == "policy":
			return self.play_policy(game)
		elif self.loaded == "algorithm":
			return self.play_alg(game)
		else:
			raise Exception("Du har ikke indlæst en agent.")
		
	def play_policy(self, game):

		with torch.no_grad():
			if self.hot_one:
				board = torch.Tensor(create_one_hot_repr(game.board)).unsqueeze(0)
			else: 
				board = torch.Tensor(game.board).view(16).unsqueeze(0)

			p = self.policy(board)
			p[p<tiny_number] = tiny_number
			
			choice = torch.argmax(p)
			change = game.move(game.board, choice)[2]
			while change == 0:
				try:
					p[0, choice] = 0
				except IndexError:
					p[choice] = 0
				choice = torch.argmax(p)
				change = game.move(game.board, choice)[2]
			
			return choice
	
	def play_alg(self, game):

		choice = self.algorithm(game, self.count)
		self.count += 1
		
		return choice
		
def sampler(probs):

	cs = np.cumsum(probs)
	idx = cs.searchsorted(np.random.random()*cs[-1], "right")
	return int(np.unravel_index(idx, probs.shape)[0])

def bootstrap(population, sample_size = 0, B = 0):
	if not B:
		B = len(population)
	if not sample_size:
		sample_size = len(population)
	
	means = np.zeros(B)
		
	for i in range(B):
		sample = np.random.choice(population, sample_size, replace = True)
		means[i] = np.mean(sample)
		
			
	return np.mean(means), np.std(means)






