import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
from torch.distributions import Categorical
import numpy as np
from itertools import count
import matplotlib.pyplot as plt

from game import Game2048

np.seterr(all="raise")

eps = np.finfo(np.float32).eps.item()  # Verdens mindste tal


class _Counter:

	c = 0

	def count(self):

		self.c += 1


class Policy(nn.Module):

	def __init__(self, layers):

		super().__init__()

		self.nlayers = len(layers)
		for i in range(self.nlayers-1):
			setattr(self, "linear%s" % (i+1), nn.Linear(layers[i], layers[i+1]))

		# Initialiserer lag
		# self.linears = [nn.Linear(layers[i], layers[i+1], bias=True) for i in range(len(layers)-1)]
		# print(self.linears)
		# self.linear1 = nn.Linear(layers[0], layers[1], bias=True)
		# self.linear2 = nn.Linear(layers[1], layers[2], bias=True)

		self.policy_hist = Variable(torch.Tensor())

		self.episode_rewards = []
	
	def forward(self, x):
		
		# Opretter sekvens af kommandoer
		seq = []
		for i in range(self.nlayers-2):
			seq.append(getattr(self, "linear%s" % (i+1)))
			seq.append(nn.Dropout(p=.6))
			seq.append(nn.ReLU())
		seq.append(getattr(self, "linear%s" % (i+2)))
		seq.append(nn.Softmax(dim=-1))
		model = torch.nn.Sequential(*seq)

		
		# Forwarder med ReLU og softmax på sidste lag
		# seq = []
		# for l in self.linears[:-1]:
		# 	seq.append(l)
		# 	seq.append(nn.Dropout(p=.6))
		# 	seq.append(nn.ReLU())
		# seq.append(self.linears[-1])
		# seq.append(nn.Softmax(dim=-1))
		# model = torch.nn.Sequential(*seq)

		return model(x)
	
	def reset(self):

		self.policy_hist = Variable(torch.Tensor())
		self.episode_rewards = []




class Learner:

	def __init__(self, layers, gamma):

		self.gamma = gamma

		self.policy = Policy(layers)
		# Optimeringsalgoritme
		self.optimizer = torch.optim.Adam(self.policy.parameters(), lr=1e-3)

		# Se self.train
		self.ep_scoremeans = None
	
	def train(self, nbatches, ngames):

		self.ep_scoremeans = np.empty(nbatches)

		for b in range(nbatches):
			print(b)
			self.ep_scoremeans[b] = self.playbatch(ngames)
	
	def playbatch(self, ngames=50):

		scores = np.empty(ngames)

		running_reward = 10
		for g in range(ngames):
			# Starter nyt spil
			game = Game2048()
			c = _Counter()
			oldscore = 0
			while True:
				c.count()
				# Foretager en tur
				action = self.select_action(game)
				change = game.play(action)
				while change == 0:
					action = self.select_action(game, False)
					change = game.play(action)
				self.policy.episode_rewards.append(game.score-oldscore)

				# Når spillet er færdigt
				if change == 2:
					break
			
			running_reward = running_reward * self.gamma + c.c * .01
			oldscore = game.score
			scores[g] = game.score
			self.finish_episode()
		
		return np.mean(scores)


	
	def select_action(self, game, include_last=True):

		# Vælger en handling baseret på 
		state = torch.from_numpy(game.board.reshape(game.n**2)).float().unsqueeze(0)
		probs = self.policy(state)
		selector = Categorical(probs)
		action = selector.sample()
		if self.policy.policy_hist.dim() != 0 and include_last:
			self.policy.policy_hist = torch.cat([self.policy.policy_hist, selector.log_prob(action)])
		elif self.policy.policy_hist.dim() != 0 and not include_last:
			self.policy.policy_hist = torch.cat([self.policy.policy_hist[:-1], selector.log_prob(action)])
		else:
			self.policy.policy_hist = selector.log_prob(action)
		# self.policy.logprobs.append(selector.log_prob(action))

		return action.item()

	def finish_episode(self):

		R = 0
		rewards = []
		# Laver liste med rewards
		for r in self.policy.episode_rewards[::-1]:
			R = r + self.gamma * R
			rewards.insert(0, R)
		
		# Standardiserer
		rewards = torch.Tensor(rewards)
		rewards = (rewards - rewards.mean()) / (rewards.std() + eps)

		# Beregner loss
		# print(rewards.shape)
		# print()
		loss = torch.mul(self.policy.policy_hist, Variable(rewards))
		loss = torch.sum(loss.mul(-1), -1)
		# for logprob, reward in zip(self.policy.logprobs, rewards):
		# 	loss.append(-logprob*reward)
		
		# print(len(self.policy.logprobs), len(rewards), len(loss))
		
		# Udfører gradientnedstigning
		self.optimizer.zero_grad()
		loss.backward()
		self.optimizer.step()
		
		self.policy.reset()


N = Learner([16, 20, 20, 4], .99)
N.train(60, 40)
plt.plot(N.ep_scoremeans, "o")
plt.show()


