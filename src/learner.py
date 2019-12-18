from os.path import dirname, realpath
from warnings import warn

import torch
from torch.autograd import Variable
from torch.distributions import Categorical

from copy import copy
import numpy as np 
from timeit import default_timer as timer
from datetime import datetime

from game import Game2048
from utilities import max_tile_distribution, sampler, create_one_hot_repr
from policy import Policy
import reward_functions as rf

# Loads a tiny number from nps float-info to avoid division by 0
tiny_number = np.finfo(np.float32).eps.item()

# Defines the devices
cpu = torch.device("cpu")
gpu = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

class PolicyLearner2048:

	def __init__(self, layers=(256, 16, 4), learnrate=1e-3, dropout=.6, thread=0):
		
		# Parameters are saved in a dictionary
		self.params = {
			"layers": layers,
			"layer_amount": len(layers) - 1,
			"layer_sizes": layers[0:-1],
			"dropout": dropout,
			"lr": learnrate,
			"startbatch": 0
		}
		self.threadstr = "T%i" % thread
		
		#Loads the policy network
		self.policy = Policy(self.params["layers"], dropout=self.params["dropout"], device=cpu)

		#Generates an optimizer with pytorch's Adam-module and the policy networks parameters
		self.optimizer = torch.optim.SGD(self.policy.parameters(), lr=learnrate)
		self.params["optimizer_info"] = str(getattr(self, 'optimizer'))

		# Saves the output and rewards in order to execute backpropagation later
		self.rewards = list()
		self.policyhist = Variable(torch.Tensor())

	def train(self, nbatches=500, batchsize=50, start_batch=0, save_model=None,
			autosave_secs=float("infinity"), rewarder=rf.ScoreChange(), determinism=[False, False], hot_one=True):

		self.params["rewarder"] = rewarder
		self.params["nbatches"] = nbatches
		self.params["batchsize"] = batchsize
		self.params["determinism"] = determinism
		start_batch = start_batch if not self.params["startbatch"] else self.params["startbatch"]
		self.params["startbatch"] = start_batch
		self.params["rewarder"] = rewarder
		self.params["hot_one"] = hot_one

		# Training data that will be returned
		tomeasure = ("scoremeans", "maxtiles", "maxtilemeans", "nmovemeans", "npropmovemeans", "playtimes", "backproptimes", "runtimes", "rewardmeans")
		rundata = {kw: np.empty(nbatches-start_batch) for kw in tomeasure}
		
		# Saves a string with a description of the parameters in order to easily document the created model
		self.info_string = "Parametre:\n" + "\n".join(["%s: %s" % (kw, self.params[kw]) for kw in self.params]) + "\n"
		print(self.threadstr, self.info_string)

		timesincesave = 0
		
		# Training is initialized and the current batch is saved in order to be able to load it agin
		for i in range(start_batch, nbatches):

			try:
				start = timer()
				useidx = i - start_batch
				
				# A batch is played and the average score and maxtile is saved
				s, m, t, p, r = self.play_batch(batchsize)

				rundata["scoremeans"][useidx] = s.mean()
				rundata["maxtiles"][useidx] = m.max()
				rundata["maxtilemeans"][useidx] = m.mean()
				rundata["nmovemeans"][useidx] = t.mean()
				rundata["npropmovemeans"][useidx] = p.mean()
				rundata["rewardmeans"][useidx] = r.mean()
				
				m_dist_string = self.threadstr + " " + max_tile_distribution(m)

				print(self.threadstr, "Batch: {0}, gns. score: {1}, Maks. tile nået: {2}".format(
					i,
					round(float(rundata["scoremeans"][useidx])),
					m_dist_string
				))
				
				# playtime
				playend = timer()
				rundata["playtimes"][useidx] = playend - start
				print(self.threadstr, "\tSpilletid: %.2f" % rundata["playtimes"][useidx])

				# When a batch is finnished a policy update is executed
				self.policy_update()

				# Batch time
				rundata["backproptimes"][useidx] = timer() - playend
				rundata["runtimes"][useidx] = rundata["playtimes"][useidx] + rundata["backproptimes"][useidx]
				print(self.threadstr, "\tBackproptid: %.2f" % rundata["backproptimes"][useidx])
				print(self.threadstr, "\tBatchtid: %.2f" % rundata["runtimes"][useidx])
				
				# Autosaves if required
				timesincesave += rundata["runtimes"][useidx]
				if timesincesave >= autosave_secs:
					print(self.threadstr, "\tModel autosaver...")
					try:
						self.save_me(save_model + 'autosave-tur' + str(i), i)
						print(self.threadstr, "\tFærdig med at gemme.")
						timesincesave = 0
					except KeyboardInterrupt:
						raise KeyboardInterrupt
					except Exception as e:
						warn(e+"\n"+str(type(e)))

			except KeyboardInterrupt:
				# Performs a clean up, if interrupted
				i -= 1
				for kw in tomeasure:
					rundata[kw] = rundata[kw][:useidx]
				break
		
		print(self.threadstr, self.info_string)
		self.params["nbatches"] = i + 1

		# Saves the model
		if save_model:
			try:
				self.save_me(save_model, i)
			except Exception as e:
				warn(e+"\n"+str(type(e)))

		return rundata

	def play_batch(self, batchsize):

		n = 5000
		
		# Arrays to save results
		scores = np.empty(batchsize)
		maxtiles = np.empty(batchsize)
		turns = np.empty(batchsize)
		propturns = np.empty(batchsize)
		
		for i in range(batchsize):
			# Starts a game
			game = Game2048(deterministic=self.params["determinism"])
			rewarder = self.params["rewarder"]	

			for turn in range(n):
				# Decides an action based on a feed forward through the policy network and executes the action
				change = self.make_choice(game)
				# Reward is given and is saved in the reward class
				rewarder.reward(game, turn)
				# If the game is lost
				if change == 2:
					break
			
			rewards = rewarder.final_reward(game, turn)

			#Saves the reward
			self.rewards.append(torch.Tensor(rewards))
			rewarder.clear()

			# Saves the score, maxtile and the last round of the game
			scores[i] = int(game.score)
			maxtiles[i] = 2 ** int(np.max(game.board))
			turns[i] = game.moves
			propturns[i] = game.propermoves
		
		# Reshapes the rewards into a one dimensional tensor
		self.rewards = torch.cat(self.rewards)
		
		return scores, maxtiles, turns, propturns, rewards

	def make_choice(self, game):

		if self.params["hot_one"]:
			board = torch.Tensor(create_one_hot_repr(game.board))
		else: 
			board = torch.Tensor(game.board).view(16)

		# Loads the gameboard as a vector
		gamestate = board.unsqueeze(0)

		# Feed-foward is executed with the policy network, and the 4 probabilites are given
		p = self.policy(Variable(gamestate))

		# A probability distribution is formed from the probabilites
		probdist = Categorical(p)
		probdist.probs[probdist.probs<tiny_number] = tiny_number
		
		# An action is executed based on the probability distribution
		with torch.no_grad():
			choice = probdist.sample()
			change = game.play(choice)
			
			while change == 0:
				probdist.probs[0, choice] = 0
				choice = probdist.sample()
				change = game.play(choice)
		
		# Saves the chosen action in the policy network in order to execute backpropagation later
		if self.policyhist.dim() != 0:
			self.policyhist = torch.cat([self.policyhist, probdist.log_prob(choice)])
		else:
			self.policyhist = probdist.log_prob(choice)
	
		return change

	def policy_update(self):

		# Scale the rewards
		self.rewards = (self.rewards - self.rewards.mean()) / (self.rewards.std() + tiny_number)

		# Performs backpropagation
		self.policy.trainRL(self.policyhist, self.rewards)

		# Resets the network
		self.batchreset()

	def batchreset(self):

		# Resets reward and policy history
		self.rewards = []
		self.policyhist = Variable(torch.Tensor())
	
	def save_me(self, saving_dir, current_batch):

		# Creates a file where the situation of the model can be saved
		save_file = open(saving_dir.replace("\\", "/"), 'w+', encoding="utf-8")
		save_file.close()

		# Saves the weights, optimizer, parameters and the current batch
		state = {
			'batch': current_batch,
			'params': self.params,
			'state_dict': self.policy.state_dict(),
			'optimizer': self.optimizer.state_dict()
		}
		torch.save(state, saving_dir)
	
	def load_model(self, loading_dir):

		# Loads an old model from file
		state = torch.load(realpath(loading_dir))
		# Removes the old trainging parameters
		if "batch" in state.keys():
			del state["params"]["batchsize"]
			del state["params"]["nbatches"]
			# Starts from the batch where the loaded model finished
			self.params["startbatch"] = state["batch"]
		else:
			self.params["startbatch"] = 0		

		# The model has to be trained on the same parameters that you want to load it with
		if self.params != state["params"]:
			print(self.threadstr, str(self.params), "\n", str(state["params"]))
			warn("Du har ændret parametrene!")
		
		# Loads the weights and the status of the optimizer
		self.policy.load_state_dict(state["state_dict"])
		self.optimizer.load_state_dict(state["optimizer"])
		






