import numpy as np
import matplotlib.pyplot as plt
from os.path import dirname, realpath

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
from torch.distributions import Categorical

from timeit import default_timer as timer
times = list()

from game import Game2048
import algorithms as alg

# Henter mindste tal fra nps float-info for at undgå division med 0
tiny_number = np.finfo(np.float32).eps.item()  

# Sætter CUDA til, hvis det er muligt 
#TODO: FÅ DET HER TIL AT VIRKE
if torch.cuda.is_available():
	device=torch.device('cuda')	
	torch.set_default_tensor_type(torch.cuda.FloatTensor)

else:
	device = torch.device('cpu')

print("Enhed muligvis brugt:", device)

class _Counter():

	c = 0

	def count(self):

		self.c += 1

class Policy(torch.nn.Module):

	def __init__(self, layers: list or tuple, with_bias=True, dropout = 0):

		#Arver fra torch'es neural-netværk-klasse
		super().__init__()
		
		#Gemmer antal layers og dropout
		self.nlayers = len(layers)
		self.dropout = dropout
		
		#Tilføjer alle lagene til klassen, så de kan tilgås direkte fra klassen
		for i in range(self.nlayers-1):
			setattr(self, "linear%s" % (i+1), nn.Linear(layers[i], layers[i+1], bias=with_bias))


		#Gemmer output og rewards, så der senere kan laves backprop
		self.rewards = list()
		self.policyhist = Variable(torch.Tensor())


	def forward(self, x):

		'''
		Tager input-arrayet og indfører det trænede policy-netværk på det.
		'''
		
		#Bygger modellen ved at hente alle lagene fra klassen selv
		seq = list()
		for i in range(self.nlayers-2):
			seq.append(getattr(self, "linear%s" % (i+1)))
			#Tilføjer  og ReLu for ikke-linearitet
			seq.append(nn.Dropout(p=self.dropout))
			seq.append(nn.ReLU())
		
		#Tilføjer sidste lag og softmax
		seq.append(getattr(self, "linear%s" % (i+2)))
		seq.append(nn.Softmax(dim=-1))
		
		#Føjer alle lag og ikke-lineariteter ind i en sekvensiel torch-nn-model
		model = nn.Sequential(*seq)
		
		#Udfører feed-forward vha. modellen
		return model(x)

	def reset(self):

		#Resetter rewards og tidligere policies
		self.rewards = []
		self.policyhist = Variable(torch.Tensor())


class PolicyLearner2048:

	def __init__(self, layers, learnrate = 1e-3, with_bias=True, dropout=.6, gamma=.99):

		# Parametrene gemmes i en dictionary
		self.params = {
			"layers": layers,
			"layer_amount": len(layers) - 1,
			"layer_sizes": layers[0:-1],
			"gamma": gamma,
			"dropout": dropout,
			"bias": with_bias,
			"lr": learnrate
		}
		
		#Henter policy-netværket
		self.policy = Policy(self.params["layers"], self.params["gamma"], self.params["dropout"])

		#Opretter en optimizer vha torches indbyggede Adam-modul og policy-netværkets parametre
		self.optimizer = torch.optim.Adam(self.policy.parameters(), lr = learnrate)
		self.params["optimizer_info"] = str(getattr(self, 'optimizer'))

		#Opretter en learning rate-justerende algoritme
		# self.lr_scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(self.optimizer, 'min', verbose=True, patience= 15, factor = 0.5)
		# self.params["lr_scheduler"] = str(getattr(self, 'lr_scheduler'))

		# Resetter score og maxtile
		self.scoremeans = None
		self.maxtilemeans = None

	def train(self, nbatches, batchsize, current_batch = 0, save_model = None, auto_save_period = 0):

		self.params["nbatches"] = nbatches
		self.params["batchsize"] = batchsize

		#Gemmer en streng med beskrivelse af parametrene, så modellen kan dokumenteres nemmere
		self.info_string = "\nNetværk: {layer_amount} layers med {layer_sizes} neuroner og bias = {bias}.\
			\nParametre: Learning-rate: {lr}, gamma: {gamma}, dropout: {dropout}.\
			\nKørsel: {nbatches} batches med størrelse {batchsize}\
			\nOptimizer: {optimizer_info}".format(**self.params).replace("\t", "")
		print(self.info_string)

		
		#Gennemsnitlig score og højeste tile gemmes i løbet af hele træningen
		self.scoremeans = np.zeros(nbatches) #NB: Skal bevares som np zeros, da dette benyttes i adaptiv lr
		self.maxtiles = np.zeros(nbatches)

		#Træning sættes igang og den batch, man er nået til, gemmes for at der kan loades igen
		for i in range(current_batch, nbatches):
			start = timer()
			self.current_batch = i 
			
			#Der spilles et batch af spil og gns. score og maxtile gemmes
			if i > 0:
				s, m, _ = self.play_batch(batchsize, self.scoremeans[i-1])
			else:
				s, m, _ = self.play_batch(batchsize)
			self.scoremeans[i] = np.mean(s)
			self.maxtiles[i] = np.max(m)
			print("Batch: {0}, gns. score: {1}, høj. tile: {2}".format(i, self.scoremeans[i], self.maxtiles[i]))
			
			#Når batchen er slut udføres policy-update
			self.policy_update()
			end = timer()
			times.append(end-start)
			print("\t\tTid: %.2f" % times[-1])
		

			#autogemmer, hvis det er påkrævet
			if i and auto_save_period and i % auto_save_period == 0 and save_model:
				print("\tModel autosaver...")
				self.save_me(save_model + 'autosave-tur' + str(i))
			
		print(self.info_string)

		if save_model:
			self.save_me(save_model)


	def play_batch(self, batchsize, comparescore=0):

		n = 5000

		#lister til at gemme resultater
		scores = np.empty(batchsize)
		maxtiles = np.empty(batchsize)
		turns = np.empty(batchsize)

		for i in range(batchsize):

			#Påbegynd spil
			game = Game2048()
				
			#Pre-spiller spillet
			#game = self.preplay_game(game, alg.runInRing, 2400)
			
			# Score hver tur
			turnscores = torch.empty(n)

			#For-loop for en sikkerheds skyld
			for turn in range(n):
				#Henter spillebrættet som vektor
				gameState = torch.Tensor(game.board, device = device).view(game.n**2).unsqueeze(0)
				
				#Finder valget baseret på feed-forward gennem policy-netværket og udfører det
				choice = self.make_choice(gameState)
				change = game.play(choice)

				turnscores[turn] = int(game.score)
				
				#Hvis spillet er tabt
				if change == 2:
					break
			#Gemmer reward
			turnscores = turnscores[:turn+1]
			# self.policy.rewards.append(turnscores)
			self.policy.rewards.append(self.discount_reward(turnscores))

			#Gemmes score, maxtiles og sidste tur
			scores[i] = game.score
			maxtiles[i] = 2**np.max(game.board)
			turns[i] = turn
		
		# Laver rewards om en til en 1-dimensionel tensor
		self.policy.rewards = torch.cat(self.policy.rewards)
		
		# import matplotlib.pyplot as plt
		# plt.plot(self.policy.rewards.numpy())
		# plt.show()
		return scores, maxtiles, turns
	
	def preplay_game(self, game, agent, score_threshold, retry_threshold = 20):
		fresh_game = game 
		count = 0
		retry_count = 0
		while game.score < score_threshold and retry_count < retry_threshold :
			direction = agent(game, count)
			count += 1
				
			if game.play(direction) == 2:
				game = fresh_game
				count = 0
				retry_count += 1  
		return game 
		

		
		
	def make_choice(self, gameState):
		#Vælger en handling

		#feed-forward udføres vha policy-netværk og de 4 log-sandsynligheder gives
		logp = self.policy(Variable(gameState))

		#En sandsynlighedsfordeling dannes ud fra disse
		probDist = torch.distributions.Categorical(logp) 
		
		#Der udføres et valg baseret på sandsynlighedsfordelingen
		choice = probDist.sample()
		
		#Gemmer den valgte handling i policy-netværket, så der kan udføres backprop senere
		if self.policy.policyhist.dim() != 0:
			self.policy.policyhist = torch.cat([self.policy.policyhist, probDist.log_prob(choice)])
		else:
			self.policy.policyhist = probDist.log_prob(choice)
		
		return choice.item()

	def discount_reward(self, raw_rewards: torch.Tensor):
		
		#Udfører reward-discount vha gamma
		R = 0
		rewards = torch.empty(len(raw_rewards))
		for i in range(len(raw_rewards)):
			R = raw_rewards[-i-1] + self.params["gamma"] * R
			rewards[i] = R
		
		return rewards

	def policy_update(self):

		#Skalerer rewards
		rewards = self.policy.rewards
		rewards = (rewards - rewards.mean()) / (rewards.std() + tiny_number)
		# plt.plot(rewards.numpy())
		# plt.show()
		
		#Udregner loss
		policyLoss = torch.sum(torch.mul(self.policy.policyhist, Variable(rewards)).mul(-1), -1)

		#Udfører backprop
		self.optimizer.zero_grad()
		policyLoss.backward()
		self.optimizer.step()
		
		#Indstiller learning raten
		# self.lr_scheduler.step(policyLoss)

		#Resetter netværket
		self.policy.reset()
	
	def save_me(self, saving_dir):
		#Opretter en fil, man kan gemme modellens situation i
		save_file = open(saving_dir, 'w+' )
		save_file.close()
		
		#Gemmer vægte, optimizer, parametre og den batch, man er nået til 
		state = {
			'batch': self.current_batch,
			'params': self.params,
			'state_dict': self.policy.state_dict(),
			'optimizer': self.optimizer.state_dict()
			}
		torch.save(state, saving_dir)
		

	
	def load_model(self, loading_dir):
		#Henter den gamle model fra fil
		state = torch.load(loading_dir)
		
		#Modellen skal være trænet med samme parametre, som du vil indlæse den med
		assert self.params == state["params"]
		
		#Loader vægtene og optimizerens status
		self.policy.load_state_dict(state["state_dict"])
		self.optimizer.load_state_dict(state["optimizer"])
		
		#Sætter igang fra den batch, hvor man afsluttede den
		self.current_batch = state["batch"]
