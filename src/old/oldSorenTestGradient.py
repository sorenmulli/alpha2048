import numpy as np
import torch

import matplotlib.pyplot as plt

from game import Game2048

np.seterr(all='raise')


class Policy(torch.nn.Module):
	def __init__(self, inputSize, hiddenSize, outputSize):
		#Arver fra torch'es neural-netværk-klasse
		super(Policy, self).__init__()

		#Sætter CUDA til, hvis det er muligt
		# if torch.cuda.is_available():
		# 	self.device=torch.device('cuda')	
		# 	torch.set_default_tensor_type(torch.cuda.FloatTensor)
		
		# else:
		# 	self.device = torch.device('cpu')

		# print("Enhed brugt: ", self.device)

		
		#Laver inputlaget
		self.inputLayer = torch.nn.Linear(inputSize, hiddenSize, bias = False)
		
		#Laver det skjulte lag
		self.hiddenLayer = torch.nn.Linear(hiddenSize, outputSize, bias = False)

		
		#Gemmer output og rewards, så der senere kan laves backprop
		self.logpList = list()
		self.rewardList = list()
		self.policyHist = torch.autograd.Variable(torch.Tensor())


	def forward(self, x):
		'''
		Tager input-arrayet og indfører det trænede policy-netværk på det.
		'''
		model = torch.nn.Sequential(
			self.inputLayer,
			torch.nn.Dropout(p = 0.6),
			torch.nn.ReLU(),
			self.hiddenLayer,
			torch.nn.Softmax(dim = -1)
		)
		result = model(x)
		return result

class PolicyLearner2048:

	def __init__(self, learningRate = 1e-3, hiddenSize = 32):
		#Henter policy-netværket
		self.policy = Policy(16, hiddenSize, 4)

		#Opretter en optimizer vha torches indbyggede Adam-modul og policy-netværkets parametre
		self.optimizer = torch.optim.Adam(self.policy.parameters(), lr = learningRate)

		#Gemmer det mindste mulige tal vha numpy - så man kan undgå division med 0
		self.tinyNumber = np.finfo(np.float32).eps.item()


	def makeChoice(self, gameState):
		#Vælger en handling

		#feed-forward udføres vha policy-netværk og de 4 log-sandsynligheder gives
		logp = self.policy(torch.autograd.Variable(gameState))

		#En sandsynlighedsfordeling dannes ud fra disse
		probDist = torch.distributions.Categorical(logp) 
		
		#Der udføres et valg baseret på sandsynlighedsfordelingen
		choice = probDist.sample()


		
		#Gemmer den valgte handling i policy-netværket, så der kan udføres backprop
		if self.policy.policyHist.dim() != 0:
			self.policy.policyHist = torch.cat([self.policy.policyHist, probDist.log_prob(choice)])
		
		else:
			self.policy.policyHist = (probDist.log_prob(choice))		
		
		return choice.item()
	
	def doPolicyUpdate(self):
		#Udfører reward-discount vha gamma på 0.99
		R = 0 
		rewards = list()
		for reward in self.policy.rewardList[::-1]:
			R = reward + 0.99 * R
			rewards.insert(0, R)

		#Skalerer rewards
		rewards = torch.Tensor(rewards)
		rewards = (rewards - rewards.mean()) / (rewards.std() + self.tinyNumber)
		
		#Udregner loss
		policyLoss = (torch.sum(torch.mul(self.policy.policyHist,  torch.autograd.Variable(rewards)).mul(-1), -1))

		#Udfører backprop
		self.optimizer.zero_grad()
		policyLoss.backward()
		self.optimizer.step()

		#Resetter netværket
		self.policy.rewardList = list()
		self.policy.logpList = list()
		self.policy.policyHist = torch.autograd.Variable(torch.Tensor())


	def playBatch(self, batchSize = 50):
		#lister til at gemme resultater
		scores = np.zeros(batchSize)
		maxtiles = np.zeros(batchSize)
		turns = np.zeros(batchSize)

		for i in range(batchSize):
			#Påbegynd spil
			game = Game2048()
			oldscore = 0

			#For-loop for en sikkerheds skyld
			for turn in range(5000):
				
								
				#Henter spillebrættet som vektor
				gameState = torch.Tensor(game.board).view(16).unsqueeze(0)
				
				#Finder valget baseret på feed-forward gennem policy-netværket
				choice = self.makeChoice(gameState)
				
				#Udfører valget
				change = game.play(choice)

				#Henter score og beregner reward som ændring i score
				score = game.score
				reward = score - oldscore 
				
				#Gemmer rewarden
				self.policy.rewardList.append(reward)


				
				#Hvis spillet er tabt
				if change == 2: break
				oldscore = score 
				
				#Gemmes score, maxtiles og sidste tur
				scores[i] = game.score
				maxtiles[i] = 2**np.max(game.board)
				turns[i] = turn

			#Når spillet er slut udføres policy-update
			self.doPolicyUpdate()
				

		return scores, maxtiles, turns

		 
def runTraining():
	scoreMeans = list()
	# maxtileMean = list()
	# turnMeans = list()

	learner = PolicyLearner2048()

	for t in range(250):
		
		score, maxtiles, turns = learner.playBatch()

		scoreMean = np.mean(score)
		maxtileMean = np.mean(maxtiles)
		turnMean = np.mean(turns)
		
		scoreMeans.append(scoreMean)
		# maxtileMean.append(maxtileMean)
		# turnMeans.append(turnMean)

		print("Batch {0}: Gns. score {1}, gns. max. tile {2}, gns. længde {3}".format(t, scoreMean, maxtileMean, turnMean))
		
		#if t % 50 == 0:
		#	torch.save(learner.policy, '/savedModels/model{0}'.format(t % 50))
	
	plt.plot(scoreMeans, 'o')
	plt.show()
	print(scoreMeans)

runTraining()