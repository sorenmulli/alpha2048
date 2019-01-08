import gym
from gym import wrappers

import numpy as np
import torch

import sorenTestAtariAgents as agents
 
class AtariGame:
	def __init__(self,):
		
		# Opretter mappe til mulighed for gemning af video mm.
		outputDir = '/tmp/test-results'
		

		# Opretter env til spillet, hvor frameSkip implementeres vælges, så kun hver fjerde frame behandles
		self.env = wrappers.Monitor(gym.make('Breakout-v0'), directory = outputDir,  force=True)

	def initSingleGame(self, preProcess = True):
		#Starter et enkelt spil 
		self.frame = self.env.reset()
		self.preProcess = preProcess
		
		if self.preProcess:
			self.frame = self.preProcessFrame(self.frame)
		self.reward = 0

	def move(self, choice, render = False):
		#foretager én handling i et enkelt spil
		
		self.frame, self.reward, done, _ = self.env.step(choice)
		
		if self.preProcess:
			self.frame = self.preProcessFrame(self.frame)


		#Afslutter spillet, hvis det er ovre
		if done: return 0
			
		#Viser spillet på skærmen
		if render: self.env.render()
		
		return 1

	def preProcessFrame(self, frame):
		#Beskærer rammen til spilleområdet og laver alle farver, der ikke er baggrund til 1
		frame = frame[32:192, 8:152]
		
		#Nedskalerer hele rammen med 2
		frame = frame[::2,::2,0]
		frame[frame != 0] = 1

		#Gør rammen til et 1D array
		cleanFrame = torch.Tensor(frame.ravel())
		return cleanFrame


	def playEpisodes(self, agent = agents.randomChoice, episodeCount = 10, render = True):
		rewards = np.empty(episodeCount)

		for i in range(episodeCount):
			
			#Resetter spillet og danner den første ramme
			frame = self.env.reset()
			reward = 0 
			

			#Spiller et enkelt spil spil
			while True:
				#Finder agentens valg
				actionChoice = agent(frame, reward)
				
				#Udfører handlingen baseret på agentens beslutning
				frame, reward, done, _ = self.env.step(actionChoice)
				rewards[i] += reward
				#Afslutter spillet, hvis det er ovre
				if done:
					print(i, rewards[i])
					break
				
				#Viser spillet på skærmen
				if render: self.env.render()
		
		return rewards
		

game = AtariGame()
game.initSingleGame()
print(game.frame.shape)

# from matplotlib import pyplot as plt

# frame = self.env.reset()
# for i in range(2):
# 	frame, reward, done, _ = self.env.step(agents.randomChoice(frame, 0))

# plt.imshow(self.preProcessFrame(frame))
# plt.show()
