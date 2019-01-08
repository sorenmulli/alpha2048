import numpy as np
import torch
import matplotlib.pyplot as plt

from sorenTestAtariGame import AtariGame

global device 

if torch.cuda.is_available():
	device=torch.device('cuda')
	torch.set_default_tensor_type(torch.cuda.FloatTensor)
else:
	device = torch.device('cpu')

np.seterr(all='raise')

def policyNetwork(x, W1, W2):
	'''
	Tager input-arrayet og indfører det trænede policy-netværk på det.
	'''

	#Ganger det skjulte lags vǽgte på
	logp = W2.mv(W1.mv(x).clamp(min=0))	

	return logp

def initialise():
	W1, W2 = torch.rand(16, 16, requires_grad=True, device = device), torch.rand(4, 16, requires_grad=True, device = device)
	learningRate = 1e-6
	return 	W1, W2, learningRate


def weightedChoice(logp):
	#Sandsynligheden udtrækkes ved sigmoid-funktionen, som fører sandsynligheden ind i {0..1}

	p = np.array(1.0 / (1.0 + torch.exp(-logp)).detach(), dtype = np.float)
	p_normalized = p / (np.sum(p))
	direction = int(np.random.choice(np.arange(4), 1, p= p_normalized))

	return direction

def playBatch(game, W1, W2, learningRate, batchSize = 10):
	newW1, newW2 = torch.clone(W1).detach(), torch.clone(W2).detach()
	
	newW1.requires_grad = False
	newW2.requires_grad = False
	
	for i in range(batchSize):

		choices = list()
		prob = list()
		while True:
			logp = policyNetwork(torch.Tensor(game.board).view(16), W1, W2)
			with torch.no_grad():
				direction = weightedChoice(logp)


			change = game.play(direction)
			while True:
				if change != 0:
					break
				else:
					with torch.no_grad():
						direction = weightedChoice(logp)
					change = game.play(direction)

			if change == 2:
				break
			
			prob.append(logp)
			directions.append(direction)


		maxValue = np.max(game.board)
		reward = int(rewardFunction(game.board, game.score, scoremean))

		for j in range(len(directions)):
			gradients = torch.Tensor([0, 0, 0, 0])
			gradients[directions[j]] = reward
			prob[j].backward(gradients)
			


		with torch.no_grad():
			newW1 += reward * learningRate * W1.grad
			newW2 += reward * learningRate * W2.grad
			# print(W1.mean(), W2.mean())
			# print(W1[W1>0].size(), W2[W2>0].size())
			# print(W1)
			W1.grad.zero_()
			W2.grad.zero_()

		score[i] = game.score
		maxTiles[i] = maxValue

		# scoremean = 

	return score, maxTiles, newW1, newW2
		
	
def run():
	W1, W2, learningRate = initialise()
	game = AtariGame()

	scoreMeans = list()

	for t in range(500):
		
		W1.requires_grad = True
		W2.requires_grad = True
		score, tile, W1, W2 = playBatch(W1, W2, learningRate, batchSize = 100, lastmean=scoreMeans[-1])

		scoreMeans.append(np.mean(score))
		
		print(t, np.mean(score))
		
	
	plt.plot(scoreMeans, 'o')
	plt.show()
	
	print(scoreMeans)
run()
