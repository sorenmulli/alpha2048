import numpy as np
import matplotlib.pyplot as plt

from os import chdir
from os.path import dirname, realpath
from datetime import datetime

# Prøver at give python højere prioritet
#try:
#	nice(-20)
#except PermissionError:
#	pass

from game import Game2048
from standardpolicy import PolicyLearner2048
import algorithms as alg


def rollingavg(values, n):

	"""
	Returnerer et løbende gennemsnit baseret på n elementer til hver side
	"""

	if n == 0:
		return values

	weightvec = np.concatenate((np.arange(1, n+1), np.arange(n-1, 0, -1)))

	values = np.concatenate((np.zeros(n), values, np.zeros(n)))

	weightedvalues = values.copy()
	
	for i in range(n, values.size-n):
		weightedvalues[i] = np.sum(values[i-n+1:i+n] * weightvec) / n**2
	
	return weightedvalues[n+1:-n-1]


def run(func, n):
	results = np.zeros(n)
	maxVals = np.zeros(n)
	
	for i in range(n):
		print(i)
		results[i], maxVals[i] = playAGame(func)

	print(np.mean(results), np.std(results), "\n", np.mean(2**maxVals), 2**np.max(maxVals))

	plt.hist(results, 100)
	plt.show()

def playAGame(func):
	game = Game2048()
	count = 0

	while True:
		

		direction = func(game, count)
		count += 1

		if game.play(direction) == 2:
			maxValue = np.max(game.board)
			break

	return game.score, maxValue

def runTraining():
	#Gemmer fil-stedet og tidspunktet til at gemme resultaterne
	resultdir = dirname(realpath(__file__)).replace('src', 'results/')
	createdTime = datetime.now().strftime('%m%d-%H%M')
	#Et standard-fil-sted til at gemme modellen
	modelsdir  = dirname(realpath(__file__)) + '/saved_models/model_{0}.pt'.format(createdTime)
	
	#Opretter learneren med policy-netværket inde i sig og træningen påbegyndes
	N = PolicyLearner2048([16, 16, 4], with_bias=True, learnrate=1e-3, dropout = 0) #Angiver dimensionerne på lagene
	N.load_model('/home/sorenwh/Dropbox/DTU/1 Semester/Intro til intelligente systemer/PROJEKT/alpha2048/src/saved_models/model_0108-1134.pt')
	N.train(350, 50, save_model = modelsdir, auto_save_period=50, current_batch=N.current_batch)  # Antal batches og antal spil pr. batch - hvis du ikke vil gemme modellen, skal du sætte denne til None
	
	
	#Skriver resultaterne med gennemsnit-scoren og info om netværket 
	file_out = open("{0}result_{1}.txt".format(resultdir, createdTime), 'w+' )
	file_out.write(str(N.params) + "\n\n\n" + str(N.scoremeans) + "\n\n\n" + str(rollingavg(N.scoremeans, N.params["batches"]//10)))
	file_out.close()
	
	#Gemmer et plot med gennemsnits-scoren og viser det 
	plt.plot(N.scoremeans, "o")
	plt.plot(rollingavg(N.scoremeans, N.params["nbatches"]//10))
	plt.savefig("{0}plot_{1}.png".format(resultdir, createdTime))
	plt.show()
	

if __name__ == "__main__":
	runTraining()
