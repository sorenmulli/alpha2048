import numpy as np

def randomGuess(game, count):
	while True:

		choice = np.random.randint(0, 4)

		if game.move(game.board, choice)[2] != 0:
			break

	return choice

def priorityList(game, count):
	
	for choice in range(4):
		
		if game.move(game.board, choice)[2] != 0:
			break

	return choice

def runInRing(game, count):
	return count % 4

def spamTwoDirs(game, count):
	choice = count % 2

	if game.move(game.board, choice)[2] == 0:
		choice = 2
		if game.move(game.board, choice)[2] == 0:
			choice = 3
	
	return choice

def oneMoveAheadScore(game, count):
	results = np.zeros(4)
	
	for i in range(4):
		a = game.move(game.board, i)
		results[i] = a[2] 

	return np.argmax(results)

def oneMoveAheadZeros(game, count):
	results = np.zeros(4)
	
	for i in range(4):
		a = game.move(game.board, i)
		results[i] = a[1].shape[0] 
	
	return np.argmax(results)
