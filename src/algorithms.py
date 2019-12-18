import numpy as np
import matplotlib.pyplot as plt

from game import Game2048
import reward_functions as rf

from copy import copy

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

def pure_montecarlo(game, _, runs=50):
	#Array to save the acummulated score of choosing each direction
	end_scores = np.zeros(4)
	
	#Tries each direction
	for i in range(4):
		
		#Only runs the monte carlo tree search if the move could result in a change 
		
		if game.move(game.board, i)[2] == 1:
			for j in range(runs):
				board = np.copy(game.board)
				board, _, change, _, _ = game.move(game.board, i)
				score = 0
				while change != 2:

					direction = np.random.randint(4)
					board, _, change, _score, _ = game.move(board, direction) 
					score += _score

				end_scores[i] += score
				change = 1

	return np.argmax(end_scores)
	
def corner_hoggin(game, _):
	score = np.zeros(4)
	change = np.zeros(4)
	for i in range(4):
		_, _ , _change, _score, _ = game.move(game.board, i)
		change[i], score[i] = _change, _score 

	if game.board[0, 0] == 0:
		if change[1]: 
			return 1
		elif change[0]:
			return 0
			
	score[change == 0] = -1
	
	if change[0:2].sum():
		return np.argmax(score[0:2])
		
	elif change[2]:
		return 2
		
	else:
		return 3
		
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
