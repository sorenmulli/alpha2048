import numpy as np


def randomChoice(frame, reward):
	return np.random.randint(0, 2)

def oneChoice(frame, reward):
	return 4

def selfplay(frame, reward):
	userIn = input("Indtast: ")
	if userIn == '':
		userIn = 0
	else:
		userIn = int(userIn)
		
	if userIn in range(2):
		return userIn

	else: return 0 