from os import chdir
from os.path import dirname, realpath
chdir(realpath(dirname(__file__)))
from datetime import datetime
import numpy as np
from sys import argv
from threading import Thread

from game import Game2048

import algorithms as alg
import utilities as utl

if len(argv) == 2:
	THREAD = int(argv[1])
else:
	THREAD = 0
THREADSTR = "T%i" % THREAD

def create_data(agent, testsize=40000):

	create_time = datetime.now().strftime('%m%d-%H%M%S')

	boards = np.empty((testsize, 256))
	moves = np.empty((testsize, 4)) 

	change = 0
	game = Game2048()
	for i in range(testsize):
		try:
			if change == 2:
				print(THREADSTR, "%i\t%i" % (i, game.score))
				game = Game2048()
				
			direction = agent.make_a_move(game)
			boards[i] = utl.create_one_hot_repr(game.board)	
			
			move = np.zeros(4)
			move[direction] = 1 

			moves[i] = move
			
			change = game.play(direction)
		except KeyboardInterrupt:
			boards = boards[:i]
			moves = moves[:i]
			i -= 1
			break

	print(THREADSTR, "Datapunkter genereret:", i+1)
	path = realpath('../pretraindata/%s_t%i_pretrain_%s' % (create_time, THREAD, agent.displayname))
	np.savez(path, boards, moves)
	return path

if __name__ == "__main__":
	create_data(utl.Agent(alg.pure_montecarlo))
