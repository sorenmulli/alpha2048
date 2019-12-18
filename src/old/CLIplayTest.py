import numpy as np
from game import Game2048

def run():

	game = Game2048(deterministic=[True, True])
	
	while True:
		prettyBoard = (2**game.board)
		prettyBoard[game.board == 0] = 0
		
		print(prettyBoard)
		print(game.score)
		try:
			direction = int(input("Angiv retning, du vil bevæge spillerpladen i: \n \t(0 for venstre, 1 for op, 2 for højre, 3 for ned)\n"))
			if direction == -1:
				raise IndentationError
		
			if game.play(direction) == 2:
				break

			print("\n\tScore: {0}\n".format(game.score))
		except IndentationError:
			import sys
			sys.exit()
		except:
			print("\tFEJL: Dit input stinker")

	print("Desværre, basse, du tabte. Din score blev {0}".format(game.score))



	pass

if __name__ == "__main__":
	run()