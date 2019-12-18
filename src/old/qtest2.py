import sys
import numpy as np

from qtest import QNN, Agent


from game import Game2048
from utilities import create_one_hot_repr
import reward_functions as rf 

if __name__ == '__main__':
	agent = Agent(gamma = 0.9, epsilon = 1.0, lr = 0.003, max_memory = 5000, replace=None)

	while agent.memory_stored < agent.max_memory:
		
		game = Game2048()
		rewarder = rf.ScoreChange()
		state = create_one_hot_repr(game.board)
		
		change = 1
		i = 0

		while change != 2:
			action = np.random.randint(4)
			change = game.play(action)
			state_new =  create_one_hot_repr(game.board)

			rewarder.reward(game, i)
			reward = rewarder.rewards[i]
			i += 1

			agent.store_transition(state, action, reward, state_new)

			state = state_new
		
		print("Random init done")

		scores = []
		eps_hist = []

		numGames = 1000
		batch_size = 15

		for i in range(numGames):
			print("Spil %s, eps %4.f" %(i+1, agent.epsilon))
			eps_hist.append(agent.epsilon)

			done = False 
			
			rewarder = rf.ScoreChange()


			game = Game2048()
			state = create_one_hot_repr(game.board)
			change = 0

			j = 0
			while change != 2:
				action == agent.choose_action(state)
				change = game.play(action)
				state_new = create_one_hot_repr(game.board)


				rewarder.reward(game, j)
				reward = rewarder.rewards[i]
				j += 1

				agent.store_transition(state, action, reward, state_new)

				state = state_new
				agent.learn(batch_size)

			scores.append(game.score)
			print("score: ", game.score)

		