import numpy as np

tiny_number = np.finfo(np.float32).eps.item() 

class Rewarder:
	def __init__(self, gamma=.95):
		self.max_turns = 5000
		
		self.gamma = gamma

		self.rewards = np.zeros(self.max_turns)

	def clear(self):
		self.rewards = np.zeros(self.max_turns)
	
	def final_reward(self, game, turn):
		# Saves the length of the game
		self.len = turn + 1 

		self.rewards = self.rewards[:self.len]
				
		if self.gamma is not None:
			self.discount_reward()

		return self.rewards

	def discount_reward(self):
		# Executes a reward discount using gamma
		R = 0
		self.discounted_rewards = np.empty(self.len)

		for i in range(self.len):
			R = self.rewards[-i-1] + self.gamma * R
			self.discounted_rewards[i] = R
		
		return self.discounted_rewards


	def __str__(self):

		return "%s: gamma = %s" % (self.__class__.__name__, self.gamma) 

class RewarderCombination(Rewarder):
	def __init__(self, rewarders, weights, product_weight=False, **kwargs):
		super().__init__(**kwargs)
		self.rewards = np.ones(self.max_turns)
		self.product_weight = product_weight
		self.rewarders = rewarders
		
		for rewarder in self.rewarders:
			rewarder.gamma = None
		
		self.weights = weights

	def reward(self, game, turn):
		for rewarder in self.rewarders:
			rewarder.reward(game, turn)
			
		
	def final_reward(self, game, turn):
		self.len = turn + 1 

		self.rewards = self.rewards[:self.len]
		self.partial_rewards = np.zeros((len(self.rewarders), self.len))		
		
		for i, rewarder in enumerate(self.rewarders):
			self.partial_rewards[i, :] = rewarder.final_reward(game, turn)
		
		if self.product_weight:
			for i, p_reward in enumerate(self.partial_rewards):
				self.rewards = self.rewards * (p_reward ** self.weights[i])
				
			self.rewards = self.rewards ** ( 1 / np.sum(self.weights) ) 
		else:
			self.rewards = np.dot(self.weights, self.partial_rewards)
		
		if self.gamma is not None:
			self.discount_reward()
		return self.rewards
		
	def clear(self):
		self.rewards = np.ones(self.max_turns)

		for reward in self.rewarders:
			reward.clear()
		
	def __str__(self):
		return "%s, rewarders = %s, weights = %s, gamma = %s, product_weight %s" % (
			self.__class__.__name__ ,
			[str(reward) for reward in self.rewarders],
			str(self.weights),
			self.gamma, 
			self.product_weight
		)
	
	
class ScoreChange(Rewarder):
	def __init__(self, **kwargs):
		super().__init__(**kwargs)
		self.old_score = 0
	
	def reward(self, game, turn):
		reward = game.score - self.old_score
		self.old_score = game.score
		
		self.rewards[turn] = reward 
	def clear(self):
		self.rewards = np.empty(self.max_turns)
		self.old_score = 0
		
class EndGameBoardSum(Rewarder):
	def __init__(self, sum_function = "lambda x: 2**x", **kwargs):
		super().__init__(**kwargs)
		self.sf = sum_function
		self.rewards = np.ones(self.max_turns)
		
	def reward(self, game, turn):
		pass
	
	def final_reward(self, game, turn):
		self.rewards = np.ones(turn+1) * np.sum(eval(self.sf)(game.board))
	
		return self.rewards
		
	def clear(self):
		self.rewards = np.ones(self.max_turns)

class EndGameScore(Rewarder):
	def reward (self, game, turn):
		pass
	def final_reward(self, game, turn):

		self.rewards= np.ones(turn+1) * game.score
		
		return self.rewards

		
class TotalScore(Rewarder):
	def __init__(self, **kwargs):
		super().__init__(**kwargs)

	
	def reward(self, game, turn):
		self.rewards[turn] = game.score
			
class BoardSum(Rewarder):
	def __init__(self, sum_function = lambda x: 2**x, **kwargs):
		super().__init__(**kwargs)
		self.sf = sum_function
		
	def reward(self, game, turn):
		self.rewards[turn] = np.sum(self.sf(game.board))

class BoardVariance(Rewarder):
	def __init__(self, corner="all", **kwargs):
		super().__init__(**kwargs)
		self.corner = corner

	def reward(self, game, turn):
		self.rewards[turn] = -game.var(self.corner)
	def __str__(self):
		return "%s: corner = %s, gamma = %s" % (self.__class__.__name__, self.corner, self.gamma)



class VarScore(BoardVariance):
	def reward(self, game, turn):
		self.rewards[turn] = -game.var(self.corner) * 2**np.max(game.board)
	def __str__(self):
		return "%s: corner = %s, gamma = %s" % (self.__class__.__name__, self.gamma, self.corner)


class DumbReward(Rewarder):
	def __init__(self, **kwargs):
		super().__init__(**kwargs)

	def reward(self, game, turn):
		self.rewards[turn] = 1

class HeuristicsReward(Rewarder):
	# Only works with n=4
	
	def __init__(self, mono_weight = .25, empty_weight = .25, mergers_weight = .25, corner_weight = .25, factor = 1, n = 4, snake_mono = True, **kwargs):	
		super().__init__(**kwargs)
		self.weights = [mono_weight, empty_weight, mergers_weight, corner_weight]
		self.n = n
		self.factor = factor
		
		self.mono_func = self._snake_mono if snake_mono else self._standard_mono
		
		
		pass
		
	def reward(self, game, turn):
		reward = self.factor * np.dot(self.weights, [self.monotonicity(game), self.mergers(game), self.empty_tiles(game), self.highest_tile_in_corner(game)])
	
		self.rewards[turn] = reward
		
	def monotonicity(self, game):
		# Returns the monotonicity og the gameboard
		mono = 0
		
		diff_board= np.diff(game.board)		
		for i, row in enumerate(diff_board):
			mono += self.mono_func(row, i)
		
		return  mono / (self.n)
	

	def mergers(self, game):
		mergers = 0 
		for row in np.vstack((game.board, game.board.T)):
			max_merges_in_a_row = 2
			for i in range(self.n - 1):
				if row[i] == row[i+1] and row[i] > 0 and max_merges_in_a_row:
					mergers += 1
					max_merges_in_a_row -= 1 
		return mergers / (8 * self.n)
		
	def empty_tiles(self, game):
		return game.zeros.size / (self.n **2-2)
	
	def highest_tile_in_corner(self, game):
		reward = 0
		if np.argmax(game.board) in (0, self.n-1, self.n * (self.n -1) , self.n**2 - 1):
			reward = 1
		return reward
	
	def _standard_mono(self, row, _):
			return int(np.all(row < 0) or np.all(row > 0))
	
	def _snake_mono(self, row, i):
			if i % 2:
				reward = int(np.all(row < 0))
			else:
				reward = int(np.all(row > 0))
			return reward
	
	def __str__(self):
		return "%s: factor = %s,  weights = %s, gamma = %s" % (self.__class__.__name__, self.factor, self.weights, self.gamma )

class MaxTileReward(Rewarder):
	def __init__(self, **kwargs):
		super().__init__(**kwargs)

	def reward(self, game, turn):
		reward=2**np.max(game.board)
		self.rewards[turn] = reward