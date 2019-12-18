import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim 

import numpy as np

class QNN(nn.Module):
	def __init__(self, lr):
		super(QNN, self).__init__()
		self.fc1 = nn.Linear(256, 512)
		self.fc2 = nn.Linear(512, 6)

		self.optimizer = optim.RMSprop(self.parameters(), lr = lr)
		self.loss = nn.MSELoss()

	def forward(self, state):
		state = torch.Tensor(state)
		state = F.relu(self.fc1(state))
		actions = F.relu(self.fc2(state))
		return actions

class Agent():
	def __init__(self, gamma, epsilon, lr, max_memory, min_epsilon=.05, replace=10000, action_space = [0, 1, 2, 3]):
		self.gamma = gamma
		self.epsilon = epsilon
		self.min_epsilon = min_epsilon

		self.action_space = action_space
		self.max_memory = max_memory

		self.steps = 0
		self.learn_step_counter = 0

		self.memory = list()
		self.memory_stored = 0
		
		# self.replace = replace

		self.Q_eval = QNN(lr)
		self.Q_next = QNN(lr)

	def store_transition(self, state, action, reward, state_next):
		if self.memory_stored < self.max_memory:
			self.memory.append([state, action, reward, state_next])
		else:
			self.memory[self.memory_stored % self.max_memory] = [state, action, reward, state_next]
		
		self.memory_stored += 1

	def choose_action(self, observation):
		
		rand = np.random.random()
		actions = self.Q_eval.forward(observation)
		
		
		#Epsilon-greedy
		if rand < 1 - self.epsilon:
			action = torch.argmax(actions).item()
		else:
			action = np.random.choice(self.action_space)
		self.steps += 1

		return action
	
	def learn(self, batch_size):
		self.Q_eval.optimizer.zero_grad()

		# if self.replace is not None and \
		# 	self.learn_step_counter % self.replace == 0:
		# 	self.Q_next.load_state_dict(self.Q_eval.state_dict())

		if self.memory_stored + batch_size < self.max_memory:
			memory_start = int(np.random.choice(range(self.memory_stored)))
		else:
			memory_start = int(np.random.choice(range(self.memory_stored-batch_size-1)))
		
		mini_batch = self.memory[memory_start: memory_start + batch_size]

		memory = np.array(mini_batch)

		Qpred = self.Q_eval.forward(list(memory[:, 0]))
		Qnext = self.Q_next.forward(list(memory[:, 3]))

		max_action = torch.argmax(Qnext, dim =1)
		rewards = torch.Tensor(list(memory[:, 2]))

		Qtarget = Qpred 
		Qtarget[:, max_action] = rewards + self.gamma*torch.max(Qnext)

		if self.steps > 500:
			if self.epsilon  - 1e-4 > self.min_epsilon:
				self.epsilon - 1e-4
			else:
				self.epsilon = self.min_epsilon
		
		loss = self.Q_eval.loss(Qtarget, Qpred)
		loss.backward()

		self.Q_eval.optimizer.step()
		self.learn_step_counter += 1