import torch
from random import choice




class Game2048:

	def __init__(self, n=4, deterministic=False):

		# Opretter spillebræt
		self.n = n
		self.deterministic = deterministic
		self.board = torch.zeros(self.n, self.n, dtype=torch.int)

		self.score = 0
		#Initialisering af brættet første gang, hvor 2 startbrikker skal tilføjes
		for i in range(2):
			# 2xn-matrix med koordinater til alle 0-elementerne på brættet
			self.zeros = self._getindices(self.board, 0)
			idx, newval, zidx = self._getnew(self.zeros)
			self.board[idx[0], idx[1]] = newval
		self.zeros = self.zeros[self.zeros != idx]
	
	def play(self, direction):

		pass
	
	@staticmethod
	def _getindices(t, n=0):

		size = t.size()[0]

		if len(t.size()) > 1:
			t = t.view(t.size()[0] * t.size()[1])

		zs = (t==0).nonzero()
		indices = torch.empty(2, len(zs))
		for i in range(len(zs)):
			n = zs[i] // size
			m = zs[i] - size * n
			indices[:, i] = torch.Tensor([n, m])
		
		return torch.transpose(indices, 0, 1)
	
	def _getnew(self, z):

		"""
		Returnerer et indeks fra z at indsætte en værdi, værdien (1 eller 2)
		og det valgte indeks i z
		"""

		if self.deterministic:
			return z[0], 1, 0
		
		zidx = int(torch.randint(len(self.zeros), (1,)))
		idx = list(self.zeros[zidx, :])
		val = 1 if int(torch.randint(10, (1,))) < 9 else 2

		return idx, val, zidx

t = torch.Tensor([[1,0,3],[4,5,6],[1,0,0]])
g = Game2048()
print(g._getindices(t))


