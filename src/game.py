import numpy as np
from copy import copy


class Game2048:
	
	def __init__(self, n=4, deterministic = False):
		
		# TODO Afslut spil
		
		# Opretter spillebræt
		self.n = n
		self.deterministic = deterministic
		#self.board = np.random.randint(0, 5, (n, n))
		# self.board = np.array([
		# 	[4, 18, 5, 6],
		# 	[1, 1, 29, 4],
		# 	[44, 8, 9, 10],
		# 	[11, 12, 13, 14]
		# ])
		self.board = np.zeros((n, n), dtype = np.int)
		
		self.score = 0 
		#Initialisering af brættet første gang, hvor 2 startbrikker skal tilføjes
		for i in range(2):
			# Nx2-matrix med koordinater til alle 0-elementerne på brættet
			self.zeros = np.where(self.board == 0)
			self.zeros = np.transpose(self.zeros)
		
			idx, newval = self._getnew(self.zeros)
			self.board[idx[0], idx[1]] = newval
		self.zeros = self.zeros[self.zeros != idx]

	
	def play(self, direction: int):
		
		"""
		Foretager et ryk på spilbrættet board baseret på direction:
		0: Venstre
		1: Op
		2: Højre
		3: Ned
		Ændrer self.board
		Returnerer 2 for tabt spil, 1 for ændret plade og 0 for uændret plade
		"""
		
		self.board, zeros, change, self.score = self.move(self.board, direction)
		self.zeros = zeros
		
		return change
	
	def move(self, board, direction: int, evaluate = True, scoreFunc = lambda x: 2*x**2):
		
		"""
		Foretager et ryk i spillet baseret på direction:
		0: Venstre
		1: Op
		2: Højre
		3: Ned
		Returnerer brætmatricen
		Returnerer en vektor med 0-koordinater
		Returnerer 2 for tabt spil, 1 for ændret plade og 0 for uændret plade
		Returnerer scoren
		"""
		
		# Tjekker inputvaliditet
		if direction not in range(4):
			raise IndexError("Trækket skal være et tal 0-3, men der blev angivet %s" % direction)
		
		iniboard = board.copy()
		score = copy(self.score)

		# Brætmatricen ændres, så der altid foretages en venstreforskydning
		board = np.rot90(board, direction)
		
		# Looper gennem rækker/søjler og foretager træk
		for i in range(self.n):
			a = board[i][board[i] != 0]
			for k in range(a.size-1):
				# Trækker sammen, hvis samme værdi ved siden af hinanden
				if a[k] == a[k+1] and a[k] != 0:
					a[k] += 1
					a[k+1:] = self._shift(a[k+1:])
					score += scoreFunc(a[k])
			# Fylder nuller ind til sidst
			r = np.zeros(self.n)
			r[:a.size] = a
			board[i] = r
		
		# Roterer brætmatricen tilbage til udgangspunktet
		board = np.rot90(board, 4-direction)
		
		# Tjekker, om en ændring er sket
		zeros = np.where(self.board==0)
		zeros = np.transpose(zeros)
		if (board != iniboard).any():
			change = 1
			idx, newval = self._getnew(zeros)
			board[idx[0], idx[1]] = newval
			zeros = zeros[zeros != idx]
		else:
			change = 0

		
		#Undersøger, om spilleren har tabt spillet	
		if zeros.size == 0 and evaluate:
			b = 0
			for i in range(4):
				b += self.move(board, i, evaluate = False)[2]
			if b == 0:
				change = 2
		
		return board, zeros, change, score
	
	def _getnew(self, z):
		
		"""
		Genererer et nyt 1 eller 2 et sted i 0-koordinatvektoren z
		"""
		if self.deterministic:
			return z[0], 1
			
		zidx = np.random.randint(z.shape[0])
		idx = z[zidx]
		val = np.random.choice([1, 2], p=[.9, .1])
		
		return idx, val
			
	@staticmethod
	def _shift(v):
		
		"""
		Forskyder en vektor v med 1 element mod venstre. Tomme pladser udfyldes med 0
		Taget herfra:
		https://stackoverflow.com/questions/30399534/shift-elements-in-a-numpy-array
		"""
		
		e = np.empty_like(v)
		e[-1] = 0
		e[:-1] = v[1:]
		return e
