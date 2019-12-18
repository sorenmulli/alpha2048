import numpy as np
from copy import copy, deepcopy


class Game2048:
	
	def __init__(self, n=4, deterministic=[False, False], scorefunc=lambda x: 2*2**x):
		
		# Opretter spillebræt
		self.n = n
		self.deterministic = deterministic  # Placering, værdi
		self.scorefunc = scorefunc
		self.board = np.zeros((n, n), dtype = np.int)
		
		self.score = 0 
		self.bsum = 0
		self.moves = 0
		self.propermoves = 0
		#Initialisering af brættet første gang, hvor 2 startbrikker skal tilføjes
		for i in range(2):
			# Nx2-matrix med koordinater til alle 0-elementerne på brættet
			self.zeros = np.where(self.board == 0)
			self.zeros = np.transpose(self.zeros)
		
			idx, newval = self._getnew(self.zeros)
			self.bsum += newval
			
			self.board[idx[0], idx[1]] = newval
		
		self.zeros = np.where(self.board == 0)
		self.zeros = np.transpose(self.zeros)

		# 2xn-matrix med alle brættets indeksværdier
		# Bruges i self.var
		self.allidx = self._coordlist()

	def _coordlist(self):

		"""
		Returnerer en 2xn-matrix med alle brættets indeksværdier
		"""

		idx = np.concatenate([[np.ones(self.n)*x, np.arange(self.n)] for x in range(self.n)], axis=1)
		
		return idx
	
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
		
		self.board, zeros, change, self.score, dsum = self.move(self.board, direction)
		self.zeros = zeros
		self.bsum += dsum
		self.moves += 1
		if change in (1, 2):
			self.propermoves += 1
		
		return change
	
	def move(self, board, direction: int, evaluate = True):
		
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
		
		board = np.copy(board)
		score = copy(self.score)

		# Brætmatricen ændres, så der altid foretages en venstreforskydning
		board = np.rot90(board, direction)
		
		# Looper gennem rækker/søjler og foretager træk
		for i in range(self.n):
			a = board[i][board[i] != 0]
			for k in range(a.size-1):
				# Trækker sammen, hvis samme værdi ved siden af hinanden
				if a[k] == a[k+1] and a[k] != 0:
					score += self.scorefunc(a[k])
					a[k] += 1
					a[k+1:] = self._shift(a[k+1:])
			# Fylder nuller ind til sidst
			r = np.zeros(self.n)
			r[:a.size] = a
			board[i] = r
		
		# Roterer brætmatricen tilbage til udgangspunktet
		board = np.rot90(board, 4-direction)
		
		# Tjekker, om en ændring er sket
		zeros = np.where(board==0)
		zeros = np.transpose(zeros)

		if (board != self.board).any():
			change = 1
			idx, newval = self._getnew(zeros)
			dsum = newval
			board[idx[0], idx[1]] = newval
		else:
			change = 0
			dsum = 0
	
		zeros = np.where(board==0)
		zeros = np.transpose(zeros)

		#Undersøger, om spilleren har tabt spillet	
		if zeros.size == 0 and evaluate:
			inichange = copy(change)
			# print(inichange, self)
			change = 2
			try:
				for row in np.vstack((board, board.T)):
					for i in range(self.n - 1):
						if row[i] and (row[i] == row[i+1]):
							# print("XXX", change, inichange)
							change = inichange
							raise IndentationError
				# print(board)
			except IndentationError:
				# print("INDENTATIONERROR")
				pass

		return board, zeros, change, score, dsum
	
	def _getnew(self, z):
		
		"""
		Genererer et nyt 1 eller 2 et sted i 0-koordinatvektoren z
		"""

		if self.deterministic[0]:  # Placering
			idx = z[0]
		else:
			zidx = np.random.randint(z.shape[0])
			idx = z[zidx]
		
		val = 1 if self.deterministic[1] else np.random.choice([1, 2], p=[.9, .1])
		
		return idx, val
	
	def var(self, corner=None):

		"""
		Beregner et variansmål for brættet. Hvis store brikker er langt fra hinanden, er variancen større
		corner angiver hjørnepræference
			None: Går ikke efter hjørner
			0: Øverst tv
			1: Øverst th
			2: Nederst th
			3: Nederst th
			"all": Går efter nærmeste hjørne
		"""

		flatboard = 2**self.board.reshape(self.n**2) / self.bsum
		corners = self.allidx.T[[0, 3, -1, -4]]
		if corner is None:
			# Massemidtpunkt
			pref = np.dot(self.allidx, flatboard)
		elif type(corner) is int:
			# Valgt hjørne
			pref = corners[corner]
		elif corner == "all":
			pref = np.dot(self.allidx, flatboard)
			dists = np.empty(4)
			for i in range(4):
				dists[i] = np.linalg.norm(pref-corners[i])
			pref = corners[np.argmin(dists)]

		# Magisk formel
		rs = np.linalg.norm(self.allidx.T-pref, axis=1)
		return np.dot(rs, flatboard)


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
	
	def __str__(self):

		return "\n".join((
			"Score: %i" % self.score,
			str(self.board)
		))


# g = Game2048()
# g.board = np.array([
# 	[1,3,1,6],
# 	[2,1,4,7],
# 	[3,4,2,1],
# 	[0,0,0,0]
# ])
# g.zeros = np.transpose(np.where(g.board==0))
# g.play(3)
# print(g)

