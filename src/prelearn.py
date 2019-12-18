import torch
import torch.nn as nn

import numpy as np
from time import time
from timeit import default_timer as timer

from policy import Policy

cpu = torch.device("cpu")
gpu = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

class Supervisor(Policy):
	
	def train_from_load(self, paths, n):

		if not hasattr(paths, "__iter__"):
			paths = [paths]
		
		Xs = []
		Ys = []
		for p in paths:
			l = np.load(p)
			Xs.append(l["arr_0"])
			Ys.append(l["arr_1"])
		X = torch.Tensor(np.vstack(Xs)).to(gpu)
		Y = torch.Tensor(np.vstack(Ys)).to(gpu)

		self.trainSL(X, Y, n)


def create_pretrained(loadpaths, paths, n=100, layers=[256, 100, 4], lr=1e-4):
	s = Supervisor(layers, lr=lr)
	start = timer()
	s.train_from_load(loadpaths, n)
	print("Tid: %.2f" % (timer()-start))
	s.save_me(paths["pretrained"])
	
	return s
