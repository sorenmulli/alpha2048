from os import chdir
from os.path import dirname, realpath
chdir(realpath(dirname(__file__)))

import matplotlib.pyplot as plt
import numpy as np

from utilities import rollingavg, bootstrap


def dataload(fname, markline="Score\tMaxtile"):

	"""
	fname: Name of log file to load in ../results/
	markline: The content of the line, after which the data starts
	"""
	
	fname = realpath("../results/" + fname)
	
	# Gets index of markline
	with open(fname) as f:
		lines = f.read().splitlines()
		mark_idx = lines.index(markline)
	
	# Loads data
	s, m = np.genfromtxt(fname, skip_header=mark_idx+1, unpack=True)

	return s, m

def multiload(*fnames):
	
	s = []
	m = []
	markline = "Score\tMaxtile"
	for f in fnames:
		f += "_log_eval.txt"
		s_, m_ = dataload(f, markline)
		s.append(s_)
		m.append(m_)
	s = np.concatenate(s)
	m = np.concatenate(m)
	return s, m

def postAnalysis(s, name):
	boot_mean, boot_std = bootstrap(s)
	
	mean = np.mean(s)
	
	conf_int = 1.96 * boot_std / np.sqrt(len(s))
	
	print("%s:\nMean: %s\nB_Mean: %s\nB_conf:%s\n\n" % (name, mean, boot_mean, conf_int))

def analysis(*fnames, agent):
	s, m = multiload(*fnames)
	mean = s.mean()
	mean_std = np.std(means(s))
	file_out = open("../latex/rapport/evals/%s.txt" % agent, "w+", encoding="utf-8")
	file_out.write("Agent: %s\n" % agent)
	file_out.write("Antal spil: %i\n" % s.size)
	file_out.write("Middelværdi: %.2f\n" % mean)
	file_out.write("Middelværdi std: %.2f\n" % mean_std)
	file_out.write("Maxtilefordeling:\n")
	mtiles, _ = np.unique(m, return_counts=True)
	mtiledist = _ / _.sum() * 100
	templ1 = "%i: %.2f\\pro\\\\"
	templ2 = "%i: %.1f\\pro\\\\"
	strs = ""
	for i in range(mtiles.size):
		if mtiledist[i] >= 9.995:
			mtiledist[i] = np.round(mtiledist[i], 1)
			strs += templ2 % (mtiles[i], mtiledist[i])
		else:
			mtiledist[i] = np.round(mtiledist[i], 2)
			strs += templ1 % (mtiles[i], mtiledist[i])
	file_out.write(strs+"\n")
	file_out.write("Fejlmargin: %.2f\n" % (1.96 * mean_std / np.sqrt(means(s).size)))
	file_out.close()



def means(scores, batchsize=50):

	np.random.shuffle(scores)
	s = scores[:scores.size - scores.size % batchsize]
	s = s.reshape((s.size//batchsize, batchsize))
	means = s.mean(axis=1)
	
	return means


def histplot(scores, agent):
	
	# if not hasattr(fname, "__iter__") or type(fname) == str:
	# 	fname = [fname]
	if not hasattr(agent, "__iter__") or type(agent) == str:
		agent = [agent]
	# markline = "Score\tMaxtile"
	# scores = []
	# maxtiles = []
	# for fname in fname:
	# 	s, m = dataload(fname, markline)
	# 	scores.append(s)
	# 	maxtiles.append(np.log2(m))
	n = int(np.sqrt(scores[0].size))
	agent = ["Agent: " + a for a in agent]

	plt.subplot(121)
	plt.title("Score distribution")
	for i in range(len(scores)):
		plt.hist(scores[i], alpha=.5, bins=n, label=agent[i])
	plt.legend(loc=2)
	plt.xlabel("Score")
	plt.ylabel("Number of games")

	plt.subplot(122)
	plt.title("Score mean distribution")
	for i, s in enumerate(scores):
		m = means(s)
		plt.hist(m, alpha=.5, label=agent[i])
	plt.legend(loc=2)
	plt.xlabel("Score means")
	plt.ylabel("Number of means")
	
	plt.show()


def trainplot(fname, agent):
	markline = "Gennemsnitsscore\tMaxtile"
	scores, maxtiles = dataload(fname, markline)
	n = scores.size
	xcoords = np.arange(n)
	n = int(np.sqrt(scores.size))
	agent = "Agent: " + agent

	# plt.subplot(121)
	plt.title("Average score by batch")
	plt.plot(xcoords, scores, ".", label=agent)
	plt.plot(*rollingavg(scores, n), label="Rolling average", linewidth=3)
	plt.legend(loc=2)
	plt.xlabel("Batch")
	plt.ylabel("Average score")

	# plt.subplot(122)
	# plt.title("Average max tile by batch")
	# plt.plot(xcoords, maxtiles, ".", label=agent)
	# plt.plot(*rollingavg(maxtiles, n), label="Rolling average", linewidth=3)
	# plt.legend(loc=2)
	# plt.xlabel("Batch")
	# plt.ylabel("Average max tile")

	plt.show()

plots = (
	{
		"func": histplot,
		"scores": [multiload("0123-080054_t0", "0123-102908_t0")[0], multiload("0121-114018_t0", "0122-222025_t0")[0], multiload("0121-114013_t0", "0122-221626_t1")[0]],
		"agent": ["Random moves", "Score (simple)", "Variance (complex)"]
	},
	# {
	# 	"func": histplot,
	# 	"scores": [multiload("0121-114018_t0", "0122-222025_t0")[0]],
	# 	"agent": "Score (simple)"
	# },
	# {
	# 	"func": histplot,
	# 	"scores": [multiload("0121-114018_t1", "0122-222025_t1")[0]],
	# 	"agent": "Variance (simple)"
	# },
	# {
	# 	"func": histplot,
	# 	"scores": [multiload("0122-142914_t0", "0122-222025_t2")[0]],
	# 	"agent": "Heuristics (simple)"
	# },
	# {
	# 	"func": histplot,
	# 	"scores": [multiload("0122-143020_t0", "0122-222025_t3")[0]],
	# 	"agent": "Combination (simple)"
	# },
	# {
	# 	"func": histplot,
	# 	"scores": [multiload("0121-114149_t1", "0122-221626_t0")[0]],
	# 	"agent": "Score (complex)"
	# },
	# {
	# 	"func": histplot,
	# 	"scores": [multiload("0121-114013_t0", "0122-221626_t1")[0]],
	# 	"agent": "Variance (complex)"
	# },
	# {
	# 	"func": histplot,
	# 	"scores": [multiload("0122-123731_t0", "0122-221626_t2")[0]],
	# 	"agent": "Heuristics (complex)"
	# },
	# {
	# 	"func": histplot,
	# 	"scores": [multiload("0122-142833_t0", "0122-221626_t3")[0]],
	# 	"agent": "Combination (complex)"
	# },
	##########
)

def run():
	for p in plots:
		p["func"](scores=p["scores"], agent=p["agent"])
		# s, m = dataload(p["fname"], "Score\tMaxtile")
		# postAnalysis(s, p["agent"])

if __name__ == "__main__":
	run()
	# analysis("0121-114018_t0", "0122-222025_t0", agent="Score (simple)")
	# analysis("0121-114018_t1", "0122-222025_t1", agent="Variance (simple)")
	# analysis("0122-142914_t0", "0122-222025_t2", agent="Heuristics (simple)")
	# analysis("0122-143020_t0", "0122-222025_t3", agent="Combination (simple)")
	# analysis("0121-114149_t1", "0122-221626_t0", agent="Score (complex)")
	# analysis("0121-114013_t0", "0122-221626_t1", agent="Variance (complex)")
	# analysis("0122-123731_t0", "0122-221626_t2", agent="Heuristics (complex)")
	# analysis("0122-142833_t0", "0122-221626_t3", agent="Combination (complex)")
	# analysis("0123-080054_t0", "0123-102908_t0", agent="Tilfældige træk")



	# {
	# 	"func": trainplot,
	# 	"fname": "0120-170133_t0_log_train.txt",
	# 	"agent": "Score (simple)"
	# },
	# {
	# 	"func": trainplot,
	# 	"fname": "0120-170133_t2_log_train.txt",
	# 	"agent": "Variance (simple)"
	# },
	# {
	# 	"func": trainplot,
	# 	"fname": "0122-103804_t1_log_train.txt",
	# 	"agent": "Heuristics (simple)"
	# },
	# {
	# 	"func": trainplot,
	# 	"fname": "0122-103804_t2_log_train.txt",
	# 	"agent": "Combination (simple)"
	# },
	# {
	# 	"func": trainplot,
	# 	"fname": "0121-082659_t0_log_train.txt",
	# 	"agent": "Score (complex)"
	# },
	# {
	# 	"func": trainplot,
	# 	"fname": "0121-082659_t1_log_train.txt",
	# 	"agent": "Variance (complex)"
	# },
	# {
	# 	"func": trainplot,
	# 	"fname": "0122-092917_t1_log_train.txt",
	# 	"agent": "Heuristics (complex)"
	# },
	# {
	# 	"func": trainplot,
	# 	"fname": "0122-103816_t0_log_train.txt",
	# 	"agent": "Combination (complex)"
	# },
