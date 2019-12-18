import numpy as np
import torch
from torch.autograd import Variable

from glob import glob
from os import chdir
from os.path import dirname, realpath
from sys import argv
from datetime import datetime
chdir(realpath(dirname(__file__)))

try:
	from git import Repo
except (ImportError, ModuleNotFoundError):
	print("Installer gitpython, din fisk")

import algorithms as alg
from game import Game2048
from utilities import evalplot, trainplot, Agent, max_tile_distribution, bootstrap
from learner import PolicyLearner2048
from policy import Policy
from prelearn import create_pretrained
import reward_functions as rf

# Defines thread
if len(argv) == 2:
	THREAD = int(argv[1])
else:
	THREAD = 0

# Defines paths
resultdir = realpath("../results") + "/"
create_time = datetime.now().strftime('%m%d-%H%M%S')
stdpath = "{dir}{time}_t{thread}_".format(dir=resultdir, time=create_time, thread=THREAD)
PATHS = {
	"plot_train": stdpath + "plot_train.png",
	"plot_eval": stdpath + "plot_eval.png",
	"log_train": stdpath + "log_train.txt",
	"log_eval": stdpath + "log_eval.txt",
	"model": stdpath + "model.pt",
	"pretrained": realpath("../pretrained_models") + "/%s_t%i_pretrained.pt" % (create_time, THREAD)
}

def run_evaluation(paths, agent, evals=10000, with_show=True):

	# Evaluates an agent

	threadstr = "T%i" % THREAD
	scores = np.empty(evals)
	maxtiles = np.empty(evals)

	agentstr = "Agent: %s\n" % agent.displayname
	file_out = open(paths["log_eval"], "w+", encoding="utf-8")
	file_out.write(agentstr)
	file_out.close()
	
	for i in range(evals):
		# Starts new game
		game = Game2048()
		change = 0
		
		while change != 2:
			choice = agent.make_a_move(game)
			change = game.play(choice) 
		
		scores[i] = game.score
		maxtiles[i] = 2 ** np.max(game.board)
		print(threadstr, i, game.score, 2**np.max(game.board))

	# String with evaluation results
	resstr = "Gns. score: {0}, std. på score: {1}\nMaxtile: {2}, gns. maxtile: {3}\nFord. af maxtile: {4}".format(
	int(np.mean(scores)), int(np.std(scores)),
	int(np.max(maxtiles)), round(np.mean(maxtiles), 2),
	max_tile_distribution(maxtiles))
	print(threadstr, resstr)

	# Bootstrap statistics
	boot_mu, boot_std = bootstrap(scores)
	boot_str = "BOOTSTRAP: Gns. score: %i, std. på score: %i" % (boot_mu, boot_std)
	
	print(threadstr, boot_str)
	
	# Writes log file
	file_out = open(paths["log_eval"], "a", encoding="utf-8")
	file_out.write(resstr+"\n"+boot_str + "\n")
	file_out.write("Score\tMaxtile\n")
	for s, m in zip(scores, maxtiles):
		file_out.write("%i\t%i\n" % (s, m))
	file_out.close()

	# Creates plot
	evalplot(scores, maxtiles, agent, paths["plot_eval"], with_show=with_show)


def run_training(paths, rewarder=rf.ScoreChange(), Nparams={},
		trainparams={}, load=None, evals=0, with_plots=True):
	# Trains an agent

	# Creates a learner object
	N = PolicyLearner2048(**Nparams, thread=THREAD)
	
	# Loads a previously saved model
	if load:
		N.load_model(load)
	
	# Does the actual training
	rundata = N.train(**trainparams, rewarder=rewarder, save_model=paths["model"])
	
	# Logs the results of the training
	file_out = open(paths["log_train"], 'w+', encoding="utf-8")
	try:
		repo = Repo(search_parent_directories=True)
		sha = repo.head.object.hexsha
		gitmsg = "Kørt på commit-sha %s.\n\n" % sha
	except:
		gitmsg = "Ukendt commit-sha.\n\n"
	file_out.write(gitmsg)
	n = int(np.sqrt(N.params["nbatches"] - N.params["startbatch"])) // 2
	file_out.write(
		"Parametre:" + "\n".join(["%s: %s" % (kw, N.params[kw]) for kw in N.params]) + "\n\n"
	)
	file_out.write("Gennemsnitsscore\tMaxtile\n")
	for s, m in zip(rundata["scoremeans"], rundata["maxtilemeans"]):
		file_out.write("%i\t%i\n" % (s, m))
	file_out.close()

	# Plots training results
	trainplot(N, rundata, n, paths["plot_train"], with_show=with_plots)

	# Evaluates agent if desired
	if evals:
		agent = Agent()
		agent.from_saved_model(paths["model"])
		run_evaluation(paths, agent, evals, with_show=with_plots)


def run(kwargs, with_plots=False, mode="train"):

	# Starts a job
	if mode == "train":
		run_training(PATHS, **kwargs, with_plots=with_plots)
	elif mode == "test":
		run_evaluation(PATHS, **kwargs, with_show=with_plots)

# All jobs to be run
schedules = (
	{
		"agent": Agent("../results/0121-082659_t0_model.pt"),
		"evals": 25000
	},
	{
		"agent": Agent("../results/0121-082659_t1_model.pt"),
		"evals": 25000
	},
	{
		"agent": Agent("../results/0122-103804_t2_model.pt"),
		"evals": 25000
	},
	{
		"agent": Agent("../results/0122-103816_t0_model.pt"),
		"evals": 25000
	},
)

# Starts the thread called from threader.py
if __name__ == "__main__":
	
	run(
		schedules[THREAD],
		with_plots=False,
		mode="test"
	)



