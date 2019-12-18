from os import chdir, system
from os.path import dirname, realpath
chdir(realpath(dirname(__file__)))

from sys import argv
from threading import Thread

# Definerer hvilken fil
files = {"main": "main", "cpd": "create_pretrain_data"}
if len(argv) == 2:
	if argv[1] in files:
		f = files[argv[1]]
	else:
		f = argv[1]
else:
	f = "main"

# Definerer antal kørsler
if f == "main":
	from main import schedules
	threads = len(schedules)
else:
	threads = 4

def scheduler():
	for i in range(threads):
		x = lambda: system("python %s.py %i" % (f, i))
		Thread(target=x).start()

if __name__ == "__main__":

	scheduler()


