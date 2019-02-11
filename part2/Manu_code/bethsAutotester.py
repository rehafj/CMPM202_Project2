import train
import test
import utils
import os

for filterNum in [64]:
	for kernalSize in [5]:
		utils.changeCKPT_DIR('./Checkpoints/filter-%s_kernal-%s/'%(filterNum, kernalSize))
		train.loadAndTrain(filters=filterNum, kernal_size=kernalSize)

		counter = 0
		for file in [f for f in os.listdir(mypath) if os.path.isfile(os.path.join(mypath, f))]:
			test(file, resultFoulder="filter-%s_kernal-%s"%(filterNum, kernalSize), outputNumber=counter)
			counter+=1
