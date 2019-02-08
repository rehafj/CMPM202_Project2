# AutoTest some stuff:
from tfRNN import trainAndGenerate
import pprint

pp = pprint.PrettyPrinter(indent=4)

modelsFolder = "models/"
pickle = ".pickle"

inputFile="lordOfTheRingsReformatted.txt"
pickle_model = "LOTR_model"
epochs = 1

string_starts = [ "Frodo ", "Sam", "When ", "It was", "As for ", "Mr. Bilbo Baggins ", "All that day "]

results = {}

for epochs in [1, 3, 5, 10, 15, 25]:
	test_name = "%s_e%s"%(pickle_model, epochs)
	pickle_model_path = modelsFolder+test_name+pickle
	trainAndGenerate(inputFile, None, None, pickle_model_path,
	                epochs=epochs, checkpoint_dir='./training_checkpoints/%s'%test_name )

	result[epochs] = {}
	for string in string_starts:
		result[epochs][string] = {}
		for temperature in [0.01, 0.1, 0.5, 1.0, 1.5, 2, 5, 10]
			result[epochs][string][temperature] = trainAndGenerate(None, pickle_model_path, string, None, temperature=temperature)

	pp.pprint(result[epochs])

print(" Final Results ")
pp.pprint(result)
