# AutoTest some stuff:
from tfRNN import trainAndGenerate
import pprint

pp = pprint.PrettyPrinter(indent=4)

modelsFolder = "models/"

results = {}
for inputFileModelName in [("lordOfTheRingsReformatted.txt", "LOTR_model", [ "Frodo ", "Sam", "When ", "It was", "As for ", "Mr. Bilbo Baggins ", "All that day "]),
	("Hamilton2.txt", "musical_model", [ "[HAMILTON]", "[ELIZA]", "[BURR]", "[COMPANY]", "[WASHINGTON]", "[LAFAYETTE]", "[HAMILTON/"]),
	("GameOfThrones.txt", "GOT_model", [ "His father ", "Ned grimaced ", "Now it was ", "It was", "Ben Stark ", "She saw ", "Joffrey "])]:

	inputFile=inputFileModelName[0]
	pickle_model = inputFileModelName[1]
	string_starts = inputFileModelName[2]

	results[pickle_model] = {}
	for epochs in [25, 50, 100, 500]:
		test_name = "%s_e%s"%(pickle_model, epochs)
		pickle_model_path = modelsFolder+test_name
		trainAndGenerate(inputFile, None, None,
		                epochs=epochs, checkpoint_dir=pickle_model_path )

		results[pickle_model][epochs] = {}
		for string in string_starts:
			results[pickle_model][epochs][string] = {}
			for temperature in [0.5, 1.0, 1.5]:
				results[pickle_model][epochs][string][temperature] = trainAndGenerate(inputFile, pickle_model_path, string, temperature=temperature, seq_length=10000)

		print(pickle_model + "-" + str(epochs)+":")
		pp.pprint(results[pickle_model][epochs])

	print(pickle_model+":")
	pp.pprint(results[pickle_model])

print(" Final Results ")
pp.pprint(results)
