In this part, we wrote the following files:
	- bilstmTrain.py
	- bilstmPredict.py
	- part3_utils.py
	- generate_plots.py

There are two copies of them:
	- In A_C_REPR folder
	- In B_D_REPR folder

The duplicate is because repr b and d require batch with size 1 and smaller learning rate from a and c.
There is also batching of size 1 for the dev accuracy computation as well as generating the predicted file (because variable char lengths)

### bilstmTrain.py ###

For running this file, there are four options:


1. python bilstmTrain.py repr trainFile modelFile

	repr: representation option - a / b / c / d
	trainFile: input file to train on
	modelFile: file to save to the model
	
	where all the outputs names are saved difaultively with the string "bilstm" in them

2. python bilstmTrain.py repr trainFile modelFile devFile testFile

	devFile: input file to "dev" on while training
	testFile: input file to test on after the training

	where the default name for the predicting file on the test is "Part3.bilstm"

3. python bilstmTrain.py repr trainFile modelFile devFile testFile name_for_outputs

	where we change all the default "bilstm" string to name_for_outputs

4. python bilstmTrain.py repr trainFile modelFile devFile testFile name_for_outputs NER_FLAG

	where NER_FLAG is bolean flag that give you the option of masking the predicted label "O" and true label "O" for the NER assignment in order to give more chance to the model to be with diagonal confusion matrix

### bilstmPredict.py ###

Just run the following line:

python bilstmPredict.py repr modelFile inputFile
	repr: representation option - a / b / c / d
	modelFile: file to load from the model
	inputFile: input file to test on

The predicted file will be saved with the name "Part3.Predicted File"
