The code requires these modules:
numpy, deap, torch, sklearn, random, matplotlib, pandas

NN.py contains the function for the base model which it constructs using the model class contained in model.py
retrain.py contains the function for the pruned model and loads the state of the model created by NN.py

*** Due to the dataset being private, it has been removed in this repository ***



WHAT TO RUN:

(NOTE: THESE TAKE A LONG TIME TO RUN. TO TEST FOR THEORY, SET population_size AND number_of_generations TO 5)

	(1) evolutionary_algorithm.py should be run first to obtain hyperparameters for the base model. This file runs NN.py 
	    for the defined population and generations to select the best individual and saves it to base_parameters.txt. 
	(2) evolutionary_algorithm_retrain.py is run next and uses the parameters found from (1) to run another GA on a pruned model.
	    The pruned model always uses the same state dictionary from the last run of the NN.py function which may be either manually run 
	    or run by the evolutionary_algorithm.py file. These hyperparameters are saved to prune_parameters.txt

To test for reproducibility, the model functions from NN.py and retrain.py can be run separately in the console with user-given arguments
to test the same parameters multiple times for multiple sets of randomly split data.






Citation for the dataset:
Mendis, B. S., Gedeon, T. D., & Koczy, L. T. (2005). Investigation of aggregation in fuzzy signatures, in Proceedings, 3rd International Conference on Computational Intelligence, Robotics and Autonomous Systems, Singapore.

http://users.cecs.anu.edu.au/~Tom.Gedeon/pdfs/Investigation%20of%20Aggregation%20in%20Fuzzy%20Signatures.pdf
