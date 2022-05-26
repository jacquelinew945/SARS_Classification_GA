import preprocessing as pp
import numpy as np

# contains parameters to be passed to the models
# number of epochs, size of our input neurons according to data read in the .csv files
# output neurons corresponding to the number of class labels
num_epoch = 150
input_neurons = pp.df_array_xTrain.shape[1]
output_neurons = np.unique(pp.df_array_yTrain).size
