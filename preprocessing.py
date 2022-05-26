from sklearn.model_selection import train_test_split
import pandas as pd
import matplotlib.pyplot as plt
import torch

# loading training data. uncomment out the number of features we wish to test
# the below contains the tested feature sets

# all 23 features
df = pd.read_csv('feature sets/SM_data.csv', header=0)

# only temp
# df = pd.read_csv('feature sets/SM_data_temp.csv', header=0)

# only BP
# df = pd.read_csv('feature sets/SM_data_BP.csv', header=0)

# only abdominal pains and nausea
# df = pd.read_csv('feature sets/SM_data_apn.csv', header=0)

# only systolic BP
# df = pd.read_csv('feature sets/SM_data_BP_sys.csv', header=0)

# only diastolic BP
# df = pd.read_csv('feature sets/SM_data_BP_dia.csv', header=0)

# only one med temp, med systolic bp and nausea
# df = pd.read_csv('feature sets/SM_data_temp_bp_naus.csv', header=0)

# only one med temp
# df = pd.read_csv('feature sets/SM_one_temp.csv', header=0)

# df = pd.read_csv('feature sets/SM_data_temp_naus.csv', header=0)

# In[ ]:

# convert string target to numeric values
#   class 0: HighBP
#   class 1: Normal
#   class 2: Pneumonia
#   class 3: SARS
df.at[df['Class'] == 'HighBP', ['Class']] = 0
df.at[df['Class'] == 'Normal', ['Class']] = 1
df.at[df['Class'] == 'Pneumonia', ['Class']] = 2
df.at[df['Class'] == 'SARS', ['Class']] = 3

# convert all string numeric values to int ['2' -> 2]
df = df.apply(pd.to_numeric)

# In[ ]:

# using sklearn to create a test-train-split
# create training and testing vars
# x is everything but the last target label
# y is the last column target label
x = df.iloc[0:, :-1]
y = df.iloc[:, -1]
# split dataset in half
NN_xVal, P_xVal, NN_yVal, P_yVal = train_test_split(x, y, test_size=0.5, stratify=y)
# split train-test for initial model phase
X_train, X_test, y_train, y_test = train_test_split(NN_xVal, NN_yVal, test_size=0.2, stratify=NN_yVal)
# creating the validation set
X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=0.2, stratify=y_train)

# split train-test for pruning phase
X_trainP, X_testP, y_trainP, y_testP = train_test_split(P_xVal, P_yVal, test_size=0.2, stratify=P_yVal)

# visualize distribution of training and testing data after split
# extract frequency of each class for train set
class_freq = y_train.value_counts()
class_freq = list(class_freq.sort_index())

# x-axis labels and length
x_axis = ('HighBP', 'Normal', 'Pneumonia', 'SARS')

# plot train set
graph = plt.bar(x_axis, class_freq)
plt.xticks(x_axis)
plt.ylim([200, 450])
plt.ylabel('Frequency')
plt.xlabel('Classes')
plt.title('Training Data Frequencies')

plt.show()

# extract frequency of each class for test set
class_freq_test = y_test.value_counts()
class_freq_test = list(class_freq_test.sort_index())

# plot test set
graph_test = plt.bar(x_axis, class_freq_test)
plt.xticks(x_axis)
plt.ylim([50, 150])
plt.ylabel('Frequency')
plt.xlabel('Classes')
plt.title('Testing Data Frequencies')

plt.show()

# converting pandas dataframe to array
# last column is target
df_array_xTrain = X_train.to_numpy()
df_array_yTrain = y_train.to_numpy()

# creating Tensors to hold inputs and outputs
X = torch.tensor(df_array_xTrain, dtype=torch.float)
Y = torch.tensor(df_array_yTrain, dtype=torch.long)
X_val = torch.tensor(X_val.to_numpy(), dtype=torch.float)
Y_val = torch.tensor(y_val.to_numpy(), dtype=torch.long)

# In[ ]:

# converting pandas dataframe to array
df_array_xTrainP = X_trainP.to_numpy()
df_array_yTrainP = y_trainP.to_numpy()

# creating Tensors to hold inputs and outputs
XP = torch.tensor(df_array_xTrainP, dtype=torch.float)
YP = torch.tensor(df_array_yTrainP, dtype=torch.long)

# loading testing data
# converting pandas dataframe to array
# last column is target
df_array_xTestP = X_testP.to_numpy()
df_array_yTestP = y_testP.to_numpy()

# creating Tensors to hold inputs and outputs
X_testP = torch.tensor(df_array_xTestP, dtype=torch.float)
Y_testP = torch.tensor(df_array_yTestP, dtype=torch.long)
