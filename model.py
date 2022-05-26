import torch

# linear classification model with 3 layers
# input layer of n_input
# hidden layer of n_hidden neurons
# output layer of n_output
class Classification(torch.nn.Module):

    def __init__(self, n_input, n_hidden, n_output):
        super(Classification, self).__init__()
        # define linear hidden layer output
        self.hidden = torch.nn.Linear(n_input, n_hidden)
        # define linear output layer output
        self.out = torch.nn.Linear(n_hidden, n_output)

    def forward(self, x):
        # get hidden layer input
        h_input = self.hidden(x)
        # defining Sigmoid activation function for hidden layer
        h_output = torch.sigmoid(h_input)
        # getting output layer output
        y_pred = self.out(h_output)

        return y_pred
