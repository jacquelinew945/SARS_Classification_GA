# In[ ]:
import torch
import numpy as np
import torch.nn.functional as F
import matplotlib.pyplot as plt
import preprocessing as pp
import params as p
from model import Classification
from sklearn.metrics import confusion_matrix, classification_report

# Function which runs the baseline model with arguments hidden neuron number, learning rate
def run_NN(hn, lr):
    # defining a neural network using customised structure
    net = Classification(p.input_neurons, hn, p.output_neurons)

    # defining loss function
    loss_func = torch.nn.CrossEntropyLoss()

    # defining optimiser
    optimiser = torch.optim.SGD(net.parameters(), lr=lr)

    # storing losses for visualisation
    all_losses = []
    all_val_losses = []
    accuracy_count = []
    val_accuracy_count = []
    epoch_count = []

    val_loss_penalty = 0

    print("Un-pruned")
    # training the network
    for epoch in range(p.num_epoch):
        # forward pass computing y by passing x to the model
        # pass Tensor of input data to Module and produces Tensor of output data
        # Y_pred contains four columns
        # index of max column indicates class of the instance
        Y_pred = net(pp.X)
        Y_val_pred = net(pp.X_val)

        # computing loss
        # passing Tensors containing pred and true Y
        # loss function returns a Tensor containing the loss
        loss = loss_func(Y_pred, pp.Y)
        val_loss = loss_func(Y_val_pred, pp.Y_val)

        all_losses.append(loss.item())
        all_val_losses.append(val_loss.item())

        # keeps track of when the validation loss increases so the individual in the GA receives a penalty
        if all_val_losses[epoch] > all_val_losses[epoch-1]:
            val_loss_penalty += 0.2

        # printing progress
        if epoch % 50 == 0:
            # convert four-column predicted Y values to one column for comparison
            _, predicted = torch.max(F.softmax(Y_pred, 1), 1)
            _, predicted_val = torch.max(F.softmax(Y_val_pred, 1), 1)

            # calculate and print accuracy
            total = predicted.size()
            total_val = predicted_val.size()

            correct = predicted.data.numpy() == pp.Y.data.numpy()
            correct_val = predicted_val.data.numpy() == pp.Y_val.data.numpy()

            accuracy_count.append(100 * sum(correct) / total)
            val_accuracy_count.append(100 * sum(correct_val) / total_val)

            epoch_count.append(epoch)

            print(
                f'Epoch [{epoch + 1} / {p.num_epoch}] Loss: {loss.item():.4f} Accuracy: {100 * sum(correct) / total} '
                f'Val. Loss: {val_loss.item():.4f} Val. Accuracy: {100 * sum(correct_val) / total_val}')

        # clear gradients before backward pass
        net.zero_grad()

        # perform backward pass: compute gradients of the loss
        loss.backward()

        # calling the step function on an optimiser to update parameters
        optimiser.step()

        # get final train and val accuracy to pass to GA
        if epoch == p.num_epoch-1:
            index = int(p.num_epoch / 50)-1
            train_pass = float(accuracy_count[index])
            val_pass = float(val_accuracy_count[index])
            final_val_loss = all_val_losses[p.num_epoch-1]

    # specify path to save un-pruned model
    PATH = "saved models/state_dict_model.pt"

    # saving un-pruned model
    torch.save(net.state_dict(), PATH)

    # In[ ]:

    # plotting losses of un-pruned model
    plt.figure()
    plt.title('Cross-entropy loss on the un-pruned Training set')
    plt.xlabel('epochs')
    plt.ylabel('loss')
    plt.plot(all_losses, 'r')

    plt.figure()
    plt.title('Cross-entropy loss on the un-pruned Validation set')
    plt.xlabel('epochs')
    plt.ylabel('loss')
    plt.plot(all_val_losses, 'b')

    # plotting accuracies of un-pruned model
    plt.figure()
    plt.title('Accuracy on the un-pruned Training set')
    plt.xlabel('epochs')
    plt.ylabel('loss')
    plt.plot(np.arange(len(accuracy_count)) * 50, accuracy_count, 'r')

    plt.figure()
    plt.title('Accuracy on the un-pruned Validation set')
    plt.xlabel('epochs')
    plt.ylabel('loss')
    plt.plot(np.arange(len(val_accuracy_count)) * 50, val_accuracy_count, 'b')

    plt.show()

    # In[ ]:

    # loading testing data
    # converting pandas dataframe to array
    # last column is target
    df_array_xTest = pp.X_test.to_numpy()
    df_array_yTest = pp.y_test.to_numpy()

    # creating Tensors to hold inputs and outputs
    X_test = torch.tensor(df_array_xTest, dtype=torch.float)
    Y_test = torch.tensor(df_array_yTest, dtype=torch.long)

    # testing the NN with test data
    Y_pred_test = net(X_test)

    # get prediction for test
    _, predicted_test = torch.max(F.softmax(Y_pred_test, 1), 1)

    # calculate accuracy for test
    total_test = predicted_test.size()
    correct_test = sum(predicted_test.data.numpy() == Y_test.data.numpy())

    # get test accuracy to pass to GA
    test_accuracy = 100 * correct_test / total_test
    test_pass = test_accuracy[0]

    print('Un-Pruned Testing Accuracy: %.2f %%' % test_accuracy)

    # confusion matrix for test
    pred_y_test = predicted_test.data.numpy()

    print(confusion_matrix(df_array_yTest, pred_y_test, labels=[0, 1, 2, 3]))
    print('')
    print(classification_report(df_array_yTest, pred_y_test, zero_division=1))

    # passing variables which are used in the genetic algorithm evaluation function
    return train_pass, val_pass, test_pass, final_val_loss, val_loss_penalty
