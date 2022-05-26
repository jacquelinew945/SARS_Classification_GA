import torch
import matplotlib.pyplot as plt
import torch.nn.functional as F
import params as p
from sklearn.metrics import confusion_matrix, classification_report
from model import Classification
import preprocessing as pp
from distinctiveness import pruning_weights
from NN import run_NN

# In[ ]:

# function to run the pruned model with arguments learning rate, min angle, max angle
def retrain_model(r_lr, ga_min_angle, ga_max_angle):
    # read best parameters from saved text file made by evolutionary_algorithm.py
    f = open("saved params/base_parameters.txt", "r")

    read_params = f.read()
    hyper_params = read_params.split(", ")
    hyper_params[0] = int(hyper_params[0].replace('[', ''))
    hyper_params[1] = float(hyper_params[1].replace(']', ''))

    best_hn = hyper_params[0]
    best_lr = hyper_params[1]

    # run model with GA chosen parameters which saves a new state dict of weights for these parameters for retraining
    run_NN(best_hn, best_lr)

    # load baseline model
    model = Classification(p.input_neurons, best_hn, p.output_neurons)
    model.load_state_dict(torch.load('saved models/state_dict_model.pt'))
    model.train()

    # prune the weights and bias of old model according to the distinctiveness technique
    new_tuple = pruning_weights(model.hidden.weight.data, model.hidden.bias.data, ga_min_angle, ga_max_angle)

    # set new model's hidden layer size to the size of the pruned network
    hidden_size = len(new_tuple[1])

    # storing losses for visualisation
    all_lossesP = []
    accuracy_countP = []
    epoch_count = []
    # re-initialise model with new hidden layer size
    modelP = Classification(p.input_neurons, hidden_size, p.output_neurons)

    # set hidden layer data to pruned layer data from the baseline model so that it will continue training
    modelP.hidden.weight.data = new_tuple[0]
    modelP.hidden.bias.data = new_tuple[1]

    # defining loss function
    loss_func = torch.nn.CrossEntropyLoss()

    # defining optimiser
    optimiser = torch.optim.SGD(modelP.parameters(), lr=r_lr)

    # re-training the pruned network on new training data
    for epoch in range(p.num_epoch):
        Y_pred = modelP(pp.XP)

        # computing loss
        # passing Tensors containing pred and true Y
        # loss function returns a Tensor containing the loss
        lossP = loss_func(Y_pred, pp.YP)

        all_lossesP.append(lossP.item())

        # printing progress
        if epoch % 50 == 0:
            # convert four-column predicted Y values to one column for comparison
            _, predicted = torch.max(F.softmax(Y_pred, 1), 1)

            # calculate and print accuracy
            total = predicted.size()

            correct = predicted.data.numpy() == pp.YP.data.numpy()

            accuracy_countP.append(100 * sum(correct) / total)

            print('Epoch [%d/%d] Loss: %.4f Accuracy: %.2f %%'
                  % (epoch + 1, p.num_epoch, lossP.item(), 100 * sum(correct) / total))

            epoch_count.append(epoch)

        # clear gradients before backward pass
        modelP.zero_grad()

        # perform backward pass: compute gradients of the loss
        lossP.backward()

        # calling the step function on an optimiser to update parameters
        optimiser.step()

        # get final train accuracy to pass to GA
        if epoch == p.num_epoch - 1:
            index = int(p.num_epoch / 50) - 1
            train_pass = float(accuracy_countP[index])

    # In[ ]:

    # plotting cross-entropy losses for training
    plt.figure()
    plt.title('Cross-entropy loss on Pruned Training')
    plt.xlabel('epochs')
    plt.ylabel('loss')
    plt.plot(all_lossesP)
    plt.show()

    pred_y = predicted.data.numpy()

    # confusion matrix for training
    print('')
    print('Confusion matrix for Pruned training')
    print(confusion_matrix(pp.df_array_yTrainP, pred_y, labels=[0, 1, 2, 3]))
    print('')
    print(classification_report(pp.df_array_yTrainP, pred_y, zero_division=1))

    # In[ ]:
    # accuracy and confusion matrix for testing

    Y_pred_testP = modelP(pp.X_testP)

    # get prediction for test
    _, predicted_test = torch.max(F.softmax(Y_pred_testP, 1), 1)

    # calculate accuracy for test
    total_test = predicted_test.size()
    correct_test = sum(predicted_test.data.numpy() == pp.Y_testP.data.numpy())

    test_accuracyP = 100 * correct_test / total_test
    test_pass = test_accuracyP[0]

    print("Test accuracy is: " + str(test_accuracyP[0]) + "%")

    # confusion matrix for test
    pred_y_test = predicted_test.data.numpy()

    print(confusion_matrix(pp.df_array_yTestP, pred_y_test, labels=[0, 1, 2, 3]))
    print('')
    print(classification_report(pp.df_array_yTestP, pred_y_test, zero_division=1))
    print('')
    print("Reduced by " + str(len(model.hidden.bias.data) - len(modelP.hidden.bias.data)) + " hidden neurons")
    print("Initial size: " + str(len(model.hidden.bias.data)) + " hidden neurons")
    print("Post-Pruning size: " + str(len(modelP.hidden.bias.data)) + " hidden neurons")

    return train_pass, test_pass
