import numpy as np
import torch

# These functions calculate the angles between pairs of vectors in pattern space between the angular range of
# 0-180 degrees by Gedeon's distinctiveness method.

# cosA = the dot product of the vectors divided by the product of vector magnitude
def angle_calc(a, b):
    # calculate vector magnitude
    a_mag = np.sqrt(a.dot(a))
    b_mag = np.sqrt(b.dot(b))

    # return the cosine angle in degrees between vectors
    return a.dot(b) / (a_mag * b_mag)


# compute similarity and returns pairs to merge and delete
def to_prune(vector, min_angle_param, max_angle_param):
    angles_to_prune = {}
    angles_to_delete = {}
    # normalise values to be between 0-1 for the vector
    norm_vector = 1 / (1 + np.exp(-vector))
    for i in range(len(norm_vector)):
        for j in range(i + 1, len(norm_vector)):
            angle = angle_calc(norm_vector[i], norm_vector[j])
            angles_to_prune[(i, j)] = angle
            angles_to_delete[(i, j)] = angle

    # 180 degrees * ((X - X minimum) / (X max - X min))
    norm = np.array([X for X in angles_to_prune.values()])
    norm = 180 * (norm - norm.min()) / (norm.max() - norm.min())

    # replaces angle values with norm angle values in the key
    angles_to_prune = {list(angles_to_prune.keys())[i]: norm[i] for i in range(len(norm))}
    # new keys which has only angles to be merged
    angles_to_prune = {X: angles_to_prune[X] for X in angles_to_prune.keys() if min_angle_param > angles_to_prune[X]}

    # 180 degrees * ((X - X minimum) / (X max - X min))
    norm2 = np.array([X for X in angles_to_delete.values()])
    norm2 = 180 * (norm2 - norm2.min()) / (norm2.max() - norm2.min())

    # replaces angle values with norm angle values in the key
    angles_to_delete = {list(angles_to_delete.keys())[i]: norm2[i] for i in range(len(norm2))}
    # new keys which has only angles to be deleted
    angles_to_delete = {X: angles_to_delete[X] for X in angles_to_delete.keys() if max_angle_param <
                        angles_to_delete[X]}

    return angles_to_prune, angles_to_delete


# for each similar merge pair, merge one and concat to the remaining pair to create a new tensor which does not have the
# pruned neuron and returns the new pruned tensor and prunes the bias at the same index accordingly
# for each similar delete pair, remove them both
def pruning_weights(model_weights, model_bias, min_angle_pass, max_angle_pass):
    pruning_vectors, deleting_vectors = to_prune(model_weights, min_angle_pass, max_angle_pass)
    pruning_vectors = list(pruning_vectors)
    deleting_vectors = list(deleting_vectors)

    new_T = model_weights
    new_B = model_bias

    # merging and deleting vectors according to distinctiveness angular separations calculated above
    for i in range(len(pruning_vectors)):
        if len(new_B) > 2:
            pair = pruning_vectors[i]
            merge1 = pair[0]
            merge2 = pair[1]

            # adds the weights together then deletes one by concatenating
            model_weights[merge2] += model_weights[merge1]
            new_T = torch.cat([new_T[0:merge1], new_T[merge1+1:]])

            # same process on the bias
            model_bias[merge2] += model_bias[merge1]
            new_B = torch.cat([new_B[0:merge1], new_B[merge1+1:]])

    for i in range(len(deleting_vectors)):
        # must be greater than 2 or it is possible that it deletes all vectors
        if len(new_B) > 2:
            pair = deleting_vectors[i]
            delete1 = pair[0]
            delete2 = pair[1]

            # deletes both by concatenating them onto the next one
            new_T = torch.cat([new_T[0:delete1], new_T[delete1+1:]])
            new_T = torch.cat([new_T[0:delete2], new_T[delete2+1:]])

            # same process for the bias
            new_B = torch.cat([new_B[0:delete1], new_B[delete1+1:]])
            new_B = torch.cat([new_B[0:delete2], new_B[delete2+1:]])

    return new_T, new_B
