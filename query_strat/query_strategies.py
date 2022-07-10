import numpy as np
from scipy.stats import entropy

'''
Design notes for custom strategies : 

Input : 
confidences:  Confidence values of all the unlabeled images
number : Number of images to be queried

Output :
Paths of all the intelligently queried images
'''

def entropy_based(confidences):

    print("Using Entropy Based")

    entropies = entropy(confidences["conf_vals"], axis = 1)

    entropies = (entropies - entropies.mean()) / entropies.std()

    assert len(confidences["loc"]) == len(entropies)

    # path_to_score = dict(zip(confidences["loc"], entropies))

    return entropies


def margin_based(confidences):

    print("Using Margin Based")
    vals = confidences['conf_vals'].copy()

    max_indices = np.argmax(vals, axis = 1)
    print("max_indices.shape: ", max_indices.shape)
    
    max_vals = []

    counter = 0
    for index in max_indices:

      max_vals.append(vals[counter, index]) 

      vals[counter, index] = -100
      counter += 1

    max_vals = np.array(max_vals)

    # print("max_indices: ", max_indices)
    # max_vals = vals[max_indices]

    # print("max_vals: ", max_vals)

    # print("vals.shape: ", vals.shape)
    # print("max_vals.shape: ", max_vals.shape)
    

    # making max to a low number that cannot be reselected
    # vals[max_indices] = -1
    second_max_vals = np.max(vals, axis = 1)
    
    # print("second_max_vals.shape: ", second_max_vals.shape)
    
    # make sure to negate below. Since lower margin is more uncertain
    difference_array = - (max_vals - second_max_vals)

    difference_array = (difference_array - difference_array.mean()) / difference_array.std()

    assert len(confidences["loc"]) == len(difference_array)

    # path_to_score = dict(zip(confidences["loc"], difference_array))

    print("difference_array.shape: ", difference_array.shape)
  
    return difference_array


def least_confidence(confidences):
    
    print("Using Least Confidence")

    difference_array = 1 - np.max(confidences['conf_vals'], axis = 1)

    difference_array = (difference_array - difference_array.mean()) / difference_array.std()
    
    assert len(confidences["loc"]) == len(difference_array)

    # path_to_score = dict(zip(confidences["loc"], difference_array))

    return difference_array
