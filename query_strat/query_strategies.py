import numpy as np
from operator import itemgetter

from query_strat.query_utils import get_samples

def least_confidence(confidences, number):
    print("Using Least Confidence Strategy")
    least_conf_indices = np.argsort(confidences['conf_vals'])
    query_paths = list(itemgetter(*least_conf_indices)(confidences['loc']))
    return query_paths[:number]

def uncertainty_sampling(confidences, number):
    
    print("Using Uncertainty Strategy")
    # tmp_query = np.array([confidences[i][1] for i in range(len(confidences))])
    difference_array = np.absolute(confidences['conf_vals'] - 0.5) 
    uncertain_elements = difference_array.argsort()[:number]
    query_paths = list(itemgetter(*uncertain_elements)(confidences['loc']))
    return query_paths

def gaussian_sampling(confidences, number):

    print("Using Gaussian Strategy")
    return get_samples(confidences['loc'], confidences['conf_vals'],number)