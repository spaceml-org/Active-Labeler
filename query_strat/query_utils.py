import numpy as np

def gaussian(x, mean, sigma):
                a = 1 / (sigma * np.sqrt(2 * np.pi))
                exp_val = -(1 / 2) * (((x - mean) ** 2) / (sigma ** 2))
                tmp = a * np.exp(exp_val)
                return tmp

def get_gaussian_weights(confidences):
    mean = 0.5
    sigma = 0.2
    return [gaussian(conf_i, mean, sigma) for conf_i in confidences]

def get_samples(image_list, confidences, number_of_samples):
    sample_wts = get_gaussian_weights(confidences)
    sample_wts_norm = np.array(sample_wts)
    sample_wts_norm /= sample_wts_norm.sum()
    picked_samples = np.random.choice(
        image_list, replace=False, size=number_of_samples, p=sample_wts_norm
    )
    return picked_samples

