import numpy as np
from sklearn.preprocessing import MinMaxScaler
from sklearn.cluster import KMeans as km


def pick_top_n(uncertainty_scores, filepaths, number):
    #this will pick the top n samples regardless of diversity

    top_indices = uncertainty_scores.argsort()[::-1][:number]

    return list(filepaths[top_indices])


def iterative_proximity_sampling(uncertainty_scores, filepaths, number, embedding):
     
    uncertainty_scores = uncertainty_scores.copy()
    filepaths = filepaths.copy()

    #hyperparams:
    bhat_size = 10
    how_much_diversity = 0.3
    
    number_of_bhats = int(np.ceil(number / bhat_size))
    centroid_pos = None
    my_scaler = MinMaxScaler(feature_range=(-how_much_diversity, how_much_diversity))
    all_indices = np.array([])
    
    for bhat_index in range(number_of_bhats):
        
        #calculate appropriate bhatsize (accounting for last batch having fewer elements)
        if bhat_index != number_of_bhats - 1:
            current_bahtsize = bhat_size
        else:
            current_bahtsize = number - (bhat_index * bhat_size)
  
        if bhat_index == 0:
            selected_indices = uncertainty_scores.argsort()[::-1][:current_bahtsize]
        else:
            distances = np.linalg.norm(centroid_pos - embedding, axis = 1)
            # higher the distance the better. 
            
            distance_scores = my_scaler.fit_transform(distances[:,None])
            total_scores = uncertainty_scores + np.squeeze(distance_scores)

            selected_indices = total_scores.argsort()[::-1][:current_bahtsize]
            
        
        uncertainty_scores[selected_indices] = -1000
        all_indices = np.append(all_indices, selected_indices).astype(int)
        # set centroid pos to the mean of the selected indices.
        centroid_pos = embedding[all_indices].mean(axis = 0)
        
    print(len(list(filepaths[all_indices])))
    
    return list(filepaths[all_indices])

def clustering_sampling(uncertainty_scores, filepaths, number, embeddings):
    initial_dict = dict()
    uncertainity_dict = dict()
    cluster_num = 3
    for i in range(len(uncertainty_scores)):
        initial_dict[filepaths[i]] = embeddings[i]
        uncertainity_dict[filepaths[i]] = uncertainty_scores[i]
    kmeans = km(n_clusters=cluster_num, random_state=0)


    stuff = kmeans.fit(list(initial_dict.values()))
    _, count = np.unique(stuff.labels_, return_counts=True)
    total = len(stuff.labels_)
    cluster_numbers = np.ceil((count / total) * number)
    clusters = [[] for i in range(cluster_num)]
    scores = [[] for i in range(cluster_num)]
    for i in range(len(stuff.labels_)):
        clusters[stuff.labels_[i]].append(filepaths[i])
        scores[stuff.labels_[i]].append(uncertainty_scores[i])
    final_files = []
    for i in range(cluster_num):
        temp_val = np.array(scores[i]).argsort()[::-1][:int(cluster_numbers[i])]
        for j in range(len(temp_val)):
            final_files.append(clusters[i][temp_val[j]])

    print(final_files)
    return final_files


def random_sampling(filepaths, number):

    filepaths = filepaths.copy()
    np.random.shuffle(filepaths)
    return list(filepaths[:number])
