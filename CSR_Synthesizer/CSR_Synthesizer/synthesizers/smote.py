import numpy as np
from random import randrange, choice
from sklearn.neighbors import NearestNeighbors
import pandas as pd

class SMOTE():
  def fit(self, target_data, sample_numbers=None, k_neighbors=None):
    self.target_data = target_data
    self.sample_numbers = sample_numbers
    self.k_neighbors = k_neighbors

    target_data_columns = target_data
    target_data = target_data.to_numpy()
    n_minority_samples, n_features = target_data.shape

    if int(sample_numbers/n_minority_samples) >= 1:
      M = int(sample_numbers/n_minority_samples)
      N = M+1
      n_synthetic_samples = N * n_minority_samples
    elif int(sample_numbers/n_minority_samples) < 1:
      N = 1
      n_synthetic_samples = N * n_minority_samples

    # n_synthetic_samples = N * n_minority_samples
    n_synthetic_samples = int(n_synthetic_samples)
    n_features = int(n_features)
    S = np.zeros(shape=(n_synthetic_samples, n_features))

    #Learn nearest neighbours
    neigh = NearestNeighbors(n_neighbors = k_neighbors)
    neigh.fit(target_data)

    #Calculate synthetic samples
    for i in range(n_minority_samples):
      #  nn = neigh.kneighbors(T[i], return_distance=False)
       nn = neigh.kneighbors(target_data[i].reshape(1, -1), return_distance=False)
       
       
       for n in range(N):
          nn_index = choice(nn[0])
          #NOTE: nn includes T[i], we don't want to select it
          while nn_index == i:
             nn_index = choice(nn[0])

          dif = target_data[nn_index] - target_data[i]
          gap = np.random.random()
          S[n + i * N, :] = target_data[i,:] + gap * dif[:]
   
    S = pd.DataFrame(S, columns=target_data_columns.columns)
    S = S.sample(len(S))
    S = S[:sample_numbers]
    return S