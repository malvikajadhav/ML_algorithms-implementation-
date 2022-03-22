import numpy as np
from collections import Counter
class KNN:
    
        
        
    def __init__(self,k=3):
        self.k = k
        
    def distance(self, x1, x2):
        return np.sqrt(np.sum((x1 - x2)**2))
        
    def fit(self, X, y):
        self.X_train = X
        self.y_train = y
    
    def predict(self,X):
        predicted_labels = [self._predicted(x) for x in X ]
        return np.array(predicted_labels)
        
    def _predicted(self,x):
        distances = [self.distance(x, xtr) for xtr in self.X_train]
        k_ind = np.argsort(distances)[:self.k]
        k_nearest_labels = [self.y_train[i] for i in k_ind]
        
        majority = Counter(k_nearest_labels).most_common(1)
        return majority[0][0] 
        
        
