import numpy as np
from sklearn import datasets
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
cmap = ListedColormap(['#FF0000','#00FF00','#0000FF'])

iris = datasets.load_iris()
X, y = iris.data, iris.target

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=1234)

plt.figure()
plt.scatter(X[:,0],X[:,1],c=y,cmap=cmap,edgecolor='k',s=20)
plt.show()

from knn import KNN

classifier = KNN(k=6)
classifier.fit(X_train, y_train)
preds = classifier.predict(X_test)

accuracy = np.sum(preds == y_test)/len(y_test)

print(accuracy)
