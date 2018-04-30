import numpy as np
import matplotlib.pyplot as plt
from matplotlib import style
from sklearn.cluster import MeanShift
from sklearn.datasets.samples_generator import make_blobs

centers = [[1,1],[5,5]]

X, _ = make_blobs(n_samples=200, centers=centers, cluster_std=1)

plt.scatter(X[:,0],X[:,1])
plt.show()


ms = MeanShift()
ms.fit(X)
labels = ms.labels_
cluster_centers = ms.cluster_centers_

n_cluseters_ = len(np.unique(labels))

print('Nubmer of estimated cluseters:', n_cluseters_)

colors = 10*['r.','g.','b.','r.','c.','k.','u.','m.']

print(colors)
print(labels)

for i in range(len(X)):
    plt.plot(X[i][0], X[i][1], colors[labels[i]], markersize=10)

plt.scatter(cluster_centers[:,0], cluster_centers[:,1], marker = "x", s=500, linewidths = 5, zorder=10)


plt.show()

##centroids = kmeans.cluster_centers_
##labels = kmeans.labels_
##


##X = np.array([[1,2],
##              [5,8],
##              [1.5,1.8],
##              [8,8],
##              [1,0.6],
##              [9,11]])
##
##             
##kmeans = KMeans(n_clusters=2)
##kmeans.fit(X)
##
##centroids = kmeans.cluster_centers_
##labels = kmeans.labels_
##
##print(centroids)
##print(labels)
##
##colors = ["g.","r."]
##
##for i in range(len(X)):
##    print('coordinate:',X[i], "label:", labels[i])
##    plt.plot(X[i][0], X[i][1], colors[labels[i]], markersize=10)
##    
##plt.scatter(centroids[:,0], centroids[:,1], marker = "x", s=500, linewidths = 5, zorder=10)
##plt.show()
##
##             
##             
##             
##
