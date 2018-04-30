import numpy as np
from math import sqrt
import matplotlib.pyplot as plt
from matplotlib import style
import warnings
from collections import Counter

style.use('fivethirtyeight')


dataset = {'k':[[1,2],[2,3],[3,1]], 'r':[[6,5],[7,7],[8,6]]}
new_features = [5,7]

##for i in dataset:
##    # dataset is a dictionary, i is a key, dataset[i] is a 2d list
##    print(i, dataset[i])
##    for ii in dataset[i]:
##        # ii is a 1d list in 2d list dataset[i]
##        print(ii)

### plot dataset
##[[plt.scatter(ii[0],ii[1],s=100,color=i) for ii in dataset[i]] for i in dataset]
##plt.scatter(new_features[0],new_features[1])
##plt.show()


### test numpy array math
##print(np.sum((np.array([[1,2],[2,3],[3,1]]) - np.array([[6,5],[7,7],[8,6]]))**2))

def k_nearest_neighbors(data, predict, k=3):
    if len(data) >= k:
        warnings.warn('K is set to a value less than total voting groups!')
    distances = []
    for group in data:
        for features in data[group]:
            # calculate euclidean distance for each feature in dataset(totally 6 in this case)
            euclidean_distance = np.linalg.norm(np.array(features)-np.array(predict))
##            print('euclidean_distance', euclidean_distance)
            distances.append([euclidean_distance, group])
##            print('distances',distances)
            
    votes = [i[1] for i in sorted(distances)[:k]]
##    print(sorted(distances)[:k])
##    print(votes)
##    print(Counter(votes).most_common(1))
    vote_result = Counter(votes).most_common(1)[0][0]
    
    return vote_result

result = k_nearest_neighbors(dataset, new_features, k=3)
print(result)

# plot dataset
[[plt.scatter(ii[0],ii[1],s=100,color=i) for ii in dataset[i]] for i in dataset]
plt.scatter(new_features[0],new_features[1], color=result)
plt.show()

