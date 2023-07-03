# Mackenzie Crawford – Student # 1000732558

import scipy.cluster.hierarchy as hcl
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.cluster.hierarchy import ward, fcluster
from Data_Ingestion_Clustering import *




"""
Review all the metrics that can be used for hierarchal clustering. 
Chose the metric that is closest to '1', this will allow for better clustering. 
In this case, euclidean distance is the most effective. 
"""
from scipy.spatial.distance import squareform, pdist

distances=["euclidean", "sqeuclidean", "cityblock", "cosine", "hamming", 
           "chebyshev", "braycurtis", "correlation"]
for distance in distances:
    dist = pdist(main_df.values, metric=distance)
    print(distance, hcl.cophenet(hcl.ward(dist), dist)[0])

"""
The resulting matrix Z is informing each step of the agglomerative clustering by informing the first 
two columns of which cluster indices were merged. The third column is the distance between those clusters, 
and the fourth column is the number of original samples contained in that newly merged cluster
"""
from scipy.spatial.distance import squareform, pdist
dist = pdist(main_df.values, metric="euclidean")
Z = hcl.ward(dist)

print(np.array_str(Z[-3:], precision=1))



"""
VISUALIZE THE CLUSTERS

Plot a dendogram to visualzie the clusters
"""
plt.figure(figsize=(25, 10))
plt.title('Hierarchical Clustering Dendrogram')
plt.xlabel('sample index')
plt.ylabel('distance')
hcl.dendrogram(
    Z,
    leaf_rotation=90., 
    leaf_font_size=8.
)
plt.axhline(y=8, c='k')
plt.axhline(y=3, c='k')
plt.axhline(y=4.5, c='k')
plt.show()



"""
ELBOW METHOD
Use the second derivatives in the 
"""
plt.figure(figsize=(25, 10))
last = Z[-12:, 2]
last_rev = last[::-1]
idxs = np.arange(1, len(last) + 1)
plt.plot(idxs, last_rev)

acceleration = np.diff(last, 2)  # 2nd derivative of the distances
acceleration_rev = acceleration[::-1]
plt.plot(idxs[:-2] + 1, acceleration_rev)

plt.ylabel("Distance")
plt.xlabel("Number of cluster")
plt.show()