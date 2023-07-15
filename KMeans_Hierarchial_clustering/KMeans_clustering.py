import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import scipy.cluster.hierarchy as sch
data=pd.read_csv("Country_clusters.csv")
#print(data.head())
# Only 1st and 2nd column requaired to make Create cluster.
clean_data=data.iloc[:,1:3]
#print(clean_data.head())
# Create KMeans model for clustering
from sklearn.cluster import KMeans
kmean=KMeans(3)# 3 is the number of clusters
prediction=kmean.fit_predict(clean_data)
print(prediction)
plt.title("Country Dendrogram")
dendrogram=sch.dendrogram(sch.linkage(clean_data,method='ward'))

