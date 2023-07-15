import numpy as np
import pandas as pd
data=pd.read_csv("Wholesale customers data.csv")
#print(data.head())
# Create Hierarchial model for clustering the dataset
from sklearn.cluster import AgglomerativeClustering
agcluster=AgglomerativeClustering(n_clusters=3,linkage='ward')
print(agcluster.fit_predict(data))