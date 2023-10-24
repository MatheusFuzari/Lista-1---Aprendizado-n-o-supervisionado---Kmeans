import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans

sector = np.array([1,2,3,4,5,6,7,8])
num_make = np.array([100,50,15,200,500,1000,375,450])
buy_data = np.column_stack([sector,num_make])
kmeans = KMeans(n_clusters=3) #number of clusters
kmeans.fit(buy_data)
sse = kmeans.inertia_
print("SSE: ", sse)
labels = kmeans.labels_
centroids = kmeans.cluster_centers_
plt.scatter(buy_data[:,0],buy_data[:,1],c=labels)
plt.title("Agrupamento kmeans")
plt.xlabel("Setor")
plt.ylabel("Numero de produtos fabricados")
plt.show()

df_buy_data = pd.DataFrame(buy_data,columns=['Setor','Numero de produtos fabricados'])
df_buy_data['Grupo']= labels
grouped_data = df_buy_data.groupby('Grupo').mean()
print(grouped_data)