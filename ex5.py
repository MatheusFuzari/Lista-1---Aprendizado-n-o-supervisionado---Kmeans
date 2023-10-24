import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans

data = {
    'Maquina': np.array([1,2,3,4,5,6,7,8,9,10]),
    'Temperatura': np.array([70.2,65.1,75.5,80.3,68.7,72.9,78.6,66.4,73.1,69.5]),
    'Vibração': np.array([12.5,8.2,15.6,10.2,11.8,14.3,9.8,8.9,13.7,12.1]),
    'Corrente':np.array([4.7,3.9,5.1,4.5,4.2,5.3,4.8,4.0,5.0,4.3]),
}

df = pd.DataFrame(data)
print(data['Temperatura'])
kmeans = KMeans(n_clusters=3) #number of clusters
kmeans.fit(df)
sse = kmeans.inertia_
print("SSE: ", sse)
labels = kmeans.labels_
centroids = kmeans.cluster_centers_
plt.scatter(data['Maquina'],data['Temperatura'],c=labels)
plt.scatter(centroids[:,0],centroids[:,2],marker='x',color='red')
plt.title("Agrupamento kmeans")
plt.xlabel("Maquina")
plt.ylabel("Temperatura")
plt.show()
fig, ax = plt.subplots(figsize=(10,6))
ax.boxplot([data['Temperatura'],data['Vibração'],data['Corrente']],labels=['Temperatura','Vibração','Corrente'])
plt.title("Agrupamento kmeans")
plt.show()

df['Grupo']= labels
grouped_data = df.groupby('Grupo').mean()
print(grouped_data)