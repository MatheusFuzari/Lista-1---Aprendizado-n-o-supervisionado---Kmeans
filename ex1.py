import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans

data = {
    'Teor Alcoólico': [3, 4, 5, 6],
    'Acidez': ['muito', 'pouco', 'médio', 'baixo'],
    'pH': [4.3, 2.8, 4.2, 3.9]
}

df = pd.DataFrame(data)
df = pd.get_dummies(df,columns=['Acidez'])
kmeans = KMeans(n_clusters=3) #number of clusters
kmeans.fit(df)
sse = kmeans.inertia_
print("SSE: ", sse)
labels = kmeans.labels_
centroids = kmeans.cluster_centers_
plt.scatter(df['Teor Alcoólico'],df['pH'],c=labels)
plt.scatter(centroids[:,0],centroids[:,2],marker='x',color='red')
plt.title("Agrupamento kmeans")
plt.xlabel("Teor Alcólico")
plt.ylabel("pH")
plt.show()

df['Grupo']= labels
grouped_data = df.groupby('Grupo').mean()
print(grouped_data)