import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans

data = {
    'Substância': ['Álcool','Gasolina','Leite','Querosene','Óleo','Vinho'],
    'Concentração': [12.5,0.1,4.0 ,1.2  ,0.5  ,15.0 ],
    'Teor Alcoólico': [50 ,0.05 ,0.01 ,0.02 ,0.01 ,12.5]
}

df = pd.DataFrame(data)
df = pd.get_dummies(df,columns=['Substância'])
print(df)
kmeans = KMeans(n_clusters=3) #number of clusters
kmeans.fit(df)
sse = kmeans.inertia_
print("SSE: ", sse)
labels = kmeans.labels_
centroids = kmeans.cluster_centers_
plt.scatter(df['Concentração'],df['Teor Alcoólico'],c=labels)
plt.scatter(centroids[:,0],centroids[:,2],marker='x',color='red')
plt.title("Agrupamento kmeans")
plt.xlabel("Concentração")
plt.ylabel("Teor Alcoólico")
plt.show()

df['Grupo']= labels
grouped_data = df.groupby('Grupo').mean()
print(grouped_data)