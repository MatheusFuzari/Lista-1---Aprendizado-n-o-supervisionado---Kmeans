import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans

week_day = np.array([1,2,3,4,5,6,7])
qnt_client = np.array([15,20,30,100,350,500,700])
buy_data = np.column_stack([week_day,qnt_client])
kmeans = KMeans(n_clusters=3) #number of clusters
kmeans.fit(buy_data)
sse = kmeans.inertia_
print("SSE: ", sse)
labels = kmeans.labels_
centroids = kmeans.cluster_centers_
plt.scatter(buy_data[:,0],buy_data[:,1],c=labels)
plt.title("Agrupamento kmeans")
plt.xlabel("Dia da semana")
plt.ylabel("Quantidade de cliente")
plt.show()

df_buy_data = pd.DataFrame(buy_data,columns=['Dia da semana','Quantidade de cliente'])
df_buy_data['Grupo']= labels
grouped_data = df_buy_data.groupby('Grupo').mean()
print(grouped_data)