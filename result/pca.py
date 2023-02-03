from sklearn.decomposition import PCA
import numpy as np
import pandas as pd
import os
import matplotlib.pyplot as plt


# load data of csv
emb_path = os.path.join("./result", "emb.csv")
df = pd.read_csv(emb_path, index_col=0, header=0)
data = df.values # transform value of dataframe to ndarray

# load labels
labels_path = os.path.join('./result/', 'labels.csv')
labels = np.loadtxt(labels_path, dtype=float, delimiter=',')

pca = PCA(n_components=2)
data_reduced = pca.fit(data)

data_pca = pca.transform(data)

# visualizatin 2D plot
fig = plt.figure()
ax = fig.add_subplot(111)
# sc = ax.scatter(data_reduced[:,0], data_reduced[:,1], c=labels, cmap='jet')
sc = ax.scatter(data_pca[:,0], data_pca[:,1], c=labels, cmap='jet')
plt.axis('off')
plt.colorbar(sc)
plt.show()
