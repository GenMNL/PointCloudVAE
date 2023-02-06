import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os

# load data of csv
emb_path = os.path.join("./result", "emb.csv")
df = pd.read_csv(emb_path, index_col=0, header=0)
data = df.values # transform value of dataframe to ndarray

# load labels
labels_path = os.path.join('./result/', 'labels.csv')
labels = np.loadtxt(labels_path, dtype=float, delimiter=',')

# visualizatin 2D plot
fig = plt.figure()
ax = fig.add_subplot(111)
sc = ax.scatter(data[:,0], data[:,1], c=labels, cmap='jet')
plt.axis('off')
plt.colorbar(sc)
plt.show()
