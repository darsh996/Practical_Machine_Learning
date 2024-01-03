from scipy.cluster.hierarchy import linkage, dendrogram
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler

milk = pd.read_csv("milk.csv",index_col=0)

scaler = StandardScaler()
milkscaled=scaler.fit_transform(milk)

linkage_method = "average"
mergings = linkage(milkscaled,
                   method=linkage_method)
dendrogram(mergings,
           labels=np.array(milk.index),
           leaf_rotation=90,
           leaf_font_size=10,
)
plt.title(linkage_method)
plt.show()

