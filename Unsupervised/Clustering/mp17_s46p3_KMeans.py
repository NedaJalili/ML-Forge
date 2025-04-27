#اگر یک محصول با وزن 125 ، اندازه 45 و رنگ آبی دیده شود مربوط به کدام دسته A یا B خواهد بود.
#A:0,B:1

import numpy as np
x= np.array([[120, 50, 1, 0],[60, 20, 2, 1],
[145, 65, 1, 0],[130, 45, 3, 0],
[50, 15, 2, 1]])
from sklearn.cluster import KMeans
model = KMeans(n_clusters=2)
model.fit(x)
print (model.labels_)
print(model.predict([[125,45,1]]))
