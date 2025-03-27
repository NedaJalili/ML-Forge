#HODA


#مثال الگوریتم kmeans 
#فرض کنیم سن مراجعه کنندگان یک فروشگاه در یک روز بصورت زیر باشد )تعدادنمونه (n=19

import numpy as np
x=np.array([[15],[15],[16],[19],[19],[20],[20],[21],
[22],[28],[35],[40],[41],[42],[43],[44],[60],[61],[65]])
from sklearn.cluster import KMeans
model=KMeans(n_clusters=2,max_iter=300)
model.fit(x)
print(model.labels_)
print(model.cluster_centers_)
