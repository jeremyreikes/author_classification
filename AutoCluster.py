#get clusters 
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
import tqdm
def AutoKMeans(X_train,X_val, X_test,n_clusters=15):
    Sum_of_squared_distances = []
    K = range(1,n_clusters)
    for k in tqdm.tqdm(K):
        km = KMeans(n_clusters=k)
        km = km.fit(X_train)
        Sum_of_squared_distances.append(km.inertia_)
    fig= plt.figure(figsize=(6,3))
    plt.plot(K, Sum_of_squared_distances, 'bx-')
    plt.xlabel('k')
    plt.ylabel('Sum_of_squared_distances')
    plt.title('Elbow Method For Optimal k')
    plt.show()
    num = input('Num clusters? ')
    km = KMeans(n_clusters=int(num))
    train_clusters = km.fit_predict(X_train)
    val_clusters = km.fit_predict(X_val)
    test_clusters = km.fit_predict(X_test)
    return train_clusters, val_clusters, test_clusters