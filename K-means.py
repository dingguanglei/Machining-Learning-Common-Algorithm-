from sklearn import datasets
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D


class Kmeans():
    """Kmeans聚类算法.

    Parameters:
    -----------
    k: int
        聚类的数目.
    max_iterations: int
        最大迭代次数. 
    varepsilon: float
        判断是否收敛, 如果上一次的所有k个聚类中心与本次的所有k个聚类中心的差都小于varepsilon, 
        则说明算法已经收敛
    """
    def __init__(self, k=2, max_iterations=500, varepsilon=0.0001):
        self.k = k
        self.max_iterations = max_iterations
        self.varepsilon = varepsilon
    
        # 计算一个样本与数据集中所有样本的距离
    def distance(self, one_sample, dataset, ord = 2 ):
        distances = np.linalg.norm((dataset - one_sample),axis = 1, ord = ord)
        return distances
    
    # 从所有样本中随机选取self.k样本作为初始的聚类中心
    def init_random_centroids(self, dataset):
        n_samples, n_features = np.shape(dataset)
        centroids = np.zeros((self.k, n_features))
        for i in range(self.k):
            centroid = dataset[np.random.choice(range(n_samples))]
            centroids[i] = centroid
        return centroids

    # 将所有样本进行归类，归类规则就是将该样本归类到与其最近的中心
    def create_clusters(self, centroids, dataset):
        clusters = [[] for _ in range(self.k)]
        for index, sample in enumerate(dataset):
             # 返回距离该样本最近的一个中心索引[0, self.k)
            centroid_i = np.argmin(self.distance(sample, centroids))
            clusters[centroid_i].append(index)
        return clusters

    # 对中心进行更新
    def update_centroids(self, clusters, dataset):
        n_features = np.shape(dataset)[1]
        centroids = np.zeros((self.k, n_features))
        for i, cluster in enumerate(clusters):
            centroid = np.mean(dataset[cluster], axis=0)
            centroids[i] = centroid
        return centroids

    # 将所有样本进行归类，其所在的类别的索引就是其类别标签
    def get_cluster_labels(self, clusters, dataset):
        y_pred = np.zeros(np.shape(dataset)[0])
        for cluster_i, cluster in enumerate(clusters):
            for sample_i in cluster:
                y_pred[sample_i] = cluster_i
        return y_pred

    # 对整个数据集X进行Kmeans聚类，返回其聚类的标签
    def predict(self, dataset):
        # 从所有样本中随机选取self.k样本作为初始的聚类中心
        centroids = self.init_random_centroids(dataset)

        # 迭代，直到算法收敛(上一次的聚类中心和这一次的聚类中心几乎重合)或者达到最大迭代次数
        for _ in range(self.max_iterations):
            # 将所有进行归类，归类规则就是将该样本归类到与其最近的中心
            clusters = self.create_clusters(centroids, dataset)
            former_centroids = centroids

            # 计算新的聚类中心
            centroids = self.update_centroids(clusters, dataset)

            # 如果聚类中心几乎没有变化，说明算法已经收敛，退出迭代
            diff = centroids - former_centroids
            if diff.any() < self.varepsilon:
                break

        return self.get_cluster_labels(clusters, dataset)


def main():
    # Load the dataset
    X, y = datasets.make_blobs(n_samples=10000, 
                               n_features=3, 
                               centers=[[3,3, 3], [0,0,0], [1,1,1], [2,2,2]], 
                               cluster_std=[0.2, 0.3, 0.3, 0.2], 
                               random_state =9)

    # 用Kmeans算法进行聚类
    clf = Kmeans(k=4)
    y_pred = clf.predict(X)


    # 可视化聚类效果
    fig = plt.figure(figsize=(12, 8))
    ax = Axes3D(fig, rect=[0, 0, 1, 1], elev=30, azim=20)
    plt.scatter(X[y_pred==0][:, 0], X[y_pred==0][:, 1], X[y_pred==0][:, 2], c = "r")
    plt.scatter(X[y_pred==1][:, 0], X[y_pred==1][:, 1], X[y_pred==1][:, 2], c = "g")
    plt.scatter(X[y_pred==2][:, 0], X[y_pred==2][:, 1], X[y_pred==2][:, 2], c = "b")
    plt.scatter(X[y_pred==3][:, 0], X[y_pred==3][:, 1], X[y_pred==3][:, 2], c = "y")
    plt.show()
    fig = plt.figure(figsize=(12, 8))
    ax = Axes3D(fig, rect=[0, 0, 1, 1], elev=30, azim=20)
    # 可视化聚类效果
    plt.scatter(X[y==0][:, 0], X[y==0][:, 1], X[y==0][:, 2], c = "r")
    plt.scatter(X[y==1][:, 0], X[y==1][:, 1], X[y==1][:, 2], c = "g")
    plt.scatter(X[y==2][:, 0], X[y==2][:, 1], X[y==2][:, 2], c = "b")
    plt.scatter(X[y==3][:, 0], X[y==3][:, 1], X[y==3][:, 2], c = "y")
    plt.show()


if __name__ == "__main__":
    main()
