# --*-- coding:utf-8 --*--
"""
参考资料：
1. https://github.com/FlameCharmander/MachineLearning
2. 《统计学习方法》
"""
import numpy as np
import matplotlib.pyplot as plt
class Node: 
    def __init__(self, data, left = None, right = None):
        self.data = data
        self.left = left
        self.right = right
 
class KdTree:  
    def __init__(self, dataset):
        self.nsamples, self.nfeatures = np.shape(dataset)
        self.kd_tree = self.create(dataset)        
        
    def create(self, dataset, depth = 0):   #创建kd树，返回根结点
        if (len(dataset) > 0):
            nsamples, nfeatures = np.shape(dataset)    #求出样本行，列
            midIndex = nsamples // 2 #中间数的索引位置
            axis = depth % nfeatures    #判断以哪个轴划分数据,每深入一层，就换下一个特征来做划分
            sortedDataSet = sorted(dataset[:], key=lambda data: data[axis]) #进行排序
            node = Node(sortedDataSet[midIndex]) #将节点数据域设置为中位数，具体参考下书本
            leftDataSet = sortedDataSet[: midIndex] #将中位数的左边创建2改副本
            rightDataSet = sortedDataSet[midIndex+1 :]
            node.left = self.create(leftDataSet, depth+1) #将中位数左边样本传入来递归创建树
            node.right = self.create(rightDataSet, depth+1)
            return node
        else:
            return None

    def search(self, x):  #搜索 
        nearest_point = None    #保存最近的点
        nearest_dist = 1e10   #保存最近的值
        
        def travel(node, depth = 0):    #递归搜索
            nonlocal nearest_point, nearest_dist
            if node != None:    #递归终止条件
                axis = depth % self.nfeatures    #计算轴
                if x[axis] < node.data[axis]:   #如果数据小于结点，则往左结点找
                    travel(node.left, depth+1)
                else:
                    travel(node.right, depth+1)
 
                #以下是递归完毕后，往父结点方向回朔，对应算法3.3(3)
                distance = self.dist(x, node.data)  #目标和节点的距离判断

                #确定当前点，更新最近的点和最近的值，对应算法3.3(3)(a)
                if (nearest_dist > distance): 
                    nearest_point = node.data
                    nearest_dist = distance
 
                #确定是否需要去子节点的区域去找（圆的判断），对应算法3.3(3)(b)
                if (abs(x[axis] - node.data[axis]) <= nearest_dist):  
                    if x[axis] < node.data[axis]:
                        travel(node.right, depth + 1)
                    else:
                        travel(node.left, depth + 1)
        travel(self.kd_tree)
        return nearest_point
    
    def dist(self, x1, x2):
        #欧式距离 ord = 2; 曼哈顿距离 ord = 1
        distance = np.linalg.norm(np.array(x1) - np.array(x2),ord = 2)
        return distance
    
if __name__ == '__main__':
    np.random.seed(7)
    dataSet = np.random.randint(0,10,(20,2)) 
    x = [5.5, 3.5]
    kdtree = KdTree(dataSet)
    nearests = np.array(kdtree.search(x))
    print("kd search:", nearests)
    plt.scatter(dataSet[:,0],dataSet[:,1],c = "b")
    plt.scatter(x[0],x[1],c="r")
    plt.plot([nearests[0],x[0]], [nearests[1],x[1]], linewidth=2.0,linestyle ="--")
    plt.show()
