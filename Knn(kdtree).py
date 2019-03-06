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
    def __init__(self):
        self.kdTree = None
        self.nfeatures = 0
        self.nsamples = 0
        
    def create(self, dataSet, depth = 0):   #创建kd树，返回根结点
        if (len(dataSet) > 0):
            self.nsamples, self.nfeatures = np.shape(dataSet)    #求出样本行，列
            midIndex = self.nsamples // 2 #中间数的索引位置
            axis = depth % self.nfeatures    #判断以哪个轴划分数据,每深入一层，就换下一个特征来做划分
            sortedDataSet = sorted(dataSet[:], key=lambda data: data[axis]) #进行排序
            node = Node(sortedDataSet[midIndex]) #将节点数据域设置为中位数，具体参考下书本
            leftDataSet = sortedDataSet[: midIndex] #将中位数的左边创建2改副本
            rightDataSet = sortedDataSet[midIndex+1 :]
            node.left = self.create(leftDataSet, depth+1) #将中位数左边样本传入来递归创建树
            node.right = self.create(rightDataSet, depth+1)
            self.kdTree = node
            return node
        else:
            return None

    def search(self, x):  #搜索 
        nearestPoint = None    #保存最近的点
        nearestValue = 1e10   #保存最近的值
        
        def travel(node, depth = 0):    #递归搜索
            nonlocal nearestPoint, nearestValue
            if node != None:    #递归终止条件
                axis = depth % self.nfeatures    #计算轴
                if x[axis] < node.data[axis]:   #如果数据小于结点，则往左结点找
                    travel(node.left, depth+1)
                else:
                    travel(node.right, depth+1)
 
                #以下是递归完毕后，往父结点方向回朔，对应算法3.3(3)
                distNodeAndX = self.dist(x, node.data)  #目标和节点的距离判断

                #确定当前点，更新最近的点和最近的值，对应算法3.3(3)(a)
                if  (nearestValue > distNodeAndX): 
                    nearestPoint = node.data
                    nearestValue = distNodeAndX
 
                #确定是否需要去子节点的区域去找（圆的判断），对应算法3.3(3)(b)
                if (abs(x[axis] - node.data[axis]) <= nearestValue):  
                    if x[axis] < node.data[axis]:
                        travel(node.right, depth + 1)
                    else:
                        travel(node.left, depth + 1)
        travel(self.kdTree)
        return nearestPoint
    
    def dist(self, x1, x2): #欧式距离的计算
        return ((np.array(x1) - np.array(x2)) ** 2).sum() ** 0.5
    
if __name__ == '__main__':
    dataSet = [[0, 1],
               [8, 7],
               [9, 8],
               [1, 1],
               [2, 2],
               [9, 7]
               ]
    x = [8, 6]
    kdtree = KdTree()
    kdtree.create(dataSet)
    nearest =  kdtree.search(x)
    print("kd search:", nearest)
    plt.scatter([0,1,2,9,8,9],[1,1,2,7,7,8],c = "b")
    plt.scatter([5],[3],c="r")
    plt.plot([nearest[0],5], [nearest[1],3], linewidth=2.0,linestyle ="--")
    plt.show()
