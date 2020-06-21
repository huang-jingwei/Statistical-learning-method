import tensorflow as tf
import  numpy as np
import random

# 加载训练mnist数据集的数据集和测试数据集
def MnistData():
    #原始的训练数据集是60000张尺寸为28*28的灰色照片，测试数据集是10000张尺寸为28*28的灰色照片
    mnist = tf.keras.datasets.mnist
    (train_data, train_label), (test_data, test_label) = mnist.load_data()
    train_data = train_data.reshape(60000, 784)
    test_data = test_data.reshape(10000, 784)
    #修改label的格式，默认格式为uint8，是不能显示负数的，将其修改为int8格式
    train_label=np.array(train_label,dtype='int8')
    test_label =np.array(test_label,dtype='int8')
    return (train_data, train_label), (test_data, test_label)

#kmaens模型
#input parameter:
# data：原始数据集
#k：聚类的类别总数，epoch：迭代训练次数的上限，error：中心点上下两轮的误差运行范围
#output parameter:
# center:返回聚类后各个类别的中心点坐标
def kmeans(data,k,epoch=3000,error=100):
    featureNum=data.shape[1]   #样本向量的特征维度
    classNum=k                 #人为预定的类别总数
    center=np.zeros(shape=(classNum,featureNum))  #定义类别中心点
    for i in range(classNum): #初始化类别中心点
        # 在数据集中随意选取一个点位类别中心点
        index=random.randint(0,data.shape[0]-1)
        center[i]=data[index]

    # 开始kmeans的迭代训练，迭代次数上限为epoch
    for i in range(epoch):
        centerFeatureCount=np.zeros(shape=(classNum,featureNum))   #记录每个类别的样本点特征向量的总和
        centerNumCount=np.zeros(classNum)   #记录每个类别的样本点的总个数
        for index in range(data.shape[0]):  #在一轮迭代过程中，遍历所有样本点
            #distance为样本点与不同类别中心点的距离
            #kmeans的距离为欧式距离平方
            distance=np.sum(np.square(data[index]-center),axis=1)
            curLable=np.argmin(distance)                #与其距离最小的中心点所属的类别就是该样本点的类别
            centerFeatureCount[curLable]+=data[index]   #记录该类别的样本点特征向量之和
            centerNumCount[curLable] +=1                #记录该类别的样本点个数
        #更新后中心点坐标
        nextCenter=np.zeros(shape=(classNum,featureNum))   #定义下一步的中心点坐标
        for j in range(classNum):   #每个类别的样本点求均值，得到下一轮迭代的中心点
            nextCenter[j]=centerFeatureCount[j]/centerNumCount[j]
        #相邻两轮的中心点偏差，并且打印出来
        nextError = np.sum(np.sum(np.abs(center - nextCenter), axis=1))
        print(' %d epoch,Center point deviation is %f ' % (i, nextError))
        #若中心点坐标变化较小时，停止迭代，否则继续进行下一轮迭代训练
        if nextError<error:
            break
        else:
            center=nextCenter
    return center

if __name__=="__main__":
    # 加载mnist数据集
    (train_data, train_label), (test_data, test_label) = MnistData()
    #用测试数据集进行kmeans距离
    center=kmeans(data=test_data,k=10)