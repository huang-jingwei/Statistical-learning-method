import tensorflow as  tf
import numpy as np

# 加载训练mnist数据集的数据集和测试数据集
def MnistData():
    #原始的训练数据集是60000张尺寸为28*28的灰色照片，测试数据集是10000张尺寸为28*28的灰色照片
    mnist = tf.keras.datasets.mnist
    (train_data, train_label), (test_data, test_label) = mnist.load_data()
    train_data = train_data.reshape(60000, 784)
    test_data = test_data.reshape(10000, 784)
    #修改label的格式，默认格式为uint8，是不能显示负数的，将其修改为int32格式
    train_data=np.array(train_data,dtype='int32')
    test_data =np.array(test_data,dtype='int32')
    return (train_data, train_label), (test_data, test_label)

#knn算法（不包括kd树）
def knn(train_data,train_label,test_data, test_label,k):
    count=0  #记录knn模型预测成功的次数
    for i in range(test_data.shape[0]):      #对测试集每个实例点都进行knn算法处理
        # 计测测试集输入点与训练数据集所有点的算欧式距离
       distance=np.sqrt(np.sum(np.square(test_data[i] - train_data),axis=1))
       #np.argsort函数返回的是数组中数组值从小到大对应的索引值
       index=np.argsort(distance) #argsort(distance)是将所计算得到距离从小到大排列，提取其对应的索引
       selectIndex=index[:k]                    #选取出与当前输入点距离最小的前k个的实例点对应的索引
       classNumber=len(np.unique(train_label))  #出重处理，得到train_label所存在不同的元素的数量
       labelList=[0]*classNumber                #初始化列表,计算不同label出现的次数
       for j in selectIndex:   #记录train_label在selectIndex索引对应的label出现的次数
           # train_label[j]：selectIndex当前索引在训练数据集中对应的种类
           labelList[train_label[j]] +=1
       # np.argmax(labelList),找到出现次数最多的种类对应的索引（索引是0到9，对应不同的种类）
       predict=np.argmax(labelList)   #出现次数最多的种类，等同于预测的种类
       if predict ==test_label[i]:
           count +=1
       if i %100==0 and i!=0:     #每预测一百次，就打印当前时刻模型的预测准确率
            acc=count/i
            print(' %d epoch,model accuracy is %f: '%(i,acc))
if __name__=="__main__":
    # 加载mnist数据集
    (train_data, train_label), (test_data, test_label)=MnistData()
    #knn模型
    knn(train_data,train_label,test_data, test_label,k=25)