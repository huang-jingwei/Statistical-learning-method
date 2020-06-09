import  tensorflow as  tf
import numpy as np

# 加载训练mnist数据集的数据集和测试数据集
#因为感知机是二分类模型，故选取label=0和1的样本点，并且将label=0改成label=-1
def MnistData():
    #原始的训练数据集是60000张尺寸为28*28的灰色照片，测试数据集是10000张尺寸为28*28的灰色照片
    mnist = tf.keras.datasets.mnist
    (train_data, train_label), (test_data, test_label) = mnist.load_data()
    train_data = train_data.reshape(60000, 784)
    test_data = test_data.reshape(10000, 784)
    #提取出label=0和label=1的样本点
    train_data = train_data [np.where((train_label==0)  | (train_label== 1))]
    train_label = train_label[np.where((train_label==0) |  (train_label == 1))]
    test_data  = test_data[np.where((test_label == 0)  |  (test_label== 1))]
    test_label = test_label[np.where((test_label== 0)  |  (test_label == 1))]
    #修改label的格式，默认格式为uint8，是不能显示负数的，将其修改为int8格式
    train_label=np.array(train_label,dtype='int8')
    test_label =np.array(test_label,dtype='int8')
    #将label=0改成label=-1
    train_label[np.where(train_label== 0)] = -1
    test_label [np.where(test_label == 0)] = -1
    return (train_data, train_label), (test_data, test_label)

#采用随机梯度下降方法，训练感知机模型
#epoch：迭代次数上限,learnRate:学习率
def perceptron(train_data,train_label,test_data, test_label,epoch=3000,learnRate=0.5):
    w=np.random.rand(1,784)  #初始化模型参数
    b=np.random.rand(1)
    for i in range(epoch):   #开始迭代训练，迭代次数上限是epoch
        for j in range(len(train_label)): #遍历数据集
            if train_label[j] * (np.dot(train_data[j], w.T) + b) < 0:  # 检测到误分类点
                w_grad=-train_label[j]*train_data[j]  #损失函数对w，b参数的偏导数
                b_grad=-train_label[j]
                w=w-learnRate*w_grad   #更新w，b参数
                b=b-learnRate*b_grad
        if i %100==0:   #每迭代训练一百次，就打印模型的分类准确率
            acc=modelTest(test_data, test_label,w,b)
            print(' %d epoch,model accuracy is %f: '%(i,acc))
    return w,b

#测试模型
def modelTest(test_data, test_label,w,b):
    acc=0   #记录测试集中分类准确点的数量
    for i in range(len(test_label)):
        if test_label[i]*(np.dot(test_data[i],w.T)+b)>0:  #检测到模型的分类准确点
            acc=acc+1
    return acc/len(test_label) * 100


if __name__=="__main__":
    # 加载mnist数据集中label=0和label=+1的数据，并且将label=0改成label=-1
    (train_data, train_label), (test_data, test_label)=MnistData()
    #训练模型
    w,b=perceptron(train_data, train_label, test_data, test_label)
