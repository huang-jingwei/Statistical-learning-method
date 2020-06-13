import  tensorflow as  tf
import numpy as np

#加载训练mnist数据集的数据集和测试数据集
def MnistData():
    #原始的训练数据集是60000张尺寸为28*28的灰色照片，测试数据集是10000张尺寸为28*28的灰色照片
    mnist = tf.keras.datasets.mnist
    (train_data, train_label), (test_data, test_label) = mnist.load_data()
    train_data = train_data.reshape(60000, 784)
    test_data = test_data.reshape(10000, 784)

    #图像色素点数据在0~255之间
    #data数据集进行归一化，这样数据范围在0~1之间
    train_data=train_data/255
    test_data=test_data/255
    return (train_data, train_label), (test_data, test_label)

#logistics模型训练
#采用随机梯度下降方法，训练logistics模型模型
#epoch：迭代次数上限,learnRate:学习率
def logistics(train_data,train_label,test_data, test_label,epoch=3000,learnRate=0.005):
   dataNum = len(train_label)           # 获取原始标签数据集的样本个数
   # np.unique(train_label)对标签数据集去重处理，返回处理后的数据
   classNum=len(np.unique(train_label)) #label数据集中类别的总数，此时classNum=10

   #对权值向量和输入向量进行扩充
   #在原始数据集合权值向量最后一列单位列向量
   train_data_one=np.ones(shape=(len(train_label),1))
   test_data_one=np.ones(shape=(len(test_label),1))
   train_data=np.c_[train_data,train_data_one]
   test_data=np.c_[test_data,test_data_one]

   #对标签数据集进行onehot处理
   train_label=one_hot(train_label)

   # 初始化多分类模型参数
   #十组类别的模型参数
   w=np.random.rand(classNum,train_data.shape[1])

   for i in range(epoch):  # 开始迭代训练，迭代次数上限是epoch
       z = np.dot(train_data, w.T)        # z:(60000,10) 即60000*10维的矩阵
       h = 1 / (1 + np.exp(-z))           # sigmoid非线性处理
       error = h - train_label            # 误差
       w_grad = np.dot(error.T, train_data) / dataNum #最大似然函数对参数w,b的偏导数

       # 参数w,b更新
       w= w - learnRate * w_grad

       if i %100==0 :   #每迭代训练一百次，就打印模型的分类准确率
           acc=modelTest(test_data, test_label,w)
           print(' %d epoch,model accuracy is %f '%(i,acc))
   return w

#one-hot处理
#one-hot处理的核心想法由于是多分类，我们的类别有10个类，所以需要训练10个分类器，每个分类器都是一个二分类器
# 例如：对于数字0的分类器来说，我们将标签为0的数据的标签重新改成正类1，
# 将非0标签对应的数据的标签改为负类0，即变为一个二分类问题，
# 其他分类器一样，将标签是对应分类的类别的标签改为1， 其他置为0
def one_hot(label):
    dataNum = len(label)  # 获取原始标签数据集的样本个数
    # 对标签数据集去掉重复元素，再计算此时元素个数，此时classNum=10
    classNum=len(np.unique(label)) #label数据集中类别的总数
    label_one_hot=np.zeros(shape=(dataNum,classNum))  #生成零矩阵
    for i in range(dataNum):        #按照onehot处理规则进行赋值
        label_one_hot[i,label[i]]=1
    return label_one_hot


#sigmoid函数
def sigmoid(z):
    return 1/(1+np.exp(-z))

#logistics模型测试
def modelTest(test_data, test_label,w):
    acc = 0  # 记录测试集中分类准确点的数量
    for i in range(len(test_label)):
        sample = test_data[i]      #提取出当前样本点的特征向量
        label = test_label[i]      #提取出当前样本点的标签向量
        linear=np.dot(sample,w.T)  #样本数据和模型参数进行矩阵相乘，进行线性变换
        prob=sigmoid(linear)       #对线性变化数据进行sigmoid处理
        predict=np.argmax(prob)    #概率值最大的类别即为预测的类别
        if predict==label:         #若模型预测的类别与样本的真实类别一致，计数器加一
            acc  +=1
    return acc / len(test_label) * 100


if __name__=="__main__":
    # 加载mnist数据集中label=0和label=+1的数据，并且将label=0改成label=-1
    (train_data, train_label), (test_data, test_label)=MnistData()
    #训练模型
    w=logistics(train_data,train_label,test_data, test_label,epoch=5000,learnRate=0.5)