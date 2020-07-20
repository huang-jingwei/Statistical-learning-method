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

#函数功能：找到当前标签集中占数目最大的标签
def majorLabelClass(label):
   labelClass=np.unique(label)                #对原始标签数据进行去重
   labelClassNum=np.zeros(len(labelClass))    #初始化0矩阵，用来记录每个类别标签出现的次数
   for index in range(len(label)):
       labelClassNum[label[index]]=labelClassNum[label[index]] +1
   maxValueIndex=np.argmax(labelClassNum)     #出现次数最多类别的下标
   maxValue=labelClassNum[maxValueIndex]      #出现次数最多的类别的出现次数
   return maxValueIndex,maxValue


#函数功能：计算数据集的经验熵
#参考公式：李航《统计学习方法》第二版 公式5.7
#参数说明：label：训练数据集的标签数据集
def calculation_H_D(label):
    labelClass = np.unique(label)              #对原始标签数据进行去重
    HD=0                                       #初始化数据集的经验熵
    for labelValue in labelClass:              #遍历所有类别的数据集
        subLabelSet=label[label==labelValue]   #把标签数据集中等于labelValue的标签全部提取出来
        HD +=(-1)*(len(subLabelSet)/len(label))*np.log(len(subLabelSet)/len(label))
    return HD

#函数功能：计算经验条件熵
#参考公式：李航《统计学习方法》第二版 公式5.7
#参数说明：trainDataFeature:训练数据集被提取出的的一列特征数据，label：训练数据集的标签数据集
def calculation_H_D_A(trainDataFeature,label):
    dataValueClass = np.unique(trainDataFeature)                 #对特征数据进行去重,得到当前特征维度下特征向量所有可能的取值
    HDA=0                                                        #初始化当前特征维度的经验条件熵
    for dataValue in dataValueClass:                             #遍历特征维度所有可能的取值
        subDatalSet=trainDataFeature[trainDataFeature==dataValue]#把特征维度中等于dataValue的数据全部提取出来
        subLabelSet = label[trainDataFeature == dataValue]       #把上述子数据集对应的标签数据集提取出来
        HDA +=(len(subDatalSet)/len(label))*calculation_H_D(subLabelSet)
    return HDA




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