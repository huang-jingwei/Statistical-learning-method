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
    #为了二叉树模型的简便性，对像素值做0~1处理，像素值大于255/2的令其为1，反之为0
    train_data[train_data < 255 / 2] = 0
    train_data[train_data >= 255/2]  = 1
    test_data[test_data < 255 / 2]   = 0
    test_data[test_data   >= 255/2]  = 1
    return (train_data, train_label), (test_data, test_label)

#函数功能：找到当前标签集中占数目最大的标签
#参数说明： labelClassNum：原始数据集中标签有多少个类别，原始的mnist数据集有10个类别
def majorLabelClass(label,labelClassNum=10):
   labelClass=np.unique(label)                      #对原始标签数据进行去重,得到label所有可能的取值，并且数值是升序排序
   labelClassNum=np.zeros(labelClassNum)            #初始化0矩阵，用来记录每个类别标签出现的次数
   for labelVal in labelClass:                      #遍历label所有可能的取值
       labelSubSet=label[np.where(label==labelVal)] #提取出标签数据集中label==labelVal的数据，构成子数据集
       labelClassNum[labelVal]=len(labelSubSet)
   maxValueIndex=np.argmax(labelClassNum)           #出现次数最多类别的下标,，对应着标签的取值
   return maxValueIndex                             #返回出现次数最多的标签



#函数功能：计算数据集的经验熵
#参考公式：李航《统计学习方法》第二版 公式5.7
#参数说明：label：训练数据集的标签数据集
def calculation_H_D(label):
    labelClass = np.unique(label)                      #对原始标签数据进行去重,得到label所有可能的取值，并且数值是升序排序
    HD=0                                               #初始化数据集的经验熵
    for labelValue in labelClass:                      #遍历label所有可能的取值
        subLabelSet=label[np.where(label==labelValue)] #提取出标签数据集中label==labelValue的数据，构成子数据集
        prob=len(subLabelSet)/len(label)               #该子集所占比例
        HD +=(-1)*prob*np.log(prob)
    return HD

#函数功能：计算经验条件熵
#参考公式：李航《统计学习方法》第二版 公式5.8
#参数说明：trainDataFeature:训练数据集被提取出的的一列特征数据，label：训练数据集的标签数据集
def calculation_H_D_A(trainDataFeature,label):
    dataValueClass = np.unique(trainDataFeature)                           #对特征数据进行去重,得到当前特征维度下特征向量所有可能的取值
    HDA=0                                                                  #初始化当前特征维度的经验条件熵
    for dataValue in dataValueClass:                                       #遍历特征维度所有可能的取值
        subDatalSet=trainDataFeature[np.where(trainDataFeature==dataValue)]#把特征维度中等于dataValue的数据全部提取出来
        subLabelSet = label[np.where(trainDataFeature == dataValue)]       #把上述子数据集对应的标签数据集提取出来
        prob=len(subDatalSet)/len(trainDataFeature)                        #该子集所占比例
        HDA +=prob*calculation_H_D(subLabelSet)
    return HDA

#函数功能：得到最佳的特征维度
#基本思路：最佳的特征划分维度就是条件经验熵最大的特征维度
#参考公式：李航《统计学习方法》第二版 公式5.9
def calcBestFeature(trainData, trainLabel):
    featureNum=trainData.shape[1]               #特征维度的数量
    informationGain=np.zeros(featureNum)        #初始化0矩阵，记录每一个特征维度的信息增益
    dataHD=calculation_H_D(trainLabel)          #数据集的经验熵
    for featureIndex in range(featureNum):
        informationGain[featureIndex]=dataHD-calculation_H_D_A(trainData[:,featureIndex],trainLabel)
    maxValueIndex = np.argmax(informationGain)  #条件经验熵最大的特征维度的下标
    maxValue = informationGain[maxValueIndex]   #获取最大的条件经验熵
    return  maxValueIndex,maxValue              #返回信息增益最大的特征向量下标、以及此时的信息增益


#函数功能：更新数据集和标签集，删除掉数据集中特征索引为featureIndex的特征维度数据
#参数说明：
# trainData:要更新的原始数据集
# trainLabel: 要更新的原始标签集
# featureIndex: 要去除的特征索引
# a:data[A]== a时，说明该行样本时要保留的
def getSubDataArr(trainData, trainLabel,featureIndex, a):
    newLabel=trainLabel[np.where(trainData[:,featureIndex]==a)]  #提取出data[：,A]== a的训练数据和标签数据
    newData=trainData[np.where(trainData[:,featureIndex]==a)]
    np.delete(arr=newData,obj=featureIndex,axis=1)               #删除featureIndex对应的特征维度
    return newData, newLabel                                     #返回更新后的数据集和标签集



#函数功能：训练决策树模型
#基本思路：采用ID3算法,参考李航《统计学习方法》第二版 算法5.2
#参数说明：dataSet=(train_data, train_label)，为元组结构
#Epsilon:信息增益的阈值,labelClassNum：原始数据集中标签有多少个类别，原始的mnist数据集有10个类别
def createTree(dataSet,labelClassNum=10,epsilon=0.05):
    trainData=dataSet[0]
    trainLabel=dataSet[1]

    #数据集为空集时，特征维度已经无法再进行划分，就返回占大多数的类别
    if trainData.shape[1]==0:
        return majorLabelClass(trainLabel)

    labelClass=np.unique(trainLabel)               #对特征数据进行去重,得到当前特征维度下特征向量所有可能的取值
    labelClassNum=np.zeros(labelClassNum)          #初始化0矩阵，用来记录每个label出现的次数

    if len(labelClass) == 1:                       #数据集中只有一个类别时，此时不需要再分化
        return  labelClass[0]                      #返回标记作为该节点的值，返回后这就是一个叶子节点


    for labelVal in labelClass:                    #遍历标签数据集所有可能的取值计算每个类别出现的次数
        labelSet=trainLabel[trainLabel==labelVal]  #统计每个类别出现的次数
        labelClassNum[labelVal]=len(labelSet)

    #计算出当前信息最大的信息增益对应的特征维度
    #参数说明：Ag：特征维度的下标索引，EpsilonGet：对应的信息增益
    Ag, EpsilonGet = calcBestFeature(trainData, trainLabel)

    # 如果Ag的信息增益比小于阈值Epsilon，则置T为单节点树，并将D中实例数最大的类Ck
    # 作为该节点的类，返回T
    if EpsilonGet<epsilon:
        return  majorLabelClass(trainLabel)

    #否则，对Ag的每一可能值ai，依Ag=ai将D分割为若干非空子集Di，将Di中实例数最大的
    #类作为标记，构建子节点，由节点及其子节点构成树T，返回T
    #在数据预处理对数据做过二值化处理，Ag的可能取值ai要么为0，要么为1
    treeDict = {Ag:{}}

    # 函数说明：getSubDataArr(trainDataList, trainLabelList, Ag, 0)
    # 在当前数据集中删除掉当前的feature，返回新的数据集和标签集
    treeDict[Ag][0] = createTree(getSubDataArr(trainData, trainLabel, Ag, 0))
    treeDict[Ag][1] = createTree(getSubDataArr(trainData, trainLabel, Ag, 1))

    return treeDict

#函数功能：基于所得到的决策树模型，对样本的标签进行预测
#参数说明：testSample：测试样本，tree：决策树模型
def labelPredict(testSample,tree):
    while True:
        # 获取树模型最顶层的key、value
        #在这个程序中，key代表的是当前节点，value对应的是下一节点或者标签类别
        key, value = tree.items()

        if type(tree[key]).__name__ == 'dict':#如果当前的value是字典，说明还需要遍历下去
            dataVal =testSample[key]          #提取出测试样本在该特征维度的数值，取值为0或1
            del testSample[key]               #去除掉测试样本在该特征维度的数值
            tree=value[dataVal]               #树节点向下移动
            if type(tree).__name__ == 'int':  #树节点移动到了叶子节点，返回该节点值，也就是分类值
                return tree
        else:                                #如果当前value不是字典，那就返回分类值
            return tree[key]

#函数说明：决策树模型测试函数
def modelTest(test_data, test_label,tree):
    errorCount = 0                                     #计数器，记录模型预测错误的次数
    for index in range(len(test_label)):
        predict=labelPredict(test_data[index],tree)  #树模型对该样本数据的标签预测值
        if predict !=test_label[index]:              #预测得到的标签与真实标签不一致时，计数器加一
            errorCount=errorCount+1
    # 返回准确率
    print("模型预测的错误率：",errorCount/len(test_label))

if __name__=="__main__":
    # 加载mnist数据集中label=0和label=+1的数据，并且将label=0改成label=-1
    print("开始加载数据")
    (train_data, train_label), (test_data, test_label)=MnistData()
    print("数据加载结束")

    #训练决策树模型
    print("开始训练模型")
    dataSet=(train_data, train_label)       #将训练数据集合标签和标签数据集组合构成元组类型
    tree=createTree((dataSet))
    print(tree)
    print("结束训练模型")

    #模型预测
    print("开始测试模型")
    modelTest(test_data, test_label, tree)
    print("结束测试模型")
