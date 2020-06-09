import tensorflow as  tf
import numpy as np

# 加载训练mnist数据集的数据集和测试数据集
def MnistData():
    #原始的训练数据集是60000张尺寸为28*28的灰色照片，测试数据集是10000张尺寸为28*28的灰色照片
    mnist = tf.keras.datasets.mnist
    (train_data, train_label), (test_data, test_label) = mnist.load_data()
    train_data = train_data.reshape(60000, 784)
    test_data = test_data.reshape(10000, 784)
    #对数据集做01处理
    train_data[train_data<=255/2]=0;train_data[train_data>255/2]=1
    test_data[test_data<=255/2]=0;test_data[test_data>255/2]=1
    return (train_data, train_label), (test_data, test_label)

#朴素贝叶斯模型的模型训练
#对训练数据集进行训练，获取先验概率和条件概率分布
#为防止浮点数太小而产生下溢出，对概率值做对数处理
def getProbability(train_data, train_label):
    print("训练模型开始")
    classNumber=len(np.unique(train_label))  #对训练数据集的label进去去重，返回互异的元素个数,此时classNumber=10
    featureNumber=train_data.shape[1]         #样本点的特征维度，featureNumber=784

    #把原数据集不同类别label的数据取出
    dataItem =[None]*classNumber
    for  i in range(classNumber):
        # 提取出train_data在train_label中元素等于不同i的元素
        # 比如item=train_data[train_label==i]，提取出train_data在label为1的全部元素
        item=train_data[train_label==i]
        dataItem[i]=item

    logPreProbability=np.zeros(classNumber)    #用来记录每个类别的先验概率
    for i in range(classNumber):   #计算先验概率，做对数化处理
        item=dataItem[i]           #提取出label=i的全部数据
        logPreProbability[i]=(item.shape[0]+1)/(len(train_label)+10)  #拉普拉斯光滑处理，防止某个类别概率为0
    logPreProbability=np.log(logPreProbability)         #对先验概率做对数化处理

    #类条件概率 P（X=x|Y = y）=P（X=x,Y = y）/P（Y = y）
    #其中P（Y = y）为先验概率，在上一步骤已求
    #数据集被进行预处理，每个特征位置只会存在0,1两个元素，要么是0要么是1
    logLikelyHood=np.zeros(shape=(classNumber,featureNumber,2))   #用来记录类条件概率
    for i in range(classNumber):        #计算类条件概率，做对数化处理
        item=dataItem[i]                #提取出label=i的全部数据
        for j in range(item.shape[0]):  #对这个label类别下的数据进行遍历
            sample=item[j]              #提取这个类别的样本点
            for k in range(featureNumber): #记录当前类别下的类条件概率
                logLikelyHood[i,k,sample[k]] +=1
        # 计算类条件概率，做对数化处理
        logLikelyHood[i]=np.log((logLikelyHood[i]+1)/item.shape[0])  #拉普拉斯平滑处理，防止出现概率值为0
    print("训练模型结束")
    return logPreProbability,logLikelyHood

#朴素贝叶斯模型
#模型测试
def naiveBayes(test_data, test_label,logPreProbability,logLikelyHood):
    print("模型测试开始")
    classNumber = len(np.unique(train_label))  # 对训练数据集的label进去去重，返回互异的元素个数,此时classNumber=10
    featureNumber = train_data.shape[1]  # 样本点的特征维度，featureNumber=784

    count=0     #记录模型预测准确的样本的总个数
    for i in range(len(test_label)):   #遍历测试数据集
        sample=test_data[i]      #提取出该样本
        prob=[0]*classNumber     #记录该样本点属于每个类别的概率
        for j in range(classNumber):   #分别计算该样本在不同类别下出现的概率
            prob[j]=logPreProbability[test_label[j]]   #当前类别对应先验概率
            for k in range(featureNumber):
                #因为概率值做了对数化处理，此时概率值的乘法变成了概率值对数的加法
                prob[j]+=logLikelyHood[j,k,sample[k]]  #当前类别下的类条件概率

        predict=np.argmax(prob)     #概率最大的种类，等同于预测的种类
        if predict==test_label[i]:  #预测正确，计数加一
            count +=1
        if i %100==0 and i !=0:    #每测试一百个样本点，就打印模型准确率
            acc = count / i
            print(' %d epoch,model accuracy is %f: ' % (i, acc))
    print("模型测试结束")

if __name__=="__main__":
    # 加载mnist数据集
    (train_data, train_label), (test_data, test_label)=MnistData()
    #对训练数据集进行训练，获取先验概率和条件概率分布
    logPreProbabilit,logLikelyHood=getProbability(train_data, train_label)
    #开始测试模型
    naiveBayes(test_data, test_label,logPreProbabilit,logLikelyHood)