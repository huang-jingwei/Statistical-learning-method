import  tensorflow as  tf
import numpy as np

#加载训练mnist数据集的数据集和测试数据集
#因为传统的adaboost是二分类模型，故选取label=0和1的样本点，并且将label=0改成label=-1
def MnistData():
    mnist = tf.keras.datasets.mnist                #通过TensorFlow加载mnist数据集
    (train_data, train_label), (test_data, test_label) = mnist.load_data()
    #原始的训练数据集是60000张尺寸为28*28的灰色照片，测试数据集是10000张尺寸为28*28的灰色照片
    train_data = train_data.reshape(60000, 784)    #原始数据集为三维向量，将其转化为二维向量
    test_data = test_data.reshape(10000, 784)
    train_label=train_label.reshape(-1)            #将标签数据集转化为行向量
    test_label=test_label.reshape(-1)
    #提取出label=0和label=1的样本点
    train_data=train_data [np.where((train_label==0)|(train_label== 1))]
    train_label=train_label[np.where((train_label==0)|(train_label == 1))]
    test_data=test_data[np.where((test_label == 0)|(test_label== 1))]
    test_label=test_label[np.where((test_label== 0)|(test_label == 1))]
    #修改label的格式，默认格式为uint8，是不能显示负数的，将其修改为int8格式
    train_label=np.array(train_label,dtype='int8')
    test_label =np.array(test_label,dtype='int8')
    train_label[np.where(train_label== 0)] = -1  #将label=0改成label=-1
    test_label [np.where(test_label == 0)] = -1
    train_data=train_data/255                    #数据进行归一化，使得图像像素点数值在0~1之间
    test_data=test_data/255
    return (train_data, train_label), (test_data, test_label)

class adaboost():
    def __init__(self,train_data, train_label,test_data, test_label,epoch=8):
        self.train_data=train_data                                           #训练数据集
        self.train_label=train_label                                         #训练标签集
        self.test_data = test_data                                           #测试数据集
        self.test_label = test_label                                         #测试标签集
        self.sampleNum=self.train_data.shape[0]                              #训练数据集的样本数目
        self.featureNum=self.train_data.shape[1]                             #特征向量的特征维度
        self.D=np.ones(shape=(self.sampleNum,self.featureNum))/self.sampleNum#初始化样本在每个维度上的分布权值
        self.epoch=epoch                                                     #弱分类器的总个数
        self.alpha=np.ones(shape=(epoch,self.featureNum))       #初始化各个弱分类在不同维度上的权值
        self.v=np.ones(shape=(epoch,self.featureNum))           #初始化各个弱分类在不同维度上的最优阈值v
        self.direction=np.ones(shape=(epoch,self.featureNum))   #初始化各个弱分类在不同维度上的最佳划分方向direction

    # direction1(+1)：x < v，y = +1；x > v,y = -1
    # direction2(-1)：x > v,y = +1；x < v，y = -1
    # 其中v是阈值轴，y为，x为实例点
    #在self.direction矩阵中，+1代表direction1，-1代表direction2

    #寻找direction1情形下(x<v y=1；x > v,y = -1)情况下的阈值v，分类误差率em,分类器占比
    #参数说明：axis:训练数据集选定的特征维度
    def lessFindBestV(self,axis):
        parameter = np.arange(0,1,0.1)                 #生成阈值的可能取值列表,原数据经过归一化，数值范围在0~1之间
        error = np.zeros(len(parameter))               #初始化各个可能取值阈值下的分类误差率
        for v_index in range(len(parameter)):          #遍历阈值轴v
            for index in range(self.sampleNum):        #遍历所有训练样本
                #记录样本点预测分类错误率，即错误预测样本点的权值之和
                if self.train_data[index,axis]<=parameter[v_index] and self.train_label[index]!=1:
                    error[v_index] =error[v_index]+self.D[index,axis]
                elif self.train_data[index,axis]>parameter[v_index] and self.train_label[index]!=-1:
                    error[v_index] = error[v_index] + self.D[index,axis]

        bestV=parameter[np.argmin(error)]  #样本点分类误差率最小对应的阈值v就是最佳阈值
        em=np.min(error)                   #对应的最佳的预测误差率
        alpha=0.5*np.log((1-em)/em)        #在当前维度上，最佳弱分类器对应的分类器的占比
        return  bestV,em,alpha

    # 寻找direction2情形下(x<v y=-1；x > v,y = +1)情况下的阈值v，分类误差率em
    #参数说明：axis:训练数据集选定的特征维度
    def moreFindBestV(self,axis):
        parameter = np.arange(0,1,0.1)                #生成阈值的可能取值列表,原数据经过归一化，数值范围在0~1之间
        error = np.zeros(len(parameter))              #初始化各个可能取值阈值下的分类误差率
        for v_index in range(len(parameter)):         #遍历阈值轴v
            for index in range(self.sampleNum):       #遍历所有训练样本
                # 记录样本点预测分类错误率，即错误预测样本点的权值之和
                if self.train_data[index,axis]<=parameter[v_index] and self.train_label[index]!=-1:
                    error[v_index] =error[v_index]+self.D[index,axis]
                elif self.train_data[index,axis]>parameter[v_index] and self.train_label[index]!=+1:
                    error[v_index] =error[v_index]+self.D[index,axis]

        bestV=parameter[np.argmin(error)]  #样本点分类误差率最小对应的阈值v就是最佳阈值
        em=np.min(error)                   #对应的最佳的预测误差率
        alpha=0.5*np.log((1-em)/em)        #在当前维度上，最佳弱分类器对应的分类器的占比
        return  bestV,em,alpha

    def modelTrain(self):
        #self.epoch为迭代训练（弱分类器总个数）的上限
        for i in range(self.epoch):               #迭代训练
            print("第%d次模型训练:训练开始"%(i+1))
            for axis in range(self.featureNum):   #对训练数据集所有的特征维度进行遍历

                # 两种方向下，最佳的阈值v，分类误差率em,分类器占比alpha
                lessBestV, lessError, lessAlpha = self.lessFindBestV(axis)
                moreBestV, moreError, moreAlpha = self.moreFindBestV(axis)

                # 预测错误差率最少时，对应的阈值v，方向direction，分类器占比alpha最优解
                #更新对应的阈值v，方向direction，分类器占比alpha矩阵
                if lessError <= moreError:
                    self.v[i, axis] = lessBestV
                    self.direction[i, axis] = +1
                    self.alpha[i,axis]=lessAlpha
                else:
                    self.v[i, axis] = moreBestV
                    self.direction[i, axis] = -1
                    self.alpha[i, axis] = moreAlpha

            print("第%d次模型训练:训练结束" % (i+1))

            # 更新训练数据集的权值分布
            self.newD(i)

            #测试所得模型
            print("第%d次模型测试:测试开始" % (i+1))
            self.modelTest(i)
            print("第%d次模型测试:测试结束" % (i+1))

    #在训练轮数（number）后，更新训练数据集的权值分布
    def newD(self,number):
        gm=self.trainDataPredict(number)     #训练数据集在每个特征维度上的标签预测值
        for axis in range(self.featureNum):  #遍历所有特征维度
            gm_axis=gm[:,axis]               #获取出在当前维度下，训练数据集的标签预测值
            # 计算当前维度上的归一化因子
            zm = np.sum(self.D[:,axis] * \
                        np.exp(-self.alpha[number,axis] * self.train_label * gm_axis))

            # 更新训练数据集在当前维度上分布权值
            for index in range(self.sampleNum):  # 遍历训练数据集所以样本点
                self.D[index, axis] = self.D[index, axis]/ zm * np.exp(
                    -self.alpha[number,axis] * self.train_label[index] * gm_axis[index])

    # 在训练轮数（number）后，更新训练数据集每个实例点的预测数值
    # direction1(+1)：x < v，y = +1；x > v,y = -1
    # direction2(-1)：x > v,y = +1；x < v，y = -1
    # 其中v是阈值轴，y为，x为实例点
    #在self.direction矩阵中，+1代表direction1，-1代表direction2
    #训练数据集在每个特征维度上的标签预测值
    def trainDataPredict(self,number):
        gm=np.zeros(shape=(self.sampleNum,self.featureNum)) #初始化每个实例点的预测数值
        for index in range(self.sampleNum):                 # 遍历所有样本
            for axis in range(self.featureNum):             # 遍历所有特征维度
                if self.direction[number,axis]==1:          # direction1(+1)：x < v，y = +1；x > v,y = -1
                    if self.train_data[index,axis]<=self.v[number,axis]:
                        gm[index,axis]+=self.alpha[number,axis]*1
                    else:
                        gm[index,axis] += self.alpha[number, axis] * (-1)
                else:                                      # direction2(-1)：x > v,y = +1；x < v，y = -1
                    if self.train_data[index,axis]>=self.v[number,axis]:
                        gm[index,axis]+=self.alpha[number,axis]*1
                    else:
                        gm[index,axis] += self.alpha[number, axis] * (-1)
        gm=np.sign(gm)      #对求和的数据进行sign函数处理就是该样本的预测
        return gm

    # 在训练轮数（times）后，更新训练数据集每个实例点的预测数值
    # direction1(+1)：x < v，y = +1；x > v,y = -1
    # direction2(-1)：x > v,y = +1；x < v，y = -1
    # 其中v是阈值轴，y为，x为实例点
    #在self.direction矩阵中，+1代表direction1，-1代表direction2
    def modelTest(self,times):
        predict = np.zeros(len(self.test_label))           # 初始化每个实例点的预测数值
        for index in range(len(self.test_label)):          # 遍历测试数据集所有样本
            for number in range(times):                    # 遍历现有训练好的所有弱分类器
                for axis in range(self.featureNum):        # 遍历测试数据集的所有特征维度
                    if self.direction[number, axis] == 1:  # 方向1
                        if self.test_data[index, axis] <= self.v[number, axis]:
                            predict[index] += self.alpha[number, axis] * 1
                        else:
                            predict[index] += self.alpha[number, axis] * (-1)
                    else:  # 方向2
                        if self.test_data[index, axis] >= self.v[number, axis]:
                            predict[index] += self.alpha[number, axis] * 1
                        else:
                            predict[index] += self.alpha[number, axis] * (-1)
            predict[index] = np.sign(predict[index])     # 对不同维度上的得到预测值进行求和，再进行sign函数处理就是该样本的预测

            if index %100==0 and index!=0:               # 每测试100个样本点就打印一次模型准确率
                errorCount = 0                           # 记录模型预测错误的样本点个数
                for i in range(index):
                    if predict[i]!=self.test_label[i]:   #标签预测值与真实标签不一致，计数器加1
                        errorCount +=1
                print("模型预测错误率为:%f" % (errorCount/index*100))

if __name__=="__main__":
    # 加载mnist数据集中label=0和label=+1的数据，并且将label=0改成label=-1
    print("加载数据集")
    (train_data, train_label), (test_data, test_label)=MnistData()
    train_data=train_data[:2000]      #加载部分训练数据集
    train_label=train_label[:2000]
    print("数据集加载结束")

    print("加载adboost模型")
    model=adaboost(train_data, train_label,test_data, test_label)
    print("训练模型")
    model.modelTrain()