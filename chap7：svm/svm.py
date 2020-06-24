import  tensorflow as  tf
import numpy as np
import random

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


#支持向量机的类
class SVM():
    #参数初始化
    def __init__(self,train_data, train_label, sigma = 10, C = 200, toler = 0.001):
        self.train_data=train_data                           #训练数据集，格式为：m*n
        self.train_label=train_label                         #训练标签集
        self.m, self.n = np.shape(self.train_data)           #m：训练集数量    n：样本特征数目
        self.sigma = sigma                                   #高斯核分母中的σ
        self.C = C                                           #软间隔的惩罚参数
        self.toler = toler                                   #松弛变量
        self.b = 0                                           #SVM模型分类超平面对应的偏置b
        self.k = self.GaussKernel()                          #高斯核函数（初始化时提前计算）
        self.alpha = np.zeros(self.m)                         # 每个样本的alpha系数，格式为：1*m
        self.supportVecIndex = []                            #存储训练数据集中支持向量的下标索引
        # self.E用来存放，每个alpha变量对应样本的标签的预测误差的Ei，格式为：m*1
        self.E = [[i, 0] for i in range(self.train_label.shape[0])]

    #高斯核函数
    #对应《统计学习方法》 第二版 公式7.90
    # k[i][j] = Xi * Xj，核函数矩阵格式：训练集长度m * 训练集长度m
    def GaussKernel(self):
        k = np.zeros(shape=(self.m,self.m))     # 初始化高斯核结果矩阵，核函数矩阵为大小为m*m的对称矩阵，m为样本个数
        for i in range(self.m):
            x = self.train_data[i]               # 得到式7.90中的x
            for j in range(i, self.m):
                z = self.train_data[j]           # 获得式7.90中的z
                # step1:先计算||x - z||^2
                step1 = np.dot((x-z),(x-z).T)
                # step2:分子除以分母后去指数，得到的即为高斯核结果
                step2 = np.exp(-1 * step1 / (2 * self.sigma ** 2))
                k[i,j] = step2           # 将Xi*Xj的结果存放入k[i][j]和k[j][i]中
                k[j,i] = step2
        return k                          # 返回高斯核矩阵

    #单独计算两个向量的高斯核函数
    #对应《统计学习方法》 第二版 公式7.90
    #两个向量的高斯核函数对应一个常数
    def SinglGaussKernel(self, x1, x2):
        step1=np.dot((x1 - x2),(x1 - x2).T)                  # step1:先计算||x - y||^2
        step2 = np.exp(-1 * step1 / (2 * self.sigma ** 2))   # step2:分子除以分母后去指数，得到的即为高斯核结果
        return step2   # 返回结果

    #第i个alpha变量对应的样本的标签预测数值g(xi)
    #对应《统计学习方法》 第二版 公式7.104
    def calc_gxi(self, i):
        gxi = 0                                                       # 初始化g(xi)
        index=[]                                                       #初始化一个空列表
        for i in range(len(self.alpha)):                               # index获得非零α的下标，,即支持向量的下标索引
            if self.alpha[i]!=0:
                index.append(i)
        for j in index:                                                #遍历非零α的下标
            gxi += self.alpha[j] * self.train_label[j] * self.k[j,i]   #g(xi)求和
        gxi += self.b                                                  #求和结束后再单独加上偏置b
        return gxi

    #函数g(x)对输入xi的预测值与真实输出yi之差
    #对应《统计学习方法》 第二版 公式7.105
    #关键变量说明：
    # gxi:第i个alpha变量对应的样本的标签预测数值g(xi)
    def calcEi(self, i):
        gxi = self.calc_gxi(i)         #计算预测数值g(xi)
        Ei= gxi - self.train_label[i]  #预测数值g(xi)与真实输出yi之差
        return Ei

    #判断第i个alpha是否满足KKT条件
    #对应《统计学习方法》 第二版 公式7.111，7.112，7.113
    #输出参数说明：
    # True：满足
    # False：不满足
    def isSatisfyKKT(self, i):
        gxi = self.calc_gxi(i)    #第i个样本标签的预测数值
        yi = self.train_label[i]  #第i个样本的真实标签

        # 依据公式7.111
        if (abs(self.alpha[i]) < self.toler) and (yi * gxi >= 1):
            return True
        # 依据公式7.112
        elif (-self.toler<self.alpha[i] and self.alpha[i] < self.C + self.toler) \
                and (abs(yi * gxi - 1) < self.toler):
            return True
        # 依据公式7.113
        elif (abs(self.alpha[i] - self.C) < self.toler) and (yi * gxi <= 1):
            return True
        return False

    #输入变量: E1: 第一个变量的ei,i: 第一个变量α1的下标
    #输出变量：E2:第二个变量的ei，第二个变量α2的下标
    def getAlphaJ(self, E1, i):
        E2 = 0          # 初始化e2
        maxE1_E2 = -1   # 初始化|E1-E2|为-1
        maxIndex = -1   # 初始化第二个变量的下标

        #获得Ei非0的对应索引组成的列表，列表内容为非0Ei的下标i
        #如果Ei全为0，则证明为第一次迭代训练
        nozeroE = [i for i, Ei in enumerate(self.E) if Ei != 0]
        for j in nozeroE:                           # 对每个非零Ei的下标i进行遍历
            E2_tmp = self.calcEi(j)                 # 计算E2
            if  abs(E1 - E2_tmp) > maxE1_E2:        # 如果|E1-E2|大于目前最大值
                maxE1_E2 = abs(E1 - E2_tmp)         # 更新最大值
                E2 = E2_tmp                         # 更新最大值E2
                maxIndex = j                        # 更新最大值E2的索引j
        if maxIndex == -1:           # 如果列表中没有非0元素了（对应程序最开始运行时的情况）
            maxIndex = i
            while maxIndex == i:     # 获得随机数，如果随机数与第一个变量的下标i一致则重新随机
                maxIndex = int(random.uniform(0, self.m))
            E2 = self.calcEi(maxIndex)     # 获得E2
        return E2, maxIndex                # 返回第二个变量的E2值以及其索引

    #训练模型
    def train(self, epoch=100):
        # parameterChanged：单次迭代中有参数改变则增加1
        parameterChanged = 1    # 状态变量，用来判断alpha相邻两次迭代训练是否发生变化
        iterStep = 0            # iterStep：迭代次数，超过设置次数上限epoch还未收敛则强制停止


        # 如果没有达到限制的迭代次数以及上次迭代中有参数改变则继续迭代
        # parameterChanged==0时表示上次迭代没有参数改变，如果遍历了一遍都没有参数改变，说明
        # 达到了收敛状态，可以停止了
        while (iterStep < epoch) and (parameterChanged > 0):
            print('iter:%d:%d' % (iterStep, epoch))   # 打印当前迭代轮数
            iterStep += 1                             # 迭代步数加1
            parameterChanged = 0                      # 新的一轮将参数改变标志位重新置0

            for i in range(self.train_data.shape[0]): # 大循环遍历所有样本，用于找SMO中第一个变量
                # 查看第一个遍历是否满足KKT条件，如果不满足则作为SMO中第一个变量从而进行优化
                if self.isSatisfyKKT(i) == False:
                    # 如果下标为i的α不满足KKT条件，则将该变量视作第一个优化变量，进行优化
                    E1 = self.calcEi(i)             # 选择第1个优化变量
                    E2, j = self.getAlphaJ(E1, i)   # 选择第2个优化变量

                    y1 = self.train_label[i]        # 获得两个优化变量的真实标签
                    y2 = self.train_label[j]

                    alphaOld_1 = self.alpha[i].copy()  # 复制α值作为old值
                    alphaOld_2 = self.alpha[j].copy()

                    # 依据标签是否一致来生成第二个优化变量α2的约束上下界限L和H
                    # 对应《统计学习方法》 第二版 公式7.103以及7.104之间的公式
                    if y1 != y2:
                        L = max(0, alphaOld_2 - alphaOld_1)
                        H = min(self.C, self.C + alphaOld_2 - alphaOld_1)
                    else:
                        L = max(0, alphaOld_2 + alphaOld_1 - self.C)
                        H = min(self.C, alphaOld_2 + alphaOld_1)
                    # 如果两者相等，说明该变量无法再优化，直接跳到下一次循环
                    if L == H:
                        continue

                    # 计算α的新值
                    # 对应《统计学习方法》 第二版 公式7.106更新α2值
                    # 先获得几个k值，用来计算事7.106中的分母η
                    k11 = self.k[i,i]                                          #先获得不同位置的核函数
                    k22 = self.k[j,j]
                    k21 = self.k[j,i]
                    k12 = self.k[i,j]
                    r=k11 + k22 - 2 * k12                                        #对应《统计学习方法》 第二版 公式7.107的分母η
                    alphaNew_2 = alphaOld_2 + y2 * (E1 - E2) / r                 # 依据式7.106更新α2，该α2还未经剪切
                    if alphaNew_2 < L:                                           # 剪切α2，对应《统计学习方法》 第二版 公式7.108
                        alphaNew_2 = L
                    elif alphaNew_2 > H:
                        alphaNew_2 = H
                    alphaNew_1 = alphaOld_1 + y1 * y2 * (alphaOld_2 - alphaNew_2) # 更新α1，依据式7.109

                    # 对应《统计学习方法》 第二版 公式7.115和7.116
                    #更新偏执b，计算b1和b2
                    b1New = -1 * E1 - y1 * k11 * (alphaNew_1 - alphaOld_1) \
                            - y2 * k21 * (alphaNew_2 - alphaOld_2) + self.b
                    b2New = -1 * E2 - y1 * k12 * (alphaNew_1 - alphaOld_1) \
                            - y2 * k22 * (alphaNew_2 - alphaOld_2) + self.b

                    # 依据α1和α2的值范围确定新b
                    if (alphaNew_1 > 0) and (alphaNew_1 < self.C):
                        bNew = b1New
                    elif (alphaNew_2 > 0) and (alphaNew_2 < self.C):
                        bNew = b2New
                    else:
                        bNew = (b1New + b2New) / 2

                    # 将更新后的各类值写入，进行更新
                    self.alpha[i] = alphaNew_1
                    self.alpha[j] = alphaNew_2
                    self.b = bNew

                    self.E[i] = self.calcEi(i)
                    self.E[j] = self.calcEi(j)

                    # 如果α2的改变量过于小，就认为该参数未改变，不增加parameterChanged值
                    # 反之则自增1
                    if abs(alphaNew_2 - alphaOld_2) >= 0.00001:
                        parameterChanged += 1

                # 打印迭代轮数，i值，该迭代轮数修改α数目
                print("iter: %d i:%d, pairs changed %d" % (iterStep, i, parameterChanged))

        # 全部计算结束后，重新遍历一遍α，查找里面的支持向量
        for i in range(self.m):
            # 如果α>0，说明是支持向量
            if self.alpha[i] > 0:
                # 将支持向量的索引保存起来
                self.supportVecIndex.append(i)

    #对样本的标签进行预测
    def predict(self,sample):
        result = 0
        # 遍历所有支持向量，计算求和式
        #因为非支持向量的系数alpha为0
        for i in self.supportVecIndex:
            tmp = self.SinglGaussKernel(self.train_data[i], sample)      #先单独将核函数计算出来
            result += self.alpha[i] *tmp                                 #对每一项子式进行求和，最终计算得到求和项的值
        result = np.sign(result+self.b)                                  #求和项计算结束后加上偏置b,再进行sign函数处理
        return result

    #svm模型测试
    def test(self, test_data, test_label):
        errorCnt = 0                               # 错误计数值
        for i in range(test_data.shape[0]):        # 遍历测试集所有样本
            result = self.predict(test_data[i])    # 获取预测结果
            if result != test_label[i]:            # 如果预测与标签不一致，错误计数值加一
                errorCnt += 1
        acc=1 - errorCnt / test_data.shape[0]      #模型预测准确率为
        print("模型预测准确率为:%f" %(acc))         #打印预测准确率
        return   acc                               # 返回正确率

if __name__=="__main__":
    # 加载mnist数据集中label=0和label=+1的数据，并且将label=0改成label=-1
    (train_data, train_label), (test_data, test_label)=MnistData()

    #初始化SVM类
    print('start init SVM')
    svm = SVM(train_data[:1000], train_label[:000],sigma = 10, C = 200, toler = 0.001)

    # 开始训练
    print('start to train')
    svm.train()

    # 开始测试
    print('start to test')
    accuracy = svm.test(test_data[:1000], test_label[:1000])
    print('the accuracy is:%d' % (accuracy * 100), '%')