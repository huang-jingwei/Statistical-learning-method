github链接：[https://github.com/gdutthu/Statistical-learning-method](https://github.com/gdutthu/Statistical-learning-method)
知乎专栏链接：[https://zhuanlan.zhihu.com/c_1252919075576856576](https://zhuanlan.zhihu.com/c_1252919075576856576)

**算法总结：**
1、一个重要定理（贝叶斯定理）、一个重要前提假设（特征相互独立）
（可以通俗的理解为：**贝叶斯公式 + 条件独立假设 = 朴素贝叶斯方法**）。
2、可以进行多分类工作。
3、朴素贝叶斯和逻辑斯特回归在建模工作的时候，都采用$log$来求解概率。这两者的不同在于：
**朴素贝叶斯：** 
（1）生成模型；
（2为了防止在计算$p(x_{1},x_{2},...,x_{n})=p(x_{1}).p(x_{2})...p(x_{n})$因为数值太小而出现浮点数下溢。因此进行对公式取对数，变成了$log  p(x_{1},x_{2},...,x_{n})=logp(x_{1})+logp(x_{2})+...+logp(x_{n})$。但是**这个步骤不是必须的。**

**逻辑斯特回归：** 
（1）判别模型。
（2）对最大似然函数进行$log$，结合梯度下降方法便于更好的求解参数$w,b$。

相关问题链接：[https://www.zhihu.com/question/265995680](https://www.zhihu.com/question/265995680)
# 1 贝叶斯定理
https://blog.csdn.net/guoyunfei20/article/details/78911721
## 1.1 正向概率
**正向概率通俗地讲指的是我们往往是上帝视角，即了解了事情的全貌再做判断。**
试着思考这个问题，已知袋子里面有 N 个球，其中 M 个黑球，剩下的均为白球。那么把手伸进去摸一个球，问摸出黑球的概率是多少。因为我们此时已经对系统有了全局的认识，这个时候计算起来就非常地简单。
![在这里插入图片描述](C:\Users\hellp\Desktop\统计学习方法\image\白球黑球问题.png)
如：一个袋子里有10个球，其中6个黑球，4个白球；那么随机抓一个黑球的概率是0.6！

## 1.2 逆向概率
**逆向概率：在没有得到一个系统全部信息时，要我们根据已有信息（先验知识，在概率模型里面表现为先验概率），对系统信息进行反推。** 
思考一个问题：酒鬼有90%概率外出喝酒，只有可能在A、B、C三个酒吧，概率相等，警察想去抓酒鬼，已知去了前两个酒吧都没抓到他，求去第三个酒吧抓到酒鬼的概率。

**先验概率：通过经验来判断事情发生的概率**，比如酒鬼有90%概率外出喝酒，就是先验概率。

**贝叶斯定理表达式如下**
$$P(Y | X)=\frac{P(Y) P(X | Y)}{P(X)}$$
其中，$P(Y | X)$为待求的后验概率，$P(X | Y)$为类条件概率，$P(X )$为证据，$P_{\text {prior}}=P(Y)$为先验概率

那么为了求解例子这个问题，我们首先先要对事件进行一些简单的假设
事件$X$为：酒鬼外出喝酒，$\overline{X}$：酒鬼不喝酒
事件$Y$为：酒鬼在前面两个酒吧被抓，$\overline{Y}$：酒鬼在前面两个酒吧没有被抓
那么根据先验知识，我们可得
$P(X)=\frac{90}{100}$,$P(\overline{X})=1-P(X)=\frac{10}{100}$
如果酒鬼没有喝酒，那么酒鬼就不可能被抓
$P(Y | \overline{X})=0$，$P(\overline{Y} | \overline{X})=1$
如果酒鬼外出喝酒，如果在前面两个酒吧没被抓，那么酒鬼必然是在第三个酒吧喝酒
$P(\overline{Y} | X)=\frac{1}{3}$
那么原问题：已知去了前两个酒吧都没抓到他，求去第三个酒吧抓到酒鬼的概率。也就是警察去在前两个酒吧没抓到酒鬼的条件下，在第三个酒吧抓到酒鬼（酒鬼外出喝酒）的概率
$$\begin{aligned}
P( X | \overline{Y})&=\frac{P(X)P(\overline{Y} | X)}{P(\overline{Y})}\\
&=\frac{P(X)P(\overline{Y} | X)}{P(X)P(\overline{Y} | X)+P(\overline{X})P(\overline{Y} | \overline{X})}\\
&=\frac{\frac{90}{100}*\frac{1}{3}}{\frac{90}{100}*\frac{1}{3}+\frac{10}{100}*1}\\
&=\frac{3}{4}
\end{aligned}$$

# 2 提出模型
在上面一小节中，我们得到贝叶斯公式如下：
$$P(Y | X)=\frac{P(Y) P(X | Y)}{P(X)}$$
在实际数据集的特征向量$x=\left(x^{(1)}, x^{(2)}, \cdots\right.$$\left.x^{(n)}\right)^{\mathrm{T}}$中，不同的特征维度可能存在多种依赖关系（依赖形式包括但不限于线性、指数、周期等等）。
因为$x^{(1)}, x^{(2)},..., x^{(n)}$的不同的组合情况有$2^{n}-1$种.
$$c_{n}^{0}+c_{n}^{1}+...+c_{n}^{n}   =  2^{n}-1$$
那么计算类条件概率$P(X | Y)=P(x^{(1)}, x^{(2)},..., x^{(n)}| Y)$就变成了$np$难得问题
为了解决这个问题，研究员提出了**一个强假设：特征向量中不同特征之间相互无光。** 那么类条件概率就变成了以下情形
$$\begin{aligned}
P(x^{(1)}, x^{(2)},..., x^{(n)}| Y) &=  P(x^{(1)}| Y) *P( x^{(2)}| Y) *... * P( x^{(n)}| Y)\\
&=\prod_{j=1}^{n} P\left(x^{(j)} | Y\right)
\end{aligned}$$
进一步可得
$$\begin{aligned}
P\left(X=x | Y=c_{k}\right) &=P\left(X^{(1)}=x^{(1)}, \cdots, X^{(n)}=x^{(n)} | Y=c_{k}\right) \\
&=\prod_{i=1}^{n} P\left(X^{(j)}=x^{(j)} | Y=c_{k}\right)
\end{aligned}$$
此时我们只需要单独计算每个特征维度的类条件概率，就可以得到该特征总的类条件概率。
注：常见的贝叶斯模型的从属关系
![在这里插入图片描述](C:\Users\hellp\Desktop\统计学习方法\image\贝叶斯模型.png)





# 3 算法流程
那么结合上面所讲的内容，可以看出朴素贝叶斯算法可以分为三大阶段，四个步骤（算法步骤在第三小节中给出）。具体如下
![在这里插入图片描述](C:\Users\hellp\Desktop\统计学习方法\image\朴素贝叶斯的计算流程.png)
**输入：**
训练数据$T=\left\{\left(x_{1}, y_{1}\right),\left(x_{2}, y_{2}\right), \cdots,\left(x_{N}, y_{N}\right)\right\}$，其中$x_{i}=\left(x_{i}^{(1)}, x_{i}^{(2)}, \cdots\right.$$\left.x_{i}^{(n)}\right)^{\mathrm{T}}$，$x_{i}^{(j)}$是第$i$个样本的第$j$个特征，$x_{i}^{(j)} \in\left\{a_{j 1}, a_{j 2}, \cdots, a_{j S_{j}}\right\}$，$a_{j l}$是第$j$个特征可能取到的第$l$个值，$j=1,2, \cdots, n, l=1,2, \cdots, S_{j}, y_{i} \in\left\{c_{1}, c_{2}, \cdots, c_{K}\right\}$；实例$x$;
**输出：**
实例$x$的分类。
step1：计算先验概率以及条件概率
$$P\left(Y=c_{k}\right)=\frac{\sum_{i=1}^{N} I\left(y_{i}=c_{k}\right)}{N}, \quad k=1,2, \cdots, K$$
$$P\left(X^{(j)}=a_{j l} | Y=c_{k}\right)=\frac{\sum_{i=1}^{N} I\left(x_{i}^{(j)}=a_{j l}, y_{i}=c_{k}\right)}{\sum_{i=1}^{N} I\left(y_{i}=c_{k}\right)}$$
$$j=1,2, \cdots, n ; \quad l=1,2, \cdots, S_{j} ; \quad k=1,2, \cdots, K$$
step2:对于给定的实例$x=\left(x^{(1)}, x^{(2)}, \cdots, x^{(n)}\right)^{\mathrm{T}}$，计算
$$P\left(Y=c_{k}\right) \prod_{j=1}^{n} P\left(X^{(j)}=x^{(j)} | Y=c_{k}\right), \quad k=1,2, \cdots, K$$
step3：确定实例$x$的类别
$$y=\arg \max _{c_{k}} P\left(Y=c_{k}\right) \prod_{j=1}^{n} P\left(X^{(j)}=x^{(j)} | Y=c_{k}\right)$$

**注意：**
在前面分析的朴素贝叶斯与逻辑斯特回归模型的不同点时，指出为了防止在step2，step3出现因为因为概率的数值太小而出现浮点数下溢。可以对step2，step3进行对公式取对数，变成了$log  p(x_{1},x_{2},...,x_{n})=logp(x_{1})+logp(x_{2})+...+logp(x_{n})$。但是**这个步骤不是必须的。**
# 4 代码附录
在这里采用mnist数据集进行朴素贝叶斯法多分类实验，采用TensorFlow2.0进行加载数据（懒得写函数加载模块了hhh）。在代码环节中，对测试集中的所有实例点都进行了测试，所需时间较长。如果想要测试部分样本点，稍微修改下代码即可。
**注意点：**

1、在求解先验概率时，为防止部分类别的概率为0（或太小），故增加了拉普拉斯平滑处理。
2、结合上文所提到的，在计算概率时，为防止概率数值太小而出现浮点数下溢，所以增加了对数化处理步骤。

```python
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
```

