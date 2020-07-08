import numpy as np
import matplotlib.image as mping
import matplotlib.pyplot as plt
import matplotlib as mpl

#函数功能：对目标矩阵进行svd分解，提取前n个主成分
#参数说明：n:需要提取前n个特征值,pic:目标矩阵
def image_svd(n, pic):
    a, b, c = np.linalg.svd(pic)                #分别为原始矩阵经SVD分解后所得：U,σ，V矩阵
    svd = np.zeros((a.shape[0],c.shape[1]))     #生成所需的图像矩阵，初始化为零矩阵
    for i in range(0, n):                       #提取出前n个特征值
        svd[i, i] = b[i]
    img = np.matmul(a, svd)                    #将处理后的矩阵分别进行矩阵乘法，将其合并起来
    img = np.matmul(img, c)
    img[ img >= 255] = 255                     #将处理后的矩阵像素值的数值限制在0~255之间
    img[  0 >= img ] = 0
    img = img.astype(np.uint8)
    return img


if __name__=="__main__":
    plt.rcParams['font.sans-serif'] = ['SimHei']  # 用来正常显示中文标签
    plt.rcParams['axes.unicode_minus'] = False    # 用来正常显示负号

    fileName = 'sample.jpg'                       # 样本图像的名称
    img = mping.imread(fileName)                  # 加载原始样本图像信息
    print("图像的原始尺寸:",img.shape)             # 打印样本图像的原始尺寸

    # 新建figure对象,用来展示原始样本图像信息
    fig = plt.figure()
    plt.imshow(img)                       # 将原始的样本照片打印出来
    plt.title("原始样本图像" )             # 图像标题
    plt.axis('on')                        # 显示坐标轴

    # 分别获取图像矩阵的rgb三通道数据
    r = img[:, :, 0]
    g = img[:, :, 1]
    b = img[:, :, 2]

    # 新建figure对象,用来展示原始样本经SVD处理后的图像信息
    plt.figure(figsize=(50, 100))                     #定义所需显示的画布的尺寸
    for i in range(1,31):                             #逐次提取出前30个重要特征值
        r_img = image_svd(i, r)                       #对rgb三通道信息矩阵分别进行SVD分解
        g_img = image_svd(i, g)
        b_img = image_svd(i, b)
        pic = np.stack([r_img, g_img, b_img], axis=2)  #将经SVD分解处理后的rgb三通道矩阵合并
        print("图像的SVD分解，使用前 %d 个特征值"%(i))   #记录此次SVD分解，提取多少个特征值
        plt.subplot(5, 6, i)                           #打印处理后的目标图像
        plt.title("使用前 %d 个特征值" % (i))           #打印图像的一些常规设置
        plt.axis('off')
        plt.imshow(pic)
    plt.suptitle("图像的SVD分解")
    plt.subplots_adjust()
    plt.show()
