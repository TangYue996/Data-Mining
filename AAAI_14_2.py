# -*- coding: utf-8 -*-

import re
import time
import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from sklearn.cluster import Birch
from sklearn.cluster import DBSCAN
from sklearn.metrics import silhouette_samples, silhouette_score
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.feature_extraction.text import CountVectorizer
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import nltk
from nltk.tokenize import sent_tokenize, word_tokenize
from nltk.tokenize import WordPunctTokenizer
import codecs
 
# 导入数据集函数，返回聚类的数据与对应ID
def loadDataSet(filename):
    dataset = pd.read_csv(filename, encoding='utf-8', usecols=[2])
    m, n = dataset.shape  # 获取行、列
    data = dataset.values[:,-1]
    dataID = dataset.values[:,0]
    return data.reshape((m,1)), dataID.reshape((m,1))

# numpy 转化为 list
def ndarrayToList(dataArr):
    dataList = []
    m, n = dataArr.shape
    for i in range(m):
        for j in range(n):
            dataList.append(dataArr[i,j])
    return dataList
 
# 去掉字符串、特殊符号
def removeStr(listData):
    strData = "".join(listData)
    removeStrData = re.sub("[\s+\!\,$^*()+\"\'\\]+:|[+——！，,《》“”〔【】；：。？、�./-~@#￥……&*（）]+", "",strData)
    return removeStrData

# 对数据集分词、去停用词
def wordSplit(data):
    # stopword = open('F:/Data_Mining/Data_set/AAAI_14/stopwords.txt', 'r', encoding = 'utf-8').read()
    stopwords = [line.strip() for line in open('F:/Data_Mining/Data_set/AAAI_14/stopwords.txt','r',encoding = 'utf-8').readlines()]
    word = ndarrayToList(data)
    #print(word)
    m = len(word)
    #print(m)
    wordList = []
    
    for i in range(0,m):
        print(i)
        #rowListRemoveStr = removeStr(word[i])    # 去特殊符号
        #print(rowListRemoveStr)
        #line = line.strip()  #去前后的空格
        #word[i] = re.sub(r"[0-9\s+\.\!\/_,$%^*()?;；:-【】+\"\']+|[+——！，;:。？、~@#￥%……&*（）]+", " ", word[i]) #去标点符号
        rowList = [eachWord for eachWord in word_tokenize(word[i])]  # 分词
        print(rowList)
        removeStopwordList = []
        for eachword in rowList:
            if eachword not in stopwords and eachword != '\t' and eachword != ' ' :
                removeStopwordList.append(eachword)
        wordList.append(removeStopwordList)
    return wordList

# 保存文件
def saveFile(filename, dataSplit):
    with open(filename, 'a', encoding = 'utf-8') as fr:
        for line in dataSplit:
            strLine = ' '.join(line)
            fr.write(strLine)    
            fr.write('\n')
        fr.close()

# 计算 tf-idf 值
def TFIDF(wordList):
    corpus = []   # 保存预料
    for i in range(2, len(wordList)):
        wordList[i] = " ".join(wordList[i])
        corpus.append(wordList[i])
    # 将文本中的词语转换成词频矩阵,矩阵元素 a[i][j] 表示j词在i类文本下的词频
    vectorizer = CountVectorizer(token_pattern='(?u)\\b\\w+\\b')  # 
    # 该类会统计每个词语tfidf权值
    transformer = TfidfTransformer()
    # 第一个fit_transform是计算tf-idf 第二个fit_transform是将文本转为词频矩阵
    tfidf = transformer.fit_transform(vectorizer.fit_transform(corpus))
    # 获取词袋模型中的所有词语
    word = vectorizer.get_feature_names()
    # 将tf-idf矩阵抽取出来，元素w[i][j]表示j词在i类文本中的tf-idf权重  
    weight = tfidf.toarray()
    
    return word,weight
 
# 对生成的 tfidf 矩阵做PCA降维
def matrixPCA(weight):
    pca = PCA(n_components = 2)  # 初始化PCA
    pcaMatrix = pca.fit_transform(weight)        # 返回降维后的数据
    print("降维之前的权重维度：",weight.shape)
    print("降维之后的权重维度：",pcaMatrix.shape)
    return pcaMatrix
 
# birch聚类 
def birch(matrix,k):
    clusterer = Birch(n_clusters = k)  # 分成簇的个数
    y = clusterer.fit_predict(matrix)    # 聚类结果
    return y

# k-means聚类
def kmeans(matrix, k):
    clusterer = KMeans(n_clusters = k, init = 'k-means++')
    y = clusterer.fit_predict(matrix)
    return y

# DBSCAN聚类
def dbscan(matrix):
    clusterer = DBSCAN(eps = 4, min_samples = 2)
    y = clusterer.fit_predict(matrix)
    return y

# 计算轮廓系数
def Silhouette(matrix, y):
    silhouette_avg = silhouette_score(matrix, y)   # 平均轮廓系数
    sample_silhouette_values = silhouette_samples(matrix, y)  # 每个点的轮廓系数
    print(silhouette_avg)
    return silhouette_avg, sample_silhouette_values

# 画图
def Draw(silhouette_avg, sample_silhouette_values, y, k):
    fig, ax1 = plt.subplots(1)
    fig.set_size_inches(8, 6)
    # 第一个 subplot 放轮廓系数点
    # 范围是[-1, 1]
    ax1.set_xlim([-0.1, 1])
    # 后面的 (k + 1) * 10 是为了能更明确的展现这些点
    #ax1.set_ylim([1, 10])
    y_lower = 10
 
    for i in range(k): # 分别遍历这几个聚类
        ith_cluster_silhouette_values = sample_silhouette_values[y == i]
        ith_cluster_silhouette_values.sort()
        print(ith_cluster_silhouette_values)

        size_cluster_i = ith_cluster_silhouette_values.shape[0]
        y_upper = y_lower + size_cluster_i
        # color = cm.Spectral(float(i)/k)
        ax1.fill_betweenx(np.arange(y_lower, y_upper),
                          0,
                          ith_cluster_silhouette_values,
                          facecolor = 'gray',
                          #edgecolor = 'gray',
                          #alpha=0.7
                          )
        # 在轮廓系数点这里加上聚类的类别号
        ax1.text(-0.05, y_lower + 0.5 * size_cluster_i, str(i))
        # 计算下一个点的 y_lower y轴位置
        y_lower = y_upper + 10
    # 在图里搞一条垂直的评论轮廓系数虚线
    ax1.axvline(x=silhouette_avg, color='red', linestyle="--")
    plt.show()

if __name__ == "__main__":
    
    # start time
    start = time.perf_counter()
    k = 10  # 聚成12类
    # jieba.load_userdict('F:/Data_Mining/Data_set/AAAI_14/user_dict.txt')  # 添加分词字典
    data, dataId = loadDataSet('F:/Data_Mining/Data_set/AAAI_14/AAAI-14 Accepted Papers - Papers.csv')
    #print(data)
    dataSplit = wordSplit(data)
    print('分词完成')
    saveFile('F:/Data_Mining/Data_set/AAAI_14/word-split.txt', dataSplit)  # 保存分词结果
    word, weight = TFIDF(dataSplit)  # 生成 tfidf 矩阵
    weightPCA = weight
 
    # 将原始矩阵降维，降维后效果反而没有不降维的好
    # weightPCA = matrixPCA(weight) 
    # y = birch(weightPCA, k)
    y = kmeans(weightPCA, k)
    # y = dbscan(weightPCA)

    silhouette_avg, sample_silhouette_values = Silhouette(weightPCA, y) # 轮廓系数
    Draw(silhouette_avg, sample_silhouette_values, y, k)
    
    elapsed = (time.perf_counter() - start)
    print('Time use', elapsed)

