
from sklearn.datasets import fetch_20newsgroups
from sklearn.feature_extraction.text import CountVectorizer 
from sklearn.feature_extraction.text import TfidfTransformer 
from sklearn.naive_bayes import MultinomialNB 
from sklearn.pipeline import Pipeline 
from sklearn.linear_model import SGDClassifier 
from sklearn import metrics
from sklearn.model_selection import GridSearchCV 
import numpy as np
from sklearn.cluster import KMeans
from sklearn.cluster import Birch
from sklearn.cluster import DBSCAN

categories = ['alt.atheism', 'comp.graphics', 'comp.os.ms-windows.misc', 'comp.sys.ibm.pc.hardware', 'comp.sys.mac.hardware', 'comp.windows.x',
              'misc.forsale', 'rec.autos', 'rec.motorcycles', 'rec.sport.baseball' ,
              'rec.sport.hockey', 'sci.crypt', 'sci.electronics', 'sci.med', 
              'sci.space', 'soc.religion.christian', 'talk.politics.guns', 'talk.politics.mideast', 'talk.politics.misc', 'talk.religion.misc']

# 加载数据集
twenty_train = fetch_20newsgroups(subset='train', categories=categories, shuffle=True, random_state=42) 
# print(twenty_train.target_names )
print(len(twenty_train.data))
#print(len(twenty_train.filenames))

# 提取tfidf特征
count_vect = CountVectorizer() 
X_train_counts = count_vect.fit_transform(twenty_train.data) 
count_vect.vocabulary_.get(u'algorithm') 

tf_transformer = TfidfTransformer(use_idf=False).fit(X_train_counts) 
X_train_tf = tf_transformer.transform(X_train_counts) 
tfidf_transformer = TfidfTransformer() 
X_train_tfidf = tfidf_transformer.fit_transform(X_train_counts) 

# 训练
clf = MultinomialNB().fit(X_train_tfidf, twenty_train.target) 
docs_new = ['God is love', 'OpenGL on the GPU is fast'] 
X_new_counts = count_vect.transform(docs_new) 
X_new_tfidf = tfidf_transformer.transform(X_new_counts) 

# 预测
predicted = clf.predict(X_new_tfidf) 
for doc, category in zip(docs_new, predicted): 
    print('%r => %s' % (doc, twenty_train.target_names[category])) 

text_clf = Pipeline([('vect', CountVectorizer()),('tfidf', TfidfTransformer()),('clf', MultinomialNB())]) 
text_clf = text_clf.fit(twenty_train.data, twenty_train.target) 

# 用测试机评估模型好坏
twenty_test = fetch_20newsgroups(subset='test', categories=categories, shuffle=True, random_state=42)
print(len(twenty_test.data))
docs_test = twenty_test.data 
predicted = text_clf.predict(docs_test) 
np.mean(predicted == twenty_test.target) 

text_clf = Pipeline([('vect', CountVectorizer()), ('tfidf', TfidfTransformer()), ('clf', SGDClassifier(loss='hinge', penalty='l2', alpha=1e-3, n_iter=5))]) 
_ = text_clf.fit(twenty_train.data, twenty_train.target) 
predicted = text_clf.predict(docs_test) 
np.mean(predicted == twenty_test.target) 

print(metrics.classification_report(twenty_test.target, predicted, target_names=twenty_test.target_names)) 
print(metrics.confusion_matrix(twenty_test.target, predicted))


parameters = {'vect__ngram_range': [(1, 1), (1, 2)], 'tfidf__use_idf': (True, False), 'clf__alpha': (1e-2, 1e-3)} 
gs_clf = GridSearchCV(text_clf, parameters, n_jobs=-1) 
gs_clf = gs_clf.fit(twenty_train.data[:400], twenty_train.target[:400]) 
twenty_train.target_names[gs_clf.predict(['God is love'])] 

best_parameters, score, _ = max(gs_clf.cv_results_, key=lambda x: x[1]) 
for param_name in sorted(parameters.keys()): 
    print("%s: %r" % (param_name, best_parameters[param_name])) 

