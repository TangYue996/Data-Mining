
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

categories = ['alt.atheism', 'comp.graphics', 'comp.os.ms-windows.misc', 'comp.sys.ibm.pc.hardware', 'comp.sys.mac.hardware', 'comp.windows.x',
              'misc.forsale', 'rec.autos', 'rec.motorcycles', 'rec.sport.baseball' ] 
              #, 'rec.sport.hockey', 'sci.crypt', 'sci.electronics', 'sci.med', 
              #'sci.space', 'soc.religion.christian', 'talk.politics.guns', 'talk.politics.mideast', 'talk.politics.misc', 'talk.religion.misc'

twenty_train = fetch_20newsgroups(subset='train', categories=categories, shuffle=True, random_state=42) 
twenty_train.target_names 
len(twenty_train.data) 
len(twenty_train.filenames)

count_vect = CountVectorizer() 
X_train_counts = count_vect.fit_transform(twenty_train.data) 
# X_train_counts.shape 
# Out[28]: (2257, 35788) 
count_vect.vocabulary_.get(u'algorithm') 
# Out[29]: 4690 

tf_transformer = TfidfTransformer(use_idf=False).fit(X_train_counts) 
X_train_tf = tf_transformer.transform(X_train_counts) 
# X_train_tf.shape 
# Out[33]: (2257, 35788)
tfidf_transformer = TfidfTransformer() 
X_train_tfidf = tfidf_transformer.fit_transform(X_train_counts) 
# X_train_tfidf.shape 
# Out[36]: (2257, 35788) 

# k-means聚类
k = 10
clusterer = KMeans(n_clusters = k, init = 'k-means++')
y = clusterer.fit_predict(X_train_tfidf)



# clf = MultinomialNB().fit(X_train_tfidf, twenty_train.target) 
clf = KMeans().fit(X_train_tfidf, twenty_train.target) 

docs_new = ['God is love', 'OpenGL on the GPU is fast'] 
X_new_counts = count_vect.transform(docs_new) 
X_new_tfidf = tfidf_transformer.transform(X_new_counts) 
predicted = clf.predict(X_new_tfidf) 
for doc, category in zip(docs_new, predicted): 
    print('%r => %s' % (doc, twenty_train.target_names[category])) 

text_clf = Pipeline([('vect', CountVectorizer()),('tfidf', TfidfTransformer()),('clf', MultinomialNB())]) 
text_clf = text_clf.fit(twenty_train.data, twenty_train.target) 


twenty_test = fetch_20newsgroups(subset='test', categories=categories, shuffle=True, random_state=42) 
docs_test = twenty_test.data 
predicted = text_clf.predict(docs_test) 
np.mean(predicted == twenty_test.target) 
#Out[51]: 0.83488681757656458 

text_clf = Pipeline([('vect', CountVectorizer()), ('tfidf', TfidfTransformer()), ('clf', SGDClassifier(loss='hinge', penalty='l2', alpha=1e-3, n_iter=5))]) 
_ = text_clf.fit(twenty_train.data, twenty_train.target) 
predicted = text_clf.predict(docs_test) 
np.mean(predicted == twenty_test.target) 
# Out[56]: 0.9127829560585885 

print(metrics.classification_report(twenty_test.target, predicted, target_names=twenty_test.target_names)) 
print(metrics.confusion_matrix(twenty_test.target, predicted))
# Out[59]: array([[261, 10, 12, 36], [ 5, 380, 2, 2], [ 7, 32, 353, 4], [ 6, 11, 4, 377]]) 


parameters = {'vect__ngram_range': [(1, 1), (1, 2)], 'tfidf__use_idf': (True, False), 'clf__alpha': (1e-2, 1e-3)} 
gs_clf = GridSearchCV(text_clf, parameters, n_jobs=-1) 
gs_clf = gs_clf.fit(twenty_train.data[:400], twenty_train.target[:400]) 
twenty_train.target_names[gs_clf.predict(['God is love'])] 
# Out[64]: 'soc.religion.christian'

best_parameters, score, _ = max(gs_clf.cv_results_, key=lambda x: x[1]) 
for param_name in sorted(parameters.keys()): 
    print("%s: %r" % (param_name, best_parameters[param_name])) 

