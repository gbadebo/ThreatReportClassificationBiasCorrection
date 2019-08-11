
# coding: utf-8
from multistream import main   #import this for bias correction
# In[3]:
from collections import defaultdict

import pandas as pd
import sys
splitratio = sys.argv[1]
biastype = sys.argv[2]
featuretype = sys.argv[3] #w2v or now2v
classifier_type = sys.argv[4] #svm or knn
def readFile(filename):
    try:
        with open(filename, 'r') as myfile:
            data=myfile.read().replace('\n', '')
            return data
    except:
        print filename + " no such file"
        return ""
df=pd.read_csv("labelattack.csv")
tactics = df.columns.values.tolist()
'''data_dict = dict.fromkeys(tactics)

for k in data_dict:
    data_dict[k] = ""

for col in df.columns.values.tolist():
    for name in df[col].T.tolist():
        if str(name) != "nan": 
            filename =  str(name).replace(" ","_").replace('/','').replace(".","")
            ##print filename
            ##print readFile(filename)
            filename = str("data/")+filename+".txt"
            data_dict[col]  = data_dict[col] + " " + readFile(filename)
'''
trainlist=[]
trainlabel=[]
for col in df.columns.values.tolist():
    for name in df[col].T.tolist():
        if str(name) != "nan": 
            filename =  str(name).replace(" ","_").replace('/','').replace(".","")
            ##print filename
            ##print readFile(filename)
            filename = str("/home/gbaduz/Downloads/attack project/Profile-master/Bias_Correction/data/")+filename+".txt"
            trainlist.append( readFile(filename))
            trainlabel.append(col.strip())
##print trainlist
##print trainlabel   
dfmap=pd.read_csv("mapper.txt",sep='\t')
mytrainlabel = []
#translate mitre labl to mandiant labels
for l in trainlabel:
    for i in [7,6,5,4,3,2]:
        mylist = [tactic.strip() for tactic in dfmap.loc[i][3].split(",")]
        if l in mylist:
            mytrainlabel.append(i+1)
            break


print  mytrainlabel


'''
all you need to do now is to add new test data  to test dir and update the filename in the testfiles
'''
testfolder = "/home/gbaduz/Downloads/attack project/Profile-master/Bias_Correction/test"
#testfolder = "/home/gbaduz/Downloads/attack project/cyberthreatextraction/scrapers/symantec_reports"
testfiles = pd.read_csv(testfolder+"/testfiles", header=None,sep='\t')
testtactics = testfiles[0].T.tolist()
test_data_dict = dict.fromkeys(testtactics)



for k in test_data_dict:
    test_data_dict[k] = ""
for f in testfiles[0].T.tolist():
    test_data_dict[f] =  test_data_dict[f] +  readFile(str(testfolder)+"/"+f)

##print test_data_dict
import re

'''alldata = []
for k in data_dict:
    alldata.extend(re.split(r'[;,\s]\s*', data_dict[k]))

for k in test_data_dict:
    alldata.extend(re.split(r'[;,\s]\s*', test_data_dict[k]))
##print set(alldata)  
'''

alldata = []
for k in trainlist:
    temp = re.split(r'[;,\s]\s*', k)
    alldata.append(temp)

for k in testfiles[0].T.tolist():
    temp = re.split(r'[;,\s]\s*', test_data_dict[k])
    alldata.append(temp)
    



import os.path
import gensim
# let X be a list of tokenized texts (i.e. list of lists of tokens)
if os.path.exists("./expmodel"):
    print "expmodel exist"
    model = gensim.models.Word2Vec.load("./expmodel")
    w2v = dict(zip(model.wv.index2word, model.wv.syn0))
else:
    print "expmodel does not exist"
    model = gensim.models.Word2Vec(alldata, size=400)
    model.save("./expmodel")
    w2v = dict(zip(model.wv.index2word, model.wv.syn0))


import nltk
import string
import os
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import HashingVectorizer
from nltk.stem.porter import PorterStemmer


#nltk.download('punkt')
stemmer = PorterStemmer()

def stem_tokens(tokens, stemmer):
    stemmed = []
    for item in tokens:
        stemmed.append(stemmer.stem(item))
    return stemmed

def tokenize(text):
    #text = unicode(text, errors='replace')
    tokens = re.split(r'[;,\s]\s*',text)
    stems = stem_tokens(tokens, stemmer)
    return stems


class TfidfEmbeddingVectorizer(object):
    def __init__(self, word2vec):
        self.word2vec = word2vec
        self.word2weight = None
        self.dim = len(word2vec.itervalues().next())

    def fit(self, X, y):
        tfidf = TfidfVectorizer(tokenizer=tokenize, stop_words='english',decode_error='ignore',max_features = 1000)
        tfidf.fit(X)
        # if a word was never seen - it must be at least as infrequent
        # as any of the known words - so the default idf is the max of 
        # known idf's
        max_idf = max(tfidf.idf_) # if word not seen uses this as default that is what lambda is for. idf is global for each word in the corpus. idf means inverse doc freq which is inverse of num of docs that contain the term log N_totalno of doc/freqterm in doc
        self.word2weight = defaultdict(
            lambda: max_idf,#lamba means use this as default if w is not in corpus
            [(w, tfidf.idf_[i]) for w, i in tfidf.vocabulary_.items()])
        
        return self

    def transform(self, X):
        return np.array([
                np.mean([self.word2vec[w] #* self.word2weight[w]
                         for w in words if w in self.word2vec] or
                        [np.zeros(self.dim)], axis=0)
                for words in X
            ])


'''for i in range(0, len(cvedatalist)):#to build model
    trainlist.append(cvedatalist[i])
    mytrainlabel.append( cvelabel_list[i])'''



#tfidf = HashingVectorizer(tokenizer=tokenize, stop_words='english',decode_error='ignore', analyzer='word')
#tfidf = CountVectorizer(tokenizer=tokenize, stop_words='english',decode_error='ignore', binary=True)
if featuretype == 'now2v':
    tfidf = TfidfVectorizer(tokenizer=tokenize, stop_words='english',decode_error='ignore',max_features = 200)
    
else:
    tfidf = TfidfEmbeddingVectorizer(w2v)
#newdict = data_dict.copy()
#newdict.update(test_data_dict)
tfidf.fit( trainlist,mytrainlabel)
tfs = tfidf.transform(trainlist)
print tfs.shape
if featuretype == 'now2v':
    features_by_gram = defaultdict(list)
    for f, w in zip(tfidf.get_feature_names(), tfidf.idf_):
        features_by_gram[len(f.split(' '))].append((f, w))
    top_n = 200
    for gram, features in features_by_gram.iteritems():
        top_features = sorted(features, key=lambda x: x[1], reverse=True)[:top_n]
        top_features = [f[0] for f in top_features]
        print '{}-gram top:'.format(gram), top_features

response = tfidf.transform([test_data_dict[x] for x in testfiles[0].T.tolist()])
#print response.shape
if featuretype == 'now2v':
    tfs = tfs.toarray()
    response = response.toarray()
#from sklearn.neighbors import KNeighborsClassifier
#knn=KNeighborsClassifier(algorithm='auto', leaf_size=30,
#           metric_params=None, n_jobs=1, n_neighbors=5, p=2,
#           weights='uniform')
#knn.fit(tfs.toarray(), trainlabel)
##print knn.predict(response.toarray())
#result = knn.kneighbors(response.toarray(), 5,return_distance=True)




#alldata =  np.vstack((tfs.toarray(), response.toarray()))
alldata =  np.vstack((tfs, response))
alllabels = []
alllabels.extend(mytrainlabel)
alllabels.extend(testfiles[1].T.tolist())
#print "len " + str(len(alllabels))

from sklearn.cross_validation import train_test_split
from collections import Counter
print Counter(alllabels)
X_train, x_test, y_train, y_test = train_test_split(alldata, alllabels, train_size=float(splitratio),stratify=alllabels)
import pandas as pd 

#alldf = pd.DataFrame(alldata)
#mylab = pd.DataFrame(alllabels)
#mylab.applymap(str)
#mylab.columns = ['class']
#resultdf = pd.concat([alldf, mylab], axis=1)
#resultdf.to_csv(testfolder+"/file_path.csv",index=False)

from sklearn import svm
myresult = []
if classifier_type == "svm":
	clf = svm.SVC(probability=True)
	clf.fit(X_train, y_train) 
	myresult = clf.predict(x_test)


from sklearn.neighbors import KNeighborsClassifier
if classifier_type == "knn":
	print "knn"
	knn=KNeighborsClassifier(algorithm='auto', leaf_size=1, 
		   metric_params=None, n_jobs=1, n_neighbors=1, p=1)
	knn.fit(X_train, y_train) 
	myresult = knn.predict(x_test)

#print "endscore"
mycount = 0
myresultcount =0
for ypred,ytruth in zip(myresult,y_test):
    mycount = mycount + 1
    
    if int(ypred)==int(ytruth):
        myresultcount = myresultcount + 1
        
print "No kmm Final acc: " + str(myresultcount*1.0/mycount *100.00)
print "for life cycle"
mycount = 0
myresultcount =0
for ypred,ytruth in zip(myresult,y_test):
    mycount = mycount + 1
    
    if int(ypred)==8 and int(ytruth)==8:
        myresultcount = myresultcount + 1
    if int(ypred)!=8 and int(ytruth)!=8:
        myresultcount = myresultcount + 1
        
print "No kmm Final acc lifecycle: " + str(myresultcount*1.0/mycount *100.00)
if classifier_type == "knn":
    quit()
#important to you.
#This is all you need. Generate a train file and test file 
datafile_name = "emb_apt_apt_" + str(biastype) + "_" + str(splitratio) + "_" + str(featuretype)
mytraindf = pd.DataFrame(X_train)
mytrainlbldf = pd.DataFrame(y_train)
datadf = pd.concat([mytraindf,mytrainlbldf],axis=1)
datadf.to_csv(datafile_name + "_source_stream.csv",header=False, index=False)


mytestdf = pd.DataFrame(x_test)
mytestlbldf = pd.DataFrame(y_test)
testdatadf = pd.concat([mytestdf,mytestlbldf],axis=1)
testdatadf.to_csv(datafile_name + "_target_stream.csv",header=False, index=False)
print "Running FC"
# ['kmm','kliep','arulsif']:
main(datafile_name, biastype)  #call this
