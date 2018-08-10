import numpy as np
import cPickle as pickle
import re
import scipy.sparse as sp
from nltk.stem.porter import PorterStemmer
from nltk.stem.snowball import SnowballStemmer
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfVectorizer

f = open('../datasets/Eurlex_XML_CNN/eurlex_raw_text.p')
[train, test, vocab, catgy] = pickle.load(f)

def clean_str(string):
    # remove stopwords
    # string = ' '.join([word for word in string.split() if word not in cachedStopWords])
    string = re.sub(r"[^A-Za-z0-9(),!?\'\`]", " ", string)
    string = re.sub(r"\'s", " \'s", string)
    string = re.sub(r"\'ve", " \'ve", string)
    string = re.sub(r"n\'t", " n\'t", string)
    string = re.sub(r"\'re", " \'re", string)
    string = re.sub(r"\'d", " \'d", string)
    string = re.sub(r"\'ll", " \'ll", string)
    string = re.sub(r",", " , ", string)
    string = re.sub(r"!", " ! ", string)
    string = re.sub(r"\(", " \( ", string)
    string = re.sub(r"\)", " \) ", string)
    string = re.sub(r"\?", " \? ", string)
    string = re.sub(r"\s{2,}", " ", string)
    return string.strip().lower()


def load_data(data, M=0, N=0):
  x_text = [clean_str(doc['text']) for doc in data]
  x_text = [s.split(" ") for s in x_text]
  labels = [doc['catgy'] for doc in data]
  row_idx, col_idx, val_idx = [], [], []
  for i in xrange(len(labels)):
    l_list = list(set(labels[i])) # remove duplicate cateories to avoid double count
    for y in l_list:
       row_idx.append(i)
       col_idx.append(y)
       val_idx.append(1)
    m = max(row_idx) + 1
    n = max(col_idx) + 1
    if(M and N):
      if(N > n):
        #y_te = y_te.resize((np.shape(y_te)[0], np.shape(y_tr)[1]))
        Y = sp.csr_matrix((val_idx, (row_idx, col_idx)), shape=(m, N)).todense()
      elif(N < n):
        Y = sp.csr_matrix((val_idx, (row_idx, col_idx)), shape=(m, N)).todense()
        Y = Y[:, :N]
    else:
        Y = sp.csr_matrix((val_idx, (row_idx, col_idx)), shape=(m, n)).todense()
  return [x_text, Y, m, n]

trn_sents, Y_trn, m, n = load_data(train)
tst_sents, Y_tst, m, n = load_data(test, M=m, N=n)

def multi_delete(list_, args):
  indexes = sorted(list(args), reverse=True)
  for index in indexes:
      # print(index)
      del list_[int(index)]
  return list_

def bow2tfidf(bow):
  numDoc = np.shape(bow)[0]
  binaryMat = np.ceil(bow/np.sum(bow, axis=1).reshape((bow.shape[0], 1)))
  numDocwW = np.sum(binaryMat, axis=0)
  idf = np.log(numDoc*(1/numDocwW))
  tfidf = np.multiply(bow, idf.reshape((1, idf.shape[1])))
  return tfidf, idf

def gettfidf(trn_sents, tst_sents):
  v1 = TfidfVectorizer()
  vectorizer = CountVectorizer()
  docs_tr = []
  docs_te = []
  for doc in trn_sents:
        docs_tr.append(" ".join(doc))
  for doc in tst_sents:
        docs_te.append(" ".join(doc))

  # docs = docs_tr + docs_te
  vectorizer.fit(docs_tr)
  # data = vectorizer.transform(docs).todense()
  # data_ind = np.argsort(np.sum(data,axis=0)).squeeze()[0, -5000:]
  Xtr_bw = vectorizer.transform(docs_tr).todense()
  data_ind = np.argsort(np.sum(Xtr_bw,axis=0))[0, -5000:]
  Xtr_bw = Xtr_bw[:,list(data_ind)].squeeze()
  Xte_bw = vectorizer.transform(docs_te).todense()[:,list(data_ind)].squeeze()

  XtrTfidf, idf = bow2tfidf(Xtr_bw)
  XteTfidf = np.multiply(Xte_bw, idf)

  np.save("xTr_tfidf", XtrTfidf)
  np.save("xTe_tfidf", XteTfidf)
  return XtrTfidf, XteTfidf

# notInDict, Y_trn, Y_tst = catgy_emb(Y_trn, Y_tst, catgy, model)
notInDict, Xtr, Xte = getEmb(trn_sents, tst_sents, model)





  # emb = np.zeros((len(ds), model[model.keys()[1]].shape[0]))
  # notInDict = []
  # for i, doc in enumerate(ds):
  #   for w in doc:
  #     if w in model.keys():
  #       emb[i] += model[w]
  #     else:
  #       if w in stem_keys:
  #         print("+++ in stem " + w + " ++++")
  #         emb[i] += model[keys[stem_keys.index(w)]]
  #       else:
  #         if w in stem_keys_sb:
  #           print("---- in stemsb " + w + " ----")
  #           emb[i] += model[keys[stem_keys_sb.index(w)]]
  #         else:
  #           if w not in notInDict:
  #             print("============ " + w + " =========")
  #             notInDict.append(w)
  #   print("Doc Done")
  # return notInDict, emb

