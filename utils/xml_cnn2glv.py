import numpy as np
import cPickle as pickle
import re
import scipy.sparse as sp
from nltk.stem.porter import PorterStemmer
from nltk.stem.snowball import SnowballStemmer
from sklearn.feature_extraction.text import CountVectorizer
vectorizer = CountVectorizer()

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

def loadGloveModel(gloveFile):
        print "Loading Glove Model"
        f = open(gloveFile,'r')
        model = {}
        for line in f:
                line = line.decode('utf-8')
                splitLine = line.split()
                #print(splitLine[0])
                word = splitLine[0]
                embedding = np.array([float(val) for val in splitLine[1:]])
                model[word] = embedding
        print "Done.",len(model)," words loaded!"
        return model

model = loadGloveModel('../datasets/Eurlex/eurlex_docs/glove.6B.300d.txt')

trn_sents, Y_trn, m, n = load_data(train)
tst_sents, Y_tst, m, n = load_data(test, M=m, N=n)

keys = model.keys()
stem_keys = []
stem_keys_sb = []
stemmer = PorterStemmer()
stemmer_sb = SnowballStemmer("english")

for w in model.keys():
  stem_keys.append(stemmer.stem(w))

for w in model.keys():
  stem_keys_sb.append(stemmer_sb.stem(w))
def multi_delete(list_, args):
    indexes = sorted(list(args), reverse=True)
    for index in indexes:
        # print(index)
        del list_[int(index)]
    return list_

def getEmb(trn_sents, tst_sents, model):
  notInDict = []
  docs_tr = []
  docs_te = []
  for doc in trn_sents:
        docs_tr.append(" ".join(doc))
  for doc in tst_sents:
        docs_te.append(" ".join(doc))

  docs = docs_tr + docs_te
  vectorizer.fit(docs)
  fn = vectorizer.get_feature_names()
  Xtr = vectorizer.transform(docs_tr).todense()
  Xte = vectorizer.transform(docs_te).todense()
  data = np.concatenate((Xtr, Xte), axis=0)
  wf = np.sum(Xtr, axis=0)
  del_ax = np.argwhere(wf<5)
  del_lst = []
  for i in del_ax:
    del_lst.append(i[1])
  Xtr = np.delete(Xtr, del_lst, axis=1)
  Xte = np.delete(Xte, del_lst, axis=1)
  fn = multi_delete(fn, del_lst)


  Wt = np.zeros((len(fn), model[model.keys()[1]].shape[0]))
  for i, w in enumerate(fn):
    if w in model.keys():
      Wt[i] = model[w]
    elif w in stem_keys:
      print("+++ in stem " + w + " ++++")
      Wt[i] = model[keys[stem_keys.index(w)]]
    # elif w in stem_keys_sb:
    #   print("---- in stemsb " + w + " ----")
    #   Wt[i] = model[keys[stem_keys_sb.index(w)]]
    elif w not in notInDict:
      print("============ " + w + " ========= " + str(i))
      notInDict.append(w)
  np.save("xTr_bog", Xtr)
  np.save("xTe_bog", Xte)
  np.save("ft_Wt", Wt)

  Xtr = np.dot(Xtr, Wt)/np.sum(Xtr, axis=1).reshape(Xtr.shape[0], 1)
  Xte = np.dot(Xte, Wt)/np.sum(Xte, axis=1).reshape(Xte.shape[0], 1)
  np.save("xTr_emb", Xtr)
  np.save("xTe_emb", Xte)

  return notInDict, Xtr, Xte

# tst_emb = getEmb(tst_sents, model)

def catgy_emb(Y_trn, Y_tst, catgy, model):
  notInDict = []
  Wt = np.zeros((len(catgy), model[model.keys()[1]].shape[0]))
  for i, wrd in enumerate(catgy):
    cat = wrd.split("_")
    for cat_ in cat:
      cat__ = cat_.split("-")
      for w in cat__:
        w = w.split("'")[0]
        if w in model.keys():
          Wt[i] += model[w]
        elif w in stem_keys:
          print("+++ in stem " + w + " ++++")
          Wt[i] += model[keys[stem_keys.index(w)]]
        elif w in stem_keys_sb:
          print("---- in stemsb " + w + " ----")
          Wt[i] += model[keys[stem_keys_sb.index(w)]]
        elif w not in notInDict:
          print("============ " + w + " ========= " + str(i))
  np.save("lblTr", Y_trn)
  np.save("lblTe", Y_tst)
  np.save("lbl_Wt", Wt)

  Wt = Wt[:Y_tst.shape[1], :]
  Y_tst = np.dot(Y_tst, Wt)/np.sum(Y_tst, axis=1).reshape(Y_tst.shape[0], 1)
  Y_trn = np.dot(Y_trn, Wt)/np.sum(Y_trn, axis=1).reshape(Y_trn.shape[0], 1)

  np.save("lblTr_emb", Y_trn)
  np.save("lblTe_emb", Y_tst)

  return notInDict, Y_trn, Y_tst
notInDict, Y_trn, Y_tst = catgy_emb(Y_trn, Y_tst, catgy, model)
# notInDict, Xtr, Xte = getEmb(trn_sents, tst_sents, model)





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

