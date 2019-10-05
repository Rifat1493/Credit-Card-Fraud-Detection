np.random.seed(1)
df= pd.read_csv('Data.csv')


list1=df["Dataset B"].tolist()

a=np.array(list1)

import matplotlib.pyplot as plt
rng = np.random.RandomState(10)  # deterministic random data

plt.hist(a, bins='auto')  # arguments are passed to np.histogram
plt.title("Histogram with 'auto' bins")
plt.show()



import pandas as pd
import numpy as np
import re

df= pd.read_csv('data_opioid.csv',encoding='ISO-8859-1')


#if we want to remove null values permanently from the corpus then the following code would be necessary
con=df["text"].isna() 
rev_con=np.invert(con)
df=df[rev_con]   



df.to_csv(path_or_buf='data_opioid_temp.csv',encoding='utf-8')

df= pd.read_csv('data_opioid_temp.csv',encoding='ISO-8859-1')

df = df.drop('Unnamed: 0', 1)

###end of null value removing

df1= pd.DataFrame()
df2= pd.DataFrame()
df3= pd.DataFrame()

df1=df
regex_HS=re.compile('h[aei]*r+oin')
regex_M=re.compile('mor+[ph|f]+[ei]*n')
regex_neg=re.compile('denies|history of h[ae]*r+oin|history of mor+[ph|f]+[ei]*n|history of IVDU')

list_text= df["text"].tolist()

HS=[]
HSA=[]
MSI=[]
MA=[]
Neg=[]

print(len(list_text))

count=0
for a in range(len(list_text)):
    
  
        strr=str(list_text[a])
        count+=1
        string=strr.lower()
        
        
        
        if regex_neg.search(string):
                Neg.append(1)
                HS.append(0)
                HSA.append(0)
                MSI.append(0)
                MA.append(0)
                continue
        
        elif regex_HS.search(string):
          if re.compile('suicide').search(string):
           if re.compile('attempt').search(string):
                Neg.append(0)
                HS.append(0)
                HSA.append(1)
                MSI.append(0)
                MA.append(0)
                continue
           else:
                Neg.append(0)
                HS.append(1)
                HSA.append(0)
                MSI.append(0)
                MA.append(0)
                continue
              
 
        elif regex_M.search(string):
             if re.compile('suicidal').search(string):
                 if re.compile('ideation[.]*').search(string):
                    Neg.append(0)
                    HS.append(0)
                    HSA.append(0)
                    MSI.append(1)
                    MA.append(0)
                    continue
        elif regex_M.search(string):
             if re.compile('accidental').search(string):
                Neg.append(0)
                HS.append(0)
                HSA.append(0)
                MSI.append(0)
                MA.append(1)
                continue
        Neg.append(0)
        HS.append(0)
        HSA.append(0)
        MSI.append(0)
        MA.append(0)
 

        
df["Heroin suicide"]=pd.Series(HS)
df["Heroin suicide attempt"]=pd.Series(HSA)
df["Morphine suicidal ideation"]=pd.Series(MSI)
df["Morphine accidental"]=pd.Series(MA)
df["Negation"]=pd.Series(Neg)


df1.to_csv(path_or_buf='Data_opioid_matrix.csv',encoding='utf-8')
regex_HS=re.compile('h[aei]*r+oin')
HS=[]
for a in range(len(list_text)):
    
  
        strr=str(list_text[a])
        
        string=strr.lower()
        
        
        

        if regex_HS.search(string):
          if re.compile('suicide|suicides|suicided').search(string):
                Neg.append(0)
                HS.append(1)
                HSA.append(0)
                MSI.append(0)
                MA.append(0)
                continue

print(len(HS))




######## New coding Schema ############

              

import pandas as pd
import numpy as np
import re

df= pd.read_csv('data.csv',encoding='ISO-8859-1')


#if we want to remove null values permanently from the corpus then the following code would be necessary
con=df["text"].isna() 
rev_con=np.invert(con)
df=df[rev_con]   



df.to_csv(path_or_buf='data_opioid_temp.csv',encoding='utf-8')

df= pd.read_csv('data_opioid_temp.csv',encoding='ISO-8859-1')

df = df.drop('Unnamed: 0', 1)

###end of null value removing

df1= pd.DataFrame()
df2= pd.DataFrame()
df3= pd.DataFrame()

df1=df
RH=re.compile('h[aei]*r+oin')
RB=re.compile('benzo|benzos')
RBZ=re.compile('benzodiazepines|benzodiazepine')
RM=re.compile('m[ae]thadon[e]*')
RL=re.compile('librium')
RO=re.compile('Alcohol use|smoking|depression|history of substance abuse|history of polysubstance abuse')

RN=re.compile('No history of|No h/o of|No prior use of|No past use of|No use of|Denies|deny|denied of herion|Denies|deny|denied of substance abuse')

list_text= df["text"].tolist()

HB=[]
HBZ=[]
MB=[]
MZ=[]
HL=[]
NEG=[]
OTH=[]
for i in range(len(list_text)):
    HB.append(0)
    HBZ.append(0)
    MB.append(0)
    MZ.append(0)
    HL.append(0)
    NEG.append(0)
    OTH.append(0)

print(len(NEG))




print(len(list_text))


for a in range(len(list_text)):
    
        check=0
        strr=str(list_text[a])
        
        string=strr.lower()

        if RH.search(string) and RB.search(string) and RO.search(string) :
                HB[a]=1
                check=1
        if RH.search(string) and RBZ.search(string) and RO.search(string) :
                HBZ[a]=1
                check=1
        if RM.search(string) and RB.search(string) and RO.search(string) :
                MB[a]=1
                check=1
        if RM.search(string) and RBZ.search(string) and RO.search(string) :
                MZ[a]=1
                check=1
        if RH.search(string) and RL.search(string) and RO.search(string) :
                HL[a]=1
                check=1
        if RN.search(string):
                NEG[a]=1
                check=1
        
        if(check==0):
            OTH[a]=1
        
df["HeroinBenzo"]=pd.Series(HB)
df["HeroinBenzodia"]=pd.Series(HBZ)
df["MethadoneBenzo"]=pd.Series(MB)
df["MethadoneBenzodia"]=pd.Series(MZ)
df["HeroinLibrium"]=pd.Series(HL)
df["Negation"]=pd.Series(NEG)
df["Others"]=pd.Series(OTH)


con=df["Others"]==0
rev_con=np.invert(con)
df=df[con]   

df = df.drop('Unnamed: 0', 1)

list_text= df["Negation"].tolist()

clas=[]

for a in list_text:
    if a==1:
        clas.append("NO Overdose")
    else:
        clas.append("Overdose")
     
        
df["Class"]=pd.Series(clas)

df.to_csv(path_or_buf='temp_data.csv',encoding='utf-8')
df= pd.read_csv('processed_data.csv',encoding='ISO-8859-1')

list_text= df["Negation"].tolist()

clas=[]

for a in list_text:
    if a==1:
        clas.append("NO Overdose")
    else:
        clas.append("Overdose")
     
        
df["Class"]=pd.Series(clas)


df.to_csv(path_or_buf='processed_data.csv',encoding='utf-8')



## Decison tree ###
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

np.random.seed(2)

X = df.iloc[:,7:13].values
Y = df.iloc[:, 13:14].values
from sklearn.preprocessing import LabelEncoder, OneHotEncoder

# Encoding the Dependent Variable
labelencoder_y = LabelEncoder()
Y = labelencoder_y.fit_transform(Y)

# Splitting the dataset into the Training set and Test set

from sklearn.cross_validation import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size = 0.3, random_state = 0)



#fitting Decision tree classifier 
from sklearn.tree import DecisionTreeClassifier
classifier = DecisionTreeClassifier(criterion = 'gini', random_state = 0)
classifier.fit(X_train, y_train)

# Predicting the Test set results
y_pred = classifier.predict(X_test)

# Making the Confusion Matrix
from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y_test, y_pred)

LABELS = ["No Overdose", "Overdose"]
plt.figure(figsize=(5, 5))
sns_plot=sns.heatmap(cm, xticklabels=LABELS, yticklabels=LABELS, annot=True, fmt="d");
plt.title("ANN")
plt.ylabel('True class')
plt.xlabel('Predicted class')
plt.show()

from sklearn.metrics import precision_recall_fscore_support
precision_recall_fscore_support(y_test, y_pred, average='binary')


### ANN ####


from keras.models import Sequential
from keras.layers import Dense
import numpy
# fix random seed for reproducibility
numpy.random.seed(2)

model = Sequential()
model.add(Dense(12, input_dim=6, activation='relu'))
model.add(Dense(8, activation='relu'))
model.add(Dense(1, activation='sigmoid'))
# Compile model
model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
# Fit the model
model.fit(X_train, y_train, validation_data=(X_test, y_test), epochs=30, batch_size=128, verbose=2)
scores = model.evaluate(X_test, y_test, verbose=0)
print("Accuracy: %.2f%%" % (scores[1]*100))


#out=model.predict_classes(X_test)

y_prob = model.predict(X_test) 

y_pred = np.argmax(y_prob, axis=1)

check_list=y_prob.tolist()

new_list1=[]
for i in check_list:
    if i[0]>=.50:
        new_list1.append(1)
    
    else:
        new_list1.append(0)


y_pred=np.array(new_list1)


from sklearn.metrics import confusion_matrix
confusion_matrix(y_test, y_pred)

from sklearn.metrics import precision_recall_fscore_support
precision_recall_fscore_support(y_test, y_pred, average='binary')


#visualize the confusion matrix
import seaborn as sn
import pandas  as pd
 
 
df_cm = pd.DataFrame(cm, range(2),
                  range(2))
plt.figure(figsize = (10,7))
sn.set(font_scale=1.4)#for label size
sn.heatmap(df_cm, annot=True,annot_kws={"size": 4})# font size
plt.show()


model.save_weights("kerasmodel.h5")


import pickle

# save the model to disk
filename = 'DT_model.sav'
pickle.dump(classifier, open(filename, 'wb'))
 
# some time later...
 
# load the model from disk
loaded_model = pickle.load(open(filename, 'rb'))
result = loaded_model.score(X_test, y_test)
print(result)



######Sir's Model#####
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import re
import nltk
from nltk.corpus import stopwords 
from nltk.stem.wordnet import WordNetLemmatizer
from gensim import corpora
from nltk.corpus import wordnet

np.random.seed(2)

def tokenize_only(text):
    text=str(text)
    tokens = [word for sent in nltk.sent_tokenize(text) for word in nltk.word_tokenize(sent)]
    filtered_tokens = []
        # filter out any tokens not containing letters (e.g., numeric tokens, raw punctuation)
    for token in tokens:
        if re.match("[a-z]", token):#for topic model use len(token)<2
           #token=''.join([j for j in token if j in include])
           token= re.sub(r'[^\x00-\x7F]+','', token)
           filtered_tokens.append(token)
          
    final_data=" ".join([i for i in filtered_tokens])
    return final_data


df=pd.read_excel('training_Data.xlsx')  # doctest: +SKIP

df1=pd.DataFrame()
df2=pd.DataFrame()
df.to_csv(path_or_buf='./Annotated corpus/binary100+98.tsv',sep='\t',encoding='utf-8')


Y = df.iloc[:, 5:6].values
from sklearn.preprocessing import LabelEncoder
from keras.utils import np_utils
# Encoding the Dependent Variable
labelencoder_y = LabelEncoder()
Y = labelencoder_y.fit_transform(Y)
Y = np_utils.to_categorical(Y)
Y=df["Class:intent"].tolist()

doc_complete=df["text"]


stop =(stopwords.words('english'))

def clean(doc):
    doc2=str(doc)
    doc2=doc2.lower() 
    doc2=tokenize_only(doc2)
    stop_free = " ".join([i for i in doc2.split() if i not in stop])
   
    
    return stop_free



doc_clean = [clean(doc).split() for doc in doc_complete]


list1=[]
for doc in doc_clean:
    list1=list1+doc
words = list(set(list1))


#words.append("ENDPAD")
word2idx = {w: i for i, w in enumerate(words)}





X=[]
for doc in doc_clean:
    temp=[]
    for j in doc:
        temp.append(word2idx[j])
        
    X.append(temp)
    
    
    
maxlen = max([len(s) for s in X])

# Need not to convert it to Np.array
from keras.preprocessing.sequence import pad_sequences
X = pad_sequences(maxlen=2505, sequences=X, padding="post",value=7231)

Y=np.array(Y)


from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.3)



#creating word embeddings
from numpy import array
from numpy import asarray
from numpy import zeros

print(len(words))

vocab_size=len(words)+1
embeddings_index = dict()
f = open('glove.6B.50d.txt',encoding="utf-8")
for line in f:
	values = line.split()
	word = values[0]
	coefs = asarray(values[1:], dtype='float32')
	embeddings_index[word] = coefs
f.close()
print('Loaded %s word vectors.' % len(embeddings_index))
# create a weight matrix for words in training docs
embedding_matrix = zeros((vocab_size, 50))
for word, i in word2idx.items():
	embedding_vector = embeddings_index.get(word)
	if embedding_vector is not None:
		embedding_matrix[i] = embedding_vector




"""my_mod=model.fit(X_train, y_train,
          epochs=20,
          batch_size=128)
score = model.evaluate(X_test, y_test, batch_size=128)
y_prob = model.predict(X_train) 
y_classes = y_prob.argmax(axis=-1)
from sklearn.metrics import confusion_matrix
confusion_matrix(y_train, y_classes)"""


from keras.models import Sequential,Input,Model
from keras.layers import Dense
from keras.layers import Flatten,Dropout,LSTM,Activation,GRU,SimpleRNN
from keras.layers.convolutional import Conv1D
from keras.layers.convolutional import MaxPooling1D
from keras.layers.embeddings import Embedding
from keras.preprocessing import sequence
               
# create the model CNN 12,1,15,14
model = Sequential()

model.add(Embedding(vocab_size, 50, weights=[embedding_matrix], input_length=25, trainable=False))
#model.add(Embedding(926, 32, input_length=25))
model.add(Dropout(0.25))
model.add(Conv1D(filters=32, kernel_size=3, padding='same', activation='relu'))
model.add(MaxPooling1D(pool_size=2))
model.add(Flatten())
model.add(Dense(250, activation='relu'))
model.add(Dense(1, activation='sigmoid'))
model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
#print(model.summary())              

#create another model CRNN 12,12,1,15
model = Sequential()

model.add(Embedding(vocab_size, 50, weights=[embedding_matrix], input_length=25, trainable=False))
model.add(Conv1D(filters=32, kernel_size=3, padding='same', activation='relu'))
model.add(MaxPooling1D(pool_size=2))
model.add(GRU(300))
model.add(Dense(1, activation='sigmoid'))
model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])

#create RNN
model = Sequential()

model.add(Embedding(vocab_size, 50, weights=[embedding_matrix], input_length=2505, trainable=False))
model.add(GRU(300))
model.add(Dropout(0.5))
model.add(Dense(3, activation='softmax'))
model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])



#binary classification###
model = Sequential()
model.add(Embedding(vocab_size, 50, weights=[embedding_matrix], input_length=2505, trainable=False))
model.add(GRU(1000))
model.add(Dropout(0.5))
model.add(Dense(1, activation='sigmoid'))
model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])

model.fit(X_train, y_train, validation_data=(X_test, y_test), epochs=3, batch_size=128, verbose=2)
scores = model.evaluate(X_test, y_test, verbose=0)
print("Accuracy: %.2f%%" % (scores[1]*100))

#create RCNN

model = Sequential()

model.add(Embedding(vocab_size, 50, weights=[embedding_matrix], input_length=25, trainable=False))

model.add( SimpleRNN(300, activation="relu", return_sequences=True))

model.add(Conv1D(filters=32, kernel_size=3, padding='same', activation='relu'))
model.add(MaxPooling1D(pool_size=2))
model.add(Flatten())
model.add(Dropout(0.5))
model.add(Dense(1, activation='sigmoid'))
model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])


#create model CNNA

from keras.layers import Add
from keras.layers.core import Permute

from keras.models import *


def attention_3d_block(inputs):
    # inputs.shape = (batch_size, time_steps, input_dim)
    
    a = Permute((2, 1))(inputs)
   # a = Reshape((input_dim, 40))(a) # this line is not useful. It's just to know which dimension is what.
    a = Dense(25, activation='softmax')(a)

    a_probs = Permute((2, 1), name='attention_vec')(a)
    output_attention_mul = Add()([inputs, a_probs])
    return output_attention_mul

input = Input(shape=(25,))
model = Embedding(input_dim=vocab_size, output_dim=50, input_length=25)(input)
#model = Dropout(0.25)(model)
model = Conv1D(filters=32, kernel_size=3, padding='same', activation='relu')(model)
attention_mul = attention_3d_block(model)
attention_mul = Flatten()(attention_mul)
output = Dense(1, activation='sigmoid')(attention_mul)
model = Model(input,output)
model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])




# Fit the model

model.fit(X_train, y_train, validation_data=(X_test, y_test), epochs=10, batch_size=128, verbose=2)
scores = model.evaluate(X_test, y_test, verbose=0)
print("Accuracy: %.2f%%" % (scores[1]*100))


#out=model.predict_classes(X_test)

y_prob = model.predict(X_test) 

y_pred = np.argmax(y_prob, axis=1)

check_list=y_prob.tolist()

new_list1=[]
for i in check_list:
    if i[0]>=.50:
        new_list1.append(1)
    
    else:
        new_list1.append(0)


y_pred=np.array(new_list1)


from sklearn.metrics import confusion_matrix
confusion_matrix(y_test, y_pred)

from sklearn.metrics import precision_recall_fscore_support
precision_recall_fscore_support(y_test, y_pred, average='binary')


#visualize the confusion matrix
import seaborn as sn
import pandas  as pd
 
 
df_cm = pd.DataFrame(cm, range(2),
                  range(2))
plt.figure(figsize = (10,7))
sn.set(font_scale=1.4)#for label size
sn.heatmap(df_cm, annot=True,annot_kws={"size": 4})# font size
plt.show()




idx2word = {v: k for k, v in word2idx.items()}
print(idx2word[343])

#### New Model ####


df= pd.read_csv('all_data.csv',encoding='ISO-8859-1')


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import re
import nltk
from nltk.corpus import stopwords 
from nltk.stem.wordnet import WordNetLemmatizer
from gensim import corpora
from nltk.corpus import wordnet

np.random.seed(2)

def tokenize_only(text):
    text=str(text)
    tokens = [word for sent in nltk.sent_tokenize(text) for word in nltk.word_tokenize(sent)]
    filtered_tokens = []
        # filter out any tokens not containing letters (e.g., numeric tokens, raw punctuation)
    for token in tokens:
        if re.match("[a-z]", token):#for topic model use len(token)<2
           #token=''.join([j for j in token if j in include])
           token= re.sub(r'[^\x00-\x7F]+','', token)
           filtered_tokens.append(token)
          
    final_data=" ".join([i for i in filtered_tokens])
    return final_data



Y = df.iloc[:, 5:6].values
from sklearn.preprocessing import LabelEncoder,OneHotEncoder
from keras.utils import np_utils
# Encoding the Dependent Variable
labelencoder_y = LabelEncoder()
Y = labelencoder_y.fit_transform(Y)
#Y = np_utils.to_categorical(Y)
#Y=df["Class:intent"].tolist()
Y=Y.reshape(-1, 1)
onehotencoder = OneHotEncoder(categorical_features = [0])
Y = onehotencoder.fit_transform(Y).toarray()



doc_complete=df["text"]


stop =(stopwords.words('english'))

def clean(doc):
    doc2=str(doc)
    doc2=doc2.lower() 
    doc2=tokenize_only(doc2)
    stop_free = " ".join([i for i in doc2.split() if i not in stop])
   
    
    return stop_free



doc_clean = [clean(doc) for doc in doc_complete]
from sklearn.feature_extraction.text import CountVectorizer

vectorizer = CountVectorizer()
X = vectorizer.fit_transform(doc_clean).toarray()
                          


from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.4)
from sklearn import svm

clf = svm.SVC( kernel='rbf')
clf.fit(X_train, y_train)  

y_pred=clf.predict(X_test)
from sklearn.metrics import confusion_matrix
confusion_matrix(y_test, y_pred)


from sklearn.neighbors import KNeighborsClassifier
classifier = KNeighborsClassifier(n_neighbors = 3, metric = 'minkowski', p = 2)
classifier.fit(X_train, y_train)




from sklearn.ensemble import RandomForestClassifier
 
clf = RandomForestClassifier(n_estimators=100, max_depth=3,
                             random_state=0)
clf.fit(X_train, y_train)
# Predicting the Test set results
y_pred = classifier.predict(X_test)



y_pred1=np.argmax(y_pred, axis=1)
y_test1=np.argmax(y_test, axis=1)
# Making the Confusion Matrix
from sklearn.metrics import confusion_matrix
confusion_matrix(y_test1, y_pred1)



from sklearn.metrics import precision_recall_fscore_support
precision_recall_fscore_support(y_test, y_pred, average='binary')




from sklearn import datasets 
from sklearn.metrics import confusion_matrix 
from sklearn.model_selection import train_test_split 
  
# loading the iris dataset 
iris = datasets.load_iris() 
  
# X -> features, y -> label 
X = iris.data 
y = iris.target 
  
# dividing X, y into train and test data 
X_train, X_test, y_train, y_test = train_test_split(X, y, random_state = 0) 
  
# training a DescisionTreeClassifier 
from sklearn.tree import DecisionTreeClassifier 
dtree_model = DecisionTreeClassifier(max_depth = 2).fit(X_train, y_train) 
dtree_predictions = dtree_model.predict(X_test) 
  
# creating a confusion matrix 
cm = confusion_matrix(y_test, dtree_predictions) 

###### Bangla NER Page""""
accuracy_rate = (4+4+4+2+10+2+39+1987)/2320
                
                
                
                
                
                
count=0
for i in y_true:
    if(i=="O"):
      count+=1


fl_read=open("train_corpus.txt","r",encoding="utf-8")
lines=fl_read.readlines()
j=0
a=1
for i in lines:
    j+=1
    if(j<5212):
      name=str(a)
    else:
      if(i[0]!="।"):
          f=1
      else:
       f3 = open("Data/"+name+".txt","a",encoding="utf-8")
       f3.write(i)
       f3.close()  
       a+=1
       j=0
       continue
    
    f3 = open("Data/"+name+".txt","a",encoding="utf-8")
    f3.write(i)
    f3.close()    
  
fl_read=open("test_data.txt","r",encoding="utf-8")
lines=fl_read.readlines()
#if (" O\n") in b:
count=0

for b in lines:
    
    if  b!='\n':
        count+=1
        
        
        
fl_read.close()
##word wmbeddings
word_list=[]

fEmbeddings = open("bn.tsv", encoding="utf-8")
for line in fEmbeddings:
    line=line.strip().split()
    if("." in line[0]):
        c=1
    else:
        word_list.append(line[1])

from gensim.models import Word2Vec
model = Word2Vec.load('bn.bin')

fl2=open("bn_w2v_300.txt","a",encoding="utf-8")
for word in word_list:
    
    b=model[word]
    fl2.write(word)
    for a in b:
      fl2.write(" "+str(a))
     
    fl2.write("\n")
    
    
fl2.close()     




#word embeddings loading##
from gensim.models import Word2Vec
new_model = Word2Vec.load('./model/bn_w2v_model.bin')
print(new_model["তিন"])





###preprocessing###
import nltk
from nltk.probability import FreqDist

fl2=open("2014-01-04.txt","r",encoding="utf-8")


word_list=[]
def tokenize_only(text):
    text=str(text)
    
    tokens = [word for sent in nltk.sent_tokenize(text) for word in nltk.word_tokenize(sent)]
    
      
    for k in tokens:
        word_list.append(k)
    
    return tokens
    
data = [tokenize_only(doc)for doc in fl2]

fdist = FreqDist(word_list)
n=fdist.most_common(1000)

data.pop(0)

fl_out=open("temp_corpus.txt","a",encoding="utf-8")
#fl_out.write("")

#fl_out.write("-DOCSTART-\n\n")

for sent in data:
    if(len(sent)==0):
        continue
    elif(len(sent)>2):
        if(sent[1]=="title"):
           continue

    for word in sent:
        if(word in ['<','>','news','/news','title','/title',' ']):
            continue
        elif (word[-1]=="।"):
            
            fl_out.write(word[:-1]+'\tO'+"\n")
            fl_out.write(word[-1]+'\tO'+"\n\n")
            
        else:
            fl_out.write(word+'\tO'+"\n")

fl_out.close()

####calculating the frequency####
fl_read=open("annotated data/3.txt","r",encoding="utf-8")
lines=fl_read.readlines()
fdist = FreqDist(lines)
n=fdist.most_common(1000)
fl_read.close()


print(fdist[1])


f1 = open("train_corpus.txt","r",encoding="utf-8")
b=f1.readlines()
f3 = open("train_corpus.txt","w",encoding="utf-8")
f3.write("")
f2 = open("train_corpus.txt","a",encoding="utf-8")
for line in b:
 f2.write(line.replace('গ্রামের\tO\n','গ্রামের\tB-LOC\n'))

f1.close()
f2.close()
f3.close()
f2 = open("test_data.txt","r",encoding="utf-8")
f3= open("predicted_label_t.txt","a",encoding="utf-8")

b= f2.readlines()
for i in b:
    if(i=="\n"):
        f3.write(i)
    else:
        f3.write(i.strip()+"\tO\n")
    
f2.close()
f3.close()

count=0