import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB

np.random.seed(1)
LABELS = ["Normal", "Fraud"]
# importing the data
df = pd.read_csv('sample_data.csv', encoding='ISO-8859-1')


df = df.drop(['Time'], axis=1)  # removing the unnecessary column
df['Amount'] = StandardScaler().fit_transform(df['Amount'].values.reshape(-1, 1))

X = df.iloc[:,:29].values
Y = df["Class"].tolist()
Y = np.array(Y)


X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.3)


# building the model
classifier = GaussianNB()
classifier.fit(X_train, y_train)

# Predicting the Test set results
y_pred = classifier.predict(X_test)

# Making the Confusion Matrix
from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y_test, y_pred)

plt.figure(figsize=(5, 5))
sns_plot=sns.heatmap(cm, xticklabels=LABELS, yticklabels=LABELS, annot=True, fmt="d");
plt.title("NB train:test=70:30")
plt.ylabel('True class')
plt.xlabel('Predicted class')
plt.show()

from sklearn.metrics import precision_recall_fscore_support
precision_recall_fscore_support(y_test, y_pred, average='binary')

#### KNN#####

# Fitting K-NN to the Training set
from sklearn.neighbors import KNeighborsClassifier
classifier = KNeighborsClassifier(n_neighbors = 2, metric = 'minkowski', p = 2)
classifier.fit(X_train, y_train)

# Predicting the Test set results
y_pred = classifier.predict(X_test)

# Making the Confusion Matrix
from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y_test, y_pred)

plt.figure(figsize=(5, 5))
sns_plot=sns.heatmap(cm, xticklabels=LABELS, yticklabels=LABELS, annot=True, fmt="d");
plt.title("KNN train:test=70:30")
plt.ylabel('True class')
plt.xlabel('Predicted class')
plt.show()

from sklearn.metrics import precision_recall_fscore_support
precision_recall_fscore_support(y_test, y_pred, average='binary')


#### SVM #####

from sklearn.svm import SVC
classifier = SVC(decision_function_shape='ovo')
classifier.fit(X_train, y_train)

# Predicting the Test set results
y_pred = classifier.predict(X_test)

# Making the Confusion Matrix
from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y_test, y_pred)

plt.figure(figsize=(5, 5))
sns_plot=sns.heatmap(cm, xticklabels=LABELS, yticklabels=LABELS, annot=True, fmt="d");
plt.title("SVM train:test=70:30")
plt.ylabel('True class')
plt.xlabel('Predicted class')
plt.show()

from sklearn.metrics import precision_recall_fscore_support
precision_recall_fscore_support(y_test, y_pred, average='binary')