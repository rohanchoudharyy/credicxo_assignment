import numpy as np
import pandas as pd

dataset = pd.read_csv('musk_csv.csv')
print("Missing values in the csv = ", dataset.isnull().sum().sum())

dataset.head()

X = dataset.iloc[:, 3:169].values
Y = dataset.iloc[:, -1].values
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.2, random_state=0)

from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test)

import tensorflow.keras
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense

classifier = Sequential()

classifier.add(Dense(units = 120, kernel_initializer = 'he_uniform', activation = 'relu', input_dim = 166))

classifier.add(Dense(units = 1, kernel_initializer = 'glorot_uniform', activation = 'sigmoid'))

classifier.summary()

classifier.compile(optimizer = 'adam', loss = 'binary_crossentropy', metrics = ['accuracy'])

model = classifier.fit(X_train, y_train, validation_split = 0.2, batch_size = 50, nb_epoch = 40)

y_pred = classifier.predict(X_test)
y_pred = (y_pred > 0.5)

classifier.save_weights("classifier.h5")
print(model.history.keys())

import matplotlib.pyplot as plt

# summarize history for accuracy
plt.plot(model.history['accuracy'])
plt.plot(model.history['val_accuracy'])
plt.title('model accuracy')
plt.ylabel('accuracy')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper left')
plt.show()

# summarize history for loss
plt.plot(model.history['loss'])
plt.plot(model.history['val_loss'])
plt.title('model loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper left')
plt.show()

from sklearn.metrics import f1_score, precision_score, recall_score, confusion_matrix, accuracy_score
print(confusion_matrix(y_test, y_pred))
loss, acc = classifier.evaluate(X_test, y_test, verbose=0)
print("Accuracy:",acc," and Loss:",loss)
print("F1 score: ", f1_score(y_pred, y_test,average="macro"))
print("Precision: ", precision_score(y_pred, y_test,average="macro"))
print("Recall: ", recall_score(y_pred, y_test,average="macro"))