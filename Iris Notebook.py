#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns


# In[2]:


iris = pd.read_csv('iris.csv')
iris.head()


# In[3]:


X = iris.drop('species', axis=1)


# In[4]:


y = iris['species']


# In[5]:


y.unique()


# In[6]:


from sklearn.preprocessing import LabelBinarizer
encoder = LabelBinarizer()


# In[7]:


y = encoder.fit_transform(y)


# In[8]:


#y


# In[9]:


from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler


# In[10]:


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=101)


# In[11]:


scaler = MinMaxScaler()


# In[12]:


scaler.fit(X_train)


# In[13]:


scaled_X_train = scaler.transform(X_train)
scaled_X_test = scaler.transform(X_test)


# In[14]:


import tensorflow as tf
import keras
print(tf.__version__)
print(keras.__version__)


# In[15]:


from keras.models import Sequential
from keras.layers import Dense


# In[16]:


model = Sequential()

model.add(Dense(units=4, activation='relu', input_shape=[4,]))
model.add(Dense(units=4, activation='relu', input_shape=[4,]))
model.add(Dense(units=4, activation='relu', input_shape=[4,]))
model.add(Dense(units=4, activation='relu', input_shape=[4,]))

model.add(Dense(units=3, activation='softmax'))

model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])


# In[17]:


from keras.callbacks import EarlyStopping
early_stop = EarlyStopping(patience=10)


# In[18]:


model.summary()


# In[19]:


model.fit(x=scaled_X_train, y=y_train, epochs=300, validation_data=(scaled_X_test, y_test), callbacks=[early_stop])


# In[20]:


metrics = pd.DataFrame(model.history.history)


# In[21]:


metrics[['loss', 'val_loss']].plot()


# In[22]:


model.evaluate(scaled_X_test, y_test, verbose=0)


# In[23]:


epochs = len(metrics)
print(epochs)


# In[24]:


scaled_X = scaler.fit_transform(X)


# In[26]:


model = Sequential()

model.add(Dense(units=4, activation='relu', input_shape=[4,]))
model.add(Dense(units=4, activation='relu', input_shape=[4,]))
model.add(Dense(units=4, activation='relu', input_shape=[4,]))
model.add(Dense(units=4, activation='relu', input_shape=[4,]))

model.add(Dense(units=3, activation='softmax'))

model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])


# In[27]:


model.fit(scaled_X, y, epochs=epochs)


# In[28]:


model.save("final_iris_model.h5")


# In[29]:


import joblib


# In[30]:


joblib.dump(scaler, 'iris_scaler.pkl')


# In[31]:


from keras.models import load_model
flower_model = load_model('final_iris_model.h5')
flower_scaler = joblib.load("iris_scaler.pkl")


# In[32]:


iris.head(1)


# In[33]:


flower_example = {"sepal_length": 5.1,
                  "sepal_width": 3.5,
                  "petal_length": 1.4,
                  "petal_width": 0.2}


# In[34]:


encoder.classes_


# In[35]:


def return_prediction(model, scaler, sample_json):
    
    
    
    s_len = sample_json["sepal_length"]
    s_wid = sample_json["sepal_width"]
    p_len = sample_json["petal_length"]
    p_wid = sample_json["petal_width"]
    
    flower = [[s_len, s_wid, p_len, p_wid]]
    flower = scaler.transform(flower)
    
    classes = np.array(['setosa', 'versicolor', 'virginica'])
    
    class_ind = model.predict_classes(flower)[0]
    
    return classes[class_ind]


# In[36]:


return_prediction(flower_model, flower_scaler, flower_example)


# # Code for Deployment

# In[37]:


from keras.models import load_model
import joblib
import numpy as np

flower_model = load_model('final_iris_model.h5')
flower_scaler = joblib.load("iris_scaler.pkl")

def return_prediction(model, scaler, sample_json):
    
    
    
    s_len = sample_json["sepal_length"]
    s_wid = sample_json["sepal_width"]
    p_len = sample_json["petal_length"]
    p_wid = sample_json["petal_width"]
    
    flower = [[s_len, s_wid, p_len, p_wid]]
    flower = scaler.transform(flower)
    
    classes = np.array(['setosa', 'versicolor', 'virginica'])
    
    class_ind = model.predict_classes(flower)[0]
    
    return classes[class_ind]


# In[ ]:




