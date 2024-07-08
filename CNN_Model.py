#!/usr/bin/env python
# coding: utf-8

# In[56]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')
import seaborn as sns
sns.set(style="whitegrid")
import os
import glob as gb
import cv2
import tensorflow as tf
import keras
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout, BatchNormalization


# In[31]:


training_dataset_path = '.\gender-classification-dataset/Training'
validation_dataset_path = '.\gender-classification-dataset/Validation'
Prediction_dataset_path = '.\gender-classification-dataset/Prediction'


# In[32]:


for folder in  os.listdir(training_dataset_path) : 
    files = gb.glob(pathname= str( training_dataset_path +'//'+ folder + '/*.jpg'))
    print(f'For training data , found {len(files)} in folder {folder}')


# In[33]:


for folder in  os.listdir(validation_dataset_path) : 
    files = gb.glob(pathname= str( validation_dataset_path +'//'+ folder + '/*.jpg'))
    print(f'For testing data , found {len(files)} in folder {folder}')


# In[34]:


files = gb.glob(pathname= str( Prediction_dataset_path + '/*.jpg'))
print(f'For Prediction data , found {len(files)}')


# In[35]:


code = {'male':0 ,'female':1}

def getcode(n) : 
    for x , y in code.items() : 
        if n == y : 
            return x    


# In[36]:


size = []
for folder in  os.listdir(training_dataset_path) : 
    files = gb.glob(pathname= str( training_dataset_path +'//'+ folder + '/*.jpg'))
    for file in files: 
        image = plt.imread(file)
        size.append(image.shape)
pd.Series(size).value_counts()


# In[37]:


size = []
for folder in  os.listdir(validation_dataset_path) : 
    files = gb.glob(pathname= str( validation_dataset_path +'//'+ folder + '/*.jpg'))
    for file in files: 
        image = plt.imread(file)
        size.append(image.shape)
pd.Series(size).value_counts()


# In[38]:


size = []
files = gb.glob(pathname= str(Prediction_dataset_path +'/*.jpg'))
for file in files: 
    image = plt.imread(file)
    size.append(image.shape)
pd.Series(size).value_counts()


# In[39]:


#resize images 
size = 100


# In[40]:


X_train = []
y_train = []
for folder in  os.listdir(training_dataset_path) : 
    files = gb.glob(pathname= str( training_dataset_path +'//' + folder + '/*.jpg'))
    for file in files: 
        image = cv2.imread(file)
        image_array = cv2.resize(image , (size,size))
        X_train.append(list(image_array))
        y_train.append(code[folder])


# In[41]:


print(f'we have {len(X_train)} items in X_train')


# In[42]:


plt.figure(figsize=(20,20))
for n , i in enumerate(list(np.random.randint(0,len(X_train),36))) : 
    plt.subplot(6,6,n+1)
    plt.imshow(X_train[i])   
    plt.axis('off')
    plt.title(getcode(y_train[i]))


# In[43]:


X_test = []
y_test = []
for folder in  os.listdir(validation_dataset_path) : 
    files = gb.glob(pathname= str(validation_dataset_path + '//' + folder + '/*.jpg'))
    for file in files: 
        image = cv2.imread(file)
        image_array = cv2.resize(image , (size,size))
        X_test.append(list(image_array))
        y_test.append(code[folder])
        


# In[44]:


print(f'we have {len(X_test)} items in X_test')


# In[45]:


plt.figure(figsize=(20,20))
for n , i in enumerate(list(np.random.randint(0,len(X_test),36))) : 
    plt.subplot(6,6,n+1)
    plt.imshow(X_test[i])    
    plt.axis('off')
    plt.title(getcode(y_test[i]))


# In[46]:


X_pred = []
files = gb.glob(pathname= str(Prediction_dataset_path + '/*.jpg'))
for file in files: 
    image = cv2.imread(file)
    image_array = cv2.resize(image , (size,size))
    X_pred.append(list(image_array)) 


# In[47]:


print(f'we have {len(X_pred)} items in X_pred')


# In[48]:


plt.figure(figsize=(20,20))
for n , i in enumerate(list(np.random.randint(0,len(X_pred),36))) : 
    plt.subplot(6,6,n+1)
    plt.imshow(X_pred[i])    
    plt.axis('off')


# # BUILDING THE CNN MODEL

# In[49]:


X_train = np.array(X_train)
X_test = np.array(X_test)
y_train = np.array(y_train)
y_test = np.array(y_test)
X_pred_array = np.array(X_pred)


print(f'X_train shape  is {X_train.shape}')
print(f'X_test shape  is {X_test.shape}')
print(f'y_train shape  is {y_train.shape}')
print(f'y_test shape  is {y_test.shape}')
print(f'X_pred shape  is {X_pred_array.shape}')


# In[57]:


Model = keras.models.Sequential([
    Conv2D(32, kernel_size=(3, 3), activation='relu', input_shape=(100, 100, 3)),
    BatchNormalization(),
    Conv2D(32, kernel_size=(3, 3), activation='relu'),
    BatchNormalization(),
    MaxPooling2D(pool_size=(2, 2)),
    Dropout(0.25),
    
    Conv2D(64, kernel_size=(3, 3), activation='relu'),
    BatchNormalization(),
    Conv2D(64, kernel_size=(3, 3), activation='relu'),
    BatchNormalization(),
    MaxPooling2D(pool_size=(2, 2)),
    Dropout(0.25),

    Conv2D(128, kernel_size=(3, 3), activation='relu'),
    BatchNormalization(),
    Conv2D(128, kernel_size=(3, 3), activation='relu'),
    BatchNormalization(),
    MaxPooling2D(pool_size=(2, 2)),
    Dropout(0.25),
    
    Flatten(),
    Dense(512, activation='relu'),
    BatchNormalization(),
    Dropout(0.5),
    Dense(1, activation='sigmoid')
])


# In[58]:


Model.compile(optimizer ='adam',loss='binary_crossentropy',metrics=['accuracy'])


# In[59]:


print('Model Details are : ')
print(Model.summary())


# In[ ]:


ThisModel = Model.fit(X_train, y_train, epochs=5,batch_size=64)


# In[63]:


ModelLoss, ModelAccuracy = modell.evaluate(X_test, y_test, batch_size=1)

print('Test Loss is {}'.format(ModelLoss))
print('Test Accuracy is {}'.format(ModelAccuracy ))


# In[51]:


Model.save('my_model2.h5')


# In[62]:


modell = keras.models.load_model("my_model2.h5")


# In[ ]:


y_pred = KerasModel.predict(X_test)

print('Prediction Shape is {}'.format(y_pred.shape))


# In[ ]:


y_result = KerasModel.predict(X_pred_array)

print('Prediction Shape is {}'.format(y_result.shape))


# In[ ]:


plt.figure(figsize=(20,20))
for n , i in enumerate(list(np.random.randint(0,len(X_pred),36))) : 
    plt.subplot(6,6,n+1)
    plt.imshow(X_pred[i])    
    plt.axis('off')
    plt.title(getcode(np.argmax(y_result[i])))


# In[ ]:




