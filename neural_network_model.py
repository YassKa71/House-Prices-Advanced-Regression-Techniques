
import numpy as np
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow import initializers
from keras.models import Sequential
from keras.layers import Dense
import keras.optimizers
import pandas as pd 
import math
from sklearn.metrics import r2_score
from sklearn.decomposition import PCA as pca

import Data_Advanced_Cleaning as dt


dt.final_train_data_x = dt.final_train_data_x.iloc[:,1:]
dt.final_test_data_x = dt.final_test_data_x.iloc[:,1:]

###PCA

pcas = pca(.9)
pcas.fit(dt.final_train_data_x)
print("Number of PCA components are : {0}".format(pcas.n_components_))

##################################### Transform the Data ##################
dt.final_train_data_x = pcas.fit_transform(dt.final_train_data_x)
dt.final_test_data_x  = pcas.transform(dt.final_test_data_x)

train_x, test_x, train_y, test_y = train_test_split((dt.final_train_data_x), (dt.final_train_data_y), test_size = 0.1, random_state = 123)

print(train_x.shape)
print(train_y.shape)

model = Sequential()
num_columns= (train_x).shape[1]
num_rows=(train_x).shape[0]


#1/5/7/10 
''' 
model.add(Dense(num_columns, input_dim=num_columns, activation = 'relu' ,kernel_initializer=initializers.RandomNormal(stddev=0.01), bias_initializer=initializers.Zeros())) # (features,)
model.add(Dense(300,activation = 'relu' ,kernel_initializer=initializers.RandomNormal(stddev=0.01))) # output node with 0.009 lerning rate
model.add(Dense(100,activation = 'relu' ,kernel_initializer=initializers.RandomNormal(stddev=0.01))) # output node with 0.009 lerning rate
model.add(Dense(1, activation='sigmoid', kernel_initializer=initializers.RandomNormal(stddev=0.01))) # output node with 0.009 lerning rate
'''

model.add(Dense(num_columns, input_dim=num_columns, activation = 'sigmoid' ,kernel_initializer=initializers.RandomNormal(stddev=0.01), bias_initializer=initializers.Zeros())) # (features,)
model.add(Dense(100,activation = 'sigmoid' ,kernel_initializer=initializers.RandomNormal(stddev=0.01)))
model.add(Dense(75,activation = 'sigmoid' ,kernel_initializer=initializers.RandomNormal(stddev=0.01)))
model.add(Dense(50,activation = 'sigmoid' ,kernel_initializer=initializers.RandomNormal(stddev=0.01)))
model.add(Dense(10,activation = 'sigmoid' ,kernel_initializer=initializers.RandomNormal(stddev=0.01)))
model.add(Dense(1,activation = 'sigmoid' ,kernel_initializer=initializers.RandomNormal(stddev=0.01))) # output node with 0.00009 lerning rate

model.summary() # see what the model looks like


model.compile(
    #optimizer='rmsprop',  # Optimizer
    optimizer=keras.optimizers.Adam(learning_rate = 0.0009), 
    # Loss function to minimize
    loss='mse',
    metrics=[tf.keras.metrics.RootMeanSquaredError()]
    # List of metrics to monitor
    # metrics=['mae'],
)

print("Fit model on training data")
history = model.fit(
    train_x,
    train_y,
    validation_data=(test_x,test_y),
    batch_size=5,
    epochs=100,
    verbose=2,
    # We pass some validation for
    # monitoring validation loss and metrics
    # at the end of each epoch
)

history_dict = history.history
def showdata():
    loss_values = history_dict['loss'] # you can change this
    val_loss_values = history_dict['val_loss'] # you can also change this
    epochs = range(1, len(loss_values) + 1) # range of X (no. of epochs)
    plt.plot(epochs, loss_values, 'bo', label='Training loss')
    plt.plot(epochs, val_loss_values, 'orange', label='Validation loss')
    plt.title('Training and validation loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()
    # plt.show()

#Saving the model
model_json = model.to_json()
with open("model.json", "w") as json_file:
    json_file.write(model_json)
model.save_weights("model.h5")


pred_vect = model.predict((dt.final_test_data_x))
pred_vect = dt.stdScaler_y.inverse_transform(pred_vect)
pred_vect=[math.exp(element) for element in pred_vect ]
print(pred_vect)
#pred_vect =[ '%.0f' % elem for elem in pred_vect ]
#print(pred_vect)
sub_data = { 'SalePrice': pred_vect}
df_sub = pd.DataFrame(sub_data)
df_sub.to_csv('Submission.csv', index=False)

real_y = pd.read_csv('sample_submission.csv')
real_y = real_y.iloc[:,-1]
print(r2_score(real_y, pred_vect))

real_y_log=np.array([math.log(element) for element in real_y])
pred_vect_log=np.array([math.log(element) for element in pred_vect])
diff= np.abs(real_y_log - pred_vect_log) 
RMSE = np.sqrt(diff.sum() / np.size(pred_vect))
print("The RMSE using NN is: ",RMSE)
