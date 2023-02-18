
import numpy as np
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
from tensorflow import initializers
from keras.models import Sequential
from keras.layers import Dense
import keras.optimizers 

import Data_Advanced_Cleaning as dt

train_x, test_x, train_y, test_y = train_test_split(dt.final_train_data_x, dt.final_train_data_y, test_size = 0.2, random_state = 123)

model = Sequential()
num_columns= (dt.final_train_data_x).shape[1]
num_rows=(dt.final_train_data_x).shape[0]


#1/5/7/10  
model.add(Dense(num_columns, input_shape=(num_columns,), activation = 'relu' ,kernel_initializer=initializers.RandomNormal(stddev=0.01), bias_initializer=initializers.Zeros())) # (features,)
model.add(Dense(5,activation = 'relu' ,kernel_initializer=initializers.RandomNormal(stddev=0.01)))
model.add(Dense(10,activation = 'relu' ,kernel_initializer=initializers.RandomNormal(stddev=0.01)))
model.add(Dense(20,activation = 'relu' ,kernel_initializer=initializers.RandomNormal(stddev=0.01)))
model.add(Dense(5,activation = 'relu' ,kernel_initializer=initializers.RandomNormal(stddev=0.01)))
model.add(Dense(1,activation = 'relu' ,kernel_initializer=initializers.RandomNormal(stddev=0.01))) # output node with 0.009 lerning rate


model.summary() # see what the model looks like


model.compile(
    #optimizer='rmsprop',  # Optimizer
    optimizer=keras.optimizers.Adam(learning_rate = 0.009), 
    # Loss function to minimize
    loss='mse',
    # List of metrics to monitor
    # metrics=['mae'],
)

print("Fit model on training data")
history = model.fit(
    train_x,
    train_y,
    validation_data=(test_x,test_y),
    batch_size=1460,
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


pred_vect = model.predict(dt.final_test_data_x)
pred_vect = dt.stdScaler_y.inverse_transform(pred_vect)[0]
pred_vect = [round(element,0) for element in pred_vect]
Id = dt.final_test_data_x['Id'].values
sub_data = {'Id': Id, 'SalePrice': pred_vect}
df_sub = pd.DataFrame(sub_data)
df_sub.to_csv('Submission.csv', index=False)


print(2)
