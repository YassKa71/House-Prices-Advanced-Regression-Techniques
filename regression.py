import numpy as np
import matplotlib.pyplot as plt
from tensorflow import initializers
import pandas as pd
from keras.models import Sequential
from keras.layers import Dense
import keras.optimizers 


train=pd.read_csv('train.csv')
train_array=train.to_numpy()

#print(train["LotShape"].value_counts())
#print(train["LotConfig"].value_counts())
print(train.dtypes)