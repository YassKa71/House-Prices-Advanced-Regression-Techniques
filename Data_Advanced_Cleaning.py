#!/usr/bin/env python
# coding: utf-8

# In[93]:


import numpy as np
import scipy
import matplotlib.pyplot as plt
import math
#from tensorflow import initializers
import seaborn as sns
from sklearn import preprocessing
from sklearn.compose import ColumnTransformer
import pandas as pd
#from keras.models import Sequential
#from keras.layers import Dense
#import keras.optimizers 


# In[94]:


train=pd.read_csv('train.csv')
test = pd.read_csv('test.csv')
y_test = pd.read_csv('sample_submission.csv')

print(train.shape)
train_copy=train.copy()
test_copy = test.copy()

test_copy = pd.merge(test_copy, y_test, left_on = 'Id', right_on = 'Id')
frames = [train_copy, test_copy]
data = pd.concat(frames)
print(data.shape)
data.head()


# In[95]:


####Printing types of data
pd.set_option('display.max_rows',81)
print(train_copy.dtypes)
####Droping columns with many NAN values
#print(train_copy.isna().sum())
data=data.drop(columns=['Alley', 'PoolQC','Fence','MiscFeature'])
print(data.shape)


# In[96]:


def object_col_with_nan():
    L = []
    for col in list(data.dtypes.where(data.dtypes == np.object).dropna().index):
        if data[col].isna().sum() > 0:
            L.append(col)
    return L
object_col_with_nan()

def float_col_with_nan():
    L = []
    for col in list(data.dtypes.where(data.dtypes == np.float64).dropna().index):
        if data[col].isna().sum() > 0:
            L.append(col)
    return L

def int_col_with_nan():
    L = []
    for col in list(data.dtypes.where(data.dtypes == np.int64).dropna().index):
        if data[col].isna().sum() > 0:
            L.append(col)
    return L

print(object_col_with_nan())
print(numerical_col_with_nan())


# In[97]:


####Replace NAN values for int and float types by mean
def replace_with_mean_float(columns):
    for column in columns:
        mean_column= data[column].mean()
        data[column]=data[column].fillna(mean_column)
    
def replace_with_mean_int(columns):
    for column in columns:
        mean_column= round(data[column].mean(),0)
        data[column]=data[column].fillna(mean_column)
    
replace_with_mean_int(int_col_with_nan())
replace_with_mean_float(float_col_with_nan())
print(data.shape)


# In[98]:


####Replacing NAN values for categorical types
#If a certain category represents more than 60% of the column values, we replace NAN values in the column by this category. Else, we replace NaN values by the category: "None_Categorical".
def replace_by_value(column):
    max_count=data[column].value_counts().max()
    id_max_count=data[column].value_counts().idxmax()[0]
    percentage=(max_count*100)/2919
    if percentage>=60:
        data[column]=data[column].fillna(id_max_count)
    else: data[column]=data[column].fillna("Unspecified")

list_Na=object_col_with_nan()
for column in list_Na:
    replace_by_value(column) 
print(data.shape)
#droping the remaining line with a NaN value
print(data.isna().sum())
data=data.dropna()
print(data.shape)


# In[99]:


#### One-Hot-Encoder

categorical_features = data.dtypes.where(data.dtypes!=np.int64 ).dropna()
categorical_features = list(categorical_features.where(categorical_features!=np.float64 ).dropna().index)
#print(categorical_features)


encoder = preprocessing.OneHotEncoder(handle_unknown='ignore')
one_hot_features = encoder.fit_transform(data[categorical_features])
one_hot_names = encoder.get_feature_names_out()


print("Type of one_hot_columns is:",type(one_hot_features))

one_hot_df = pd.DataFrame.sparse.from_spmatrix(one_hot_features)
one_hot_df.columns = one_hot_names 
#print(one_hot_df.head())


# In[104]:


#### Dealing with skewness
stdScaler_x = preprocessing.StandardScaler()
stdScaler_y = preprocessing.StandardScaler()

train_copy= data.iloc[:1460,:]
train_copy_x = pd.DataFrame(data.iloc[:1460,:-1])
train_copy_y = pd.DataFrame(data.iloc[:1460,-1])
test_copy_x = pd.DataFrame(data.iloc[1460:,:-1])


numerical_features= list(train_copy_x.dtypes.where(train_copy_x.dtypes== np.int64).dropna().index)
numerical_features = numerical_features + list(train_copy_x.dtypes.where(train_copy_x.dtypes== np.float64 ).dropna().index)
#numerical_features=numerical_features[30:40]
#print(numerical_features)


#Scaling the train and test data using the training distribution
dataScaled_x = pd.DataFrame(stdScaler_x.fit_transform(train_copy_x[numerical_features]), columns=numerical_features)
dataScaled_y = pd.DataFrame(stdScaler_y.fit_transform(train_copy_y[['SalePrice']]), columns=['SalePrice'])
testScaled_x = pd.DataFrame(stdScaler_x.transform(test_copy_x[numerical_features]), columns=numerical_features)

#Defining the columns with highly skewed data
def highly_skewed_data(numerical_features, dataScaled):
    transform_cols=[]
    for column in numerical_features:
        skew= scipy.stats.skew(dataScaled[column], axis=0)
        if skew>=1 or skew<=-1:
            transform_cols.append(column)
    return transform_cols

transform_cols_x = highly_skewed_data(numerical_features, dataScaled_x)
transform_cols_y = highly_skewed_data(['SalePrice'], dataScaled_y)
print(transform_cols_y)

#Renaming the columns in transform_cols 
def take_log_col(col):
    if col in transform_cols_x or col in transform_cols_y: return col + '_log1p'
    else: return col

#applying the log transformation to our training and testing data
def add_log1p_col(train_copy, transform_cols):
    for col in transform_cols:
        col_log1p = col + '_log1p'
        train_copy[col_log1p] = train_copy[col].apply(math.log1p)

add_log1p_col(train_copy_x, transform_cols_x)
add_log1p_col(train_copy_y, transform_cols_y)
add_log1p_col(test_copy_x, transform_cols_x)


#Scaling the newly added features using the training distribution
numerical_features_log1p = numerical_features
numerical_features_log1p[:] = [take_log_col(col) for col in numerical_features_log1p]
dataScaled_log1p_x = pd.DataFrame(stdScaler_x.fit_transform(train_copy_x[numerical_features_log1p]), columns=numerical_features_log1p)
testScaled_log1p_x= pd.DataFrame(stdScaler_x.transform(test_copy_x[numerical_features_log1p]), columns=numerical_features_log1p)
dataScaled_log1p_y = pd.DataFrame(stdScaler_y.fit_transform(train_copy_y[[take_log_col('SalePrice')]]), columns=[take_log_col('SalePrice')])



        
    # Now let's plot the numerical features, but take the transformed values for the columns we applied log1p to
    #ax = sns.boxplot(data=dataScaled_log1p, orient="h")
    #ax.set_title("Box plots for min-max scaled features")
    


# In[105]:


def show(dataScaled, train_copy, title, col):
    plt.figure(figsize = (8,8))
    ax = sns.boxplot(data=dataScaled, orient="h")
    ax.set_title("Box plots for standard scaled features")
    
    plt.figure(figsize = (8,8))
    sns.distplot(train_copy[col]).set_title(title)
    plt.show()
    
show(dataScaled_x, train_copy_x,"Distribution without log(1 + price)",'LotArea' )
show(dataScaled_log1p_x, train_copy_x,"Distribution with log(1 + price)",'LotArea_log1p' )
show(dataScaled_y, train_copy_y,"Distribution without log(1 + price)",'SalePrice' )
show(dataScaled_log1p_y, train_copy_y,"Distribution with log(1 + price)",'SalePrice_log1p' )


# In[109]:


final_train_data_x = pd.merge(dataScaled_log1p_x, one_hot_df, left_index = True, right_index = True )
final_test_data_x = pd.merge(testScaled_log1p_x, one_hot_df, left_index = True, right_index = True )
final_train_data_y= dataScaled_log1p_y


# In[110]:


final_train_data_x.head()
print(final_train_data_x.shape)
print(final_test_data_x.shape)


# In[1]:


final_test_data_x.head()


# In[ ]:




