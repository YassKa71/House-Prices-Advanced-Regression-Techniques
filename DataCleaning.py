#%%
import numpy as np
import scipy
import matplotlib.pyplot as plt
import math
from tensorflow import initializers
import seaborn as sns
from sklearn import preprocessing
from sklearn.compose import ColumnTransformer
import pandas as pd
from keras.models import Sequential
from keras.layers import Dense
import keras.optimizers 


##### Reading and Making a copy of train data
train=pd.read_csv('train.csv')
train_copy=train.copy()
train_array=train.to_numpy()




####Printing types of data
pd.set_option('display.max_rows',81)
print(train_copy.dtypes)




####Droping columns with many NAN values
#print(train_copy.isna().sum())
train_copy=train_copy.drop(columns=['Alley', 'PoolQC','Fence','MiscFeature'])




####Replace NAN values for int and float types by mean

def replace_with_mean_float(column):
    mean_column= train_copy[column].mean()
    train_copy[column]=train_copy[column].fillna(mean_column)
    
def replace_with_mean_int(column):
    mean_column= round(train_copy[column].mean(),0)
    train_copy[column]=train_copy[column].fillna(mean_column)
    
   
replace_with_mean_float('LotFrontage')
replace_with_mean_int('GarageYrBlt')
replace_with_mean_float('MasVnrArea')




####Replacing NAN values for categorical types


#If a certain category represents more than 60% of the column values, we replace NAN values in the column by this category. Else, we replace NaN values by the category: "None_Categorical".
def replace_by_value(column):
    max_count=train_copy[column].value_counts().max()
    id_max_count=train_copy[column].value_counts().idxmax()[0]
    percentage=(max_count*100)/1461
    if percentage>=60:
        train_copy[column]=train_copy[column].fillna(id_max_count)
    else: train_copy[column]=train_copy[column].fillna("None_Categorical")

list_Na=["MasVnrType", "FireplaceQu", "GarageType", "GarageFinish","GarageQual", "GarageCond","BsmtQual","BsmtCond","BsmtExposure","BsmtFinType1","BsmtFinType2"]
for column in list_Na:
    replace_by_value(column)
 
#droping the remaining line with a NaN value     
train_copy=train_copy.dropna()





#### One-Hot-Encoder

categorical_features = train_copy.dtypes.where(train_copy.dtypes!=np.int64 ).dropna()
categorical_features = list(categorical_features.where(categorical_features!=np.float64 ).dropna().index)
#print(categorical_features)


encoder = preprocessing.OneHotEncoder(handle_unknown='ignore')
one_hot_features = encoder.fit_transform(train_copy[categorical_features])
one_hot_names = encoder.get_feature_names_out()


print("Type of one_hot_columns is:",type(one_hot_features))

one_hot_df = pd.DataFrame.sparse.from_spmatrix(one_hot_features)
one_hot_df.columns = one_hot_names 
#print(one_hot_df.head())





#### Dealing with skewness
min_max_scaler = preprocessing.StandardScaler()

numerical_features= list(train_copy.dtypes.where(train_copy.dtypes== np.int64).dropna().index)
numerical_features = numerical_features + list(train_copy.dtypes.where(train_copy.dtypes== np.float64 ).dropna().index)
#numerical_features=numerical_features[30:40]
#print(numerical_features)

dataScaled = pd.DataFrame(min_max_scaler.fit_transform(train_copy[numerical_features]), columns=numerical_features)
#viz_2=sns.violinplot(data=data, y=['price','number_of_reviews'])
#ax = sns.boxplot(data=dataScaled, orient="h")
#ax.set_title("Box plots for min-max scaled features")
def highly_skewed_data(numerical_features, dataScaled):
    transform_cols=[]
    for column in numerical_features:
        skew= scipy.stats.skew(dataScaled[column], axis=0)
        if skew>=1:
            transform_cols.append(column)
    return transform_cols

transform_cols=highly_skewed_data(numerical_features,dataScaled)
print(transform_cols)
        
    

#transform_cols = ['BsmtFinSF2','LowQualFinSF','BsmtHalfBath','KitchenAbvGr','EnclosedPorch','3SsnPorch','ScreenPorch','PoolArea','MiscVal']
for col in transform_cols:
    col_log1p = col + '_log1p'
    train_copy[col_log1p] = train_copy[col].apply(math.log1p)


min_max_scaler = preprocessing.StandardScaler()

# Now let's plot the numerical features, but take the transformed values for the columns we applied log1p to
numerical_features_log1p = numerical_features
def take_log_col(col):
    if col in transform_cols: return col + '_log1p'
    else: return col
numerical_features_log1p[:] = [take_log_col(col) for col in numerical_features_log1p]

dataScaled_log1p = pd.DataFrame(min_max_scaler.fit_transform(train_copy[numerical_features_log1p]), columns=numerical_features_log1p)
#ax = sns.boxplot(data=dataScaled_log1p, orient="h")
#ax.set_title("Box plots for min-max scaled features")

sns.distplot(train_copy['LotArea_log1p']).set_title("Distribution without log(1 + price)")

#print(train_copy["LowQualFinSF"].value_counts())
mean=train_copy['EnclosedPorch'].mean()
median=train_copy['EnclosedPorch'].median()
print(median)
print(mean)
print(scipy.stats.skew(train_copy['LotArea'], axis=0))
print(scipy.stats.skew(train_copy['LotArea_log1p'], axis=0))



# %%
