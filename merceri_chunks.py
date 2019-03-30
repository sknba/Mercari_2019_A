#przed wykonaniem kodu upewnij się, że masz około 20 GB miejsca na dysku
#zajmie to prawdopodobnie około 1,5h
#importing necessary libraries
import pandas as pd
import numpy as np
from scipy import sparse
from sklearn.preprocessing import LabelEncoder
from dummyPy import OneHotEncoder

#creating classes
labelencoder_c1 = LabelEncoder()
labelencoder_c2 = LabelEncoder()
labelencoder_c3 = LabelEncoder()
ohe = OneHotEncoder(['category_1', 'category_2', 'category_3'])

#fitting the label encoder
d1 = pd.read_csv('train.tsv', sep = '\t')
d2 = pd.read_csv('test_stg2.tsv', sep = '\t')
category_list = pd.DataFrame(pd.concat([d1['category_name'], d2['category_name']]))
category_list['category_name'] = category_list['category_name'].astype('str')
category_list['category_name'] = category_list['category_name'].replace('nan', 'no_category1/no_category2/no_category3')
categories = category_list['category_name'].str.split('/', 3, expand = True)
category_list['category_1'] = categories[0]
category_list['category_2'] = categories[1]
category_list['category_3'] = categories[2]
labelencoder_c1.fit(category_list['category_1'])
labelencoder_c2.fit(category_list['category_2'])
labelencoder_c3.fit(category_list['category_3'])


#transforming the training set to fit the onehotencoder
d1 = d1.drop('price', axis = 1)
d1['category_name'] = d1['category_name'].astype('str')
d1['category_name'] = d1['category_name'].replace('nan', 'no_category1/no_category2/no_category3')
categories = d1['category_name'].str.split('/', 3, expand = True)
d1['category_1'] = categories[0]
d1['category_2'] = categories[1]
d1['category_3'] = categories[2]
d1['category_1'] = labelencoder_c1.transform(d1['category_1'])
d1['category_2'] = labelencoder_c2.transform(d1['category_2'])
d1['category_3'] = labelencoder_c3.transform(d1['category_3'])
d1_filtered = d1[['item_condition_id',
                    'shipping',
                    'category_1',
                    'category_2',
                    'category_3']].copy()
ohe.fit(d1_filtered)


#transforming the test set to fit the onehotencoder
d2['category_name'] = d2['category_name'].astype('str')
d2['category_name'] = d2['category_name'].replace('nan', 'no_category1/no_category2/no_category3')
categories = d2['category_name'].str.split('/', 3, expand = True)
d2['category_1'] = categories[0]
d2['category_2'] = categories[1]
d2['category_3'] = categories[2]
d2['category_1'] = labelencoder_c1.transform(d2['category_1'])
d2['category_2'] = labelencoder_c2.transform(d2['category_2'])
d2['category_3'] = labelencoder_c3.transform(d2['category_3'])
d2_filtered = d2[['item_condition_id',
                    'shipping',
                    'category_1',
                    'category_2',
                    'category_3']].copy()
ohe.fit(d2_filtered)

del categories, category_list, d1, d2, d1_filtered, d2_filtered


#preprocessing the training set
for dataset in pd.read_csv('train.tsv', sep = '\t', chunksize = 10000):    
    dataset['category_name'] = dataset['category_name'].astype('str')
    dataset['category_name'] = dataset['category_name'].replace('nan', 'no_category1/no_category2/no_category3')
    categories = dataset['category_name'].str.split('/', 3, expand = True)
    dataset['category_1'] = categories[0]
    dataset['category_2'] = categories[1]
    dataset['category_3'] = categories[2]
    dataset_filtered = dataset[['item_condition_id',
                            'shipping',
                            'category_1',
                            'category_2',
                            'category_3']].copy()
    dataset_filtered['category_1'] = labelencoder_c1.transform(dataset_filtered['category_1'])
    dataset_filtered['category_2'] = labelencoder_c2.transform(dataset_filtered['category_2'])
    dataset_filtered['category_3'] = labelencoder_c3.transform(dataset_filtered['category_3'])
    dataset_filtered = ohe.transform(dataset_filtered)
    dataset_filtered.to_csv('train_preprocessed.csv', mode = 'a', index = False, header = False)
#test = pd.read_csv('train_preprocessed.csv')
#test = test.head(11000)
    
    
#preprocessing the test set
for dataset_test in pd.read_csv('test_stg2.tsv', sep = '\t', chunksize = 10000):
    dataset_test['category_name'] = dataset_test['category_name'].astype('str')
    dataset_test['category_name'] = dataset_test['category_name'].replace('nan', 'no_category1/no_category2/no_category3')
    categories_test = dataset_test['category_name'].str.split('/', 3, expand = True)
    dataset_test['category_1'] = categories_test[0]
    dataset_test['category_2'] = categories_test[1]
    dataset_test['category_3'] = categories_test[2]
    dataset_test_filtered = dataset_test[['item_condition_id',
                                'shipping',
                                'category_1',
                                'category_2',
                                'category_3']].copy()
    dataset_test_filtered['category_1'] = labelencoder_c1.transform(dataset_test_filtered['category_1'])
    dataset_test_filtered['category_2'] = labelencoder_c2.transform(dataset_test_filtered['category_2'])
    dataset_test_filtered['category_3'] = labelencoder_c3.transform(dataset_test_filtered['category_3'])
    dataset_test_filtered = ohe.transform(dataset_test_filtered)
    dataset_test_filtered.to_csv('test_stg2_preprocessed.csv', mode = 'a', index = False, header = False)

"""od tego momentu jeszcze nie działa

#Preparing X and y
#Sparse matrix
######################3
from scipy import sparse
y_train = dataset_final.iloc[:, 2].values
dataset_final = dataset_final.drop('price', axis = 1)
X_sparse = sparse.csr_matrix(dataset_final.values)
X_sparse_test = sparse.csr_matrix(dataset_test_final)

# Feature Scaling
######################
from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
X_sparse = sc.fit_transform(X_sparse)
X_sparse_test = sc.transform(X_sparse_test)

#Fitting Linear Regression
#############################
from sklearn.linear_model import LinearRegression
regressor = LinearRegression()
regressor.fit(X_sparse, y_train)
regressor.score(X_sparse, y_train)

#Predicting the results of the test set
##########################################
y_pred = regressor.predict(X_sparse_test)
"""