#przed wykonaniem kodu upewnij się, że masz około 20 GB miejsca na dysku
#zajmie to prawdopodobnie około 1,5h
#importing necessary libraries

import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder
from dummyPy import OneHotEncoder
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler

#creating classes
labelencoder_c1 = LabelEncoder()
labelencoder_c2 = LabelEncoder()
labelencoder_c3 = LabelEncoder()
ohe = OneHotEncoder(['category_1', 'category_2', 'category_3'])
sc = StandardScaler()

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
#d1 = d1.drop('price', axis = 1)
d1['category_name'] = d1['category_name'].astype('str')
d1['category_name'] = d1['category_name'].replace('nan', 'no_category1/no_category2/no_category3')
categories = d1['category_name'].str.split('/', 3, expand = True)
d1['category_1'] = categories[0]
d1['category_2'] = categories[1]
d1['category_3'] = categories[2]
d1['category_1'] = labelencoder_c1.transform(d1['category_1'])
d1['category_2'] = labelencoder_c2.transform(d1['category_2'])
d1['category_3'] = labelencoder_c3.transform(d1['category_3'])
d1_filtered = d1[['price',
                  'item_condition_id',
                  'shipping',
                  'category_1',
                  'category_2',
                  'category_3']]
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
                    'category_3']]
ohe.fit(d2_filtered)

del categories, category_list, d1, d2, d1_filtered, d2_filtered


#preprocessing the training set (148 chunks) (.copy() removed)
i = 0
for dataset in pd.read_csv('train.tsv', sep = '\t', chunksize = 10000):
    print(i)    
    dataset['category_name'] = dataset['category_name'].astype('str')
    dataset['category_name'] = dataset['category_name'].replace('nan', 'no_category1/no_category2/no_category3')
    categories = dataset['category_name'].str.split('/', 3, expand = True)
    dataset['category_1'] = categories[0]
    dataset['category_2'] = categories[1]
    dataset['category_3'] = categories[2]
    dataset_filtered = dataset[['price',
                            'item_condition_id',
                            'shipping',
                            'category_1',
                            'category_2',
                            'category_3']]
    dataset_filtered['category_1'] = labelencoder_c1.transform(dataset_filtered['category_1'])
    dataset_filtered['category_2'] = labelencoder_c2.transform(dataset_filtered['category_2'])
    dataset_filtered['category_3'] = labelencoder_c3.transform(dataset_filtered['category_3'])
    dataset_filtered = ohe.transform(dataset_filtered)
   # dataset_filtered = pd.concat([dataset_filtered, dataset['price']], axis = 1, ignore_index = True)
    dataset_filtered.to_csv('train_preprocessed.csv', mode = 'a', index = False, header = False)
    i = i + 1
#test = pd.read_csv('train_preprocessed.csv')
#test = test.head(11000)
    
    
#preprocessing the test set (probably around 450 chunks)
j = 0
for dataset_test in pd.read_csv('test_stg2.tsv', sep = '\t', chunksize = 10000):
    print(j)
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
                                'category_3']]
    dataset_test_filtered['category_1'] = labelencoder_c1.transform(dataset_test_filtered['category_1'])
    dataset_test_filtered['category_2'] = labelencoder_c2.transform(dataset_test_filtered['category_2'])
    dataset_test_filtered['category_3'] = labelencoder_c3.transform(dataset_test_filtered['category_3'])
    dataset_test_filtered = ohe.transform(dataset_test_filtered)
    dataset_test_filtered.to_csv('test_stg2_preprocessed.csv', mode = 'a', index = False, header = False)
    j = j + 1
del categories, categories_test, dataset, dataset_filtered, dataset_test, dataset_test_filtered

#testing ground
nans = d1['price'][d1['price'] == 'nan']
sample = pd.read_csv('train_preprocessed3.csv', nrows = 1001)
nulls = sample.isnull().values
rows_null = sample.iloc[:, :][nulls]
"""