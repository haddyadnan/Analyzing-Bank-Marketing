import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler, PolynomialFeatures, QuantileTransformer
from imblearn.over_sampling import SMOTE

def preprocess(df):#, linear = False):

    '''
    parameters
    ----------
    df : A pandas dataframe
          should contain the data

    Return
    ------
    X_train, X_test, y_train, y_test
      A pandas dataframe of the train and test set that has been transformed with PCA and minority class over sampling
    
    '''
    #perform feature engineering and data preprocessing
    d = {'yes':1, 'no':0}
    y = df.y.replace(d)
    df['new_pdays'] = df['pdays'].apply(lambda x: 1 if x < 16 else 0 if x > 30 else 2)
    df['new_pdays'] = df['previous'] * df['new_pdays']
    df['new_emp_rate'] = df['emp.var.rate'].apply(lambda x: 0 if x > 0 else 1)
    df['empxemp'] = (df['nr.employed'] / df['euribor3m'])# + df[emp.var.rate]
    
    #Turn categorical columns to dummies
    X = pd.get_dummies(df.drop(['y','cons.conf.idx','previous'], axis = 1))
    
    #Split data into train and test set
    X_train, X_test, y_train, y_test = train_test_split(X,y, test_size = .1, random_state = 12)
    print('The shape of the train set is {}, the shape of the test set is {}'.format(X_train.shape, X_test.shape))
    
   # if linear:
#         print('Generating  polynomials')

#         poly = PolynomialFeatures(2)
#         X_train = poly.fit_transform(X_train)
#         X_test = poly.transform(X_test)
        
    #Scale the data
    S = StandardScaler()
    
    #Apply the transformation on the train and test set 
    X_train = S.fit_transform(X_train)
    X_test = S.transform(X_test)
    
    #Apply PCA on the train and test set
    pca = PCA(10)
    x_train_pca = pca.fit_transform(X_train)
    x_test_pca = pca.transform(X_test)
    
    #Oversample the minority class
    sm = SMOTE(k_neighbors=10, random_state= 10)
    
    X_m, y_m = sm.fit_sample(x_train_pca, y_train)
    
    return X_m, x_test_pca, y_m, y_test