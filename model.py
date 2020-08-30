from sklearn.model_selection import RandomizedSearchCV, cross_val_score, StratifiedKFold, KFold
from sklearn.metrics import f1_score, classification_report
from sklearn.linear_model import LogisticRegression
import warnings
warnings.filterwarnings('ignore')
from sklearn.neural_network import MLPClassifier
from xgboost import XGBClassifier
from hyperopt import hp, tpe
from hyperopt.fmin import fmin
import numpy as np

class model():

  def __init__(self, X_train, X_test, y_train, y_test):

    '''
    This class implements Logistic Regression, Multi layer Perceptron and extreme gradient boosting algorithm
    Parameters
    ----------
    X_train : Dataframe, numpy 2D array
               The train set to be used for training
    X_test :  Dataframe, Numpy 2D array
              The hold out set, to be used for validating the model perfomancce
    y_train : pandas series, numpy 1D array
               Labels for the train set
    y_test : pandas series, numpy 1D array
              Label for the test set

    Methods
    -------
    logit : To fit the data using logistic regression
    MLP : To fit the data using Multi layered perceptron
    XGB : To fit the data using extreme gradient boosting

    Return
    ------
    Score
      A 5 fold cross validation score
    
    Example
    ------
    M = model(X_train, X_test, y_train, y_test)
    M.logit() #To fit logistic regression
    '''

    self.X_train = X_train
    self.X_test = X_test
    self.y_train = y_train
    self.y_test = y_test


  def evaluate(self, X_train, X_test, y_train, y_test, model):

    '''
    Evaluate the performance of the model

    Parameters
    ---------
    X_train : Dataframe, numpy 2D array
               The train set to be used for training
    X_test :  Dataframe, Numpy 2D array
              The hold out set, to be used for validating the model perfomancce
    y_train : pandas series, numpy 1D array
               Labels for the train set
    y_test : pandas series, numpy 1D array
              Label for the test set
    model : instance
            A fitted instance of the model

    Return
    ------
    f1_score on the train set and test set
    classification report of the test set

    Example
    -------
    evaluate(X_train, X_test, y_train, y_test)
    

    '''

    #Obtain train and test f1 score
    train_score = f1_score(y_train, model.predict(X_train))
    test_score =  f1_score(y_test, model.predict(X_test))

    #Print scores and classification report
    print (f'train score is {train_score}, test_score is {test_score}')
    print('----------------------------------------------------------')
    print(classification_report(y_test, model.predict(X_test)))

  
  def objective_xgb(self, params):

    '''
    Define optimization objective for XGBOOST

    Parameters
    ---------
    params : dict
            Model parameters to be optimized

    Return:
    ------
    Cross validationn score

    '''

    #set parameters to tune
    params = {
        'max_depth': int(params['max_depth']),
        'gamma': "{:.3f}".format(params['gamma']),
        'learning_rate': "{:.3f}".format(params['learning_rate']),
        'colsample_bytree': '{:.3f}'.format(params['colsample_bytree']),
    }
    
    #fit model with parameters
    xgb = XGBClassifier(random_state=23, **params, n_estimators=300)
    #get cv score
    skf=StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    kf = KFold(n_splits=5, shuffle=True, random_state=42)
    
    #Obtain cross validation score
    score_skf = cross_val_score(xgb, self.X_train, self.y_train, scoring='f1', cv = skf).mean()
    score_kf = cross_val_score(xgb, self.X_train, self.y_train, scoring='f1', cv = kf).mean()

    #print scores
    print("stratifiedKFold score {}, Kfold_score {}, params {}".format(score_skf,score_kf, params))
    return score_skf


  def logit(self):
    '''
    Fit Logistic regression model

    Return
    ------
    Cross validation score
    
    '''
    
    #Call the model
    print('fitting Logistic regression...')
    lr = LogisticRegression(random_state= 10, max_iter = 10000, )
    #fit model
    lr.fit(self.X_train, self.y_train)
    #Obtain and print AUC Score for test and train
    self.evaluate(self.X_train, self.X_test, self.y_train, self.y_test, lr)
    
    '''Hyper Parameter Search and Cross Validation for logistic Regression'''
    
    print('Searching for best hyperparameter... ')
    #Set params
    params = {'C': np.linspace(0.0001,0.001,20)}
    
    #Init and fit grid search
    lr_grid = RandomizedSearchCV(LogisticRegression(random_state = 10, max_iter = 10000), params,
                                 scoring='f1', cv =10,  n_iter = 20)
    lr_grid.fit(self.X_train, self.y_train)
    print('--------------------------------------\nDONE')
    print(f'Best Score {lr_grid.best_score_} Best Param {lr_grid.best_params_}')
    self.evaluate(self.X_train, self.X_test, self.y_train, self.y_test,lr_grid.best_estimator_ )
    
    print('Running Cross Val \nReturning 5fold CV Scores')
    print('--------------------------------------')
    
    #stratified kfold
    skf=StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    kf = KFold(n_splits=5, shuffle=True, random_state=42)
    
    skf_score = cross_val_score(lr_grid.best_estimator_, self.X_train, self.y_train, scoring = 'f1', cv = skf).mean()
    kf_score = cross_val_score(lr_grid.best_estimator_, self.X_train, self.y_train, scoring = 'f1', cv = kf).mean()

    
    print('--------------------------------------\nDONE')
    print('StratifiedKfold Score: {}, KFold Score: {}'.format(skf_score, kf_score))
  

  def MLP(self):
    '''
    Fit MLP model
    
    Return
    ------
    Cross validation score
    
    '''
    
    #Call the model
    print('fitting MLP...')
    mlp = MLPClassifier(random_state= 10, early_stopping= True, learning_rate= 'adaptive')
    #fit model
    mlp.fit(self.X_train, self.y_train)
    #Obtain and print AUC Score for test and train
    self.evaluate(self.X_train, self.X_test, self.y_train, self.y_test, mlp)
    
    '''Hyper Parameter Search and Cross Validation for logistic Regression'''
    
    print('Searching for best hyperparameter... ')
    #Set params
    params = {'hidden_layer_sizes': np.arange(100,600,100), 
             'learning_rate_init': np.linspace(0.001,0.003,5)}
    
    #Init and fit grid search
    mlp_grid = RandomizedSearchCV(MLPClassifier(random_state= 10, early_stopping= True, learning_rate= 'adaptive'), params,
                                 scoring='f1', cv =10,  n_iter = 20, n_jobs = 12)
    mlp_grid.fit(self.X_train, self.y_train)
    print('--------------------------------------\nDONE')
    print(f'Best Score {mlp_grid.best_score_} Best Param {mlp_grid.best_params_}')
    self.evaluate(self._train, self.X_test, self.y_train, self.y_test,mlp_grid.best_estimator_ )
    
    print('Running Cross Val \nReturning 5fold CV Scores')
    print('--------------------------------------')
    
    #stratified kfold
    skf=StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    kf = KFold(n_splits=5, shuffle=True, random_state=42)
    
    skf_score = cross_val_score(mlp_grid.best_estimator_, self.X_train, self.y_train, scoring = 'f1', cv = skf).mean()
    kf_score = cross_val_score(mlp_grid.best_estimator_, self.X_train, self.y_train, scoring = 'f1', cv = kf).mean()

    
    print('--------------------------------------\nDONE')
    print('StratifiedKfold Score: {}, KFold Score: {}'.format(skf_score, kf_score))


  def XGB(self):
    '''
    Fit XGBOOST
    
    Return
    ------
    Cross validation score
    '''
    
    #Call the model
    print('fitting xgboost...')
    xgb = XGBClassifier(random_state= 10, n_estimators = 1000,  use_best_model = True, verbosity=0)
    #fit model
    xgb.fit(self.X_train, self.y_train, eval_set = [(self.X_test, self.y_test)], eval_metric = 'auc', early_stopping_rounds = 100, verbose = 0 )
    #Obtain and print AUC Score for test and train
    self.evaluate(self.X_train, self.X_test, self.y_train, self.y_test, xgb)
    
    '''Hyper Parameter Search and Cross Validation for logistic Regression'''
    
    print('Searching for best hyperparameter... ')
    #Set params
    
    space_xgb = {
        'max_depth': hp.quniform('max_depth', 2, 5, 1),
        'colsample_bytree': hp.uniform('colsample_bytree', 0.3, 1.0),
        'gamma': hp.uniform('gamma', 0.0, 0.5),
        'learning_rate': hp.uniform('learning_rate', 0.01, 0.5)
    }

    print('Running Cross Val \nReturning 5fold CV Scores')
    print('--------------------------------------')
    
    best_xgb = fmin(fn=self.objective_xgb,
            space=space_xgb,
            algo=tpe.suggest,
            max_evals=10)
    
    best_xgb['max_depth'] = int(best_xgb['max_depth'])
    
    print('--------------------------------------\nDONE')
    print(f'Best Params {best_xgb}')
    
    skf=StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    kf = KFold(n_splits=5, shuffle=True, random_state=42)
    xgb = XGBClassifier(n_estimators=10, random_state = 42, **best_xgb)

    skf_score = cross_val_score(xgb, self.X_train, self.y_train, scoring = 'f1', cv = skf).mean()
    kf_score = cross_val_score(xgb, self.X_train, self.y_train, scoring = 'f1', cv = kf).mean()
    
    
    print('--------------------------------------\nDONE')
    print('StratifiedKfold Score: {}, KFold Score: {}'.format(skf_score, kf_score))