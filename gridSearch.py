
from sklearn.svm import LinearSVR, SVC, SVR
from sklearn.metrics import mean_squared_error,  mean_absolute_error
from sklearn.neighbors import KNeighborsRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import AdaBoostRegressor, RandomForestRegressor, ExtraTreesRegressor, GradientBoostingRegressor
from sklearn.model_selection import GridSearchCV,RandomizedSearchCV
import lightgbm as lgb
from sklearn.model_selection import train_test_split
from catboost import CatBoostRegressor
from ngboost import NGBRegressor
from ngboost.distns import Exponential, Normal
import numpy as np

class TreeSearchCV():
    def __init__(self,N_job=3,SetSeed=1):
        self.N_job=N_job
        self.SetSeed=SetSeed
    def tree_Randomized_search_regressor(self,N_job,SetSeed,model=['ADB','DT','RF','ET','GBM','LGB','CBM','NGB'],default_grid=None,base=None):
        models=dict()
        if 'ADB' in model:
            ab = AdaBoostRegressor(random_state=SetSeed)
            if default_grid!=None:
                ab_hyperparameter_grid=default_grid['ADB']
            else:
                ab_hyperparameter_grid = {
                    'learning_rate': [0.01, 0.05, 0.1, 0.3, 1],
                    'loss': ['ls', 'lad', 'huber', 'linear', 'square', 'exponential'],
                    'n_estimators': [100, 500, 900]}
            models['AdaBoostRegressor'] = GridSearchCV(ab,
                                             param_grid=ab_hyperparameter_grid,
                                             cv=3, scoring='neg_mean_absolute_error',# n_iter=10,
                                             return_train_score=True, n_jobs=N_job#, random_state=SetSeed
                                                       )
        if 'DT' in model:
            dt = DecisionTreeRegressor(random_state=SetSeed)
            if default_grid!=None:
                dt_hyperparameter_grid=default_grid['DT']
            else:
                dt_hyperparameter_grid = {
                    'criterion': ['mse', 'friedman_mse', 'mae'],
                    'max_depth': [2, 3, 5, 10, 15],
                    'max_features': ['auto', 'sqrt', 'log2', None],
                    'min_samples_leaf': [1, 2, 4, 6, 8],
                    'min_samples_split': [2, 4, 6, 10],
                    'splitter': ['best', 'random']}
            models['DecisionTreeRegressor'] = GridSearchCV(dt,
                                             param_grid=dt_hyperparameter_grid,
                                             cv=3, scoring='neg_mean_absolute_error',# n_iter=10,
                                             return_train_score=True, n_jobs=N_job#, random_state=SetSeed
                                                           )
        if 'RF' in model:
            rf = RandomForestRegressor(random_state=SetSeed)
            if default_grid!=None:
                rf_hyperparameter_grid=default_grid['RF']
            else:
                rf_hyperparameter_grid = {
                    'criterion': ['mse', 'mae'],
                    'max_depth': [2, 5, 10, 15, 20],
                    'max_features': ['auto', 'sqrt', 'log2'],
                    'min_samples_leaf': [1, 2, 4, 8],
                    'min_samples_split': [2, 4, 10]}
            models['RandomForestRegressor'] =  GridSearchCV(rf,
                                             param_grid = rf_hyperparameter_grid,
                                             cv=3,scoring='neg_mean_absolute_error',#n_iter = 10,
                                             return_train_score = True, n_jobs=N_job#,random_state=SetSeed
                                                            )
        if 'ET' in model:
            et = ExtraTreesRegressor(random_state=SetSeed)
            if default_grid!=None:
                et_hyperparameter_grid=default_grid['ET']
            else:
                et_hyperparameter_grid = {'bootstrap': [False,True],
                        'criterion':['mse','mae'],
                        'max_depth': [2, 3, 5, 10, None],
                        'min_samples_leaf':  [1, 2, 4, 8] ,
                        'min_samples_split':  [2, 4, 10],
                        'max_features': ['auto', 'sqrt', 'log2']}
            models['ExtraTreesRegressor'] = GridSearchCV(et,
                                             param_grid = et_hyperparameter_grid,
                                             cv=3,scoring='neg_mean_absolute_error',#n_iter = 10,
                                             return_train_score = True,
                                              n_jobs=N_job#,random_state=SetSeed
                                                         )
        if 'GBM' in model:
            gb = GradientBoostingRegressor(random_state=SetSeed)
            if default_grid!=None:
                gb_hyperparameter_grid=default_grid['GBM']
            else:
                gb_hyperparameter_grid = {'loss': ['ls', 'lad','huber', 'quantile'],
                       'learning_rate':[0.01,0.05,0.1,0.3,1],
                        'criterion':['friedman_mse','mse','mae'],
                        'max_depth': [2, 3, 5, 10, None],
                        'min_samples_leaf':  [1, 2, 4, 8] ,
                        'min_samples_split':  [2, 4, 10],
                        'max_features': ['auto', 'sqrt', 'log2']}
            models['GradientBoostingRegressor'] =  GridSearchCV(gb,
                                             param_grid = gb_hyperparameter_grid,
                                             cv=3,scoring='neg_mean_absolute_error',#n_iter = 10,
                                             return_train_score = True, n_jobs=N_job#,random_state=SetSeed
                                                                )

        if 'LGB' in model:
            lgbm = lgb.LGBMRegressor(random_state=SetSeed)
            if default_grid!=None:
                lgbm_hyperparameter_grid=default_grid['LGB']
            else:
                lgbm_hyperparameter_grid = {
                        'boosting_type': ['gbdt', 'dart'],
                        'learning_rate': [0.01, 0.05, 0.1, 0.3, 1],
                        'max_depth': [2, 3, 5, 10, None],
                        'min_split_gain': [0, .2]}
            models['LightGBMRegressor'] = GridSearchCV(lgbm,
                                             param_grid=lgbm_hyperparameter_grid,
                                             cv=3, scoring='neg_mean_absolute_error',# n_iter=10,
                                             return_train_score=True, n_jobs=N_job#, random_state=SetSeed
                                                         )
        if 'CBM' in model:
            cat = CatBoostRegressor(random_state=SetSeed)
            if default_grid!=None:
                cat_hyperparameter_grid=default_grid['CBM']
            else:
                cat_hyperparameter_grid = {
                    'learning_rate': [0.01, 0.05, 0.1, 0.3, 1],
                    'depth': [2, 3, 5, 10],
                    'l2_leaf_reg': [1, 3, 5, 7, 9]}
            models['CatBoostRegressor'] = GridSearchCV(cat,
                                             param_grid=cat_hyperparameter_grid,
                                             cv=3, scoring='neg_mean_absolute_error',# n_iter=10,
                                             return_train_score=True, n_jobs=N_job#, random_state=SetSeed
                                                       )
        if 'NGB' in model:
            ngb = NGBRegressor(random_state=SetSeed)
            if default_grid!=None:
                ngb_param_grid=default_grid['NGB']
            else:
                if base==None:
                    rf = RandomForestRegressor(random_state=SetSeed)
                    et = ExtraTreesRegressor(random_state=SetSeed)
                    base=[rf,et]
                ngb_param_grid = {'learning_rate': [0.01, 0.05, 0.1, 0.3, 1],
                                  'Base__min_samples_leaf': [1, 2, 4, 8],
                                  'Base__min_samples_split': [2, 4, 10],
                                  'minibatch_frac': [1.0, 0.5],
                                  'Base': base
                                  }
            models['NGBRegressor']  = GridSearchCV(ngb,
                                         param_grid=ngb_param_grid,
                                         cv=3, scoring='neg_mean_absolute_error',# n_iter=10,
                                         return_train_score=True, n_jobs=N_job#, random_state=SetSeed
                                                   )

        return models


    def pred_performace(df_pred):
        gold_data, pred_data = df_pred['target'].values, df_pred['prediction'].values
        performance = dict()
        performance['mse'] = mean_squared_error(gold_data, pred_data)
        performance['rms'] = np.sqrt(performance['mse'])
        performance['mae'] = mean_absolute_error(gold_data, pred_data)
        performance['cor'] = sum((gold_data-np.mean(gold_data))*(pred_data-np.mean(pred_data)))/(np.sqrt(sum((gold_data-np.mean(gold_data))**2))*np.sqrt(sum((pred_data-np.mean(pred_data))**2)))
        return performance