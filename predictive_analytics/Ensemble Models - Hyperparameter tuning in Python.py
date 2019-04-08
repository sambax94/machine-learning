# importing required libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn import svm
from sklearn.datasets import make_blobs
from sklearn.model_selection import train_test_split
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import classification_report
from sklearn.svm import SVC
from sklearn.preprocessing import LabelEncoder
import math
from sklearn.metrics import mean_squared_error
from sklearn.metrics import mean_absolute_error

from sklearn.ensemble import GradientBoostingRegressor
from sklearn.ensemble import AdaBoostRegressor
from sklearn.ensemble import RandomForestRegressor

from sklearn.model_selection import KFold

##-------------------------------- UNFILTERED DATA -------------------------------------------

init_data = pd.read_excel('data.xlsx')
init_data.loc[0:2,]

## Removing the sequence number
init_data = init_data.iloc[:,1:]

## Splitting the data into target (y) and predictors (X)
y = init_data.iloc[:,23]
X = init_data.iloc[:,0:22]

# Split the dataset in training and testing
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=1)

##-------------------------------- GBM
# Setting the hyperparameters by cross-validation
gbm_parameters = [{'n_estimators': [100,500,1000],
                    'max_depth': [6,10,14],
                    'max_features': [4,8,12],
                    'min_samples_leaf': [5,10,15]}]

reg = GridSearchCV(GradientBoostingRegressor(), gbm_parameters, cv=5, scoring = 'neg_mean_squared_error')
reg.fit(X_train, y_train)

print("Best parameters set found on development set:")
print(reg.best_params_)

#Best parameters set found on development set:
#{'max_depth': 6, 'max_features': 8, 'min_samples_leaf': 10, 'n_estimators': 100}

print("Best score found on development set:")
print(reg.best_score_)

#Best score found on development set:
#-13200.8723277

y_pred = reg.predict(X_test)
reg.score(X_test,y_test) ## -25033.623467464509

mean_squared_error(y_test,y_pred) ## 25033.623467464509
mean_absolute_error(y_test,y_pred) ## 78.056978019234634
math.sqrt(mean_squared_error(y_test,y_pred)) ## 158.22017402172364

##-------------------------------- Adaboost
# Setting the hyperparameters by cross-validation
ab_parameters = [{'n_estimators': [10,50,100,200],
                    'learning_rate': [0.01,0.1,0.3]}]

reg = GridSearchCV(AdaBoostRegressor(), ab_parameters, cv=5, scoring='neg_mean_squared_error')
reg.fit(X_train, y_train)

print("Best parameters set found on development set:")
print(reg.best_params_)

#Best parameters set found on development set:
#{'learning_rate': 0.01, 'n_estimators': 10}

print("Best score found on development set:")
print(reg.best_score_)

#Best score found on development set:
#-15271.2828469

# Final Adaboost model accuracy using the hyper-parameters obtained from grid search
y_pred = reg.predict(X_test)
reg.score(X_test,y_test) ## -30981.828701366718

mean_squared_error(y_test,y_pred) ## 30981.828701366718
mean_absolute_error(y_test,y_pred) ## 91.344153999094743
math.sqrt(mean_squared_error(y_test,y_pred)) ## 176.01655803181336

##-------------------------------- Random Forest
# Setting the hyperparameters by cross-validation
rf_parameters = [{'n_estimators': [100,500,1000],
                    'max_depth': [6,10,14],
                    'max_features': [4,8,12],
                    'min_samples_leaf': [5,10,15]}]

reg = GridSearchCV(RandomForestRegressor(), rf_parameters, cv=5, scoring='neg_mean_squared_error')
reg.fit(X_train, y_train)

print("Best parameters set found on development set:")
print(reg.best_params_)

#Best parameters set found on development set:
#{'max_depth': 10, 'max_features': 12, 'min_samples_leaf': 5, 'n_estimators': 500}

print("Best score found on development set:")
print(reg.best_score_)

#Best score found on development set:
#-13730.5379757

# Final RandomForest model accuracy using the hyper-parameters obtained from grid search
y_pred = reg.predict(X_test)
reg.score(X_test,y_test) ## -27606.666886122777
 
mean_squared_error(y_test,y_pred) ## 27606.666886122777
mean_absolute_error(y_test,y_pred) ## 81.366218859746553
math.sqrt(mean_squared_error(y_test,y_pred)) ## 166.1525410161481

##-------------------------------- NESTED CROSS VALIDATION

y = init_data.iloc[:,24]
X = init_data.iloc[:,0:23]
NUM_TRIALS = 10

gbm = GradientBoostingRegressor()
ab = AdaBoostRegressor()
rf = RandomForestRegressor()

# Arrays to store scores
non_nested_scores_gbm = np.zeros(NUM_TRIALS)
nested_scores_gbm = np.zeros(NUM_TRIALS)

non_nested_scores_ab = np.zeros(NUM_TRIALS)
nested_scores_ab = np.zeros(NUM_TRIALS)

non_nested_scores_rf = np.zeros(NUM_TRIALS)
nested_scores_rf = np.zeros(NUM_TRIALS)

# Loop for each trial
for i in range(NUM_TRIALS):
    print('\n Iteration number :',i)
    # Choose cross-validation techniques for the inner and outer loops,
    # independently of the dataset.
    # E.g "LabelKFold", "LeaveOneOut", "LeaveOneLabelOut", etc.
    inner_cv = KFold(n_splits=4, shuffle=True, random_state=i)
    outer_cv = KFold(n_splits=4, shuffle=True, random_state=i)
    
    ## GBM
    # Non_nested parameter search and scoring
    clf = GridSearchCV(estimator=gbm, param_grid=gbm_parameters, cv=inner_cv)
    clf.fit(X, y)
    non_nested_scores_gbm[i] = clf.best_score_
    
    # Nested CV with parameter optimization
    nested_score_gbm = cross_val_score(clf, X=X, y=y, cv=outer_cv)
    nested_scores_gbm[i] = nested_score_gbm.mean()
    
    ## Adaboost
    # Non_nested parameter search and scoring
    clf = GridSearchCV(estimator=ab, param_grid = ab_parameters, cv=inner_cv)
    clf.fit(X, y)
    non_nested_scores_ab[i] = clf.best_score_

    # Nested CV with parameter optimization
    nested_score_ab = cross_val_score(clf, X=X, y=y, cv=outer_cv)
    nested_scores_ab[i] = nested_score_ab.mean()

    ## Random Forest
    # Non_nested parameter search and scoring
    clf = GridSearchCV(estimator=rf, param_grid = rf_parameters, cv=inner_cv)
    clf.fit(X, y)
    non_nested_scores_rf[i] = clf.best_score_

    # Nested CV with parameter optimization
    nested_score_rf = cross_val_score(clf, X=X, y=y, cv=outer_cv)
    nested_scores_rf[i] = nested_score_rf.mean()
    
score_difference_gbm = non_nested_scores_gbm - nested_scores_gbm
score_difference_ab = non_nested_scores_ab - nested_scores_ab
score_difference_rf = non_nested_scores_rf - nested_scores_rf

print("GBM : \n \
      Average Accuracy for GBM [Nested] : {0:.4f} \n \
      Average Accuracy for GBM [Non-Nested] : {1:.4f} \n \
      Average difference of {2:.4f} with std. dev. of {3:.4f}."
      .format(nested_scores_gbm.mean(), 
              non_nested_scores_gbm.mean(), 
              score_difference_gbm.mean(), 
              score_difference_gbm.std()))

print("Adaboost : \n \
      Average Accuracy for Adaboost [Nested] : {0:.4f} \n \
      Average Accuracy for Adaboost [Non-Nested] : {1:.4f} \n \
      Average difference of {2:.4f} with std. dev. of {3:.4f}."
      .format(nested_scores_ab.mean(), 
              non_nested_scores_ab.mean(), 
              score_difference_ab.mean(), 
              score_difference_ab.std()))

print("Random Forest : \n \
      Average Accuracy for Random Forest [Nested] : {0:.4f} \n \
      Average Accuracy for Random Forest [Non-Nested] : {1:.4f} \n \
      Average difference of {2:.4f} with std. dev. of {3:.4f}."
      .format(nested_scores_rf.mean(), 
              non_nested_scores_rf.mean(), 
              score_difference_rf.mean(), 
              score_difference_rf.std()))
