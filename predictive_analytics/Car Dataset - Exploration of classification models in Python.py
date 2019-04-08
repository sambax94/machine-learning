# Loading libraries
import numpy as np
import pandas as pd
import sklearn
from sklearn import svm
from sklearn.datasets import make_blobs
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix,classification_report
from sklearn.svm import SVC
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import StandardScaler
from sklearn.naive_bayes import MultinomialNB
from sklearn import metrics
from sklearn import tree
from sklearn import neighbors, datasets
from sklearn import preprocessing
from sklearn.model_selection import train_test_split
from sklearn.model_selection import StratifiedKFold
from sklearn.model_selection import GridSearchCV
from sklearn.tree import DecisionTreeClassifier, export_graphviz
from sklearn.metrics import classification_report, accuracy_score, confusion_matrix
from sklearn import linear_model
from sklearn.model_selection import cross_val_score
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
import seaborn as sns
from sklearn.tree import export_graphviz
import subprocess
from pandas import Series, DataFrame
from sklearn.model_selection import KFold

# Importing data 
car_data = np.loadtxt('car.data.txt',dtype=np.str,delimiter=',')

# Categorical data preparation
# Separating attributes and target fields from the data
X = np.concatenate([np.array(pd.get_dummies(car_data[:,0])),
                     np.array(pd.get_dummies(car_data[:,1])),
                     np.array(pd.get_dummies(car_data[:,2])),
                     np.array(pd.get_dummies(car_data[:,3])),
                     np.array(pd.get_dummies(car_data[:,4])),
                     np.array(pd.get_dummies(car_data[:,5]))],axis=1)

y = car_data[:,6]

# Split the dataset in training and testing
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=1, stratify=y)

# -------------------------- SVM Model ----------------------------------------

# Set the parameters byfor cross-validation
tuned_parameters = [{'kernel': ['rbf'], 'gamma': [1e-3, 1e-4],
                     'C': [1, 10, 100, 1000]},
                    {'kernel': ['linear'], 'C': [1, 10, 100, 1000]},
                    {'kernel':['poly'],'degree':[1,2,3,4,5,6,7,8],
                     'C': [1, 10, 100, 1000]}]

# Execute cross validation to find the best set of parameters and hyper parameters
clf = GridSearchCV(SVC(), tuned_parameters, cv=5, scoring='accuracy', refit=True)
clf.fit(X_train, y_train)

# Best parameters
print("Best parameters set found on development set:")
print(clf.best_params_)
print(clf.best_score_)
# Best parameters set found on development set:
# {'C': 1000, 'degree': 3, 'kernel': 'poly'}


# Showing grid scores on development set
print("Grid scores on development set:")
print()
means = clf.cv_results_['mean_test_score']
stds = clf.cv_results_['std_test_score']
for mean, std, params in zip(means, stds, clf.cv_results_['params']):
    print("%0.3f (+/-%0.03f) for %r"
          % (mean, std * 2, params))


#Train final model
clf_final = SVC(**clf.best_params_, random_state=1)
clf_final = clf_final.fit(X_train, y_train)
scores = cross_val_score(clf, X_train, y_train, cv=10)
print ('Accuracy on the Train dataset:',round(scores.mean()*100,3),'%')
#Accuracy on the Train dataset: 99.855 %

# Evaluating the algorithm
y_true, y_pred = y_test, clf.predict(X_test)
clf.score(X_test, y_test)

print(classification_report(y_true, y_pred))
print(confusion_matrix(y_true, y_pred))

# 100% accuracy
# -------------------------- Naive Bayes Model --------------------------------

# Performing a GridSearch cross-validation to find the optimal hyperparameters
tuned_parameters = [{'alpha': list(np.arange(0.01,100, 1)),
                    'fit_prior': ['True', 'False']}]

# Execute cross validation to find the best set of parameters and hyper parameters
clf = GridSearchCV(MultinomialNB(), tuned_parameters, cv=5, scoring='accuracy',refit=True)
clf.fit(X_train, y_train)

# Best parameters
print("Best parameters set found on development set:")
print(clf.best_params_)
print(clf.best_score_)
print(clf.best_estimator_)
# {'alpha': 0.01, 'fit_prior': 'True'}
# 0.8523878437047757

# Showing grid scores on development set
print("Grid scores on development set:")
print()
means = clf.cv_results_['mean_test_score']
stds = clf.cv_results_['std_test_score']
for mean, std, params in zip(means, stds, clf.cv_results_['params']):
    print("%0.3f (+/-%0.03f) for %r"
          % (mean, std * 2, params))
  
# Using the best model from GridSearchCV to predict on the test dataset
# The best hyperparamters are alpha = 0.01 and fit_prior = True
    
#Train final model
scores = cross_val_score(clf, X_train, y_train, cv=10)
print ('Accuracy on the Train dataset:',round(scores.mean()*100,3),'%')
#Accuracy on the Train dataset: 99.855 %

# Evaluating the algorithm
y_true, y_pred = y_test, clf.predict(X_test)
clf.score(X_test, y_test)

print(classification_report(y_true, y_pred))
print(confusion_matrix(y_true, y_pred))

# -------------------------- Decision Tree ------------------------------------

#Hyperparameter selection
max_depths = np.linspace(1, 20, 20, endpoint=True)
max_features = list(range(1,21))
min_samples_splits = np.linspace(1, 0, 10, endpoint=False)
min_samples_leafs = np.linspace(.5, 0, 5, endpoint=False)
criterion_options = ["entropy", "gini"]

parameter_grid_tree = {'max_depth': max_depths,
                  'criterion' :criterion_options}

dtc = DecisionTreeClassifier()

grid_search_tree = GridSearchCV(dtc, param_grid = parameter_grid_tree,cv=10, refit = True)                          
grid_search_tree.fit(X_train, y_train)
print ("Best Score: {}".format(grid_search_tree.best_score_))
print ("Best params: {}".format(grid_search_tree.best_params_))

#Best Score: 0.9797395079594791
#Best params: {'criterion': 'entropy', 'max_depth': 12.0}


#Train final model
clf_final = tree.DecisionTreeClassifier(**grid_search_tree.best_params_, random_state=1)
clf_final = clf_final.fit(X_train, y_train)
scores = cross_val_score(clf_final, X_train, y_train, cv=10)
print ('Accuracy on the Train dataset:',round(scores.mean()*100,3),'%')
#Accuracy on the Train dataset: 97.688 %

#Test the model 
y_true, y_pred = y_test, clf_final.predict(X_test) #Can use the gridsearch object as well
print('Accuracy on the Test dataset:',round(accuracy_score(y_true, y_pred)*100,2),'%')
print(classification_report(y_true, y_pred))

confusion = confusion_matrix(y_test, y_pred)
print (confusion)

# -------------------------- Logistic Regression ------------------------------

#Using GridSearch to get best possible parameters
param_grid={'C':[1,10,100,1000],
           'solver':['newton-cg', 'lbfgs','sag'],
           'multi_class':['multinomial']}

grid=GridSearchCV(estimator=LogisticRegression(),
                  param_grid=param_grid,cv=5,refit=True)
grid.fit(X_train,y_train)


#Identifying best hyperparameter from GridSearch
print(grid.best_params_)
print(grid.best_score_)
#{'C': 100, 'multi_class': 'multinomial', 'solver': 'lbfgs'}
#0.9312590448625181

#Test the model 
scores = cross_val_score(grid, X_train, y_train, cv=10)
print ('Accuracy on the Train dataset:',round(scores.mean()*100,3),'%')
#Accuracy on the Train dataset: 97.688 %

#Test the model 
y_true, y_pred = y_test, grid.predict(X_test)
print('Accuracy on the Test dataset:',round(accuracy_score(y_true, y_pred)*100,2),'%')
print(classification_report(y_true, y_pred))
confusion = confusion_matrix(y_test, y_pred)
print (confusion)


# -------------------------- K-Nearest Neighbour Model-------------------------
# Running the KNN classifier
knns = neighbors.KNeighborsClassifier()

# Creating grid of hyper parameters for grid search
parameters = {'n_neighbors': [1,3,5],
              'metric' : ["euclidean", "cityblock","manhattan"], 
              'weights' : ['uniform', 'distance']}

# Creating 10 Kfolds
kfolds = StratifiedKFold(10)

# Doing the grid search of the 10 kfolds
clf1 = GridSearchCV(knns, parameters, cv=10, refit=True)# kfolds.split(standardized_X, y))
clf1.fit(X_train, y_train)

print(clf1.best_params_)
# {'metric': 'euclidean', 'n_neighbors': 5, 'weights': 'uniform'}


# Accuracy on training data
print(clf1.score(X_train,y_train))
# 0.9377713458755427

# Accuracy on testing data
print(clf1.score(X_test,y_test))
# 0.8959537572254336

# Classification report on testing data
print(classification_report(y_test, clf1.predict(X_test)))

# Confusion MAtrix
print(confusion_matrix(y_test,clf1.predict(X_test)))

# -------------------------- Nested Cross Validation---------------------------
# -----------------------------------------------------------------------------

NUM_TRIALS = 10

# Set up possible values of parameters to optimize over
# SVM
p_grid_svm        = [{'kernel': ['rbf'], 'gamma': [1e-3, 1e-4],
                   'C': [1, 10, 100, 1000]},
                    {'kernel': ['linear'], 'C': [1, 10, 100, 1000]},
                    {'kernel':['poly'],'degree':[1,2,3,4,5,6,7,8],
                    'C': [1, 10, 100, 1000]}]
svm = SVC()


# Naive Bayes
p_grid_naiveB = [{'alpha': list(np.arange(0.01,100, 1)),
                    'fit_prior': ['True', 'False']}]
nb = MultinomialNB()

# Decision Tree
max_depths        = np.linspace(1, 20, 20, endpoint=True)
criterion_options = ["entropy", "gini"]
p_grid_DT         = {'max_depth': max_depths,'criterion' :criterion_options}
dtc = DecisionTreeClassifier()

# Logistic Regression
p_grid_lr ={'C':[1,10,100,1000],
           'solver':['newton-cg', 'lbfgs','sag'],
           'multi_class':['multinomial']}
lr = LogisticRegression()

# K-NN
p_grid_knn        = {'n_neighbors': [1,3,5],
                    'metric' : ["euclidean", "cityblock","manhattan"], 
                    'weights' : ['uniform', 'distance'] }
knns = neighbors.KNeighborsClassifier()

# Arrays to store scores
non_nested_scores_svm = np.zeros(NUM_TRIALS)
nested_scores_svm = np.zeros(NUM_TRIALS)

non_nested_scores_nb = np.zeros(NUM_TRIALS)
nested_scores_nb = np.zeros(NUM_TRIALS)

non_nested_scores_dt = np.zeros(NUM_TRIALS)
nested_scores_dt = np.zeros(NUM_TRIALS)

non_nested_scores_lr = np.zeros(NUM_TRIALS)
nested_scores_lr = np.zeros(NUM_TRIALS)

non_nested_scores_knn = np.zeros(NUM_TRIALS)
nested_scores_knn = np.zeros(NUM_TRIALS)



# Loop for each trial
for i in range(NUM_TRIALS):
    print('\n Iteration number :',i)
    # Choose cross-validation techniques for the inner and outer loops,
    # independently of the dataset.
    # E.g "LabelKFold", "LeaveOneOut", "LeaveOneLabelOut", etc.
    inner_cv = KFold(n_splits=4, shuffle=True, random_state=i)
    outer_cv = KFold(n_splits=4, shuffle=True, random_state=i)
    
    ## SVM
    # Non_nested parameter search and scoring
    clf = GridSearchCV(estimator=SVC(), param_grid=p_grid_svm, cv=inner_cv)
    clf.fit(X, y)
    non_nested_scores_svm[i] = clf.best_score_
    
    # Nested CV with parameter optimization
    nested_score_svm = cross_val_score(clf, X=X, y=y, cv=outer_cv)
    nested_scores_svm[i] = nested_score_svm.mean()
    
    ## Naive Bayes
    # Non_nested parameter search and scoring
    clf = GridSearchCV(estimator=nb, param_grid = p_grid_naiveB, cv=inner_cv)
    clf.fit(X, y)
    non_nested_scores_nb[i] = clf.best_score_

    # Nested CV with parameter optimization
    nested_score_nb = cross_val_score(clf, X=X, y=y, cv=outer_cv)
    nested_scores_nb[i] = nested_score_nb.mean()

    ## Decision Tree
    # Non_nested parameter search and scoring
    clf = GridSearchCV(estimator=dtc, param_grid = p_grid_DT, cv=inner_cv)
    clf.fit(X, y)
    non_nested_scores_dt[i] = clf.best_score_

    # Nested CV with parameter optimization
    nested_score_dt = cross_val_score(clf, X=X, y=y, cv=outer_cv)
    nested_scores_dt[i] = nested_score_dt.mean()

    ## Logistic Regression
    # Non_nested parameter search and scoring
    clf = GridSearchCV(estimator=lr, param_grid = p_grid_lr , cv=inner_cv)
    clf.fit(X, y)
    non_nested_scores_lr[i] = clf.best_score_

    # Nested CV with parameter optimization
    nested_score_lr = cross_val_score(clf, X=X, y=y, cv=outer_cv)
    nested_scores_lr[i] = nested_score_lr.mean()

    ## KNN
    # Non_nested parameter search and scoring
    clf = GridSearchCV(estimator=knns, param_grid = p_grid_knn , cv=inner_cv)
    clf.fit(X, y)
    non_nested_scores_knn[i] = clf.best_score_

    # Nested CV with parameter optimization
    nested_score_knn = cross_val_score(clf, X=X, y=y, cv=outer_cv)
    nested_scores_knn[i] = nested_score_knn.mean()


score_difference_svm = non_nested_scores_svm - nested_scores_svm
score_difference_nb = non_nested_scores_nb - nested_scores_nb
score_difference_dt = non_nested_scores_dt - nested_scores_dt
score_difference_lr = non_nested_scores_lr - nested_scores_lr
score_difference_knn = non_nested_scores_knn - nested_scores_knn

print("SVM : \n \
      Average Accuracy for SVM [Nested] : {0:.4f} \n \
      Average Accuracy for SVM [Non-Nested] : {1:.4f} \n \
      Average difference of {2:.4f} with std. dev. of {3:.4f}."
      .format(nested_scores_svm.mean(), 
              non_nested_scores_svm.mean(), 
              score_difference_svm.mean(), 
              score_difference_svm.std()))

print("Naive Bayes : \n \
      Average Accuracy for Naive Bayes [Nested] : {0:.4f} \n \
      Average Accuracy for Naive Bayes [Non-Nested] : {1:.4f} \n \
      Average difference of {2:.4f} with std. dev. of {3:.4f}."
      .format(nested_scores_nb.mean(), 
              non_nested_scores_nb.mean(), 
              score_difference_nb.mean(), 
              score_difference_nb.std()))

print("Decision Tree : \n \
      Average Accuracy for Decision Tree [Nested] : {0:.4f} \n \
      Average Accuracy for Decision Tree [Non-Nested] : {1:.4f} \n \
      Average difference of {2:.4f} with std. dev. of {3:.4f}."
      .format(nested_scores_dt.mean(), 
              non_nested_scores_dt.mean(), 
              score_difference_dt.mean(), 
              score_difference_dt.std()))

print("Logistic Regression : \n \
      Average Accuracy for Logistic Regression [Nested] : {0:.4f} \n \
      Average Accuracy for Logistic Regression [Non-Nested] : {1:.4f} \n \
      Average difference of {2:.4f} with std. dev. of {3:.4f}."
      .format(nested_scores_lr.mean(), 
              non_nested_scores_lr.mean(), 
              score_difference_lr.mean(), 
              score_difference_lr.std()))

print("K-NN: \n \
      Average Accuracy for K-NN [Nested] : {0:.4f} \n \
      Average Accuracy for K-NN [Non-Nested] : {1:.4f} \n \
      Average difference of {2:.4f} with std. dev. of {3:.4f}."
      .format(nested_scores_knn.mean(), 
              non_nested_scores_knn.mean(), 
              score_difference_knn.mean(), 
              score_difference_knn.std()))
