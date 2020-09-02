"""
A script to grid search all parameters provided in parameters.py
including both classifiers and regressors.
Note that the execution of this script may take hours to search the 
best possible model parameters for various algorithms, feel free
to edit parameters.py on your need ( e.g remove some parameters for 
faster search )
"""

import pickle

from emotion_recognition import EmotionRecognizer
from parameters import classification_grid_parameters, regression_grid_parameters

# emotion classes you want to perform grid search on
emotions = ['sad', 'neutral', 'happy']
# number of parallel jobs during the grid search
n_jobs = 4

best_estimators = []

for model, params in classification_grid_parameters.items():
    if model.__class__.__name__ == "KNeighborsClassifier":
        # in case of a K-Nearest neighbors algorithm
        # set number of neighbors to the length of emotions
        params['n_neighbors'] = [len(emotions)]
    d = EmotionRecognizer(model, emotions=emotions)
    d.load_data()
    best_estimator, best_params, cv_best_score = d.grid_search(params=params, n_jobs=n_jobs)
    best_estimators.append((best_estimator, best_params, cv_best_score))
    print(f"{emotions} {best_estimator.__class__.__name__} achieved {cv_best_score:.3f} cross validation accuracy score!")

print(f"[+] Pickling best classifiers for {emotions}...")
pickle.dump(best_estimators, open(f"grid/best_classifiers.pickle", "wb"))

best_estimators = []

for model, params in regression_grid_parameters.items():
    if model.__class__.__name__ == "KNeighborsRegressor":
        # in case of a K-Nearest neighbors algorithm
        # set number of neighbors to the length of emotions
        params['n_neighbors'] = [len(emotions)]
    d = EmotionRecognizer(model, emotions=emotions, classification=False)
    d.load_data()
    best_estimator, best_params, cv_best_score = d.grid_search(params=params, n_jobs=n_jobs)
    best_estimators.append((best_estimator, best_params, cv_best_score))
    print(f"{emotions} {best_estimator.__class__.__name__} achieved {cv_best_score:.3f} cross validation MAE score!")

print(f"[+] Pickling best regressors for {emotions}...")
pickle.dump(best_estimators, open(f"grid/best_regressors.pickle", "wb"))



# Best for SVC: C=0.001, gamma=0.001, kernel='poly'
# Best for AdaBoostClassifier: {'algorithm': 'SAMME', 'learning_rate': 0.8, 'n_estimators': 60}
# Best for RandomForestClassifier: {'max_depth': 7, 'max_features': 0.5, 'min_samples_leaf': 1, 'min_samples_split': 2, 'n_estimators': 40}
# Best for GradientBoostingClassifier: {'learning_rate': 0.3, 'max_depth': 7, 'max_features': None, 'min_samples_leaf': 1, 'min_samples_split': 2, 'n_estimators': 70, 'subsample': 0.7}
# Best for DecisionTreeClassifier: {'criterion': 'entropy', 'max_depth': 7, 'max_features': None, 'min_samples_leaf': 1, 'min_samples_split': 2}
# Best for KNeighborsClassifier: {'n_neighbors': 5, 'p': 1, 'weights': 'distance'}
# Best for MLPClassifier: {'alpha': 0.005, 'batch_size': 256, 'hidden_layer_sizes': (300,), 'learning_rate': 'adaptive', 'max_iter': 500}