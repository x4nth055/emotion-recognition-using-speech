from sklearn.ensemble import AdaBoostClassifier, RandomForestClassifier, GradientBoostingClassifier, BaggingClassifier
from sklearn.ensemble import AdaBoostRegressor, RandomForestRegressor, GradientBoostingRegressor, BaggingRegressor
from sklearn.svm import SVC, SVR
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor
from sklearn.neighbors import KNeighborsClassifier, KNeighborsRegressor
from sklearn.neural_network import MLPClassifier, MLPRegressor

classification_grid_parameters = {
    SVC():  {
        'C': [0.0005, 0.001, 0.002, 0.01, 0.1, 1, 10],
        'gamma' : [0.001, 0.01, 0.1, 1],
        'kernel': ['rbf', 'poly', 'sigmoid']
    },
    RandomForestClassifier():   {
        'n_estimators': [10, 40, 70, 100],
        'max_depth': [3, 5, 7],
        'min_samples_split': [0.2, 0.5, 0.7, 2],
        'min_samples_leaf': [0.2, 0.5, 1, 2],
        'max_features': [0.2, 0.5, 1, 2],
    },
    GradientBoostingClassifier():   {
        'learning_rate': [0.05, 0.1, 0.3],
        'n_estimators': [40, 70, 100],
        'subsample': [0.3, 0.5, 0.7, 1],
        'min_samples_split': [0.2, 0.5, 0.7, 2],
        'min_samples_leaf': [0.2, 0.5, 1],
        'max_depth': [3, 7],
        'max_features': [1, 2, None],
    },
    KNeighborsClassifier(): {
        'weights': ['uniform', 'distance'],
        'p': [1, 2, 3, 4, 5],
    },
    MLPClassifier():    {
        'hidden_layer_sizes': [(200,), (300,), (400,), (128, 128), (256, 256)],
        'alpha': [0.001, 0.005, 0.01],
        'batch_size': [128, 256, 512, 1024],
        'learning_rate': ['constant', 'adaptive'],
        'max_iter': [200, 300, 400, 500]
    },
    BaggingClassifier():    {
        'n_estimators': [10, 30, 50, 60],
        'max_samples': [0.1, 0.3, 0.5, 0.8, 1.],
        'max_features': [0.2, 0.5, 1, 2],
    }
}

regression_grid_parameters = {
    # SVR():  {
    #     'C': [0.0005, 0.001, 0.002, 0.01, 0.1, 1, 10],
    #     'gamma' : [0.001, 0.01, 0.1, 1],
    #     'kernel': ['rbf', 'poly', 'sigmoid']
    # },
    RandomForestRegressor():   {
        'n_estimators': [10, 40, 70, 100],
        'max_depth': [3, 5, 7],
        'min_samples_split': [0.2, 0.5, 0.7, 2],
        'min_samples_leaf': [0.2, 0.5, 1, 2],
        'max_features': [0.2, 0.5, 1, 2],
    },
    GradientBoostingRegressor():   {
        'learning_rate': [0.05, 0.1, 0.3],
        'n_estimators': [40, 70, 100],
        'subsample': [0.3, 0.5, 0.7, 1],
        'min_samples_split': [0.2, 0.5, 0.7, 2],
        'min_samples_leaf': [0.2, 0.5, 1],
        'max_depth': [3, 7],
        'max_features': [1, 2, None],
    },
    KNeighborsRegressor(): {
        'weights': ['uniform', 'distance'],
        'p': [1, 2, 3, 4, 5],
    },
    MLPRegressor():    {
        'hidden_layer_sizes': [(200,), (200, 200), (300,), (400,)],
        'alpha': [0.001, 0.005, 0.01],
        'batch_size': [64, 128, 256, 512, 1024],
        'learning_rate': ['constant', 'adaptive'],
        'max_iter': [300, 400, 500, 600, 700]
    },
    BaggingRegressor():    {
        'n_estimators': [10, 30, 50, 60],
        'max_samples': [0.1, 0.3, 0.5, 0.8, 1.],
        'max_features': [0.2, 0.5, 1, 2],
    }
}