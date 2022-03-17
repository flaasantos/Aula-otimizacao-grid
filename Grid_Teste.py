import numpy as np
from sklearn import datasets
from sklearn import svm
from sklearn.model_selection import cross_val_score, GridSearchCV, RandomizedSearchCV

iris = datasets.load_iris()
features = iris.data
target = iris.target

C = np.linspace(0.01, 20, num=50)
kernel = ['rbf', 'poly', 'sigmoid']
degree = [1, 2, 3] #GRAU DO POLINÔMIO DO KERNEL POLY
hyperparameters = dict(C=C, kernel=kernel, degree=degree)
clf = svm.SVC()
search = GridSearchCV(clf, hyperparameters)
best_model = search.fit(features, target)
print(best_model.best_estimator_)

#GridSearchCV implements a “fit” and a “score” method. It also implements “score_samples”, “predict”,
# “predict_proba”, “decision_function”, “transform” and “inverse_transform” if they are implemented in the estimator used.