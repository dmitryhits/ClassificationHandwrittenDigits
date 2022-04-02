import pandas as pd
import tensorflow as tf
import numpy as np
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import Normalizer


def fit_predict_eval(model, features_train, features_test, target_train, target_test):
    model.fit(features_train, target_train)
    y_pred = model.predict(features_test)
    score = accuracy_score(target_test, y_pred)
    print(f'Model: {model}\nAccuracy: {score}\n')


if __name__ == "__main__":
    (x_train, y_train), _ = tf.keras.datasets.mnist.load_data(path="mnist.npz")
    x_train = x_train.reshape((x_train.shape[0], -1))
    features, targets = x_train[:6000], y_train[:6000]
    x_train, x_test, y_train, y_test = train_test_split(features, targets, test_size=0.3, random_state=40)
    # print(f'Classes: {np.unique(y_train)}')
    # print(f"Features' shape: {x_train.shape}")
    # print(f"Target's shape: {y_train.shape}")
    # print(f"min: {x_train.min()}, max: {x_train.max()}")
    train_set = pd.Series(y_train.flatten())
    # print(f"x_train shape: {x_train.shape}")
    # print(f"x_test shape: {x_test.shape}")
    # print(f"y_train shape: {y_train.shape}")
    # print(f"y_test shape: {y_test.shape}")
    # print("Proportion of samples per class in train set:")
    # print(train_set.value_counts(normalize=True))
    normalizer = Normalizer()
    x_train_norm = normalizer.transform(x_train)
    x_test_norm = normalizer.transform(x_test)
    neighbours = KNeighborsClassifier()
    neighbours_params = dict(n_neighbors=[3, 4], weights=['uniform', 'distance'], algorithm=['auto', 'brute'])
    neighbours_search = GridSearchCV(estimator=neighbours, param_grid=neighbours_params, scoring='accuracy', n_jobs=-1)
    neighbours_search.fit(x_train_norm, y_train)
    y_pred_neighbors = neighbours_search.best_estimator_.predict(x_test_norm)
    accuracy_neighbors = accuracy_score(y_test, y_pred_neighbors)

    forest = RandomForestClassifier(random_state=40)
    forest_params = dict(n_estimators=[300, 500], max_features=['auto', 'log2'],
                         class_weight=['balanced', 'balanced_subsample'])
    forest_search = GridSearchCV(estimator=forest, param_grid=forest_params, scoring='accuracy', n_jobs=-1)
    forest_search.fit(x_train_norm, y_train)
    y_pred_forest = forest_search.best_estimator_.predict(x_test_norm)
    accuracy_forest = accuracy_score(y_test, y_pred_forest)

    print("K-nearest neighbours algorithm")
    print(f"best estimator: {neighbours_search.best_estimator_}")
    print(f"accuracy: {accuracy_neighbors}")
    print()
    print("Random forest algorithm")
    print(f"best estimator: {forest_search.best_estimator_}")
    print(f"accuracy: {accuracy_forest}")


    # models = [KNeighborsClassifier(), DecisionTreeClassifier(random_state=40),
    #           LogisticRegression(random_state=40), RandomForestClassifier(random_state=40)]


    # for model in models:
        # fit_predict_eval(model, x_train, x_test, y_train, y_test)

        # fit_predict_eval(model, x_train_norm, x_test_norm, y_train, y_test)
    # print('The answer to the 1st question: yes')

    # print(f'The answer to the 2nd question: KNeighborsClassifier-0.953, RandomForestClassifier - 0.937')
