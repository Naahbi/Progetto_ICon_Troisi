from typing import Dict, Any

from sklearn.model_selection import train_test_split
from sklearn.model_selection import GridSearchCV

from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from xgboost import XGBClassifier

SEED = 42
TEST_SIZE = 0.2


# Metodo che allena un modello di regressione, e restituisce il modello.
def train_logistic_regression(matrix, dataframe, max_iter=1000):
    model = LogisticRegression(max_iter=max_iter)

    x_train, x_test, y_train, y_test = train_test_split(matrix, dataframe['true_label_enc'], test_size=TEST_SIZE,
                                                        random_state=SEED)

    model.fit(x_train, y_train)

    return model


# Metodo che esegue una grid search per la ricerca dei migliori parametri per il modello di regressione lineare,
# restituendo il modello e i migliori parametri.
def train_best_logistic_regression(matrix, dataframe, max_iter=1000):
    x_train, x_test, y_train, y_test = train_test_split(matrix, dataframe['true_label_enc'], test_size=TEST_SIZE,
                                                        random_state=SEED)

    param_grid = {
        'C': [0.001, 0.01, 0.1, 1, 10, 100],
        'penalty': ['l1', 'l2'],
        'solver': ['liblinear', 'lbfgs', 'saga']
    }

    grid_search = GridSearchCV(
        LogisticRegression(max_iter=max_iter),
        param_grid,
        cv=5,  # 5-fold cross validation
        scoring='accuracy',
        n_jobs=-1
    )

    grid_search.fit(x_train, y_train)

    best_params = grid_search.best_params_
    best_model = grid_search.best_estimator_

    return best_model, best_params


# Metodo che allena un DecisionTreeClassifier, restituendo il modello
def train_decision_tree_classifier(matrix, dataframe, max_depth=10):
    model = DecisionTreeClassifier(
        criterio='entropy',
        max_depth=max_depth,
        random_state=SEED
    )
    x_train, x_test, y_train, y_test = train_test_split(matrix, dataframe['true_label_enc'], test_size=TEST_SIZE,
                                                        random_state=SEED)

    model.fit(x_train, y_train)

    return model


# Metodo che esegue una grid search per la ricerca dei migliori parametri per l'albero decisionale
def train_best_decision_tree_classifier(matrix, dataframe):
    x_train, x_test, y_train, y_test = train_test_split(matrix, dataframe['true_label_enc'], test_size=TEST_SIZE,
                                                        random_state=SEED)
    param_grid = {
        'criterion': ['gini', 'entropy'],
        'max_depth': [None, 10, 20, 30],
        'min_samples_split': [2, 5, 10],
        'min_samples_leaf': [1, 2, 4]
    }

    grid_search = GridSearchCV(
        DecisionTreeClassifier(random_state=SEED),
        param_grid,
        cv=5,
        scoring='accuracy',
        n_jobs=-1
    )

    grid_search.fit(x_train, y_train)

    best_model = grid_search.best_params_
    best_params = grid_search.best_params_

    return best_model, best_params


# Metodo che restituisce un modello di gradient boost allenanto
def train_xgb_model(matrix, dataframe, n_estimators=100, max_depth=6, learning_rate=0.01):
    x_train, x_test, y_train, y_test = train_test_split(matrix, dataframe['true_label_enc'], test_size=TEST_SIZE,
                                                        random_state=SEED)
    xgb_model = XGBClassifier(
        objective='multi:softmax',
        num_class=3,
        eval_metric='logloss',
        n_estimators=n_estimators,
        max_depth=max_depth,
        learning_rate=learning_rate,
        random_state=SEED
    )
    xgb_model.fit(x_train, y_train)

    return xgb_model


# Metodo che esegue una grid search per la ricerca dei migliori parametri per il gradient boosting
def train_best_xgb_model(matrix, dataframe):
    x_train, x_test, y_train, y_test = train_test_split(matrix, dataframe['true_label_enc'], test_size=TEST_SIZE,
                                                        random_state=SEED)
    param_grid = {
        'n_estimators': [100, 200, 300],
        'learning_rate': [0.01, 0.1, 0.2],
        'max_depth': [4, 6, 8],
    }

    grid_search = GridSearchCV(
        XGBClassifier(objective='multi:softmax', num_class=3, eval_metrics='mlogloss', random_state=SEED),
        param_grid,
        cv=3,
    )
    grid_search.fit(x_train, y_train)

    best_model = grid_search.best_estimator_
    best_params = grid_search.best_params_

    return best_model, best_params


