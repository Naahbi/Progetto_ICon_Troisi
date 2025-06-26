from datetime import datetime

from pyswip import Prolog
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
from xgboost import XGBClassifier

from supervised_ML_models import SEED, train_best_decision_tree_classifier, TEST_SIZE, train_best_logistic_regression

import numpy as np


# Metodo per ottenere i pesi, per classe, di un decision_tree_classifier
def get_weights_tree(matrix, dataframe, class_names, max_depth=10, min_samples_split=2, min_samples_leaf=2):
    x_train, x_test, y_train, y_test = train_test_split(matrix, dataframe['true_label_enc'], test_size=TEST_SIZE,
                                                        random_state=SEED)
    label_map = {name: i for i, name in enumerate(class_names)}

    tree_weights_per_class = {}
    for cls in class_names:
        class_idx = label_map[cls]
        y_bin = (y_train == class_idx).astype(int)
        model = DecisionTreeClassifier(
            criterion='gini',
            max_depth=max_depth,
            random_state=SEED,
            min_samples_leaf=min_samples_leaf,
            min_samples_split=min_samples_split
        )
        model.fit(x_train, y_bin)
        tree_weights_per_class[cls] = model.feature_importances_
    return tree_weights_per_class


# Metodo per ottenere i pesi, per classe, di un gradient booster
def get_weights_xgboost(matrix, dataframe, class_names, n_estimators, max_depth, learning_rate):
    x_train, x_test, y_train, y_test = train_test_split(matrix, dataframe['true_label_enc'], test_size=TEST_SIZE,
                                                        random_state=SEED)
    label_map = {name: i for i, name in enumerate(class_names)}
    xgb_weights_per_class = {}
    for cls in ['left', 'center', 'right']:
        class_idx = label_map[cls]
        y_bin = (y_train == class_idx).astype(int)

        xgb_model = XGBClassifier(
            objective='binary:logistic',
            eval_metric='logloss',
            n_estimators=n_estimators,
            max_depth=max_depth,
            learning_rate=learning_rate,
            random_state=SEED
        )
        xgb_model.fit(x_train, y_bin)

        xgb_weights_per_class[cls] = xgb_model.feature_importances_
    return xgb_weights_per_class


# Metodo che normalizza una matrice per il suo massimo
def max_normalization(matrix):
    return matrix / np.max(np.abs(matrix), axis=1, keepdims=True)


# Metodo che calcola e restituisce la somma pesata dei pesi di tre modelli, regression, tree, xgboost,
# per ogni feature della KB
def compute_final_weights_from_models(feature_names, matrix, dataframe, classes):
    tree_weights_per_class = get_weights_tree(matrix, dataframe, classes)
    tree_weights = np.array([
        tree_weights_per_class['center'],
        tree_weights_per_class['left'],
        tree_weights_per_class['right']
    ])
    xgb_weights_per_class = get_weights_xgboost(matrix, dataframe, classes)
    xgb_weights = np.array([
        xgb_weights_per_class['center'],
        xgb_weights_per_class['left'],
        xgb_weights_per_class['right']
    ])

    best_logReg_model = train_best_logistic_regression(matrix, dataframe)

    # normalizziamo i pesi su scala [0,1]
    regression_weights = max_normalization(best_logReg_model.coef_)
    tree_weights = max_normalization(tree_weights)
    xgb_weights = max_normalization(xgb_weights)

    final_weights = (0.2 * regression_weights + 0.3 * tree_weights + 0.5 * xgb_weights)
    return final_weights


# Metodo che costruisce la knowledge_base
def build_knowledge_base(feature_name, matrix, dataframe, classes):
    kb = {}
    final_weights = compute_final_weights_from_models(feature_name, matrix, dataframe, classes)
    n_classes, n_features = final_weights.shape

    for i in range(n_features):
        term = feature_name[i]
        kb[term] = {classes[c]: final_weights[c, i] for c in range(n_classes)}
    return kb


# Metodo che restituisce la classe di un testo calcolata dalla kb
def classify_text(vectorizer, text, knowledge_base, classes):
    feature_names = vectorizer.get_feature_names_out()
    x = vectorizer.transform([text])
    final_weights = np.array([knowledge_base.get(f, 0) for f in feature_names])
    scores = final_weights @ x.toarray().flatten()
    max_idx = scores.argmax()

    return classes[max_idx]


# Funzione che salva in file .pl i fact della knowledge base
def export_kb_to_prolog(file_path, feature_names, final_weights, classes):
    with open(file_path, 'w', encoding='utf-8') as f:
        for class_idx, class_name in enumerate(classes):
            for i, term in enumerate(feature_names):
                weight = final_weights[class_idx, i]
                line = f"term_weight('{term}','{class_name}',{weight}).\n"
                f.write(line)


# Metodo che genera una query di classificazione utilizzabile in linguaggio prolog
def generate_prolog_query(text, vectorizer, class_predicate='classify_text'):
    analyzer = vectorizer.build_analyzer()

    tokens = analyzer(text)

    valid_tokens = [t for t in tokens if t in vectorizer.vocabulary_]

    quoted_tokens = [f"'{token}'" for token in valid_tokens]
    token_list = "[" + ", ".join(quoted_tokens) + "]"

    query = f"{class_predicate}({token_list}, Class)."

    return query


# Metodo per classificazione di un testo attraverso KB usando prolog
def classify_text_prolog(rules_dir, facts_dir, text, vectorizer):
    prolog = Prolog()

    prolog.consult(rules_dir)
    prolog.consult(facts_dir)

    query = generate_prolog_query(text, vectorizer)
    results = list(prolog.query(query))

    if results:
        return results[0]['Class']
    else:
        return 'Unknown'
