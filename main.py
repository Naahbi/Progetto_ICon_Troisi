from sklearn.model_selection import train_test_split

from datasetManager import download_dataset, get_TF_IDF_matrix
from models_evaluation import print_model_classification_report
from supervised_ML_models import train_best_logistic_regression, train_best_decision_tree_classifier, train_xgb_model, \
    SEED, TEST_SIZE

if __name__ == '__main__':
    df = download_dataset(directory='resources/articles_link.csv')
    matrix, vectorizer = get_TF_IDF_matrix(dataframe=df)

    x_train, x_test, y_train, y_test = train_test_split(matrix, df['true_label_enc'], test_size=TEST_SIZE,
                                                        random_state=SEED)


    # regressione
    lr_model, lr_params = train_best_logistic_regression(matrix=matrix, dataframe=df)

    # tree
    tree_model, tree_param = train_best_decision_tree_classifier(matrix=matrix, dataframe=df)

    # xgboost
    xgb_model, xgb_params = train_xgb_model(matrix=matrix, dataframe=df)

    print_model_classification_report(lr_model, x_test, y_test)
    print_model_classification_report(tree_model, x_test, y_test)
    print_model_classification_report(xgb_model, x_test, y_test)