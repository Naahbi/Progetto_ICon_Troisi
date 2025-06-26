from sklearn.metrics import classification_report, ConfusionMatrixDisplay, accuracy_score, \
    confusion_matrix, log_loss
import matplotlib.pyplot as plt

from knowledge_base import classify_text


# Funzione che stampa il classification report di un modello già allenato, passando x ed y per il testing
def print_model_classification_report(model, x_test, y_test):
    y_pred = model.predict(x_test)
    print(classification_report(y_pred, y_test))


# Funzione che stampa la confusion matrix di un modello
def print_model_confusion_matrix(model, x_test, y_test):
    y_pred = model.predict(x_test)
    ConfusionMatrixDisplay.from_predictions(y_test, y_pred, xticks_rotation='vertical')
    plt.title("Confusion Matrix")
    plt.grid(False)
    plt.show()


# Funzione che stampa
def evaluate_knowledge_base(vectorizer, kb, x_test, y_test, label_name):
    y_true = y_test.tolist()
    y_pred = []

    y_true = [label_name[label] for label in y_true]
    for text in x_test:
        pred_class = classify_text(vectorizer, text, kb, label_name)
        y_pred.append(pred_class)

    print(y_true)
    print("Accuracy:", accuracy_score(y_true, y_pred))
    print(classification_report(y_true, y_pred, labels=label_name))
    print("Confusion matrix:\n", confusion_matrix(y_true, y_pred, labels=label_name))


# Funzione per valutare una rete bayesiana attraverso classification report e logloss
def evaluate_belief_network(b_net, x_test_df, y_test_df, classes):
    y_pred = b_net.predict(x_test_df)
    pred_classes = y_pred['bias_class'].to_list()
    print(classification_report(y_test_df, pred_classes))

    y_pred_prob = b_net.predict_probability(x_test_df)
    y_pred_prob = y_pred_prob[classes]
    logloss = log_loss(y_test_df, y_pred_prob)

    print(f'Log-loss : {logloss:.4f}')
