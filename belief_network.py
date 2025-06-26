import pandas as pd
from pgmpy.estimators import BayesianEstimator
from pgmpy.models import DiscreteBayesianNetwork
from pgmpy.inference import Inference

from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer

from supervised_ML_models import SEED, TEST_SIZE


# Metodo che costruisce e addestra una belief network, restituendola
def train_belief_network(matrix, dataframe):
    x_train, x_test, y_train, y_test = train_test_split(matrix, dataframe['true_label_enc'], test_size=TEST_SIZE,
                                                        random_state=SEED)
    vectorizer = CountVectorizer(binary=True)
    x_train_vec = vectorizer.fit_transform(x_train)

    features = vectorizer.get_feature_names_out()

    x_train_df = pd.DataFrame(x_train_vec.toarray(), columns=features)
    x_train_df['label'] = y_train.values

    model = DiscreteBayesianNetwork([('label', feat) for feat in features])
    model.fit(x_train_df, estimator=BayesianEstimator, pseudo_counts=1)

    return model
