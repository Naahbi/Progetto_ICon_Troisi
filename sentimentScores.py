import pandas as pd
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
from scipy.sparse import hstack


# Metodo per ricevere sentiment score di un testo
def get_sentiment_score(analyzer, text):
    scores = analyzer.polarity_scores(text)
    return scores['compound']


# Metodo che restituisce una sparce matrix TF-IDF contenente i sentiment score del testo.
def get_sentiment_matrix(dataframe, matrix):
    analyzer = SentimentIntensityAnalyzer()
    sentiment_score = dataframe['text'].apply(lambda x: get_sentiment_score((x, analyzer)))
    sentiment_df = pd.DataFrame({'sentiment_score': sentiment_score})

    sentiment_matrix = sentiment_df.values
    final_matrix = hstack([matrix, sentiment_matrix])
    return final_matrix
