import pandas as pd
from newspaper import Article
from tqdm import tqdm
import spacy
from sklearn.feature_extraction.text import TfidfVectorizer


def download_dataset(directory):
    df = pd.read_csv('resources/articles_link.csv')
    articles = []
    mapping = {'center': 0, 'left': 1, 'right': 2}
    for idx, row in tqdm(df.iterrows(), total=len(df), desc='Download Articles'):
        record = []
        error = False
        for bias in ['Left', 'Right', 'Center']:
            url = row[bias]
            try:
                article = Article(url)
                article.download()
                article.parse()

                if article.text and len(article.text) > 50:
                    record.append({
                        'bias': bias.lower(),
                        'url': url,
                        'text': article.text,
                        'title': article.title
                    })
            except:
                error = True
                print(f'Error downloading {url}')
                break
        if not error:
            for _ in record:
                articles.append()
    df = pd.DataFrame(articles)
    df['combined'] = df['title'].str.cat(df['text'], sep=' ', na_rep='')
    df['true_label_enc'] = df['bias'].map(mapping)
    df.to_csv('resources/articles.csv')


# Funzione di preprocessing per ottenere tokens
def text_preprocessing(text):
    nlp = spacy.load('en_core_web_sm')
    doc = nlp(text.lower())
    # estraiamo i token eliminando punteggiatura e spazi bianchi
    tokens = [
        token.lemma_ for token in doc
        if not token.is_stop and not token.is_punct and not token.is_space
    ]
    return ' '.join(tokens)


# Funzione che restituisce una matrice sparsa TF_IDF ed il vectorizer allenato
def get_TF_IDF_matrix(dataframe):
    # Istanziamo un vectorizer, in grado di restituire una matrice di token TF-IDF, che usi la log(TF)
    vectorizer = TfidfVectorizer(
        preprocessor=text_preprocessing(),
        ngram_range=(1, 2),
        min_df=5,
        max_df=0.9,
        stop_words='english',
        sublinear_tf=True
    )

    matrix = vectorizer.fit_transform(dataframe['combined'])
    return matrix, vectorizer
