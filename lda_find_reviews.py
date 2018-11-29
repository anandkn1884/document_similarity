import json
from nltk.tokenize import RegexpTokenizer
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer
from gensim import corpora, models
import gensim

tokenizer = RegexpTokenizer(r'\w+')

# create English stop words list
# en_stop = stopwords('en')
en_stop = stopwords.words('english')

# Create p_stemmer of class PorterStemmer
p_stemmer = PorterStemmer()

def preprocess(doc_set):
    texts = []

    # loop through document list
    for i in doc_set:
        # clean and tokenize document string
        raw = i.lower()
        tokens = tokenizer.tokenize(raw)

        # remove stop words from tokens
        stopped_tokens = [i for i in tokens if not i in en_stop]

        # stem tokens
        stemmed_tokens = [p_stemmer.stem(i) for i in stopped_tokens]

        # add tokens to list
        texts.append(stemmed_tokens)

    return texts

if __name__ == "__main__":

    recomms = []
    document_set = []
    train_document_set = []

    with open('review_data.json') as f:
        data = json.load(f)

    all_recommendations = ''

    for review in data["reviews"]:
        recommendations = review["recommendations"]
        for recomm in recommendations:
            document_set.append(review["name"] + '\t' + review["updated"] + '\t' + recomm)

    search_review = "Store with a choice of freshly baked bread"

    train_document_set = []

    # compile sample documents into a list
    for document in document_set:
        train_document_set.append(document.split('\t')[2])

    train_text = preprocess(train_document_set)

    # turn our tokenized documents into a id <-> term dictionary
    train_dictionary = corpora.Dictionary(train_text)

    # convert tokenized documents into a document-term matrix
    corpus = [train_dictionary.doc2bow(text) for text in train_text]

    test_doc_set = [search_review]
    test_text = preprocess(test_doc_set)
    test_dictionary = corpora.Dictionary(test_text)
    test_corpus = [train_dictionary.doc2bow(text) for text in test_text]

    # generate LDA model
    ldamodel = gensim.models.ldamodel.LdaModel(corpus, num_topics=4, id2word=train_dictionary, passes=20)

    test_vector = ldamodel[test_corpus[0]]
    vector2 = ldamodel[corpus[0]]

    index = 0
    review_score = {}
    while index < len(corpus):
        similarity_score = gensim.matutils.cossim(test_vector, ldamodel[corpus[index]])
        review_score.update({similarity_score : index})
        index += 1

    review_score = sorted(review_score.items(), reverse=True)

    for key, value in review_score:
        print(document_set[value], '\t', key)