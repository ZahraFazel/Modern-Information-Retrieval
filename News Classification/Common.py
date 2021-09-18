from nltk import WordNetLemmatizer, PorterStemmer
from nltk.corpus import stopwords
from string import punctuation
from math import log10


def tokenize(docs):
    tokenized_docs = []
    for doc in docs:
        tokenized_doc = {'category': doc['category'] if 'category' in doc.keys() else 0}
        all_tokens = []
        title_tokens = doc['title'].lower().translate(str.maketrans(' ', ' ', punctuation)).split(' ')
        body_tokens = doc['body'].lower().translate(str.maketrans(' ', ' ', punctuation)).split(' ')
        for token in title_tokens:
            if token != '':
                all_tokens.append(token)
        for token in body_tokens:
            if token != '':
                all_tokens.append(token)
        tokenized_doc['tokens'] = all_tokens
        tokenized_docs.append(tokenized_doc)
    return tokenized_docs


def stemming(docs):
    stemmer = PorterStemmer()
    stemmed_docs = []
    for doc in docs:
        stemmed_doc = {'category': doc['category']}
        all_tokens = []
        for token in doc['tokens']:
            all_tokens.append(stemmer.stem(token))
        stemmed_doc['tokens'] = all_tokens
        stemmed_docs.append(stemmed_doc)
    return stemmed_docs


def lemmatization(docs):
    lemmatizer = WordNetLemmatizer()
    lemmatized_docs = []
    for doc in docs:
        lemmatized_doc = {'category': doc['category']}
        all_tokens = []
        for token in doc['tokens']:
            all_tokens.append(lemmatizer.lemmatize(token))
        lemmatized_doc['tokens'] = all_tokens
        lemmatized_docs.append(lemmatized_doc)
    return lemmatized_docs


def stop_words_removal(docs):
    stop_words = set(stopwords.words('english'))
    new_docs = []
    for doc in docs:
        new_doc = {'category': doc['category']}
        all_tokens = []
        for token in doc['tokens']:
            if token not in stop_words:
                all_tokens.append(token)
        new_doc['tokens'] = all_tokens
        new_docs.append(new_doc)
    return new_docs


def tf_idf(docs):
    vocabulary = {}
    tf = [{} for _ in range(len(docs))]
    idf = {}
    number_of_terms = 0
    for i in range(len(docs)):
        for term in docs[i]['tokens']:
            if term not in tf[i].keys():
                tf[i][term] = 0
            tf[i][term] += 1
            if term not in vocabulary.keys():
                vocabulary[term] = number_of_terms
                number_of_terms += 1
    for term in vocabulary.keys():
        df = 0
        for i in range(len(docs)):
            if term in tf[i].keys():
                df += 1
        idf[term] = log10(len(docs) / df) if df != 0 else 0
    docs_tf_idf_matrix = [[0 for __ in range(len(vocabulary.keys()))] for _ in range(len(docs))]
    for i in range(len(docs)):
        for term in tf[i].keys():
            docs_tf_idf_matrix[i][vocabulary[term]] = tf[i][term] * idf[term]
    return vocabulary, idf, docs_tf_idf_matrix


def evaluation(results):
    precision = [0, 0, 0, 0, 0]
    recall = [0, 0, 0, 0, 0]
    fi = [0, 0, 0, 0, 0]
    confusion_matrix = [[0, 0, 0, 0, 0], [0, 0, 0, 0, 0], [0, 0, 0, 0, 0], [0, 0, 0, 0, 0], [0, 0, 0, 0, 0]]
    for result in results:
        confusion_matrix[result[0]][result[1]] += 1
    for i in range(1, 5):
        tp, fp, fn = 0, 0, 0
        for result in results:
            if result[0] == i and result[1] != i:
                fn += 1
            if result[0] == i and result[1] == i:
                tp += 1
            if result[0] != i and result[1] == i:
                fp += 1
        precision[i] = tp / (tp + fp) if tp + fp != 0 else 0
        recall[i] = tp / (tp + fn) if tp + fn != 0 else 0
        fi[i] = 2 * precision[i] * recall[i] / (precision[i] + recall[i]) if precision[i] + recall[i] != 0 else 0
    for i in range(1, 5):
        print('\tCategory {}: \n\t\tPrecision: {:.5f}\n\t\tRecall: {:.5f}'.format(i, precision[i], recall[i]))
    print('\tAccuracy: {:.5f}'.format((confusion_matrix[1][1] + confusion_matrix[2][2] + confusion_matrix[3][3] + confusion_matrix[4][4]) / len(results)))
    print('\tConfusion Matrix: ', end='')
    for i in range(1, 5):
        print('\n\t\t', end='')
        for j in range(1, 5):
            print('{: =3}  '.format(confusion_matrix[i][j]), end="")
    print('\n\tMacro Averaged FI: {:.5f}'.format(sum(fi) / 4))


def apply_text_operation(docs, text_operation):
    if text_operation == 'stemming':
        new_docs = stemming(tokenize(docs))
    elif text_operation == 'lemmatization':
        new_docs = lemmatization(tokenize(docs))
    elif text_operation == 'stop words removal':
        new_docs = stop_words_removal(tokenize(docs))
    else:
        new_docs = tokenize(docs)
    return new_docs
