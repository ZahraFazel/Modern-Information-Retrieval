from Common import apply_text_operation, evaluation
from math import log10
from copy import deepcopy


vocabulary = set()
probability_of_categories = {1: 0, 2: 0, 3: 0, 4: 0}
probability_of_terms = {}


def preprocess(docs, text_operation):
    new_docs = apply_text_operation(docs, text_operation)
    terms_of_category = {1: {}, 2: {}, 3: {}, 4: {}}
    for doc in new_docs:
        for word in doc['tokens']:
            if word not in terms_of_category[doc['category']]:
                terms_of_category[doc['category']][word] = 0
            terms_of_category[doc['category']][word] += 1
            vocabulary.add(word)
    return terms_of_category


def train(docs, alpha, text_operation):
    terms_of_category = preprocess(docs, text_operation)
    number_of_terms = {}
    for doc in docs:
        probability_of_categories[doc['category']] += 1
    for category in probability_of_categories.keys():
        probability_of_categories[category] /= len(docs)
    for term in vocabulary:
        number_of_terms[term] = {category: 0 for category in probability_of_categories.keys()}
        for category in probability_of_categories.keys():
            if term in terms_of_category[category].keys():
                number_of_terms[term][category] = terms_of_category[category][term]
    for term in vocabulary:
        probability_of_terms[term] = {category: 0.0 for category in probability_of_categories.keys()}
        for category in probability_of_categories.keys():
            probability_of_terms[term][category] = (number_of_terms[term][category] + alpha) / (len(terms_of_category[category]) + alpha * len(vocabulary))


def classify(doc, text_operation):
    tokens = apply_text_operation([doc], text_operation)[0]['tokens']
    scores = {}
    for category in probability_of_categories.keys():
        scores[category] = log10(probability_of_categories[category])
        for term in tokens:
            if term in vocabulary:
                scores[category] += log10(probability_of_terms[term][category])
    for category in scores.keys():
        if scores[category] == max(scores.values()):
            return category


def test(docs, alpha, text_operation):
    new_data = deepcopy(docs)
    results = []
    for doc in new_data:
        category = doc.pop('category')
        predicted_category = classify(doc, text_operation)
        results.append((category, predicted_category))
    print('Naive Bayes:')
    print('\tAlpha: ' + str(alpha))
    print('\tText Operation: ' + text_operation)
    evaluation(results)


def find_best_hyper_parameter(train_docs, test_docs, parameters, text_operation):
    print("Finding best alpha for Naive Bayes:")
    train(train_docs, parameters[0], text_operation)
    test(test_docs, parameters[0], text_operation)
    train(train_docs, parameters[1], text_operation)
    test(test_docs, parameters[1], text_operation)
    train(train_docs, parameters[2], text_operation)
    test(test_docs, parameters[2], text_operation)


def check_text_operations_effects(train_docs, test_docs, alpha):
    print("Checking effects of text operations on Naive Bayes:")
    train(train_docs, alpha, 'stemming')
    test(test_docs, alpha, 'stemming')
    train(train_docs, alpha, 'lemmatization')
    test(test_docs, alpha, 'lemmatization')
    train(train_docs, alpha, 'stop words removal')
    test(test_docs, alpha, 'stop words removal')


def run(train_docs, test_docs, alpha, text_operation):
    train(train_docs, alpha, text_operation)
    test(test_docs, alpha, text_operation)
