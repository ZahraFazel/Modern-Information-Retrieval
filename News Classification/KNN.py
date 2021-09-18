from Common import apply_text_operation, evaluation, tf_idf
import numpy
from copy import deepcopy


vocabulary = {}
train_matrix = []
category_of_docs = []
idf = {}
x2 = numpy.asarray([])


def train(docs, text_operation):
    new_docs = apply_text_operation(docs, text_operation)
    for i in range(len(docs)):
        category_of_docs.append(docs[i]['category'])
    output = tf_idf(new_docs)
    global vocabulary
    vocabulary = output[0]
    global idf
    idf = output[1]
    global train_matrix
    train_matrix = numpy.asarray(output[2])
    global x2
    x2 = [row @ row for row in train_matrix]


def classify(doc, text_operation, distance_metric):
    tokens = apply_text_operation([doc], text_operation)[0]['tokens']
    result = []
    test_matrix = [0 for _ in range(len(vocabulary))]
    for term in tokens:
        if term in vocabulary:
            test_matrix[vocabulary[term]] += 1
    test_matrix = numpy.asarray([test_matrix[vocabulary[term]] * idf[term] for term in vocabulary.keys()])
    if distance_metric == 'euclidean distance':
        xy = train_matrix @ test_matrix
        y2 = test_matrix @ test_matrix
        scores = list(x2 - 2 * xy + y2)
        for i in range(5):
            result.append(category_of_docs[scores.index(min(scores))])
            scores[scores.index(min(scores))] = float('inf')
    else:   # distance_metric == 'cosine similarity'
        scores = list(numpy.asarray(train_matrix) @ numpy.asarray(test_matrix))
        for i in range(5):
            result.append(category_of_docs[scores.index(max(scores))])
            scores[scores.index(max(scores))] = float('-inf')
    return result


def test(docs, mode, distance_metric):
    new_data = deepcopy(docs)
    results = []
    for doc in new_data:
        category = doc.pop('category')
        first_k_categories = classify(doc, mode, distance_metric)
        results.append((category, first_k_categories))
    return results


def print_log(k, distance_metric, all_results, text_operation):
    result = []
    for i in range(len(all_results)):
        result.append((all_results[i][0], numpy.bincount(all_results[i][1][:k]).argmax()))
    print('K-Nearest Neighbours:')
    print('\tK: ' + str(k))
    print('\tText Operation: ' + text_operation)
    print('\tDistance Metric: ' + distance_metric)
    evaluation(result)


def find_best_hyper_parameter(train_docs, test_docs, parameters, distance_metric):
    print("Finding best K for K-Nearest Neighbours:")
    train(train_docs, 'none')
    all_results = test(test_docs, 'none', distance_metric)
    for parameter in parameters:
        print_log(parameter, distance_metric, all_results, 'none')


def check_text_operations_effects(train_docs, test_docs, distance_metric):
    print("Checking effects of text operations on K-Nearest Neighbours:")
    train(train_docs, 'stemming')
    all_results = test(test_docs, 'stemming', distance_metric)
    print_log(1, distance_metric, all_results, 'stemming')
    train(train_docs, 'lemmatization')
    all_results = test(test_docs, 'lemmatization', distance_metric)
    print_log(1, distance_metric, all_results, 'lemmatization')
    train(train_docs, 'stop words removal')
    all_results = test(test_docs, 'stop words removal', distance_metric)
    print_log(1, distance_metric, all_results, 'stop words removal')


def run(train_docs, test_docs, k, distance_metric, text_operation):
    train(train_docs, 'none')
    all_results = test(test_docs, text_operation, distance_metric)
    print_log(k, distance_metric, all_results, text_operation)
