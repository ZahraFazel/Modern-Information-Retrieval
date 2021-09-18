from sklearn.ensemble import RandomForestClassifier
from Common import tokenize, tf_idf, evaluation
from copy import deepcopy
import scipy.sparse


vocabulary = {}
random_forest_classifier = None
idf = {}


def train(docs, number_of_trees, max_depth):
    output = tf_idf(tokenize(docs))
    global vocabulary
    vocabulary = output[0]
    global idf
    idf = output[1]
    train_x = scipy.sparse.csc_matrix(output[2])
    train_y = [docs[i]['category'] for i in range(len(docs))]
    global random_forest_classifier
    random_forest_classifier = RandomForestClassifier(n_estimators=number_of_trees, max_depth=max_depth)
    random_forest_classifier.fit(train_x, train_y)


def classify(doc):
    tokens = tokenize([doc])[0]['tokens']
    test_x = [0 for _ in range(len(vocabulary))]
    for term in tokens:
        if term in vocabulary:
            test_x[vocabulary[term]] += 1
    test_x = scipy.sparse.csc_matrix([test_x[vocabulary[term]] * idf[term] for term in vocabulary.keys()])
    return random_forest_classifier.predict(test_x)[0]


def test(docs, number_of_trees, max_depth):
    new_data = deepcopy(docs)
    results = []
    for doc in new_data:
        category = doc.pop('category')
        predicted_category = classify(doc)
        results.append((category, predicted_category))
    print('Random Forest:')
    print('\tNumber of trees: ' + str(number_of_trees))
    print('\tMaximum depth: ' + str(max_depth))
    evaluation(results)


def find_best_hyper_parameter(train_docs, test_docs, parameters):
    print("Finding best number of trees & depth for Random Forest:")
    train(train_docs, parameters['n_trees'][0], parameters['max_depth'][0])
    test(test_docs, parameters['n_trees'][0], parameters['max_depth'][0])
    train(train_docs, parameters['n_trees'][1], parameters['max_depth'][1])
    test(test_docs, parameters['n_trees'][1], parameters['max_depth'][1])
    train(train_docs, parameters['n_trees'][2], parameters['max_depth'][2])
    test(test_docs, parameters['n_trees'][2], parameters['max_depth'][2])


def run(train_docs, test_docs, number_of_trees, max_depth):
    train(train_docs, number_of_trees, max_depth)
    test(test_docs, number_of_trees, max_depth)
