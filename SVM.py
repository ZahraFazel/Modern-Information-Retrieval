from sklearn import svm
from Common import tokenize, tf_idf, evaluation
from copy import deepcopy
import scipy.sparse


vocabulary = {}
SVM = None
idf = {}


def train(docs, c):
    output = tf_idf(tokenize(docs))
    global vocabulary
    vocabulary = output[0]
    global idf
    idf = output[1]
    train_x = scipy.sparse.csr_matrix(output[2])
    train_y = [docs[i]['category'] for i in range(len(docs))]
    global SVM
    SVM = svm.SVC(C=c, kernel='linear', degree=3, gamma='auto', max_iter=12000)
    SVM.fit(train_x, train_y)


def classify(doc):
    tokens = tokenize([doc])[0]['tokens']
    test_x = [0 for _ in range(len(vocabulary))]
    for term in tokens:
        if term in vocabulary:
            test_x[vocabulary[term]] += 1
    test_x = scipy.sparse.csr_matrix([test_x[vocabulary[term]] * idf[term] for term in vocabulary.keys()])
    return SVM.predict(test_x)[0]


def test(docs, c):
    new_data = deepcopy(docs)
    results = []
    for doc in new_data:
        category = doc.pop('category')
        predicted_category = classify(doc)
        results.append((category, predicted_category))
    print('SVM:')
    print('\tC: ' + str(c))
    evaluation(results)


def find_best_hyper_parameter(train_docs, test_docs, parameters):
    print("Finding best C for SVM:")
    train(train_docs, parameters[0])
    test(test_docs, parameters[0])
    train(train_docs, parameters[1])
    test(test_docs, parameters[1])
    train(train_docs, parameters[2])
    test(test_docs, parameters[2])


def run(train_docs, test_docs, c):
    train(train_docs, c)
    test(test_docs, c)
