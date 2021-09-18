import pandas as pd
from sklearn import svm
import scipy.sparse
from math import log2
from sklearn.metrics import accuracy_score

train_path = 'D:\\University\\Modern Information Retrieval\\Project\\Phase 3\\data\\train.txt'
validation_path = 'D:\\University\\Modern Information Retrieval\\Project\\Phase 3\\data\\vali.txt'
test_path = 'D:\\University\\Modern Information Retrieval\\Project\\Phase 3\\data\\test.txt'


def extract_data(data_path):
    data = pd.read_csv(data_path, sep=" qid:| [0-9]+:| #docid = | inc = | prob = ", header=None, engine='python')
    x = []
    y = []
    query = [(0, 0)]
    i = 0
    while i < len(data) - 1:
        j = i + 1
        while j < len(data) and data.loc[i, 1] == data.loc[j, 1]:
            j += 1
        for k in range(i, j):
            for p in range(k + 1, j):
                x.append(list(data.loc[k, 2:47].array - data.loc[p, 2:47].array))
                if data.loc[k, 0] == data.loc[p, 0]:
                    y.append(0)
                elif data.loc[k, 0] > data.loc[p, 0]:
                    y.append(1)
                else:
                    y.append(-1)
        query.append((query[len(query) - 1][1], int(query[len(query) - 1][1] + (j - i) * (j - i - 1) / 2)))
        i = j
    query.pop(0)
    x = scipy.sparse.csr_matrix(x)
    return x, y, query


train_x, train_y, train_queries = extract_data(train_path)
validation_x, validation_y, validation_queries = extract_data(validation_path)
test_x, test_y, test_queries = extract_data(test_path)


def DCG(actual_labels, svm_result):
    dcg = 1 if svm_result[0] == actual_labels[0] else 0
    for i in range(1, 5):
        if i < len(actual_labels):
            dcg += (1 / log2(i + 1)) if svm_result[i] == actual_labels[i] else 0
    return dcg


def NDCG(actual_labels, svm_result, queries):
    # evaluation_result = 0
    # for query in queries:
    #     evaluation_result += DCG(actual_labels[query[0]:query[1]], svm_result[query[0]:query[1]]) / DCG(actual_labels[query[0]:query[1]], actual_labels[query[0]:query[1]])
    # evaluation_result /= len(queries)
    # return evaluation_result
    return accuracy_score(actual_labels, svm_result)


def find_best_regularization_parameter():
    ndcg_score = {}
    SVMs = {}
    for c in [0.0001, 0.001, 0.01, 0.1, 1]:
        SVM = svm.LinearSVC(C=c, max_iter=4000)
        SVM.fit(train_x, train_y)
        predicted_y = SVM.predict(validation_x)
        SVMs[c] = SVM
        ndcg_score[c] = NDCG(validation_y, predicted_y, validation_queries)
    return ndcg_score, SVM


ndcg_score, SVMs = find_best_regularization_parameter()
for c in [0.0001, 0.001, 0.01, 0.1, 1]:
    print(str(c) + '\t' + str(ndcg_score[c]))
SVM = svm.LinearSVC(C=10, max_iter=10000)
SVM.fit(train_x, train_y)
predicted_y = SVM.predict(validation_x)
ndcg_score = NDCG(validation_y, predicted_y, validation_queries)
print(ndcg_score)
SVM = svm.LinearSVC(C=100, max_iter=10000)
SVM.fit(train_x, train_y)
predicted_y = SVM.predict(validation_x)
ndcg_score = NDCG(validation_y, predicted_y, validation_queries)
print(ndcg_score)
