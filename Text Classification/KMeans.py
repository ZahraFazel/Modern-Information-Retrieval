import numpy
from Common import tokenize, tf_idf
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
import pandas as pd


def run(docs):
    new_docs = tokenize(docs)
    vocabulary, idf, train_matrix = tf_idf(new_docs)
    categories = [new_docs[i]['category'] for i in range(len(new_docs))]
    centroids = [numpy.random.random(len(vocabulary)) for _ in range(4)]
    clusters = [[] for _ in range(4)]
    train_matrix = numpy.asarray(train_matrix)
    temp = [row @ row for row in train_matrix]
    x2 = numpy.asarray([[temp[i] for _ in range(4)] for i in range(len(docs))])
    for i in range(100):
        for j in range(4):
            clusters[j].clear()
        xy = train_matrix @ numpy.transpose(centroids)
        temp = [row @ row for row in numpy.asarray(centroids)]
        y2 = numpy.asarray([temp for _ in range(len(docs))])
        distance = list(x2 - 2 * xy + y2)
        for j in range(len(new_docs)):
            clusters[list(distance[j]).index(min(list(distance[j])))].append(j)
        for j in range(4):
            sum_of_vectors = [0 for _ in range(len(vocabulary))]
            for k in clusters[j]:
                sum_of_vectors = numpy.add(sum_of_vectors, train_matrix[k])
            centroids[j] = numpy.asarray(sum_of_vectors / len(clusters[j]) if len(clusters[j]) > 0 else sum_of_vectors)
    print('Done')
    # tsne = TSNE(n_components=4, random_state=0)  # first try without init='pca'
    # selected_data = [clusters[i][:50] for i in range(4)]
    # X_tsne = tsne.fit_transform(selected_data)
    # xtsne = pd.DataFrame(X_tsne)
    # xtsne['label'] = [0, 1, 2, 3]
    # xtsne.info()
    # c_map = {0: 'b', 1: 'r', 2: 'g', 3: 'y'}
    # plt.figure(figsize=(10, 9))
    # plt.scatter(X_tsne[:, 0], X_tsne[:, 1], c=[c_map[_] for _ in xtsne['label']], alpha=0.5)
    # # plt.ylim(-10,10)
    # plt.show()
