import json
import datetime

import NaiveBayes
import KNN
import SVM
import RandomForest
import KMeans


training_data_path = 'train.json'
test_data_path = 'validation.json'

with open(training_data_path) as training_data_file:
    training_data = json.loads(training_data_file.read())

with open(test_data_path) as test_data_file:
    test_data = json.loads(test_data_file.read())

# NaiveBayes.run(training_data, test_data, 1, 'none')
# NaiveBayes.find_best_hyper_parameter(training_data, test_data, [0.25, 1, 4], 'none')
# NaiveBayes.check_text_operations_effects(training_data, test_data, 1)

# KNN.run(training_data[:int(len(training_data) / 2)], test_data[:int(len(test_data) / 2)], 1, 'cosine similarity', 'none')
# KNN.run(training_data[:int(len(training_data) / 2)], test_data[:int(len(test_data) / 2)], 1, 'euclidean distance', 'none')
# KNN.find_best_hyper_parameter(training_data[:int(len(training_data) / 2)], test_data[:int(len(test_data) / 2)], [1, 3, 5], 'cosine similarity')
# KNN.find_best_hyper_parameter(training_data[:int(len(training_data) / 2)], test_data[:int(len(test_data) / 2)], [1, 3, 5], 'euclidean distance')
# KNN.check_text_operations_effects(training_data[:int(len(training_data) / 2)], test_data[:int(len(test_data) / 2)], 'cosine similarity')
# KNN.check_text_operations_effects(training_data[:int(len(training_data) / 2)], test_data[:int(len(test_data) / 2)], 'euclidean distance')

# SVM.run(training_data[:int(len(training_data) / 2)], test_data[:int(len(test_data) / 2)], 1)
# SVM.find_best_hyper_parameter(training_data[:int(len(training_data) / 2)], test_data[:int(len(test_data) / 2)], [0.001, 0.01, 0.1])

# RandomForest.run(training_data[:int(len(training_data) / 2)], test_data[:int(len(test_data) / 2)], 100, 10000)
# RandomForest.find_best_hyper_parameter(training_data[:int(len(training_data) / 2)], test_data[:int(len(test_data) / 2)],
#                                        {'n_trees': [25, 50, 100], 'max_depth': [5000, 10000, 20000]})

KMeans.run(training_data[:int(len(training_data) / 2)])
