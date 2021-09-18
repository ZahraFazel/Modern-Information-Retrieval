from typing import List, Dict
import NaiveBayes


def train(training_docs: List[Dict]):
    NaiveBayes.train(training_docs, 1, 'stop words removal')


def classify(doc: Dict) -> int:
    return NaiveBayes.classify(doc, 'stop words removal')


