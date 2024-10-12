import numpy as np
from mbw.ml.tree_model.base import DecisionTree, Node, Leaf


class ClassifierLeaf(Leaf):

    def _get_prediction(self, labels):
        values, counts = np.unique(labels, return_counts=True)
        self.prediction = values[counts.argmax()]


def gini(labels):
    _, counts = np.unique(labels, return_counts=True)
    p = counts / sum(counts)
    return 1 - sum(p**2)


def entropy(labels):
    _, counts = np.unique(labels, return_counts=True)
    p = counts / sum(counts)
    return -sum(p * np.log2(p))


class DecisionTreeClassifier(DecisionTree):

    def __init__(self, criterion='gini', **kwargs):
        super().__init__(**kwargs)

        criterion = criterion.lower()
        criterion_list = ['gini', 'entropy']
        if criterion == 'gini':
            self._selected_criterion = gini

        elif criterion == 'entropy':
            self._selected_criterion = entropy

        else:
            error = f'Unknown criterion {criterion} for DecisionTreeClassifier'
            error += f' (criterion not in [{", ".join(criterion_list)}])!'
            raise ValueError(error)

    def _criterion(self, labels):
        return self._selected_criterion(labels)

    def _Leaf(self, labels):
        return ClassifierLeaf(labels)

    def _Node(self, index, t, true_branch, false_branch):
        return Node(index, t, true_branch, false_branch)
