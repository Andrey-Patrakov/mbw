import numpy as np
from mbw.ml.tree_model.base import DecisionTree, Node, Leaf
from mbw.ml.metrics.loss import mse, mae


class RegressorLeaf(Leaf):

    def _get_prediction(self, labels):
        self.prediction = np.mean(labels)


class DecisionTreeRegressor(DecisionTree):

    def __init__(self, criterion='mse', **kwargs):
        super().__init__(**kwargs)

        criterion = criterion.lower()
        criterion_list = ['mse', 'mae']
        if criterion == 'mse':
            self._selected_criterion = mse

        elif criterion == 'mae':
            self._selected_criterion = mae

        else:
            error = f'Unknown criterion {criterion} for DecisionTreeClassifier'
            error += f' (criterion not in [{", ".join(criterion_list)}])!'
            raise ValueError(error)

    def _criterion(self, labels):
        return self._selected_criterion(labels, np.mean(labels))

    def _Leaf(self, labels):
        return RegressorLeaf(labels)

    def _Node(self, index, t, true_branch, false_branch):
        return Node(index, t, true_branch, false_branch)
