from mbw.ml.ensemble.bagging import Bagging
from mbw.ml.tree_model.decision_tree_classifier import DecisionTreeClassifier
from mbw.ml.tree_model.decision_tree_regressor import DecisionTreeRegressor


class RandomForestClassifier(Bagging):
    def __init__(self,
                 n_estimators=100,
                 random_state=None,
                 max_features='sqrt',
                 **kwargs) -> None:

        super().__init__(estimator=DecisionTreeClassifier,
                         prediction='voting',
                         n_estimators=n_estimators,
                         random_state=random_state,
                         max_features=max_features,
                         **kwargs)


class RandomForestRegressor(Bagging):

    def __init__(self,
                 n_estimators=100,
                 random_state=None,
                 max_features=1/3,
                 **kwargs) -> None:

        super().__init__(estimator=DecisionTreeRegressor,
                         prediction='mean',
                         n_estimators=n_estimators,
                         random_state=random_state,
                         max_features=max_features,
                         **kwargs)
