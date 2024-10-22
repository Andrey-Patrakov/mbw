from mbw.ml.ensemble.boosting import BoostingRegressor
from mbw.ml.tree_model.decision_tree_regressor import DecisionTreeRegressor


class GradientBoostingRegressor(BoostingRegressor):

    def __init__(self,
                 estimator=None,
                 eta=0.1,
                 n_estimators=100,
                 random_state=None,
                 estimator_coefs=None,
                 **kwargs) -> None:

        if estimator is None:
            estimator = DecisionTreeRegressor

        super().__init__(estimator=estimator,
                         eta=eta,
                         n_estimators=n_estimators,
                         random_state=random_state,
                         estimator_coefs=estimator_coefs,
                         **kwargs)
