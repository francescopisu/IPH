import numpy as np
from interpret.glassbox import ExplainableBoostingClassifier
from yacs.config import CfgNode
from sklearn.preprocessing import StandardScaler

from src._typing import ArrayLike, CVScheme, Estimator


def train_model(X: ArrayLike, y: ArrayLike, conf: CfgNode) -> ExplainableBoostingClassifier:
    """
    Train an Explainable Boosting Classifier on specified data.

    Parameters
    ----------
    X: ArrayLike of shape (n_obs, n_features)
        A design matrix of feature values.
    y: ArrayLike of shape (n_obs,)
        An array of ground-truths.
    conf: CfgNode
        A yacs configuration node to access configuration values.

    Returns
    -------
    A fitted Explainable Boosting Classifier
    """
    ebm = ExplainableBoostingClassifier(random_state=conf.MISC.SEED, interactions=6,
                                        learning_rate=0.01,
                                        min_samples_leaf=2,
                                        outer_bags=25,
                                        inner_bags=25,
                                        max_bins=64,
                                        max_leaves=5)
    ebm.fit(X, y)

    return ebm
