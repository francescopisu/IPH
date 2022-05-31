"""
This file deals with implementing the necessary methods to
perform bayesian hyperparameter tuning.
"""
import optuna
from optuna.integration import OptunaSearchCV
import numpy as np
from yacs.config import CfgNode
from typing import Tuple, Dict, Any
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import make_scorer, brier_score_loss
from sklearn.model_selection import train_test_split

from src._typing import Estimator, ArrayLike, CVScheme
from src.utils.scoring import _brier_loss


def tune_model(model: Estimator,
               search_space: Dict[str, Any],
               X: ArrayLike, y: ArrayLike,
               cv: CVScheme,
               conf: CfgNode) -> Tuple[Estimator, Dict]:
    """
    Tune the given model with the specified search space.

    Parameters
    ----------
    model: Estimator
        The model to optimize.
    search_space: Dict[str, Any]
        The search space to perform optimization on.
    X: ArrayLike
        A design matrix of shape (n_obs, n_feat)
    y: ArrayLike
        A ground-truth vector
    cv: CVScheme
        A cross-validation scheme to evaluate the different
        parameters samples.
    conf: CfgNode
        A yacs configuration node.

    Returns
    -------
    Tuple[Estimator, Dict]
        A tuple consisting of the optimized model and related
        best parameters.
    """
    Xtrain, Xval, ytrain, yval = train_test_split(X, y,
                                                  test_size=conf.TUNING.EVAL_SIZE,
                                                  stratify=y,
                                                  random_state=conf.MISC.SEED)

    opt_search = OptunaSearchCV(estimator=model,
                                param_distributions=search_space,
                                cv=cv,
                                n_trials=conf.TUNING.N_TRIALS_FINAL,
                                random_state=conf.MISC.SEED,
                                refit=True,
                                scoring=conf.TUNING.METRIC,
                                n_jobs=1,
                                )
    if conf.DATA.SCALE:
        numeric_cols = X.select_dtypes(include=np.float64).columns.tolist()
        scaler = StandardScaler()
        Xtrain.loc[:, numeric_cols] = scaler.fit_transform(Xtrain[numeric_cols])
        Xval.loc[:, numeric_cols] = scaler.transform(Xval[numeric_cols])
    # X.loc[:, numeric_cols] = scaler.transform(X[numeric_cols])

    opt_search.fit(Xtrain, ytrain, **{
        "categorical_feature": [0, 1, 2, 4, 5, 6, 7],
        "eval_metric": _brier_loss,
        "eval_set": [(Xval, yval)],
        "early_stopping_rounds": 30
    })

    best = opt_search.best_estimator_
    params = opt_search.best_params_

    # refit on whole training set X, y
    best.fit(X, y)

    return best, params
