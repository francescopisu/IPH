import numpy as np
from typing import Dict, Callable, List, Tuple, Any
from functools import partial
from sklearn.metrics import (
    precision_score,
    fbeta_score,
    roc_auc_score,
    brier_score_loss,
    average_precision_score
)
from sklearn.preprocessing import StandardScaler
from yacs.config import CfgNode
from collections import defaultdict
from optuna.integration import OptunaSearchCV

from src._typing import ArrayLike, CVScheme, Estimator
from src.utils.scoring import (
    compute_conf_matrix_metric,
    compute_metrics,
    bootstrap_median_ci
)
from src.utils.io import load_data


def compute_cross_val_conf_intervals(cross_val_results: Dict[str, List[float]],
                                     alpha: float = 0.05) -> Dict[str, Tuple[float, float, float]]:
    """
    Computes confidence intervals for the cross-validation results
    with the desired significance level using the percentile method.

    Parameters
    ----------
    cross_val_results: Dict[str, List[float]]
        A dictionary of results of the cross validation procedure.
    alpha: float (default = 0.05)
        The significance level alpha for computing confidence intervals.
        The corresponding confidence level will be (1-alpha)%
        E.g.:
        alpha = 0.05
        confidence level = (1-0.05)% = 95%

    Returns
    -------
    Dict[str, Tuple[float, float, float]]
        A dictionary where keys are of the form SET_METRIC and values are
        tuple of median score, lower and upper bound of confidence interval.
    """
    results_ = dict()
    for k, scores in cross_val_results.items():
        med = np.median(scores).round(3)
        ci_lower = np.percentile(scores, alpha * 100 / 2).round(2)
        ci_upper = np.percentile(scores, 100 - (alpha * 100 / 2)).round(2)

        results_[k] = (med, ci_lower, ci_upper)

    return results_


def test_performance(conf: CfgNode,
                     model: Estimator,
                     X_test: ArrayLike, y_test: ArrayLike,
                     eval_metrics: Dict[str, Callable]) -> Dict[str, str]:
    """
    Compute test set performance of the specified model.
    In greater detail, the model is trained on the whole training set
    and used to predict probabilities (if is suppoted by the model)
    on the test data. Then, median and 95% CI are computed for each
    evaluation metric and returned in a dictionary.

    Parameters
    ----------
    conf: CfgNode
        A yacs configuration node to access configuration values.
    model : Estimator
      A fitted estimator to be tested on test data.
    X_test: ArrayLike of shape (n_obs, n_features)
        Design matrix containing feature values of test data.
    y_test: ArrayLike of shape (n_obs,)
        A vector of test data ground-truth labels.
    eval_metrics: Dict[str, Callable]
        A dictionary of evaluation metrics.
        Example:
        {
            "auc": roc_auc_score,
            "brier_loss": brier_score_loss,
            ...
        }

    Returns
    -------
    Dict[str, str]
        A dictionary of metrics values computed on the test set.
    """
    # fit scaler on train data and apply scaling to test data
    scaler = StandardScaler()
    X_train, y_train = load_data(conf, which='train')
    numeric_cols = X_train.select_dtypes(include=np.float64).columns.tolist()
    scaler.fit(X_train[numeric_cols])
    X_test.loc[:, numeric_cols] = scaler.transform(X_test[numeric_cols])

    # Predict on test set
    pred_probas_test = model.predict_proba(X_test)
    pos_class_probas = pred_probas_test[:, 1]

    # predicted labels with class. threshold of 0.5
    pred_test_labels = np.where(pos_class_probas > conf.EVAL.THRESHOLD, 1, 0)

    out = {}
    for metric_name, metric_fn in eval_metrics.items():
        print("Bootstrapping {}..".format(metric_name))
        if metric_name in ['ROCAUC', 'PRAUC']:
            # use probabilities
            preds = pos_class_probas
        else:
            # use labels
            preds = pred_test_labels

        med, conf_int = bootstrap_median_ci(target=y_test,
                                            preds=preds,
                                            metric=metric_fn,
                                            n_boot=conf.EVAL.BOOT_ROUNDS,
                                            seed=conf.MISC.SEED)
        print(f"Metric name: {metric_name}, Median value:{med} ")
        out[metric_name] = "{:.3f} [{:.3f}-{:.3f}]".format(med, conf_int[0], conf_int[1])

    return out


def nested_cv(algorithms: Dict[str, Estimator],
              parameters: Dict[str, Any],
              X: ArrayLike,
              y: ArrayLike,
              outer_cv: CVScheme,
              inner_cv: CVScheme,
              conf: CfgNode) -> List[Tuple[Estimator, Dict, Dict]]:
    """
    Perform nested cross-validation of a binary LightGBM classifier
    with inner model selection through bayesian optimization with the Optuna framework.

    Parameters
    ----------
    algorithms: Dict[str, Estimator]
        A dictionary of estimators with fixed parameters.
    parameters: Dict[str, Any]
        Fixed parameters and search spaces for each algorithm.
    X: ArrayLike of shape (n_obs, n_features)
        A design matrix of feature values.
    y: ArrayLike of shape (n_obs,)
        An array of ground-truths.
    outer_cv: CVScheme
        A cross-validation scheme for outer model evaluation.
    inner_cv: CVScheme
        A cross-validation scheme for inner model evaluation.
    conf: CfgNode
        A yacs configuration node to access configuration values.

    Returns
    -------
    Tuple[lgb.Booster, Dict, Dict]
        A tuple consisting of the best LightGBM booster and associated
        parameters after hyperparameter tuning and the results of
        outer the cross-validation evaluation, for each estimator.
    """
    results = defaultdict(list)
    preds = dict()

    eval_metrics = get_evaluation_metrics()

    for algo_name, algo in algorithms.items():
        opt_search = OptunaSearchCV(estimator=algo,
                                    param_distributions=parameters[algo_name]['space'],
                                    cv=inner_cv,
                                    n_trials=conf.TUNING.N_TRIALS,
                                    random_state=conf.MISC.SEED,
                                    refit=True,
                                    scoring=conf.TUNING.METRIC,
                                    n_jobs=1,
                                    )
        # preds[algo_name] = {"gt": [], "proba": []}
        preds[algo_name] = dict()
        curr_algo_outer_gts = []
        curr_algo_outer_probas = []

        for i, (train_idx, test_idx) in enumerate(outer_cv.split(X, y)):
            Xtrain, ytrain = X.iloc[train_idx], y.loc[train_idx]
            Xtest, ytest = X.loc[test_idx], y.loc[test_idx]

            # scale numeric variables
            numeric_cols = Xtrain.select_dtypes(include=np.float64).columns.tolist()
            scaler = StandardScaler()
            Xtrain.loc[:, numeric_cols] = scaler.fit_transform(Xtrain.loc[:, numeric_cols])
            Xtest.loc[:, numeric_cols] = scaler.transform(Xtest.loc[:, numeric_cols])

            # run hyperparameter tuning
            opt_search.fit(Xtrain, ytrain)

            # retrieve best estimator for current fold
            best = opt_search.best_estimator_

            # predict on both sets
            train_preds = best.predict_proba(Xtrain)
            test_preds = best.predict_proba(Xtest)

            # accumulate ground-truths and predicted probabilities
            # preds[algo_name]['gt'].append(ytest)
            # preds[algo_name]['proba'].append(test_preds)
            curr_algo_outer_gts.append(ytest)
            curr_algo_outer_probas.append(test_preds)

            # compute evaluation metrics
            train_scores = compute_metrics(train_preds, ytrain, eval_metrics)
            test_scores = compute_metrics(test_preds, ytest, eval_metrics)

            # save scores
            for s, scores_dict in zip(conf.EVAL.SET_NAMES, [train_scores, test_scores]):
                for metric_name, score in scores_dict.items():
                    key = "{}_{}_{}".format(algo_name, s, metric_name)
                    results[key].append(score)

        # concatenate ground-truths and predicted probabilities
        preds[algo_name] = {
            "gt_conc": np.concatenate(curr_algo_outer_gts),
            "probas_conc": np.concatenate(curr_algo_outer_probas),
            "gt": curr_algo_outer_gts,
            "probas": curr_algo_outer_probas,
        }

    # compute confidence intervals based on the percentile method.
    results_with_cis = compute_cross_val_conf_intervals(results, alpha=conf.EVAL.ALPHA)

    return results_with_cis, preds


def get_evaluation_metrics() -> Dict[str, Callable]:
    eval_metrics = {
        "Sensitivity": partial(compute_conf_matrix_metric, metric_name="tpr"),
        "Specificity": partial(compute_conf_matrix_metric, metric_name="tnr"),
        "FPR": partial(compute_conf_matrix_metric, metric_name="fpr"),
        "FNR": partial(compute_conf_matrix_metric, metric_name="fnr"),
        "Precision": precision_score,
        "PPV": partial(compute_conf_matrix_metric, metric_name="ppv"),
        "NPV": partial(compute_conf_matrix_metric, metric_name="npv"),
        "F1": partial(fbeta_score, beta=1),
        "F2": partial(fbeta_score, beta=2),
        "ROCAUC": roc_auc_score,
        "PRAUC": average_precision_score,
        "Brier": brier_score_loss
    }

    return eval_metrics
