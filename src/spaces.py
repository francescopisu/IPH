# """
# This file defines search spaces to perform bayesian optimization on
# for each learning algorithm.
# """
# import optuna
# from optuna.trial import Trial
# from yacs.config import CfgNode
# from typing import Dict, Any
#
#
# class LightGBMSpace:
#     def __init__(self, trial, conf):
#         self.trial = trial
#         self.conf = conf
#         self.fixed = {
#             'boosting': 'gbdt',
#             'objective': 'binary',
#             'metric': "None",
#             'num_threads': 1,
#             'seed': self.conf.MISC.SEED,
#             'force_row_wise': True,
#             'deterministic': True,
#             'verbosity': -1,
#             'feature_pre_filter': False
#         }
#
#     def get_space(self):
#         space = {
#             "reg_alpha": self.trial.suggest_float("lambda_l1", 0.1, 10.0),
#             "reg_lambda": self.trial.suggest_float("lambda_l2", 0.1, 10.0),
#             "num_leaves": self.trial.suggest_int("num_leaves", 2, 15),
#             "feature_fraction": self.trial.suggest_float("feature_fraction", 0.4, 1.0),
#             "bagging_fraction": self.trial.suggest_float("bagging_fraction", 0.4, 1.0),
#             "bagging_freq": self.trial.suggest_int("bagging_freq", 1, 7),
#             "min_child_samples": self.trial.suggest_int("min_child_samples", 5, 50),
#             "max_depth": self.trial.suggest_int("max_depth", 2, 20),
#             "n_estimators": self.trial.suggest_int("n_estimators", 30, 120),
#             "learning_rate": 0.1
#         }
#
#         return space
#
#
# class RandomForestSpace:
#     def __init__(self, trial, conf):
#         self.trial = trial
#         self.conf = conf
#         self.fixed = {
#             'criterion': "gini",
#             'n_jobs': 1,
#             'random_state': self.conf.MISC.SEED,
#             'verbose': 0,
#             'class_weight': "balanced",
#             'bootstrap': True
#         }
#
#     def get_space(self):
#         space = {
#             "max_leaf_nodes": self.trial.suggest_int("num_leaves", 2, 15),
#             "max_features": self.trial.suggest_float("feature_fraction", 0.4, 1.0),
#             "bagging_fraction": self.trial.suggest_float("bagging_fraction", 0.4, 1.0),
#             "max_samples": self.trial.suggest_float("max_samples", 0.4, 1.0),
#             "min_samples_leaf": self.trial.suggest_int("min_child_samples", 5, 10),
#             "max_depth": self.trial.suggest_int("max_depth", 2, 10),
#             "n_estimators": self.trial.suggest_int("n_estimators", 30, 250),
#         }
#
#         return space
#
#
# class SVCSpace:
#     def __init__(self, trial, conf):
#         self.trial = trial
#         self.conf = conf
#         self.fixed = {
#             'probability': True,
#             'class_weight': 'balanced',
#             'verbose': 0,
#             'random_state': self.conf.MISC.SEED,
#             'max_iter': 500,
#             'n_jobs': 1
#         }
#
#     def get_space(self):
#         space = {
#             "C": self.trial.suggest_float("num_leaves", 0.1, 5),
#             "kernel": self.trial.suggest_categorical("kernel", ['linear', 'rbf']),
#         }
#
#         return space
#
#
# class LogRegSpace:
#     def __init__(self, trial, conf):
#         self.trial = trial
#         self.conf = conf
#         self.fixed = {
#             'probability': True,
#             'class_weight': 'balanced',
#             'verbose': 0,
#             'random_state': self.conf.MISC.SEED,
#             'solver': 'liblinear',
#             'n_jobs': 1,
#             'max_iter': 200
#         }
#
#     def get_space(self):
#         # space = {
#         #     'penalty': self.trial.suggest_categorical('penalty', ['l1', 'l2']),
#         #     'C': self.trial.suggest_float('C', 0.1, 5)
#         # }
#         space = {
#             'penalty': optuna.distributions.CategoricalDistribution(['l1', 'l2']),
#             'C': optuna.distributions.UniformDistribution('C', 0.1, 5)
#         }
#
#         return space
#
#
# _spaces = {
#     "LGBM": LightGBMSpace,
#     "RF": RandomForestSpace,
#     "SVC": SVCSpace,
#     "LR": LogRegSpace
# }
#
#
# def get_search_space(algo: str,
#                      trial: Trial,
#                      conf: CfgNode) -> Dict[str, Any]:
#     ss = _spaces[algo](trial, conf).get_space()
#     return ss

"""
This file defines search spaces to perform bayesian optimization on
for each learning algorithm.
"""
import optuna
from optuna.trial import Trial
from yacs.config import CfgNode
from typing import Dict, Any
from config.defaults import get_defaults

PARAMETERS = {
    "RF": {
        "fixed": {
            'criterion': "gini",
            'n_jobs': 10,
            'random_state': get_defaults().MISC.SEED,
            'verbose': 0,
            'class_weight': "balanced",
            'bootstrap': True,
            'n_estimators': 100
        },
        "space": {
            "max_leaf_nodes": optuna.distributions.IntUniformDistribution(2, 15),
            "max_features": optuna.distributions.UniformDistribution(0.4, 1.0),
            # "bagging_fraction": optuna.distributions.UniformDistribution(0.4, 1.0),
            "max_samples": optuna.distributions.UniformDistribution(0.4, 1.0),
            "min_samples_leaf": optuna.distributions.IntUniformDistribution(5, 10),
            "max_depth": optuna.distributions.IntUniformDistribution(2, 10),
            # "n_estimators": optuna.distributions.IntUniformDistribution(30, 250),
        }
    },
    "LGBM": {
        "fixed": {
            'boosting_type': 'gbdt',
            'objective': 'binary',
            'metric': "None",
            'n_jobs': 1,
            'seed': get_defaults().MISC.SEED,
            'force_row_wise': True,
            'deterministic': True,
            'verbose': -1,
            'feature_pre_filter': False,
            # "learning_rate": 0.05,  # model selection within nested cross-validation
            "learning_rate": 0.02,  # tuning
            "is_unbalance": True,
        },
        "space": {
            "reg_alpha": optuna.distributions.UniformDistribution(0.1, 10.0),
            "reg_lambda": optuna.distributions.UniformDistribution(0.1, 10.0),
            "num_leaves": optuna.distributions.IntUniformDistribution(2, 15),
            "colsample_bytree": optuna.distributions.UniformDistribution(0.4, 1.0),
            "subsample": optuna.distributions.UniformDistribution(0.4, 1.0),
            "subsample_freq": optuna.distributions.IntUniformDistribution(1, 7),
            "min_child_samples": optuna.distributions.IntUniformDistribution(5, 50),
            "max_depth": optuna.distributions.IntUniformDistribution(2, 20),
            "n_estimators": optuna.distributions.IntUniformDistribution(30, 120),
        }
    },
    "LR": {
        "fixed": {
            'class_weight': 'balanced',
            'verbose': 0,
            'random_state': get_defaults().MISC.SEED,
            'solver': 'liblinear',
            'n_jobs': 1,
            'max_iter': 200
        },
        "space": {
            'penalty': optuna.distributions.CategoricalDistribution(['l1', 'l2']),
            'C': optuna.distributions.UniformDistribution(0.1, 5)
        }
    },
    "SVC": {
        "fixed": {
            'probability': True,
            'class_weight': 'balanced',
            'verbose': 0,
            'random_state': get_defaults().MISC.SEED,
            'max_iter': 500},
        "space": {
            "C": optuna.distributions.UniformDistribution(0.1, 5),
            "kernel": optuna.distributions.CategoricalDistribution(['linear', 'rbf']),
        }
    },
    "KNN": {
        "fixed": {
            "n_jobs": 1,
        },
        "space": {
            "leaf_size": optuna.distributions.IntUniformDistribution(1, 20),
            "n_neighbors": optuna.distributions.IntUniformDistribution(2, 20),
            "p": optuna.distributions.IntUniformDistribution(1, 4)
        }
    }
}
