"""
This file defines the classification algorithms to be tuned and evaluated.
"""
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from lightgbm import LGBMClassifier
from yacs.config import CfgNode
from sklearn.pipeline import Pipeline
from sklearn.compose import (
    ColumnTransformer,
    make_column_selector as selector
)
from sklearn.preprocessing import StandardScaler
from optuna.integration import OptunaSearchCV
from typing import List, Dict

from src.spaces import PARAMETERS
from src._typing import Estimator, Distribution

ALGORITHMS = {
    "RF": RandomForestClassifier(**PARAMETERS['RF']['fixed']),
    "LGBM": LGBMClassifier(**PARAMETERS['LGBM']['fixed']),
    "SVC": SVC(**PARAMETERS['SVC']['fixed']),
    "LR": LogisticRegression(**PARAMETERS['LR']['fixed']),
    "KNN": KNeighborsClassifier(**PARAMETERS['KNN']['fixed'])
}

# def get_pipelines(algorithms: Dict[str, Estimator],
#                   search_spaces: Dict[str, Distribution],
#                   conf: CfgNode) -> Dict[str, Pipeline]:
#     pipelines = dict()
#
#     preprocessor = ColumnTransformer
#
#     for algo_name, search_space in search_spaces.items():
#         algorithm = algorithms[algo_name]
#
#         # build optuna search object
#         search_obj =
#
#         pipe = Pipeline([
#             ('preproc', ),
#             ('')
#         ])
