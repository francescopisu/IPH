"""
Default configuration values for the experiments.
"""
import os
from yacs.config import CfgNode as CN

_C = CN()

# data
_C.DATA = CN()
_C.DATA.BASE_INPUT_PATH = os.path.join(os.getcwd(), 'input')
_C.DATA.TRAIN_DATA_PATH = os.path.join(_C.DATA.BASE_INPUT_PATH, 'train.csv')
_C.DATA.TEST_DATA_PATH = os.path.join(_C.DATA.BASE_INPUT_PATH, 'test.csv')
_C.DATA.EXTERNAL_DATA_PATH = os.path.join(_C.DATA.BASE_INPUT_PATH, 'external.csv')
_C.DATA.APPLY_OHE = False
_C.DATA.SCALE = True
_C.DATA.TEST_SIZE = 0.2
_C.DATA.TARGET = "symptoms"
_C.DATA.TO_SCALE = ['plaque_volume', 'iph_volume', 'lipid_minus_iph',
                    'lipid_volume', 'mixed_volume', 'calcium_volume', 'perc_lipid',
                    'perc_mixed', 'perc_calcium', 'perc_iph', 'perc_lipid_minus_iph',
                    'iph_total_lipid_ratio', 'age']
_C.DATA.CAT_FEATURES = ['symptoms', 'iph', 'hypertension', 'CAD',
                        'smoker', 'gender', 'diabetes',
                        'lipids', 'stenosis']
_C.DATA.SUBSET_DATA = False
_C.DATA.WHICH_SUBSET = "plaque"
_C.DATA.CLINICAL_SUBSET = ['hypertension', 'CAD', 'smoker', 'age', 'gender', 'diabetes',
                           'lipids', 'stenosis']
_C.DATA.PLAQUE_SUBSET = ['plaque_volume', 'iph_volume', 'lipid_minus_iph',
                         'lipid_volume', 'mixed_volume', 'calcium_volume', 'perc_lipid',
                         'perc_mixed', 'perc_calcium', 'perc_iph', 'perc_lipid_minus_iph',
                         'iph_total_lipid_ratio',
                         'iph']
_C.DATA.ONLY_RATIO = ['iph_total_lipid_ratio']

# output
_C.OUTPUT = CN()
_C.OUTPUT.BASE_OUTPUT_PATH = os.path.join(os.getcwd(), "output")
_C.OUTPUT.FITTED_MODELS_PATH = os.path.join(os.getcwd(), "output/fitted_models")
_C.OUTPUT.RESULTS_PATH = os.path.join(os.getcwd(), "output/results")
_C.OUTPUT.PARAMS_PATH = os.path.join(os.getcwd(), "output/parameters")
_C.OUTPUT.PREDS_PATH = os.path.join(os.getcwd(), "output/predictions")
_C.OUTPUT.PLOTS_PATH = os.path.join(os.getcwd(), "output/plots")
_C.OUTPUT.FITTED_MODEL_PATH = os.path.join(_C.OUTPUT.FITTED_MODELS_PATH, "model.pkl")
_C.OUTPUT.BEST_MODEL = os.path.join(_C.OUTPUT.FITTED_MODELS_PATH, "final.pkl")
_C.OUTPUT.BEST_PARAMS = os.path.join(_C.OUTPUT.PARAMS_PATH, "best_params.pkl")
_C.OUTPUT.TEST_RESULTS = os.path.join(_C.OUTPUT.RESULTS_PATH, "test_results.csv")
# _C.OUTPUT.OUTER_CV_PREDS = os.path.join(_C.OUTPUT.PREDS_PATH, "outer_cv_accumulated_preds.csv")

# evaluation
_C.EVAL = CN()
_C.EVAL.N_SPLITS = 10
_C.EVAL.N_REPEATS = 10
_C.EVAL.SHUFFLE = True
_C.EVAL.ALPHA = 0.05
_C.EVAL.BOOT_ROUNDS = 5000
# _C.EVAL.THRESHOLD = 0.5729885
_C.EVAL.THRESHOLD = 0.23723600934486413
# _C.EVAL.THRESHOLD = 0.5
_C.EVAL.SET_NAMES = ["Train", "Valid"]
_C.EVAL.ALGO_SHORT_NAMES = ["EBM"]
_C.EVAL.ALGO_LONG_NAMES = ["Explainable Boosting Classifier"]
_C.EVAL.METRIC_NAMES = ["Sensitivity", "Specificity", "FPR", "FNR",
                        "Precision", "PPV", "NPV", "F1", "F2", "ROCAUC", "Brier"]

# misc
_C.MISC = CN()
_C.MISC.SEED = 1303


def get_defaults():
    return _C.clone()
