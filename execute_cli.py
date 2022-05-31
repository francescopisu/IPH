"""
This file implements a CLI to start the training, tuning and testing of the model.
"""
from typing import Dict, Optional, List
import os
import time
import numpy as np
import pandas as pd
import click
from sklearn.model_selection import StratifiedKFold, RepeatedStratifiedKFold
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt

from config.defaults import get_defaults
from src.utils.io import load_data, load_obj, save_obj
from src.utils.misc import add_extension, show_cross_val_results
from src.evaluate import nested_cv, get_evaluation_metrics, test_performance
from src.tune import tune_model
from src.algorithms import ALGORITHMS
from src.spaces import PARAMETERS
from src.utils.plotting import plot_average_roc_curves, plot_average_pr_curves


def bootstrap(new_options: Optional[List] = None,
              mode: str = "train") -> Dict:
    """
    This function is responsible for the bootstrap phase prior to
    training or testing the model.
    It is responsible for:
        - loading the default configuration values
        - updating defaults by merging CLI argumentsc
        - loading either train or test dataset
        - instantiating or loading a model

    Parameters
    ----------
    new_options: new options coming from CLI arguments. They will be merged
        with defaults.
    mode: str (default = "train")
        Modality of execution. Options are train, test, cv and tune.
    Returns
    -------
    Dict
        A dictionary containing preprocessed data, a model and configuration data.
    """
    defaults = get_defaults()
    if new_options:
        defaults.merge_from_list(new_options)
    defaults.freeze()

    # load datasets
    which = 'train'  # default
    if mode in ['train', 'ncv']:
        which = 'train'
    elif mode == 'internal_test':
        which = 'test'
    elif mode == 'external_test':  # mode == 'external_test'
        which = 'external'

    X, y = load_data(defaults, which=which)

    # a. Get fixed parameters and search spaces for nested cross-validation
    # b. Get fixed parameters and search space for a specific algorithm to be tuned
    # c. load an already fitted model (after optimization/tuning)

    model = None
    params = None
    algorithms = None

    if mode == 'ncv':
        algorithms = ALGORITHMS
        params = PARAMETERS
    elif mode == 'tune':
        # set parameters of chosen algorithm
        params = PARAMETERS[defaults.TUNING.ALGORITHM_TO_TUNE]['space']
        model = ALGORITHMS[defaults.TUNING.ALGORITHM_TO_TUNE]
    elif mode in ['internal_test', 'external_test']:
        # load model from defaults.OUTPUT.FITTED_MODEL_PATH
        model = load_obj(defaults.OUTPUT.FITTED_MODEL_PATH)

    return {
        "data": (X, y),
        "defaults": defaults,
        "model": model,
        "params": params,
        "algorithms": algorithms
    }


@click.group()
def cli():
    pass


# @click.command()
# def data_bootstrap():
#     """
#     Command to load the data, split the data into
#     training and test set and apply basic preprocessing.
#     """
#     defaults = get_defaults()
#     load_split_preprocess(defaults)


@click.command()
@click.option("--which", default='internal')
@click.option("--fitted_model_filename", default='model.pkl')
def test(which, fitted_model_filename):
    """
    Test a fitted model on the test set.
    By default, we look for a fitted model in the output/fitted_model
    directory.

    fitted_model_filename is used as follows:
        a) to update the fitted_model_path from output/fitted_models/model.pkl (default)
        to output/fitted_models/<fitted_model_filename> in order to load the already fitted model.
        b) to derive the model name (part preceding the .extension) which is then used to
        save the test results in output/results/test_results_<model name>.csv
    """
    click.echo("Mode: test.")
    defaults = get_defaults()

    # bootstrap input
    fitted_model_path = os.path.join(defaults.OUTPUT.FITTED_MODELS_PATH, fitted_model_filename)
    new_options = ["OUTPUT.FITTED_MODEL_PATH", fitted_model_path]

    mode = "{}_test".format(which)
    boot_data = bootstrap(new_options, mode=mode)

    model = boot_data['model']
    X_test, y_test = boot_data['data']
    defaults = boot_data['defaults']

    eval_metrics = get_evaluation_metrics()
    # model = RandomForestClassifier(random_state=defaults.MISC.SEED, class_weight='balanced')

    # X_train, y_train = load_data(defaults, which='train')
    # scaler = StandardScaler()
    # numeric_cols = X_train.select_dtypes(include=np.number).columns.tolist()
    # X_train.loc[:, numeric_cols] = scaler.fit_transform(X_train[numeric_cols])

    # model.fit(X_train, y_train)

    test_results = test_performance(conf=defaults,
                                    model=model,
                                    X_test=X_test, y_test=y_test,
                                    eval_metrics=eval_metrics)
    results = pd.DataFrame(test_results.values(), index=test_results.keys(), columns=["test"])

    results_filename = "{}_results_{}.csv".format(mode, fitted_model_filename.split(".")[0])
    results_path = os.path.join(defaults.OUTPUT.RESULTS_PATH, results_filename)
    results.to_csv(results_path)


@click.command()
@click.option('--exp_name', default="exp_1")
def nested_cross_validation(exp_name):
    """
    Model selection and evaluation by means of nested cross-validation.

    fitted_model_filename is used as follows:
        a) to update the fitted_model_path from output/fitted_models/model.pkl (default)
        to output/fitted_models/<fitted_model_filename> in order to dump the fitted model.
    """
    click.echo("Mode: Nested cross-validation.")
    # defaults = get_defaults()

    # fitted_model_filename = add_extension(fitted_model_filename)

    # derive final path for fitted model as base output path for fitted models + model filename
    # fitted_model_path = os.path.join(defaults.OUTPUT.FITTED_MODELS_PATH, fitted_model_filename)
    # new_options = ["OUTPUT.FITTED_MODEL_PATH", fitted_model_path]

    # don't reserve dev set at this point since we need to do it in each cv fold
    boot_data = bootstrap(new_options=None, mode="ncv")

    parameters = boot_data['params']
    algorithms = boot_data['algorithms']
    defaults = boot_data['defaults']
    X_train, y_train = boot_data['data']

    inner_cv = StratifiedKFold(n_splits=defaults.EVAL.INNER_N_SPLITS,
                               shuffle=defaults.EVAL.SHUFFLE,
                               random_state=defaults.MISC.SEED)
    outer_cv = RepeatedStratifiedKFold(n_splits=defaults.EVAL.OUTER_N_SPLITS,
                                       n_repeats=defaults.EVAL.N_REPEATS,
                                       random_state=defaults.MISC.SEED)

    s = time.time()
    outer_results, outer_preds = nested_cv(algorithms=algorithms,
                                           parameters=parameters,
                                           X=X_train, y=y_train,
                                           outer_cv=outer_cv,
                                           inner_cv=inner_cv,
                                           conf=defaults)
    print("Execution time: %s seconds." % (time.time() - s))

    # dump results
    # fitted_model_best_params_path = os.path.join(defaults.OUTPUT.PARAMS_PATH,
    #                                              "best_params_{}.pkl".format(fitted_model_filename.split('.')[0]))

    outer_results_formatted = show_cross_val_results(outer_results, conf=defaults)

    # os.makedirs(defaults.OUTPUT.FITTED_MODELS_PATH, exist_ok=True)
    # save_obj(final_model, defaults.OUTPUT.FITTED_MODEL_PATH)

    # best_model_path = os.path.join(defaults.OUTPUT.FITTED_MODELS_PATH,
    #                                "{}.txt".format(fitted_model_filename.split('.')[0]))
    # final_model.save_model(best_model_path)

    # save_obj(best_params, fitted_model_best_params_path)

    # save_obj(outer_results, ncv_results_path)
    ncv_results_path = os.path.join(defaults.OUTPUT.RESULTS_PATH, "ncv_results_{}.csv".format(exp_name))
    outer_results_formatted.to_csv(ncv_results_path)

    # save predictions
    outer_preds_path = os.path.join(defaults.OUTPUT.PREDS_PATH, "outer_cv_accumulated_preds_{}.pkl".format(exp_name))
    save_obj(outer_preds, outer_preds_path)


@click.command()
@click.option('--algorithm', default="LGBM")
@click.option('--fitted_model_filename', default="model.pkl")
def tune(algorithm, fitted_model_filename):
    """
    Perform hyperparameter tuning of the specified algorithm.
    """
    click.echo("Mode: tuning.\n")
    defaults = get_defaults()

    fitted_model_filename = add_extension(fitted_model_filename)

    # derive final path for fitted model as base output path for fitted models + model filename
    fitted_model_path = os.path.join(defaults.OUTPUT.FITTED_MODELS_PATH, fitted_model_filename)
    new_options = ["OUTPUT.FITTED_MODEL_PATH", fitted_model_path,
                   "TUNING.ALGORITHM_TO_TUNE", algorithm]
    boot_data = bootstrap(new_options, mode="tune")

    defaults = boot_data['defaults']
    X_train, y_train = boot_data['data']
    search_space = boot_data['params']
    model = boot_data['model']

    cv = RepeatedStratifiedKFold(n_splits=defaults.EVAL.OUTER_N_SPLITS,
                                 n_repeats=defaults.EVAL.N_REPEATS,
                                 random_state=defaults.MISC.SEED)

    tuned_model, best_params = tune_model(model=model,
                                          search_space=search_space,
                                          X=X_train, y=y_train,
                                          cv=cv,
                                          conf=defaults)

    # dump fitted model
    fitted_model_best_params_path = os.path.join(defaults.OUTPUT.PARAMS_PATH,
                                                 "best_params_{}".format(fitted_model_filename))

    os.makedirs(defaults.OUTPUT.FITTED_MODELS_PATH, exist_ok=True)
    save_obj(tuned_model, defaults.OUTPUT.FITTED_MODEL_PATH)
    save_obj(best_params, fitted_model_best_params_path)


@click.command()
@click.option('--fitted_model_filename', default="model.pkl")
def get_predictions(fitted_model_filename):
    """
    Use a fitted model to predict probabilities and save it
    in the results folder.
    """
    click.echo("Mode: predicting probabilities.\n")
    defaults = get_defaults()

    fitted_model_filename = add_extension(fitted_model_filename)
    fitted_model_path = os.path.join(defaults.OUTPUT.FITTED_MODELS_PATH, fitted_model_filename)
    new_options = ["OUTPUT.FITTED_MODEL_PATH", fitted_model_path]

    # boot_data = bootstrap(new_options, mode="internal_test")
    # model = boot_data['model']
    #
    # X_test_int, y_test_int = boot_data['data']
    # internal_test_proba = model.predict_proba(X_test_int)
    # internal_test_proba = np.c_[y_test_int, internal_test_proba[:, 1]]

    boot_data = bootstrap(new_options, mode="external_test")
    model = boot_data['model']
    X_test_ext, y_test_ext = boot_data['data']

    # fit scaler on train data and transform test data
    scaler = StandardScaler()
    X_train, y_train = load_data(defaults, which='train')

    numeric_cols = X_train.select_dtypes(include=np.float64).columns.tolist()
    scaler.fit(X_train[numeric_cols])
    X_test_ext.loc[:, numeric_cols] = scaler.transform(X_test_ext[numeric_cols])

    external_test_proba = model.predict_proba(X_test_ext)
    external_test_proba = np.c_[y_test_ext, external_test_proba[:, 1]]

    # internal_test_results_path = os.path.join(defaults.OUTPUT.PREDS_PATH, "internal_test_preds.csv")
    external_test_results_path = os.path.join(defaults.OUTPUT.PREDS_PATH,
                                              f"external_test_preds_{fitted_model_filename.replace('.pkl', '')}.csv")
    # pd.DataFrame(internal_test_proba, columns=['target', 'proba']).to_csv(internal_test_results_path, index=False)
    pd.DataFrame(external_test_proba, columns=['target', 'proba']).to_csv(external_test_results_path, index=False)


@click.command()
@click.option('--exp_name')
@click.option('--exp_names', nargs=2, type=str)
@click.option('--plot_title')
@click.option('--output_filename')
def plot_ncv_roc_and_pr_curves(exp_name, exp_names, plot_title, output_filename):
    defaults = get_defaults()
    plot_path = os.path.join(defaults.OUTPUT.PLOTS_PATH, f"ncv_roc_pr_curves_{output_filename}.pdf")

    if exp_names:
        # load predictions corresponding to the specified experiment names
        exp_name1, exp_name2 = exp_names
        preds_path1 = os.path.join(defaults.OUTPUT.PREDS_PATH, f"outer_cv_accumulated_preds_{exp_name1}.pkl")
        preds_path2 = os.path.join(defaults.OUTPUT.PREDS_PATH, f"outer_cv_accumulated_preds_{exp_name2}.pkl")
        outer_cv_preds1 = load_obj(preds_path1)
        outer_cv_preds2 = load_obj(preds_path2)

        # f, axes = plt.subplots(2, 2, figsize=(16, 16))
        #
        # ax1 = plot_average_roc_curves(cv_preds=outer_cv_preds1, conf=defaults, ax=axes[0, 0])
        # ax2 = plot_average_pr_curves(cv_preds=outer_cv_preds1, conf=defaults, ax=axes[0, 1])
        # ax1.set_title("With plaque's features\n", fontsize=16)
        # ax3 = plot_average_roc_curves(cv_preds=outer_cv_preds2, conf=defaults, ax=axes[1, 0],
        #                               highlight_best=False)
        # ax4 = plot_average_pr_curves(cv_preds=outer_cv_preds2, conf=defaults, ax=axes[1, 1],
        #                              highlight_best=False)
        # ax4.set_title("Using demographic and clinical variables only\n", fontsize=16)
        #
        # ax1.text(0, 1.03, "A", transform=ax1.transAxes, size=20, weight='bold')
        # ax2.text(0, 1.03, "B", transform=ax2.transAxes, size=20, weight='bold')
        # ax3.text(0, 1.03, "C", transform=ax3.transAxes, size=20, weight='bold')
        # ax4.text(0, 1.03, "D", transform=ax4.transAxes, size=20, weight='bold')
        #
        # plt.suptitle(plot_title, fontsize=20, y=0.95)
        f = plt.figure(constrained_layout=True, figsize=(15, 17))
        f.suptitle(plot_title, fontsize=18, weight="normal")
        f.set_constrained_layout_pads(w_pad=0.2, h_pad=0.15,
                                      hspace=0.05, wspace=0. / 72.)

        subfigs = f.subfigures(nrows=2, ncols=1)

        subfigs[0].suptitle("With plaque morphology information", fontsize=15)
        # create two subplots for the first subfigure
        ax1, ax2 = subfigs[0].subplots(nrows=1, ncols=2)
        ax1 = plot_average_roc_curves(cv_preds=outer_cv_preds1, conf=defaults, ax=ax1)
        ax2 = plot_average_pr_curves(cv_preds=outer_cv_preds1, conf=defaults, ax=ax2, legend_position="lower right")

        subfigs[1].suptitle("Demographic and clinical variables only", fontsize=15)
        # create two subplots for the second subfigure
        ax3, ax4 = subfigs[1].subplots(nrows=1, ncols=2)
        ax3 = plot_average_roc_curves(cv_preds=outer_cv_preds2, conf=defaults, ax=ax3,
                                      highlight_best=False)
        ax4 = plot_average_pr_curves(cv_preds=outer_cv_preds2, conf=defaults, ax=ax4,
                                     highlight_best=False)

        ax1.text(0, 1.03, "A", transform=ax1.transAxes, size=20, weight='bold')
        ax2.text(0, 1.03, "B", transform=ax2.transAxes, size=20, weight='bold')
        ax3.text(0, 1.03, "C", transform=ax3.transAxes, size=20, weight='bold')
        ax4.text(0, 1.03, "D", transform=ax4.transAxes, size=20, weight='bold')
    else:
        # load predictions corresponding to the specified experiment name
        preds_path = os.path.join(defaults.OUTPUT.PREDS_PATH, f"outer_cv_accumulated_preds_{exp_name}.pkl")
        outer_cv_preds = load_obj(preds_path)

        f, axes = plt.subplots(1, 2, figsize=(16, 8))

        ax1 = plot_average_roc_curves(cv_preds=outer_cv_preds, conf=defaults, ax=axes[0])
        ax2 = plot_average_pr_curves(cv_preds=outer_cv_preds, conf=defaults, ax=axes[1])

        ax1.text(0, 1.03, "A", transform=ax1.transAxes, size=20, weight='bold')
        ax2.text(0, 1.03, "B", transform=ax2.transAxes, size=20, weight='bold')

        plt.suptitle(plot_title, fontsize=15)

    f.savefig(plot_path, format="pdf")


# cli.add_command(data_bootstrap)
cli.add_command(test)
cli.add_command(nested_cross_validation)
cli.add_command(tune)
cli.add_command(get_predictions)
cli.add_command(plot_ncv_roc_and_pr_curves)
# cli.add_command(plot_test_roc_and_pr_curves)

if __name__ == '__main__':
    cli()
