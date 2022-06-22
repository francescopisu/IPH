# An Interpretable Machine Learning Approach to Stroke Prediction from Anatomical Features of Carotid Plaques
This repository contains the code to reproduce the experiments of "An Interpretable Machine Learning Approach to Stroke Prediction from Anatomical Features of Carotid Plaques", (under review).

## Abstract
Coming soon..

## Repository structure:
The repository is organized as follows:
```
IPH/
│    README.md: you are here
│    requirements.txt
└─── config/
│    └─── defaults.py
└─── notebooks/
│    │    Comparing ROC AUCs.ipynb
│    │    Preprocessing and conversion.ipynb
│    │    Interpreting predictions.ipynb
│    │    Statistical Analysis.Rmd
│    └─── ROCandPRcurvesconfidenceintervals.Rmd
└─── output/
│    │    fitted_models/
│    │    parameters/
│    │    plots/
│    │    predictions/
│    └─── results/
└─── src/    
│    └─── utils/
│    │    │   data.py
│    │    │   io.py
│    │    │   misc.py
│    │    │   plotting.py
│    │    └── scoring.py
│    │    _typing.py
│    │    algorithms.py
│    │    evaluate.py
│    │    spaces.py
│    └──  tune.py
│
└─── execute_cli.py


```

In greater detail:

- **requirements.txt**: required python3 packages to reproduce the experiments. Create a conda virtual environment, activate it and install the required packages.
- **config/**: this folder deals with experiment configuration.
  - *defaults.py*: default configuration values based on the YACS framework
- **notebooks/**: python and R notebooks used to analyze experiments' results.
  - *Comparing ROC AUCs.ipynb*: comparison of areas under the ROC curves of the classifiers' performance after nested cross-validation.
  - *Preprocessing and conversion.ipynb*: basic preprocessing to training and test data and conversion of from excel to csv.
  - *Interpreting predictions.ipynb*: application of SHAP and accumulated local effect frameworks to analyze final model's predictions and misclassifications.
  - *Statistical Analysis.Rmd*: computation of baseline characteristics of the derivation and validation cohorts.
  - *ROCandPRcurvesconfidenceintervals.Rmd*: computation of ROC and precision-recall curves, and confusion matrix-related metrics associated to the best operational point.
- **output/**: this folder contains output of the experiments.
  - fitted_models/: optimized models obtained after hyperparameter tuning.
  - parameters/: parameters of each optimized model.
  - plots/: plots and figures to be included in the manuscript.
  - predictions/: predicted probabilities of the optimized models.
  - results/: results of the nested cross-validation procedure and performance on the validation cohort.
- **src/**: codebase for reproducibile experiments.
  - utils/: utilities to deal with scoring, plotting and handling data.
    - *io.py*: methods related to input-output operations.
    - *misc.py*: various utility functions.
    - *plotting.py*: plotting-related utilities (e.g., plotting ROC and precision-recall curves).
    - *scoring.py*: methods for computing evaluation metrics.
  - *_typing.py*: custom types to improve readibility
  - *algorithms.py*: definition of the learning algorithms used in the study.
  - *evaluate.py*: methods to perform nested cross-validation and to test performance of models.
  - *spaces.py*: definition of search spaces to conduct hyperparameter optimization on.
  - *tune.py*: methods to perform hyperparameter tuning
- **execute_cli.py**: CLI to execute training, tuning, testing and plotting. 


## How to use the CLI:
The `execute_cli.py` file is the entry point for all the parts of the projects. It allows starting the nested cross-validation procedure, tuning a specific algorith, testing its performance, plotting the resulting ROC and precision-recall curves after nested cross-validation and getting the predicted probabilities of a specific model.

### Compare algorithms through nested cross-validation
```
python3 execute_cli.py nested-cross-validation --exp_name <EXPERIMENT NAME>
```
This command starts the nested cross-validation and saves the results in a .csv file in `output/results/ncv_results_<EXP_NAME>.csv`.

### Optimize a specific algorithm and get an optimized model
```
python3 execute_cli.py tune --algorithm <ALGORITHM NAME> --fitted_model_filename <FILENAME OF FITTED MODEL>
```
This command optimizes the specified algorithms using the search spaces defined in `src/spaces.py` and saves the optimized model in pickle format at  `output/fitted_models/<FILENAME OF FITTED MODEL>.pkl` and best parameters at `output/parameters/best_params_<FILENAME OF FITTED MODEL>.pkl`

Accepted values for the `--algorithm` argument are the keys of the algorithms dictionary specified in `src/algorithms.py`.

### Test an optimized model on a specific set
```
python3 execute_cli.py test --which <WHICH DATASET> --fitted_model_filename <FILENAME OF FITTED MODEL>
```
This command tests the performance of an optimized model and saves the results in a .csv file in `output/predictions/<WHICH DATASET>_test_results_<FILENAME OF FITTED MODEL>.csv`. 

For each evaluation metric (which are specified in `src/evaluate.py`), median and 95% CI are computed via a configurable number of the bias-corrected and accelerated bootstrap replications (default is 5000, see EVAL.BOOT_ROUNDS parameter in `config/defaults.py`).

In general, this project is set up to handle three different datasets: training set, internal test set and external test set. The test function should be used to compute performance on either internal or external test sets. By default, the `--which` argument accepts either `internal` or `external` values, which are connected to specific dataset filenames through the default configuration file (e.g., internal value is connected to the DATA.TEST_DATA_PATH whereas external value is connected to DATA.EXTERNAL_DATA_PATH).

### Plot ROC and Precision-recall curves of the nested cross-validation
```
python3 execute_cli.py plot-ncv-roc-and-pr-curves --exp_name <EXPERIMENT NAME> --exp_names <EXPERIMENT NAMES> --plot_title <PLOT TITLE> --output_filename <FILENAME OF PLOT>
```
This command generates ROC and precision-recall curves based on predicted probabilities of the outer cross-validation of the nested procedure of the experiment specified via `--exp_name` argument. The resulting plot is saved in .pdf format at `output/plots/ncv_roc_pr_curves_<FILENAME OF PLOT>.pdf`.

Alternatively, two different experiment names can be specified via `--exp_names` argument, separated by space. For each experiment, a pair of ROC and precision-recall curves is generated in each row. In addition, an optional argument `--plot_title` can be used to specify the main title for the figure.


### Get predicted probabilities of an optimized model on a specific set
```
python3 execute_cli.py get-predictions --which <WHICH DATASET> --fitted_model_filename <FILENAME OF FITTED MODEL>
```
This command computes probabilities for each sample in the specified set and saves the output in a .csv file at `output/predictions/<WHICH DATASET>_test_preds_<FILENAME OF FITTED MODEL>.csv` (the path can be customized modifying the argument OUTPUT.PREDS_PATH). See the documentation for the `test` for accepted values of `--which` argument.

## Acknowledgements

* The code for comparing ROC AUCs using DeLong test is taken from the [Yandex School of Data Analysis github repository](https://github.com/yandexdataschool/roc_comparison).

