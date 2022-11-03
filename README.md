# An Interpretable Machine Learning Approach to Identify Subjects with Recent Cerebrovascular Symptoms from Anatomical Features of Carotid Plaques
This repository contains the code to reproduce the experiments of "An Interpretable Machine Learning Approach to Identify Patients with Recent Cerebrovascular Symptoms from Anatomical Features of Carotid Plaques", currently under review at European Heart Journal: Cardiovascular Imaging.

## Abstract
Carotid plaque structure and composition are known to be associated with stroke, but the role of subcomponents (e.g., intraplaque hemorrhage [IPH]) is yet to be determined. Consequently, learning to recognize symptomatic patients having atherosclerotic disease allows identifying plaque features that are predominant in symptomatic subjects. We hypothesized that Machine Learning (ML) could perform a more complex evaluation of a patient’s status, integrating plaque morphology data with clinical and demographic factors. However, the clinical application of ML models has suffered from the interpretability problem. Using an interpretable tree-based boosting generalized additive model trained on demographic, clinical and CT angiography-derived plaque features, we were able to identify symptomatic patients with excellent diagnostic accuracy both on internal validation (area under the curve: 0.886) and dedicated testing (0.926). Volumetric measurements of carotid plaque’s subcomponents (in particular the ratio of IPH to lipid volume) were the most important parameters for identifying symptomatic subjects.

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
  - *Statistical Analysis.ipynb*: code for all the statistical analysis reported in the manuscript.
  - *Preprocessing and conversion.ipynb*: basic preprocessing to training and test data and conversion of from excel to csv.
  - *Baseline characteristics.Rmd*: computation of baseline characteristics of the derivation and validation cohorts.
  - *LOESS analysis.Rmd*: code used to produce the LOESS curves when analyzing the relationships with the top influential features and model predictions.
- **output/**: this folder contains output of the experiments.
  - fitted_models/: optimized models obtained after hyperparameter tuning.
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


### Get predicted probabilities of an optimized model on a specific set
```
python3 execute_cli.py get-predictions --which <WHICH DATASET> --fitted_model_filename <FILENAME OF FITTED MODEL>
```
This command computes probabilities for each sample in the specified set and saves the output in a .csv file at `output/predictions/<WHICH DATASET>_test_preds_<FILENAME OF FITTED MODEL>.csv` (the path can be customized modifying the argument OUTPUT.PREDS_PATH). See the documentation for the `test` for accepted values of `--which` argument.

## Acknowledgements

* The code for comparing ROC AUCs using DeLong test is taken from the [Yandex School of Data Analysis github repository](https://github.com/yandexdataschool/roc_comparison).

* The code for producing the observed vs. predicted plot is based on the [verhulst](https://github.com/grivescorbett/verhulst) package.
