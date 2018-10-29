# ABM_callibration

Callibration of two Agent Based Models:

- **The Brock and Hommes Model**: an asset pricing model defined in Brock and Hommes (1998), describes a hetrogeneneous population of agents trading assets according to different strategies.

- **The Islands Model**: defined in Fagiolo and Dosi (2003), describes a hetrogenenous population of agents discovering and diffusing new technologies.

## Agent-Based Model Calibration using Machine Learning Surrogates

Much of the work is based on findings in https://arxiv.org/abs/1703.10639. 

This paper tackles the exploration of a vast parameter space by using supervised machine learning surrogates. It is then possible to approximate the Agent Based Model using these surrogates and use this for callibration (to some degree of accuracy).

The two models used to demonstrate this approach are given above.

## Getting Started

All code has been ran/tested using Python 2.7.

There are several packages and dependancies that I have specified in the Pipfile, however, if you are not using pipenv, these are:

numpy = "\*" <br />
pandas = "\*" <br />
scipy = "\*" <br />
seaborn = "\*" <br />
matplotlib = "\*" <br />
scikit-optimize = "\*" <br />
xgboost = "\*" <br />
conda = "\*" <br />

where "\*" denotes the most recent release.

You can automatically install all of the dependancies in the Pipfile into your pipenv using:

```shell
pipenv install
```

## Data

The only data required for either model is found in the sp500.csv file. This contains daily adjusted closing prices for the S&P 500 from December 09, 2013 to December 07, 2015, for a total of 502 observations and is used in the case of the Brock and Hommes Model.

This can be found at https://finance.yahoo.com/quote/%5EGSPC/history.

## Files

The main code for each Agent Based Model is found in bh_functions.py and island_functions.py respectively. These contain the actual code for each model along with specific functions that are applicable to each model.

Example experiments are found in bh_experiment.py and island_experiment.py which can be ran through the command line. These scripts compare the performance of a GaussianProcessRegressor to that of a GradientBoosingRegressor.

