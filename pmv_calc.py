import warnings
import os
import sys
import logging
import yaml
import numpy as np
import pandas as pd
from data import DataLoader
from utils import load_variable, save_variable, clf_metrics
from pythermalcomfort.models import pmv
from pythermalcomfort.psychrometrics import clo_dynamic, v_relative


def warn(*args, **kwargs):
    pass


def simplified_pmv_model(data, dataset="dorn"):
    if "dorn" in dataset:
        rh = "rh-ubi"
        temp = "t-ubi"
        data = data[[rh, temp, "clothing", "met"]].copy()
        data["met"] = data["met"].map(
            {
                "Sitting": 1.1,
                "Resting": 0.8,
                "Standing": 1.4,
                "Exercising": 3,
            }
        )
        data["clothing"] = data["clothing"].map(
            {
                "Very light": 0.3,
                "Light": 0.5,
                "Medium": 0.7,
                "Heavy": 1,
            }
        )
    elif "smc" in dataset:
        rh = "RH_hobo"
        temp = "Ta_hobo"
        data = data[["RH_hobo", "Ta_hobo", "clothing"]].copy()
        data["met"] = 1.1  # people voted while resting

    arr_pmv_grouped = []
    arr_pmv = []
    for _, row in data.iterrows():
        val = pmv(
            row[temp],
            row[temp],
            v_relative(0.1, row["met"]),
            row[rh],
            row["met"],
            clo_dynamic(row["clothing"], row["met"]),
        )
        if val < -1.5:
            arr_pmv_grouped.append(9.0)  # "Warmer"
        elif -1.5 <= val <= 1.5:
            arr_pmv_grouped.append(10.0)  # "No Change"
        else:
            arr_pmv_grouped.append(11.0)  # "Cooler"

        arr_pmv.append(val)

    data["PMV"] = arr_pmv
    data["PMV_grouped"] = arr_pmv_grouped

    return data["PMV_grouped"]


warnings.warn = warn

# load config file from CLI
with open(str(sys.argv[1]), "r") as f:
    config = yaml.load(f, Loader=yaml.FullLoader)

# extract parameters from config file
# data
experiment_name = config["name"]
iterations = config["iterations"]
seed = config["seed"]
dataset_name = config["dataset_name"]
dataset_str = config["dataset"]
dataset_qns_str = config["dataset_survey"]
dataset_type = config["dataset_type"]
categorical_features = config["categorical_features"]
target_column = config["target_column"]
qns_encoding_idx = config["qns_encoding_idx"]
qns_categories = config["qns_categories"]
target_values = config["target_values"]
train_test_ratio = config["train_test_ratio"]
gamma = config["gamma"]
scorer = config["scorer"]

# folder for experiment results
try:
    os.mkdir(experiment_name)
except OSError:
    pass

# logging file
logging.basicConfig(
    filename=experiment_name + ".log",
    level=logging.INFO,
    format="%(asctime)s:%(levelname)s:%(message)s",
)

# run experiment 'iterations' number of times
for i in range(0, iterations):
    iter_name = f"{experiment_name}/iter{i}_"
    logging.info(f"########## Iteration {i} ##########")

    # prepare data
    logging.info("Loading datasets ...")
    data_loader = DataLoader(
        dataset_str,
        dataset_qns_str,
        target_column,
        target_values,
        categorical_features,
        qns_encoding_idx,
        qns_categories,
        gamma,
        train_test_ratio,
        [],
        seed + i,
    )  # seed is consistent in every iteration
    logging.info(f"Test users: {data_loader.test_users}")
    logging.info(f"Features encoded: {data_loader.total_cat_features}")

    user_pmv = {}
    for user in data_loader.test_users:
        df_test = data_loader.df_full_test_encoded
        df_user_test = df_test[df_test["user_id"] == user]

        y_test = df_user_test[target_column]
        y_pred = simplified_pmv_model(df_user_test, dataset_name)

        user_pmv[user], _ = clf_metrics(
            y_test, y_pred, conf_matrix_print=False, scorer=scorer
        )

    save_variable(
        f"{experiment_name}/iter{i}_dict_test_acc_pmv_f1_micro",
        user_pmv,
    )

    logging.info(f"########## Finished iteration {i} ##########")
