import warnings
import sys
import logging
from collections import defaultdict
import yaml

from utils import save_variable
from data import DataLoader
from user_profile import UserProfile


def warn(*args, **kwargs):
    pass


warnings.warn = warn

# load config file from CLI
with open(str(sys.argv[1]), "r") as f:
    config = yaml.load(f, Loader=yaml.FullLoader)

# extract parameters from config file
# data
dataset_name = config["dataset_name"]
iterations = config["iterations"]
dataset_str = config["dataset"]
dataset_qns_str = config["dataset_survey"]
dataset_type = config["dataset_type"]
seed = config["seed"]
target_column = config["target_column"]
target_values = config["target_values"]
train_test_ratio = config["train_test_ratio"]
categorical_features = config["categorical_features"]
qns_encoding_idx = config["qns_encoding_idx"]
qns_categories = config["qns_categories"]

# modeling
model = config["model"]
scorer = config["scorer"]
use_val = config["use_val"]

# logging file
logging.basicConfig(
    filename=f"{dataset_name}_pcm.log",
    level=logging.INFO,
    format="%(asctime)s:%(levelname)s:%(message)s",
)

# initiliase data and modeling object
print("Initialising data and model objects ...")
logging.info("Initialising data and model objects ...")

data_loader = DataLoader(
    dataframe_full_str=dataset_str,
    dataframe_qns_str=dataset_qns_str,
    target_column=target_column,
    target_values=target_values,
    categorical_features=categorical_features,
    qns_encoding_idx=qns_encoding_idx,
    qns_categories=qns_categories,
    seed=seed,
)

user_profile = UserProfile(
    model=model,
    scorer=scorer,
    use_val=use_val,
    folder_str="",
    dataset_type=dataset_type,
    dataset_name=dataset_name,
)

# calculate PCM for multiple iterations and save the accuracies and hyperparam of models
metrics = ["f1_micro", "cohen_kappa"]
dict_pcm_list = {}
dict_pcm_acc_list = {}

# empty dictionary of dictionaries
for metric in metrics:
    dict_pcm_list[metric] = defaultdict(list)
    dict_pcm_acc_list[metric] = defaultdict(list)
print(f"PCM training for {dataset_name} using {model} and dataset {dataset_str}")
logging.info(f"PCM training for {dataset_name} using {model} and dataset {dataset_str}")

for i in range(0, iterations):
    print(f"Calculating iteration {i} ...")
    logging.info(f"Calculating iteration {i} ...")

    # for PCM use all the dataset without constant features
    dict_pcm, dict_pcm_acc = user_profile.find_pcm(
        data_loader.df_full_encoded, verbose=True
    )

    # each available scores
    for metric in metrics:
        # append values of each dictionary within the metric
        for user, _ in dict_pcm.items():
            dict_pcm_acc_list[metric][user].append(
                dict_pcm_acc[metric][user]
            )  # CV expected performance
            dict_pcm_list[metric][user].append(dict_pcm[user])  # model

# average of performance for each user
for metric in metrics:
    for user, acc_list in dict_pcm_acc_list[metric].items():
        # finding the model with the best score
        best_model_idx = dict_pcm_acc_list[metric][user].index(
            max(dict_pcm_acc_list[metric][user])
        )
        best_model = dict_pcm_list[metric][user][best_model_idx]
        dict_pcm_list[metric][user] = best_model

        # average metric for each user
        dict_pcm_acc_list[metric][user] = sum(acc_list) / len(acc_list)

    # Save variables
    # e.g.`dict_pcm_{model}_{dataset_type}_{dataset_name}_{scorer}.pkl`
    save_variable(
        f"dict_pcm_{model}_{dataset_type}_{dataset_name}_{metric}",
        dict_pcm_list[metric],
    )
    save_variable(
        f"dict_pcm_acc_{model}_{dataset_type}_{dataset_name}_{metric}",
        dict_pcm_acc_list[metric],
    )
