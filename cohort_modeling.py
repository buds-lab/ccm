import warnings
import os
import sys
import logging
import yaml
from utils import load_variable, save_variable, general_model, general_model_eval
from data import DataLoader
from user_profile import UserProfile
from cohort import Cohort


def warn(*args, **kwargs):
    pass


warnings.warn = warn

# load config file from CLI
with open(str(sys.argv[1]), "r") as f:
    config = yaml.load(f, Loader=yaml.FullLoader)

# extract parameters from config file
# data
dataset_name = config["dataset_name"]
experiment_name = config["name"]
iterations = config["iterations"]
dataset_str = config["dataset"]
dataset_qns_str = config["dataset_survey"]
dataset_type = config["dataset_type"]
seed = config["seed"]
target_column = config["target_column"]
target_values = config["target_values"]
train_test_ratio = config["train_test_ratio"]
test_user_data = config["test_user_data"]
categorical_features = config["categorical_features"]
feedback_sim = config["feedback_sim"]
qns_encoding_idx = config["qns_encoding_idx"]
qns_categories = config["qns_categories"]
gamma = config["gamma"]
# user profiling
divergence = config["divergence"]
qns_sim = config["qns_sim"]
# modeling
model = config["model"]
scorer = config["scorer"]
use_val = config["use_val"]
# cohort
coef = config["coefficients"]
cluster_algo = config["cluster_algo"]
cluster_num = config["cluster_num"]
cluster_assign = config["cluster_assign"]
cluster_predict_criteria = config["cluster_predict_criteria"]
cluster_forced = config["cluster_forced"]
worst_cluster = config["worst_cluster"]
# debugging
precomputed_pcm = config["precomputed_pcm"]
precomputed_test_users = config["precomputed_test_users"]

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
    test_users = (
        load_variable(iter_name + "test_users") if precomputed_test_users else []
    )
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
        test_users,
        seed + i,
    )  # seed is consistent in every iteration
    logging.info(f"Test users: {data_loader.test_users}")
    logging.info(f"Features encoded: {data_loader.total_cat_features}")
    save_variable(f"{iter_name}test_users", data_loader.test_users)

    # forced cluster or automatic clustering
    logging.info("Finding Cohorts ...")
    cohorts = Cohort(
        coef,
        cluster_algo,
        seed,
        model,
        scorer,
        use_val,
        cluster_assign,
        cluster_predict_criteria,
        test_user_data,
        iter_name,
        data_loader.ohe,
        worst_cluster,
    )

    if cluster_forced == "-1":
        logging.info("Automatic clustering ...")

        if cluster_algo in ("kprototype", "kmeans"):
            logging.info(f"Finding cohorts using: {cluster_algo}")

            # elbow plot and finding number of cluster
            cohorts.find_clusters(
                data_loader.df_qns_train_cat_encoded,
                qns_encoding_idx,
                categorical_features,
            )

            # labels for a fix number of cluster (cluster_num)
            if cluster_algo == "kprototype":
                cohorts.kprototype_clustering(
                    cluster_num,
                    data_loader.df_qns_train_cat_encoded,
                    qns_encoding_idx,
                    categorical_features,
                )
            elif cluster_algo == "kmeans":
                cohorts.kmeans_clustering(
                    cluster_num,
                    data_loader.df_qns_train,
                )
            logging.info(f"Current cluster membership: {cohorts.dict_user_cluster}")
        else:
            # feedback profiling
            logging.info("Feedback profiling ...")
            logging.info(f"Feedback similarity: {feedback_sim}")
            user_profile = UserProfile(
                feedback_sim,
                divergence,
                qns_sim,
                model,
                scorer,
                use_val,
                iter_name,
                precomputed_pcm,
                data_loader.train_users,
                dataset_type,
                dataset_name,
            )
            df_feedback_div_train = user_profile.feedback_profiling(
                data_loader.df_dist_train
            )
            df_feedback_div_test = user_profile.feedback_profiling(
                data_loader.df_dist_test
            )

            # cross-model profiling
            logging.info("Cross-model profiling ...")
            logging.info(
                f"Model: {model}, Scorer: {scorer}, Validation-Dev Set: {use_val}"
            )
            (
                df_cross_performance,
                dict_pcm,
                dict_pcm_acc,
            ) = user_profile.cross_model_performance(data_loader.df_full_train_encoded)

            # save the model and accuracies for each iteration (train users)
            save_variable(
                f"{iter_name}df_cross_model_performance_{model}", df_cross_performance
            )
            # save_variable(f'{iter_name}dict_pcm_{model}', dict_pcm) # NOTE: quite heavy to save
            save_variable(
                f"{iter_name}dict_pcm_acc_{model}_f1_micro", dict_pcm_acc["f1_micro"]
            )
            save_variable(
                f"{iter_name}dict_pcm_acc_{model}_cohen_kappa",
                dict_pcm_acc["cohen_kappa"],
            )

            # cohort creation
            logging.info("Clustering ...")
            logging.info(
                f"Coef: Feedback-{coef[0]}, Qns-{coef[1]}, CrossPerf-{coef[2]}"
            )
            df_total_sim = cohorts.add_sim_matrices(
                df_feedback_div_train, df_cross_performance
            )
            cohorts.find_clusters(df_total_sim)
            cohorts.spectral_clustering(cluster_num, df_total_sim)  # num of cohorts

    else:
        # cluster_forced based on a feature
        logging.info(f"Finding cohorts using: {cluster_forced}")
        cohorts.find_forced_cluster(
            cluster_forced, data_loader.df_qns_train_cat_encoded
        )

    # cohort models
    logging.info("Training Cohort models ...")
    cohorts.cluster_models(
        data_loader.df_full_train_encoded, data_loader.df_qns_train_cat_encoded
    )

    save_variable(f"{iter_name}dict_cluster_df", cohorts.dict_cluster_df)
    save_variable(f"{iter_name}dict_cluster_qns", cohorts.dict_cluster_qns)

    # cohort assignation
    logging.info("Cohort assignation ...")
    if cluster_forced != "-1":
        logging.info(f"Matching via {cluster_forced}")
        dict_test_label = cohorts.assign_cluster(data_loader.df_qns_test_cat_encoded)

    elif cluster_algo == "kprototype":
        logging.info(f"Matching via {cluster_algo}")
        dict_test_label = cohorts.assign_cluster(
            data_loader.df_qns_test_cat_encoded, qns_encoding_idx, categorical_features
        )

    elif cluster_assign == "questions":
        logging.info(f"Matching via {cluster_assign}")
        dict_test_label = cohorts.assign_cluster(data_loader.df_qns_test)

    elif cluster_assign in ("performance", "performance-noretrain"):
        logging.info(f"Matching via {cluster_assign}")
        dict_test_label = cohorts.assign_cluster(data_loader.df_full_test_encoded)

    save_variable(f"{iter_name}dict_test_label_{cluster_assign}", dict_test_label)

    # cohort evaluation
    logging.info("Cohort evaluation ...")
    dict_test_acc = cohorts.cluster_predict(
        data_loader.df_full_test_encoded, dict_test_label
    )
    save_variable(
        f"{iter_name}dict_test_acc_{cluster_assign}_f1_micro", dict_test_acc["f1_micro"]
    )
    save_variable(
        f"{iter_name}dict_test_acc_{cluster_assign}_cohen_kappa",
        dict_test_acc["cohen_kappa"],
    )

    # baseline holistic
    logging.info("Baseline holistic ...")
    baseline_model = general_model(
        data_loader.df_full_train_encoded,
        stratified=False,
        model=model,
        scorer=scorer,
        use_val=use_val,
        folder_str=iter_name,
    )
    dict_baseline_acc = general_model_eval(
        data_loader.df_full_test_encoded, baseline_model
    )
    # save variables
    # save_variable(f'{iter_name}baseline_model_{model}', baseline_model) # NOTE: too heavy
    save_variable(
        f"{iter_name}dict_baseline_acc_{model}_f1_micro", dict_baseline_acc["f1_micro"]
    )
    save_variable(
        f"{iter_name}dict_baseline_acc_{model}_cohen_kappa",
        dict_baseline_acc["cohen_kappa"],
    )

    # baseline pcm
    logging.info("Baseline pcm ...")
    # calculating the CV-tuned for each user is what takes most times

    if precomputed_pcm:
        # precomputing all PCM beforehand saves time
        logging.info("Loading precomputed pcm ...")
        dict_pcm_all = load_variable(
            f"models/dict_pcm_{model}_{dataset_type}_{dataset_name}_f1_micro"
        )
        dict_pcm_acc_all = {}
        dict_pcm_acc_all["f1_micro"] = load_variable(
            f"models/dict_pcm_acc_{model}_{dataset_type}_{dataset_name}_f1_micro"
        )
        dict_pcm_acc_all["cohen_kappa"] = load_variable(
            f"models/dict_pcm_acc_{model}_{dataset_type}_{dataset_name}_cohen_kappa"
        )

        # filter only the PCMs of test users (baseline)
        dict_baseline_pcm, dict_baseline_pcm_acc = {}, {}
        dict_baseline_pcm_acc["f1_micro"], dict_baseline_pcm_acc["cohen_kappa"] = {}, {}
        for user, _ in dict_pcm_all.items():
            if user in data_loader.test_users:
                dict_baseline_pcm[user] = dict_pcm_all[user]
                dict_baseline_pcm_acc["f1_micro"][user] = dict_pcm_acc_all["f1_micro"][
                    user
                ]
                dict_baseline_pcm_acc["cohen_kappa"][user] = dict_pcm_acc_all[
                    "cohen_kappa"
                ][user]
    else:
        # find pcm for each user, drop features that are constant within each user
        dict_baseline_pcm, dict_baseline_pcm_acc = user_profile.find_pcm(
            data_loader.df_full_test_encoded
        )

    # save model and accuracies (test users)
    # save_variable(f'{iter_name}dict_baseline_pcm_{model}', dict_baseline_pcm) # NOTE: too heavy
    save_variable(
        f"{iter_name}dict_baseline_pcm_acc_{model}_f1_micro",
        dict_baseline_pcm_acc["f1_micro"],
    )
    save_variable(
        f"{iter_name}dict_baseline_pcm_acc_{model}_cohen_kappa",
        dict_baseline_pcm_acc["cohen_kappa"],
    )

    logging.info(f"########## Finished iteration {i} ##########")
