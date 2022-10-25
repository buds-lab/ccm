import numpy as np
import pandas as pd

from scipy.spatial.distance import jensenshannon
from utils import load_variable, clf_metrics, train_model


class UserProfile:
    def __init__(
        self,
        feedback_sim="feedback-dist",
        divergence="jensen-shannon",
        qns_sim="cosine",
        model="rdf",
        scorer="f1_micro",
        use_val=True,
        folder_str="",
        precomputed_pcm="True",
        train_users=[],
        dataset_type="sensor_qns",
        dataset_name="cresh",
    ):
        """Load parameters"""
        self.feedback_sim = feedback_sim
        self.divergence = divergence
        self.qns_sim = qns_sim
        self.model = model
        self.scorer = scorer
        self.use_val = use_val
        self.folder_str = folder_str  # e.g., '20201008/iter0_'
        # if PCM for all uses has been calculated before
        self.precomputed_pcm = precomputed_pcm
        self.train_users = train_users
        self.dataset_type = dataset_type
        self.dataset_name = dataset_name

    def feedback_profiling(self, dataframe):
        """Calculates the appropriate user similarity from feedback datapoints"""
        df = dataframe.copy()

        if self.feedback_sim == "feedback-dist":
            # users' feedback datapoints distribution similarity
            df = self.df_affinity(df)

        return df

    def df_affinity(self, dataframe):
        """
        Calculates a square matrix [n x n] where the values are the
        divergence of two users. Identical user means a divergence of 0
        (thus the diagonal is 0).
        The input `dataframe` has the discrete distribution of
        feedback responses as columns and users as rows.
        """
        df = dataframe.copy()
        list_dist_div = []
        user_id_list = df.index.tolist()
        epsilon = 1e-6  # small constant

        # divergence
        for _, p_row in df.iterrows():
            curr_dist_div = []

            for _, q_row in df.iterrows():
                if self.divergence == "jensen-shannon":
                    # this will return a symmetric matrix
                    curr_dist_div.append(jensenshannon(p_row, q_row))
                elif self.divergence == "kl":
                    # if any class is 0, add a small constant
                    if (0.0 in p_row.values) or (0.0 in q_row.values):
                        p_row += epsilon
                        q_row += epsilon
                    curr_dist_div.append(self.kl_divergence(p_row, q_row))
                else:
                    print(f"{self.divergence} is not a valid divergence")

            list_dist_div.append(curr_dist_div)

        df_dist_div = pd.DataFrame(
            list_dist_div, index=user_id_list, columns=user_id_list
        )

        return df_dist_div

    def kl_divergence(self, p, q):
        return np.sum(np.where(p != 0, p * np.log(p / q), 0))

    def cross_model_performance(self, dataframe):
        """
        Calculates the cross-model performance of each user against all
            remaining users

        Args:
            dataframe: A dataframe with all the data and labels
        Returns:
            df_cross_performance: A square dataframe with the cross-model
                performance
            dict_pcm: Dictionary where the keys are `user_id` and the values
                the tuned model
            dict_pcm_acc: Dictionary where the keys are the `user_id` and the
                values is the accuracy the model
        """
        cross_model_perf_list = []
        df = dataframe.copy()
        user_list = df["user_id"].unique()

        # calculating the CV-tuned for each user is what takes most times
        if self.precomputed_pcm:
            dict_pcm_all = load_variable(
                f"models/dict_pcm_{self.model}_{self.dataset_type}_{self.dataset_name}_{self.scorer}"
            )
            dict_pcm_acc_all = {}
            dict_pcm_acc_all["f1_micro"] = load_variable(
                f"models/dict_pcm_acc_{self.model}_{self.dataset_type}_{self.dataset_name}_f1_micro"
            )
            dict_pcm_acc_all["cohen_kappa"] = load_variable(
                f"models/dict_pcm_acc_{self.model}_{self.dataset_type}_{self.dataset_name}_cohen_kappa"
            )

            # filter only the PCMs of train users
            dict_pcm, dict_pcm_acc = {}, {}
            dict_pcm_acc["f1_micro"], dict_pcm_acc["cohen_kappa"] = {}, {}
            for user, _ in dict_pcm_all.items():
                if user in self.train_users:
                    dict_pcm[user] = dict_pcm_all[user]
                    dict_pcm_acc["f1_micro"][user] = dict_pcm_acc_all["f1_micro"][user]
                    dict_pcm_acc["cohen_kappa"][user] = dict_pcm_acc_all["cohen_kappa"][
                        user
                    ]
        else:
            # find pcm for each user
            dict_pcm, dict_pcm_acc = self.find_pcm(df)

        # cross-pcm performance
        for _, curr_model in dict_pcm.items():
            curr_performance = []  # cross-model performance user vs user_list

            for user in user_list:
                df_user = df[df["user_id"] == user]
                df_user = df_user.drop(["user_id", "sex", "height", "weight"], axis=1)

                X = np.array(
                    df_user.iloc[:, 0 : df_user.shape[1] - 1]
                )  # minus 1 for the target column
                y = np.array(df_user.iloc[:, -1])

                y_pred = curr_model.predict(X)

                cross_acc, _ = clf_metrics(
                    y, y_pred, conf_matrix_print=False, scorer=self.scorer
                )  # acc [0, 1]
                curr_performance.append(cross_acc)

            cross_model_perf_list.append(curr_performance)

        # identical users means performance = 1
        df_cross_performance = pd.DataFrame(
            cross_model_perf_list, index=dict_pcm.keys(), columns=user_list
        )

        return df_cross_performance, dict_pcm, dict_pcm_acc

    def find_pcm(self, dataframe, verbose=False):
        """Find the personal comfort model of each user based on CV.
        Assumes a column `user_id` exists.

        Args:
            dataframe: A DataFrame with all data and labels as last column

        Returns:
            user_pcm: Dictionary with the model (value) for each user (key)
            user_pcm_acc: Dictionary with the model accuracy (value) for each
                user (key)
        """
        df = dataframe.copy()
        df = df.drop(
            ["sex", "height", "weight"], axis=1
        )  # drop features that are constant within users
        user_list = df["user_id"].unique()
        if verbose:
            print(
                f"Features used for modeling (`user_id` and the last feature are not used): {df.columns.values}"
            )

        user_pcm = {}
        user_pcm_acc = {}  # hardcoded for two metrics
        user_pcm_acc["f1_micro"] = {}
        user_pcm_acc["cohen_kappa"] = {}

        # for every user, do CV
        for user in user_list:
            df_user = df[df["user_id"] == user]
            df_user = df_user.drop(["user_id"], axis=1)

            fig_name = self.folder_str + str(user)
            model_user, model_user_acc, _ = train_model(
                dataframe=df_user,
                stratified=True,
                model=self.model,
                scorer=self.scorer,
                use_val=self.use_val,
                fig_name=fig_name,
            )
            user_pcm[user] = model_user
            user_pcm_acc["f1_micro"][user] = model_user_acc["f1_micro"]
            user_pcm_acc["cohen_kappa"][user] = model_user_acc["cohen_kappa"]

        return user_pcm, user_pcm_acc
