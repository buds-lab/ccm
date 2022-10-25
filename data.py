import os
import random
import numpy as np
import pandas as pd

from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import OneHotEncoder


class DataLoader:
    def __init__(
        self,
        dataframe_full_str,
        dataframe_qns_str,
        target_column="thermal_cozie",
        target_values=[9.0, 10.0, 11.0],
        categorical_features=[],
        qns_encoding_idx=[],
        qns_categories=[],
        gamma=0.0,
        train_test_ratio=0.8,
        precomputed_list_test_users=[],
        seed=13,
    ):
        """
        Load all default parameters and creates test and train splits.
        Assumes the dataframes have a column named `user_id` and the target
        variable is under the `thermal_cozie` column
        """
        self.df_full = pd.read_csv(dataframe_full_str)
        self.df_qns = pd.read_csv(dataframe_qns_str, index_col="user_id")
        self.target_column = target_column
        self.target_values = target_values.copy()
        self.categorical_features = categorical_features.copy()
        self.qns_encoding_idx = qns_encoding_idx.copy()
        self.qns_categories = qns_categories.copy()
        self.gamma = gamma
        self.train_test_ratio = train_test_ratio
        self.precomputed_list_test_users = precomputed_list_test_users.copy()
        self.seed = seed
        self.ohe = ""
        self.total_cat_features = categorical_features.copy()

        # train test splits
        if not self.precomputed_list_test_users:
            self.train_users, self.test_users = self.participant_train_test_split()
        else:
            self.test_users = self.precomputed_list_test_users
            self.train_users = self.train_precomputed()

        self.df_full_train, self.df_full_test = self.dataframe_train_test_split(
            self.df_full, thermal_label=True
        )
        self.df_qns_train, self.df_qns_test = self.dataframe_train_test_split(
            self.df_qns, user_index=True
        )

        # making sure the label is the last column in the un-split dataframe
        df_full_y = self.df_full.pop(self.target_column)
        self.df_full.loc[:, self.target_column] = df_full_y

        # feedback distribution calculation
        self.df_dist_all = self.feedback_dist(self.df_full)
        self.df_dist_train = self.feedback_dist(self.df_full_train)
        self.df_dist_test = self.feedback_dist(self.df_full_test)

        # feature encoding
        self.dict_encoder = self.cat_encoder()  # only `categorical_features`
        self.dict_encoder_full = self.cat_encoder(
            full=True
        )  # all categorical and qns features
        self.df_full_encoded = self.fit_cat_encoder(self.df_full)
        self.df_full_train_encoded = self.fit_cat_encoder(self.df_full_train)
        self.df_full_test_encoded = self.fit_cat_encoder(self.df_full_test)

        self.df_qns_train_cat_encoded = self.fit_cat_encoder(
            self.df_qns_train, full=True
        )
        self.df_qns_test_cat_encoded = self.fit_cat_encoder(self.df_qns_test, full=True)

        if self.qns_encoding_idx is not None:
            self.ohe = self.onehot_encoder()
            self.df_qns_train_encoded = self.fit_onehot_encoder(self.df_qns_train)
            self.df_qns_test_encoded = self.fit_onehot_encoder(self.df_qns_test)
        else:  # no need to encode
            self.ohe = None
            self.df_qns_train_encoded = self.df_qns_train
            self.df_qns_test_encoded = self.df_qns_test

    def set_seed(self):
        """Set seed"""
        random.seed(self.seed)
        np.random.seed(self.seed)
        os.environ["PYTHONHASHSEED"] = str(self.seed)

    def participant_train_test_split(self):
        """
        Generate train and test split based on the list of participants.
        Assumes the participant id is under a column named `user_id`.
        This means that a participant's data is either in the train set OR test set
        but NOT on both sides.
        """
        self.set_seed()
        df = self.df_full.copy()
        list_participants = df["user_id"].unique()
        random.shuffle(list_participants)

        # randomly choose test_participants
        test_participants = random.sample(
            set(list_participants),
            int(round((1 - self.train_test_ratio) * len(list_participants))),
        )
        train_participants = list(set(list_participants) - set(test_participants))

        return train_participants, test_participants

    def train_precomputed(self):
        """Return train users based on existing list of test users"""
        df = self.df_full.copy()
        list_participants = df["user_id"].unique()

        return list(set(list_participants) - set(self.test_users))

    def dataframe_train_test_split(
        self, dataframe, user_index=False, thermal_label=False
    ):
        """Splits a dataframe according to lists of users"""
        self.set_seed()
        df = dataframe.copy()

        # filter the data
        df_train = (
            df[df["user_id"].isin(self.train_users)]
            if not user_index
            else df[df.index.isin(self.train_users)]
        )
        df_test = (
            df[df["user_id"].isin(self.test_users)]
            if not user_index
            else df[df.index.isin(self.test_users)]
        )

        if thermal_label:
            # move thermal comfort response to the end of the dataframe
            df_train_y = df_train.pop(self.target_column)
            df_test_y = df_test.pop(self.target_column)
            df_train.loc[:, self.target_column] = df_train_y
            df_test.loc[:, self.target_column] = df_test_y

        # shuffle
        if not user_index:  # only shuffle for non-qns dataframes
            df_train = df_train.sample(frac=1, random_state=self.seed).reset_index(
                drop=True
            )
            df_test = df_test.sample(frac=1, random_state=self.seed).reset_index(
                drop=True
            )

        return df_train, df_test

    def feedback_dist(self, dataframe):
        """
        Transforms a dataframe into a new DataFrame where the columns are the
        categorical distribution probabilities of the target column values
        """
        df = dataframe.copy()
        dict_distrib = self.feedback_vector(df)
        df_distrib = self.distribution_feedback_vector(dict_distrib)
        df_distrib = df_distrib.set_index("user_id")

        return df_distrib

    def feedback_vector(self, dataframe):
        """Extracts one feedback vector for each user in `dataframe`"""
        df = dataframe.copy()
        user_id_list, dict_feedback = df["user_id"].unique(), {}

        for user_id in user_id_list:
            dict_feedback[user_id] = (
                df[df["user_id"] == user_id].loc[:, self.target_column].tolist()
            )

        return dict_feedback

    def distribution_feedback_vector(self, dict_feedback):
        """
        Calculates the probability distribution of all `classes` for each user.
        The classes are be sorted. E.g., [9, 10, 11]
        """
        df_distribution = pd.DataFrame()

        # for each user and its feedback
        for _, feedback in dict_feedback.items():
            nb_feedback = len(feedback)
            nb_classes = []

            # counts for each class value
            for label in self.target_values:
                nb_label = feedback.count(label)
                nb_classes.append(nb_label)

            # divide by total number of samples
            normalized_feedback = [x / nb_feedback for x in nb_classes]
            df_distribution = df_distribution.append(
                pd.Series(normalized_feedback), ignore_index=True
            )

        # rename columns to the classes and insert user_ids
        df_distribution.columns = self.target_values
        df_distribution.insert(
            0, "user_id", [*dict_feedback]
        )  # unpack all the keys of the dict (user_ids)

        return df_distribution

    def cat_encoder(self, full=False):
        """Create encode object for list of categorical columns"""
        df = self.df_full.copy()
        df_qns = self.df_qns.copy()
        dict_le = {}

        # encode only `categorical_features`
        for col in self.categorical_features:
            dict_le[col] = LabelEncoder().fit(df[col])

        if full and self.qns_encoding_idx is not None:  # encode qnsfeatures too
            for col in list(self.df_qns.columns[self.qns_encoding_idx]):
                dict_le[col] = LabelEncoder().fit(df_qns[col])

        return dict_le

    def fit_cat_encoder(self, dataframe, full=False):
        """Fit the categorical encoder"""
        df = dataframe.copy()

        total_cat_features = (
            list(self.df_qns.columns[self.qns_encoding_idx]) + self.categorical_features
        )

        if (
            full and self.qns_encoding_idx is not None
        ):  # apply encoding to qns and cat features
            df = df.apply(
                lambda x: self.dict_encoder_full[x.name].transform(x)
                if x.name in total_cat_features
                else x
            )
            self.total_cat_features = total_cat_features
        else:  # apply encoding only to catfeatures
            df = df.apply(
                lambda x: self.dict_encoder[x.name].transform(x)
                if x.name in self.categorical_features
                else x
            )
        return df

    def onehot_encoder(self):
        """Create one-hot encode object for qns columns"""
        cat_columns = self.df_qns.iloc[:, self.qns_encoding_idx]  # encoding columns
        ohe = OneHotEncoder(categories=self.qns_categories, sparse=False)
        ohe = ohe.fit(cat_columns)

        return ohe

    def fit_onehot_encoder(self, dataframe):
        """Fit one-hot encoder"""
        df = dataframe.copy()
        user_id_list = df.index.tolist()

        # remove categorical columns and encode
        df.drop(df.columns[self.qns_encoding_idx], axis=1, inplace=True)
        cat_columns = dataframe.iloc[:, self.qns_encoding_idx]
        cat_encoded = self.ohe.transform(cat_columns)

        # check if there are non-onehot categorical variables
        cat_qns_features = [
            value
            for value in self.categorical_features
            if value in self.df_qns.columns.values
        ]
        if cat_qns_features:  # if the list is not empty
            for cat_col in cat_qns_features:
                df[cat_col] = self.dict_encoder[cat_col].transform(df[cat_col])

        if self.gamma != 0.0:
            noise = np.random.uniform(0, self.gamma, cat_encoded.shape)
            cat_encoded = (cat_encoded + noise) / np.sum(
                cat_encoded + noise, keepdims=True, axis=1
            )

        # concatenate continuous + categorical
        df_concat = pd.DataFrame(
            np.concatenate((df, cat_encoded), axis=1), index=user_id_list
        )

        return df_concat
