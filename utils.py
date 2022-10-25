import pickle
import statistics
from collections import defaultdict
from collections import Counter

import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from matplotlib import rc
from matplotlib.colors import LinearSegmentedColormap
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import make_scorer, f1_score, cohen_kappa_score
from sklearn.metrics import confusion_matrix, classification_report
from sklearn.model_selection import train_test_split
from sklearn.model_selection import cross_val_score, GridSearchCV
from sklearn.model_selection import KFold, StratifiedKFold


def choose_tree_depth(
    clf,
    X,
    y,
    fig_name="",
    scorer="f1_micro",
    stratified=False,
    save_fig=False,
    verbose=False,
):
    """Choose the optimal depth of a tree model"""
    depths = list(range(1, 11))
    cv_scores = []

    if verbose:
        print("Finding optimal tree depth")

    for d in depths:
        if stratified:
            kf = StratifiedKFold(n_splits=5, shuffle=False)
        else:
            kf = KFold(n_splits=5, shuffle=False)

        # keep same params but depth
        clf_depth = clf.set_params(max_depth=d)

        if scorer == "f1_micro":
            scorer = "accuracy"  # accuracy = f1-micro
        elif scorer == "cohen_kappa":
            scorer = make_scorer(cohen_kappa_score)

        scores = cross_val_score(clf_depth, X, y, cv=kf, scoring=scorer)
        cv_scores.append(scores.mean())

    # changing to misclassification error and determining best depth
    error = [1 - x for x in cv_scores]  # error = 1 - scorer
    optimal_depth = depths[error.index(min(error))]

    if save_fig:
        plt.figure(figsize=(12, 10))
        plt.plot(depths, error)
        plt.xlabel("Tree Depth", fontsize=40)
        plt.ylabel("Misclassification Error", fontsize=40)
        plt.savefig(f"{fig_name}_depth.png")
        plt.close()

    if verbose:
        print(
            f"The optimal depth is: {optimal_depth} with error of {min(error)} and score {max(cv_scores)}"
        )

    return optimal_depth, max(cv_scores)


def cv_model_param(
    X, y, model, parameters, scorer="f1_micro", stratified=False, verbose=False
):
    """Choose the best combination of parameters for a given model"""
    kf = (
        StratifiedKFold(n_splits=5, shuffle=True)
        if stratified
        else KFold(n_splits=5, shuffle=True)
    )

    if scorer == "cohen_kappa":
        scorer = make_scorer(cohen_kappa_score)

    grid_search = GridSearchCV(model, parameters, cv=kf, scoring=scorer)
    grid_search.fit(X, y)

    if verbose:
        print(
            f"Best parameters set found on CV set: {grid_search.best_params_} with score of {grid_search.best_score_:.2f}"
        )

    return grid_search.best_estimator_, grid_search.best_score_


def train_model(
    dataframe,
    stratified=False,
    model="rdf",
    scorer="f1_micro",
    use_val=False,
    fig_name="",
):
    """
    Finds best set of param with K-fold CV and returns trained model and accuracy
    Assumes the label is the last column.
    """
    model_acc = {}  # hardcoded to 2 metrics
    model_acc["f1_micro"] = {}
    model_acc["cohen_kappa"] = {}
    class_report = {}
    class_report["f1_micro"] = {}
    class_report["cohen_kappa"] = {}

    # create feature matrix X and target vector y
    X = np.array(
        dataframe.iloc[:, 0 : dataframe.shape[1] - 1]
    )  # minus 1 for the target column
    y = np.array(dataframe.iloc[:, -1]).astype(
        int
    )  # casting in case the original variable was a float

    if model == "rdf":
        parameters = {
            "n_estimators": [100, 300, 500],  # [100, 200, 400, 600],
            "criterion": ["gini"],  # , 'entropy']
            "min_samples_split": [2, 3, 4],
            "min_samples_leaf": [1, 2, 3],
            "class_weight": ["balanced"],  # , 'balanced_subsample']
        }
        clf = RandomForestClassifier(
            random_state=100, warm_start=False
        )  # warm_start=true allows for partial_fit

    # cross-validation
    kf = (
        StratifiedKFold(n_splits=5, shuffle=True)
        if stratified
        else KFold(n_splits=5, shuffle=True)
    )

    if use_val:
        dev_size_percentage = 0.2
        X_cv, X_dev, y_cv, y_dev = train_test_split(
            X, y, test_size=dev_size_percentage, random_state=100
        )  # , stratify=y)
        # find params with f1_micro
        clf_cv, cv_score_f1_micro = cv_model_param(
            X_cv, y_cv, clf, parameters, scorer, stratified
        )
        # and report CV with the other metrics too
        cv_score_cohen_kappa = cross_val_score(
            clf_cv, X_cv, y_cv, cv=kf, scoring=make_scorer(cohen_kappa_score)
        ).mean()
    else:
        # find params with f1_micro
        clf_cv, cv_score_f1_micro = cv_model_param(
            X, y, clf, parameters, scorer, stratified
        )
        # and report CV with the other metrics too
        cv_score_cohen_kappa = cross_val_score(
            clf_cv, X, y, cv=kf, scoring=make_scorer(cohen_kappa_score)
        ).mean()

    # plot depth for rdf and update model
    if model == "rdf":
        # find depth
        optimal_depth, cv_score_f1_micro = (
            choose_tree_depth(clf_cv, X_cv, y_cv, fig_name, "f1_micro", stratified)
            if use_val
            else choose_tree_depth(clf_cv, X, y, fig_name, "f1_micro", stratified)
        )
        clf_cv = clf_cv.set_params(max_depth=optimal_depth)

        # calculate other metrics with new depth
        cv_score_cohen_kappa = cross_val_score(
            clf_cv, X, y, cv=kf, scoring=make_scorer(cohen_kappa_score)
        ).mean()

    # fit the model and get accuracy
    if use_val:
        clf_cv.fit(X_cv, y_cv)
        y_pred = clf_cv.predict(X_dev)
        model_acc["f1_micro"], class_report["f1_micro"] = clf_metrics(
            y_dev, y_pred, conf_matrix_print=False, scorer="f1_micro"
        )
        model_acc["cohen_kappa"], class_report["cohen_kappa"] = clf_metrics(
            y_dev, y_pred, conf_matrix_print=False, scorer="cohen_kappa"
        )

    else:  # no dev_set (use_val=False) average cv_score will be the model_acc
        model_acc["f1_micro"] = cv_score_f1_micro
        model_acc["cohen_kappa"] = cv_score_cohen_kappa
        class_report["f1_micro"] = ""
        class_report["cohen_kappa"] = ""

    return clf_cv, model_acc, class_report


def clf_metrics(test_labels, pred_labels, conf_matrix_print=False, scorer="f1_micro"):
    """Compute the confusion matrix and a particular score based on `scorer`."""
    if scorer == "f1_micro":  # [0, 1]
        metric = f1_score(test_labels, pred_labels, average="micro", zero_division=0)
    elif scorer == "cohen_kappa":  # [-1, 1]
        metric = cohen_kappa_score(test_labels, pred_labels)

    # classification report
    class_report = classification_report(
        test_labels, pred_labels, output_dict=True, zero_division=0
    )

    if conf_matrix_print:
        print(f"Confusion Matrix: \n {confusion_matrix(test_labels, pred_labels)} \n")

    return metric, class_report


def general_model(
    dataframe,
    stratified=True,
    model="rdf",
    scorer="f1_micro",
    use_val=False,
    folder_str="",
):
    """Find a general comfort model aggregating all users' data
    Args:
        dataframe: A DataFrame with all data and labels
        stratified: Boolean to use stratified or not
        model: String of algorithm to train the model
        scorer: String of metric
    Returns:
        tuned_model: trained model
        acc: model accuracy
        report: classification report
    """
    df = dataframe.copy()
    df = df.drop(["user_id"], axis=1)  # general model
    fig_name = folder_str + "general_" + model
    tuned_model, _, _ = train_model(df, stratified, model, scorer, use_val, fig_name)
    return tuned_model


def general_model_eval(dataframe, model="rdf"):
    """
    Evaluate `model` in `dataframe` and returns differen metrics and
    and class report
    """
    df = dataframe.copy()
    user_list = df["user_id"].unique()
    dict_user_acc = {}
    dict_user_acc["f1_micro"] = {}
    dict_user_acc["cohen_kappa"] = {}

    for user in user_list:  # evaluate on each user using all its data
        df_user = df[df["user_id"] == user]
        df_user = df_user.drop(["user_id"], axis=1)

        # create feature matrix X and target vector y
        X = np.array(
            df_user.iloc[:, 0 : df_user.shape[1] - 1]
        )  # minus 1 for the target column
        y = np.array(df_user.iloc[:, -1])

        # predict and get metrics
        y_pred = model.predict(X)

        # save all performance metrics
        dict_user_acc["f1_micro"][user], _ = clf_metrics(
            y, y_pred, conf_matrix_print=False, scorer="f1_micro"
        )
        dict_user_acc["cohen_kappa"][user], _ = clf_metrics(
            y, y_pred, conf_matrix_print=False, scorer="cohen_kappa"
        )

    return dict_user_acc


def compare_results(
    baseline,
    experiment_details,
    iterations=100,
    metric="f1_micro",
    fontsize=40,
    figsize=(16, 12),
    color="darkgrey",
    show_xlabel=True,
    appendix=False,
):
    """Load results for an entire experiment"""
    # experiment information
    cold_name_list = experiment_details["cold"]
    exp_name_list = experiment_details["name_list"]
    plot_name = experiment_details["name"]
    model = experiment_details["model"]
    cluster_assign_list = experiment_details["cluster_assign"]

    baseline_general_acc, baseline_pcm_acc = [], []
    linewidth = 2

    cluster_acc = {}
    for exp in cold_name_list:
        cluster_acc[exp] = []

    for i in range(0, iterations):
        # for general and baseline is enough to load from one experiment
        # general baseline
        for _, value in load_variable(
            f"{baseline}/iter{i}_dict_baseline_acc_{model}_{metric}"
        ).items():
            baseline_general_acc.append(value)
        # pcm baseline
        for _, value in load_variable(
            f"{baseline}/iter{i}_dict_baseline_pcm_acc_{model}_{metric}"
        ).items():
            baseline_pcm_acc.append(value)

        # experiments to compare
        for cold_name, cluster_assign in zip(cold_name_list, cluster_assign_list):
            for _, value in load_variable(
                f"{cold_name}/iter{i}_dict_test_acc_{cluster_assign}_{metric}"
            ).items():
                cluster_acc[cold_name].append(value)

    # figure preparation
    yticklabels = ["General-purpose", "PCM"]
    data = [baseline_general_acc, baseline_pcm_acc]

    for exp_name in exp_name_list:
        yticklabels.append(exp_name)
    for exp_data in cold_name_list:
        data.append(cluster_acc[exp_data])

    if metric == "f1_micro":
        metric_str = "F1-micro"

    # generate plot
    rc("text.latex", preamble=r"\usepackage{cmbright}")
    rc("text", usetex=True)

    figure, _ = plt.subplots(1, 1, figsize=figsize, constrained_layout=True)

    if color == "darkgrey":
        ax = sns.boxplot(data=data, color=color, orient="h", width=0.5, whis=0, fliersize=0, notch=True)
    else:
        sns.set_palette("colorblind")
        ax = sns.boxplot(data=data, orient="h", width=0.5)

    # alpha only to some plots
    for i, patch in zip(range(len(exp_name_list) + 2), ax.artists):
        r, g, b, _ = patch.get_facecolor()
        if i not in [0, 1, 2, 3, 5, 7, 9, 11, 13, 15, 17] and not appendix:
            patch.set_facecolor((r, g, b, 0.3))

    # vertical line at general's median (lower bound baseline)
    plt.vlines(
        statistics.median(data[0]),
        ymin=-0.5,
        ymax=len(data) - 0.5,
        colors="r",
        linestyles="dashed",
        linewidth=linewidth,
    )
    # vertical line at PCM's median (upper bound baseline)
    plt.vlines(
        statistics.median(data[1]),
        ymin=-0.5,
        ymax=len(data) - 0.5,
        colors="r",
        linestyles="dashed",
        linewidth=linewidth,
    )
    # fill between both baseline bounds (general-lower bound, pcm-upper bound)
    plt.fill_between(
        [statistics.median(data[0]), statistics.median(data[1])],
        [-0.5, -0.5],
        [len(data) - 0.5, len(data) - 0.5],
        color="red",
        alpha=0.3,
    )

    # write medians on top of the boxplots' median lines
    medians = [np.median(x) for x in data]
    offset_y = 0.2
    for i in range(len(medians)):
        ax.annotate(
            f"{medians[i]:.2f}",
            xy=(medians[i] + 0.04, i + offset_y),  # for horizontal
            horizontalalignment="center",
            weight="bold",
            color="black",
            fontsize=fontsize - 5,
        )

    if not show_xlabel:
        ax.set_xticklabels([])  # TODO update now that boxplots are horizontal
        ax.set_xlabel([])
    else:
        ax.set_xlabel(f"{metric_str} score", size=fontsize)
    ax.set_ylabel("Cohort approaches", size=fontsize)
    ax.tick_params(length=20, direction="inout", labelsize=fontsize)
    ax.set(yticklabels=yticklabels)
    ax.set(frame_on=False)
    
    plt.tight_layout()
    plt.show()

    figure.savefig(
        f"img/performance_comparison_{plot_name}.png", bbox_inches="tight", format="PNG"
    )

    return


def compare_results_combined(
    baseline_dict,
    experiment_details,
    iterations=100,
    metric="f1_micro",
    fontsize=40,
    figsize=(16, 12),
    height_ratio=[2,1],
    color="darkgrey",
    appendix=False,
):
    """Load results for an entire experiment"""
    # experiment information
    cold_name_dict = experiment_details["cold"]
    exp_name_dict = experiment_details["name_list"]
    plot_name = experiment_details["name"]
    model = experiment_details["model"]
    cluster_assign_dict = experiment_details["cluster_assign"]
    
    if len(baseline_dict.keys()) != 1:
        figure, axes = plt.subplots(
            2, 1, figsize=figsize, gridspec_kw={"height_ratios": height_ratio}
        )
    else:
        figure, axes = plt.subplots(
            1, 1, figsize=figsize
        )
        
    rc("text.latex", preamble=r"\usepackage{cmbright}")
    rc("text", usetex=True)
    linewidth = 2

    j = 0  # axis indices

    for key, values in cold_name_dict.items():
        baseline_general_acc, baseline_pcm_acc = [], []
        cluster_acc = {}

        cold_name_list = cold_name_dict[key]
        cluster_assign_list = cluster_assign_dict[key]
        exp_name_list = exp_name_dict[key]
        baseline = baseline_dict[key]

        for exp in cold_name_list:
            cluster_acc[exp] = []

        for i in range(0, iterations):
            # for general and baseline is enough to load from one experiment
            # general baseline
            for _, value in load_variable(
                f"{baseline}/iter{i}_dict_baseline_acc_{model}_{metric}"
            ).items():
                baseline_general_acc.append(value)
            # pcm baseline
            for _, value in load_variable(
                f"{baseline}/iter{i}_dict_baseline_pcm_acc_{model}_{metric}"
            ).items():
                baseline_pcm_acc.append(value)

            # experiments to compare
            for cold_name, cluster_assign in zip(cold_name_list, cluster_assign_list):
                for _, value in load_variable(
                    f"{cold_name}/iter{i}_dict_test_acc_{cluster_assign}_{metric}"
                ).items():
                    cluster_acc[cold_name].append(value)

        # figure preparation
        yticklabels = ["General-purpose", "PCM"]
        data = [baseline_general_acc, baseline_pcm_acc]

        for exp_name in exp_name_list:
            yticklabels.append(exp_name)
        for exp_data in cold_name_list:
            data.append(cluster_acc[exp_data])

        if metric == "f1_micro":
            metric_str = "F1-micro"

        if color == "darkgrey":
            if len(baseline_dict.keys()) != 1:
                sns.boxplot(ax=axes[j], data=data, color=color, orient="h", width=0.5, whis=0, fliersize=0, notch=True)
            else:
                sns.boxplot(ax=axes, data=data, color=color, orient="h", width=0.5, whis=0, fliersize=0, notch=True)
        elif color == "colorful":
            if j == 0:
                color_list = [
                    '#00429d', 
                    '#2a56a6', 
                    '#416baf',
                    '#5481b8',
                    '#5481b8',
                    '#6697bf',
                    '#6697bf',
                    '#a6a6a6',
                    '#a6a6a6',
                    '#ffcab9',
                    '#ffcab9',
                    '#fd9291', 
                    '#fd9291',
                    '#e75d6f',
                    '#e75d6f',
                    '#c52a52',
                    '#c52a52',
                    '#93003a',
                    '#93003a'
                ]
            else:
                color = [
                    '#00429d', 
                    '#2a56a6', 
                    '#416baf',
                    '#5481b8',
                    '#5481b8',
                    '#c52a52',
                    '#c52a52',
                    '#93003a',
                    '#93003a'
                ]

            if len(baseline_dict.keys()) != 1:
                sns.boxplot(ax=axes[j], data=data, palette=color_list, orient="h", width=0.5)
            else:
                sns.boxplot(ax=axes, data=data, palette=color_list, orient="h", width=0.5)
        
        # alpha only to some plots
        if len(baseline_dict.keys()) != 1:
            for i, patch in zip(range(len(exp_name_list) + 2), axes[j].artists):
                r, g, b, _ = patch.get_facecolor()
                if i not in [0, 1, 2, 3, 5, 7, 9, 11, 13, 15, 17] and not appendix:
                    patch.set_facecolor((r, g, b, 0.3))

        if len(baseline_dict.keys()) != 1:
            # no xticks for first subplot
            axes[0].set_xticks([])
            # vertical line at general's median (lower bound baseline)
            axes[j].vlines(
                statistics.median(data[0]),
                ymin=-0.5,
                ymax=len(data) - 0.5,
                colors="r",
                linestyles="dashed",
                linewidth=linewidth,
            )
            # vertical line at PCM's median (upper bound baseline)
            axes[j].vlines(
                statistics.median(data[1]),
                ymin=-0.5,
                ymax=len(data) - 0.5,
                colors="r",
                linestyles="dashed",
                linewidth=linewidth,
            )
            # fill between both baseline bounds (general-lower bound, pcm-upper bound)
            axes[j].fill_between(
                [statistics.median(data[0]), statistics.median(data[1])],
                [-0.5, -0.5],
                [len(data) - 0.5, len(data) - 0.5],
                color="red",
                alpha=0.3,
            )
            
            # fill for cold start and warm start
#             if j == 0:
#                 # cold start
#                 axes[j].fill_between(
#                     [0, 1],
#                     [3-0.5, 3-0.5],
#                     [14+0.5, 14+0.5],
#                     color="#212529",
#                     alpha=0.3,
#                 )
#                 # warm start
#                 axes[j].fill_between(
#                     [0, 1],
#                     [15-0.5, 15-0.5],
#                     [18+0.5, 18+0.5],
#                     color="#adb5bd",
#                     alpha=0.3,
#                 )
#             else:
#                 # cold start
#                 axes[j].fill_between(
#                     [0, 1],
#                     [3-0.5, 3-0.5],
#                     [4+0.5, 4+0.5],
#                     color="#212529",
#                     alpha=0.3,
#                 )
#                 # warm start
#                 axes[j].fill_between(
#                     [0, 1],
#                     [5-0.5, 5-0.5],
#                     [8+0.5, 8+0.5],
#                     color="#adb5bd",
#                     alpha=0.3,
#                 )

            # write medians on top of the boxplots' median lines
            medians = [np.median(x) for x in data]
            offset_y = 0.2
#             offset_y = 0.2 if len(medians) > 10 else 0.2
            for i in range(len(medians)):
                axes[j].annotate(
                    f"{medians[i]:.2f}",
                    #             xy =(i, medians[i]), # for vertical
                    xy=(medians[i] + 0.04, i + offset_y),  # for horizontal
                    horizontalalignment="center",
                    weight="bold",
                    color="black",
                    fontsize=fontsize - 5,
                )

            axes[j].tick_params(length=20, direction="inout", labelsize=fontsize)
            axes[j].set(yticklabels=yticklabels)
            if j == 1:
                # SMC
                axes[j].set_ylabel(f"{key}\n", size=fontsize, labelpad=75)
            else:
                # Dorn
                axes[j].set_ylabel(f"{key}\n", size=fontsize)
            
            # no frame around the plots
            axes[j].set(frame_on=False)

        else:
            # vertical line at general's median (lower bound baseline)
            axes.vlines(
                statistics.median(data[0]),
                ymin=-0.5,
                ymax=len(data) - 0.5,
                colors="r",
                linestyles="dashed",
                linewidth=linewidth,
            )
            # vertical line at PCM's median (upper bound baseline)
            axes.vlines(
                statistics.median(data[1]),
                ymin=-0.5,
                ymax=len(data) - 0.5,
                colors="r",
                linestyles="dashed",
                linewidth=linewidth,
            )
            # fill between both baseline bounds (general-lower bound, pcm-upper bound)
            axes.fill_between(
                [statistics.median(data[0]), statistics.median(data[1])],
                [-0.5, -0.5],
                [len(data) - 0.5, len(data) - 0.5],
                color="red",
                alpha=0.3,
            )

            # write medians on top of the boxplots' median lines
            medians = [np.median(x) for x in data]
            offset_y = 0.3 if len(medians) > 10 else 0.2
            for i in range(len(medians)):
                axes.annotate(
                    f"{medians[i]:.2f}",
                    #             xy =(i, medians[i]), # for vertical
                    xy=(medians[i], i + offset_y),  # for horizontal
                    horizontalalignment="center",
                    weight="bold",
                    color="black",
                    fontsize=fontsize - 5,
                )

            axes.tick_params(length=20, direction="inout", labelsize=fontsize)
            axes.set(yticklabels=yticklabels)
            if len(baseline_dict.keys()) != 1:
                axes.set_ylabel(f"{key}", size=fontsize)
        
        j += 1
        # end dictionary and subplot loop
        
    if len(baseline_dict.keys()) != 1:
        figure.text(
            -0.02,
            0.5,
            "Cohort approaches for each dataset",
            size=fontsize + 10,
            va="center",
            rotation="vertical",
        )
        figure.subplots_adjust(hspace=0)

    plt.xlabel(f"{metric_str} score", size=fontsize)
    plt.tight_layout()
    plt.show()

    figure.savefig(
        f"img/performance_comparison_{plot_name}.png", bbox_inches="tight", format="PNG"
    )

    return


def change_user_score(
    experiment_details,
    metric="f1_micro",
    calc="pcg",  # pcg or delta
    iterations=100,
    fontsize=40,
    figsize=(16, 16),
    fig_name="test",
):
    """user-specific percentual changes in performance metric"""

    linewidth = 2

    # experiment information
    name = experiment_details["name"]
    dataset = experiment_details["dataset"]
    model = experiment_details["model"]
    cluster_assign = experiment_details["cluster_assign"]

    # {user: list(percentual change)}
    dict_user_baseline, dict_user_cohort, dict_user_delta = (
        {},
        {},
        defaultdict(list),
    )

    # load user metric for all iterations
    for i in range(0, iterations):
        # baseline
        for user, value in load_variable(
            f"{name}/iter{i}_dict_baseline_acc_{model}_{metric}"
        ).items():
            dict_user_baseline[user] = value
        # cohort performance
        for user, value in load_variable(
            f"{name}/iter{i}_dict_test_acc_{cluster_assign}_{metric}"
        ).items():
            dict_user_cohort[user] = value
        # calculate pcg increase/decrease for each user on this iteration and append it
        for user, _ in dict_user_baseline.items():
            # pcg = (new - old)/old * 100, but values are already [0,1]
            # pcg = (new - old)/old
            if calc == "pcg":
                pcg = (
                    dict_user_cohort[user] - dict_user_baseline[user]
                ) / dict_user_baseline[user]
            elif calc == "delta":
                pcg = dict_user_cohort[user] - dict_user_baseline[user]
            else:
                print("ERROR, no calculation was specified")
            dict_user_delta[user].append(pcg)
    # end iteration loop

    # sort users
    for user, _ in dict_user_baseline.items():  # a diff dict with same keys
        dict_user_delta[int(user.replace(dataset, ""))] = dict_user_delta.pop(user)
    dict_items = dict_user_delta.items()
    sorted_items = sorted(dict_items)
    dict_user_delta_sorted = dict(sorted_items)

    # generate plot
    rc("text.latex", preamble=r"\usepackage{cmbright}")
    rc("text", usetex=True)

    figure, _ = plt.subplots(1, 1, figsize=figsize)

    ax = sns.boxplot(
        data=list(dict_user_delta_sorted.values()),
        orient="h",
        color="darkgrey",
        width=0.5,
    )

    # vertical line at 0
    plt.vlines(
        0,
        ymin=-0.5,
        ymax=len(dict_user_delta_sorted.values()) - 0.5,
        colors="r",
        linestyles="dashed",
        linewidth=linewidth,
    )

    if metric == "f1_micro":
        metric_str = "F1-micro"

    yticklabels = [f"{dataset}{key}" for key in dict_user_delta_sorted.keys()]
    ax.tick_params(length=20, direction="inout", labelsize=fontsize)
    ax.set_ylabel("User ID", size=fontsize)
    xlabel_str = (
        f"Change in {metric_str}"
        if calc == "delta"
        else f"Percentage [\%] change in {metric_str}"
    )
    ax.set_xlabel(xlabel_str, size=fontsize)
    xlim = (-1, 1) if calc == "delta" else (-5, 110)
    ax.set(yticklabels=yticklabels, xlim=xlim)

    plt.tight_layout()
    plt.show()

    figure.savefig(
        f"img/change_dist-{dataset}-{fig_name}.png",
        bbox_inches="tight",
        format="PNG",
    )


def change_user_score_combined(
    experiment_details,
    metric="f1_micro",
    calc="pcg",  # pcg or delta
    iterations=100,
    fontsize=40,
    figsize=(16, 16),
    fig_name="test",
):
    """user-specific changes in performance metric"""

    # experiment information
    model = experiment_details["model"]
    name_list = experiment_details["name"]
    dataset_list = experiment_details["dataset"]
    cluster_assign_list = experiment_details["cluster_assign"]
    ylabel_list = experiment_details["ylabel"]

    ratios = [1, 1, 0.5, 1] if len(name_list) > 2 else [1, 1]

    figure, axes = plt.subplots(
        len(name_list),
        1,
        figsize=figsize,
        sharex=True,
        gridspec_kw={"height_ratios": ratios},
    )

    rc("text.latex", preamble=r"\usepackage{cmbright}")
    rc("text", usetex=True)
    linewidth = 2

    for j, _ in enumerate(name_list):
        # {user: list(percentual change)}
        dict_user_baseline, dict_user_cohort, dict_user_delta = (
            {},
            {},
            defaultdict(list),
        )

        # load user metric for all iterations
        for i in range(0, iterations):
            # baseline
            for user, value in load_variable(
                f"{name_list[j]}/iter{i}_dict_baseline_acc_{model}_{metric}"
            ).items():
                dict_user_baseline[user] = value
            # cohort performance
            for user, value in load_variable(
                f"{name_list[j]}/iter{i}_dict_test_acc_{cluster_assign_list[j]}_{metric}"
            ).items():
                dict_user_cohort[user] = value
            # calculate pcg increase/decrease for each user on this iteration and append it
            for user, _ in dict_user_baseline.items():
                # pcg = (new - old)/old * 100, but values are already [0,1]
                # pcg = (new - old)/old
                if calc == "pcg":
                    pcg = (
                        dict_user_cohort[user] - dict_user_baseline[user]
                    ) / dict_user_baseline[user]
                elif calc == "delta":
                    pcg = dict_user_cohort[user] - dict_user_baseline[user]
                else:
                    print("ERROR, no calculation was specified")
                dict_user_delta[user].append(pcg)
        # end iteration loop

        # sort users
        for user, _ in dict_user_baseline.items():  # a diff dict with same keys
            dict_user_delta[
                int(user.replace(dataset_list[j], ""))
            ] = dict_user_delta.pop(user)
        dict_items = dict_user_delta.items()
        sorted_items = sorted(dict_items)
        dict_user_delta_sorted = dict(sorted_items)

        sns.boxplot(
            ax=axes[j],
            data=list(dict_user_delta_sorted.values()),
            orient="h",
            color="darkgrey",
            width=0.5,
        )

        # vertical line at 0
        axes[j].vlines(
            0,
            ymin=-0.5,
            ymax=len(dict_user_delta_sorted.values()) - 0.5,
            colors="r",
            linestyles="dashed",
            linewidth=linewidth,
        )

        if metric == "f1_micro":
            metric_str = "F1-micro"

        yticklabels = [
            f"{dataset_list[j]}{key}" for key in dict_user_delta_sorted.keys()
        ]
        axes[j].tick_params(length=20, direction="inout", labelsize=fontsize)
        axes[j].set_ylabel(ylabel_list[j], size=fontsize)
        xlabel_str = (
            f"Change in {metric_str}"
            if calc == "delta"
            else f"Percentage [%] change in {metric_str}"
        )

        xlim = (-1, 1) if calc == "delta" else (-5, 115)
        axes[j].set(yticklabels=yticklabels, xlim=xlim)

    # end for loop of subplots

    axes[j].set_xlabel(xlabel_str, size=fontsize)
    figure.text(
        -0.02,
        0.5,
        "Cohort approach and user ID",
        size=fontsize + 10,
        va="center",
        rotation="vertical",
    )
    figure.subplots_adjust(hspace=0)
    plt.tight_layout()
    plt.show()

    figure.savefig(
        f"img/change_dist-all_datasets-{fig_name}.png",
        bbox_inches="tight",
        format="PNG",
    )


def change_user_score_all(
    experiment_details,
    metric="f1_micro",
    calc="pcg",  # pcg or delta
    bins=20,
    iterations=100,
    fontsize=40,
    figsize=(16, 6),
    fig_name="test",
):
    """user-specific changes in performance metric"""

    # experiment information
    model = experiment_details["model"]
    exp_list = experiment_details["exp_list"]
    exp_str_list = experiment_details["name_list"]
    assign_list = experiment_details["cluster_assign"]

    # {user: list(percentual change)}
    dict_user_delta = {}

    # load cohort approaches for each datasets
    for dataset in exp_list.keys():
        # use idx delimiting the files for each dataset
        for exp, exp_str, assign in zip(
            exp_list[dataset], exp_str_list[dataset], assign_list[dataset]
        ):
            if len(exp_list.keys()) == 1:
                dict_user_delta[f"{exp_str}"] = defaultdict(list)
            else:
                dict_user_delta[f"{dataset}:{exp_str}"] = defaultdict(list)
            
            dict_user_baseline = defaultdict(list)  # general-purpose model
            dict_user_cluster = defaultdict(list)  # cohort approach

            for i in range(0, iterations):
                # baseline
                for user, value in load_variable(
                    f"{exp}/iter{i}_dict_baseline_acc_{model}_{metric}"
                ).items():
                    dict_user_baseline[user] = value

                # cohort approach performance
                for user, value in load_variable(
                    f"{exp}/iter{i}_dict_test_acc_{assign}_{metric}"
                ).items():
                    dict_user_cluster[user] = value
            # end iterations loop

            # calculate delta for each user on this iteration and append it
            for user in dict_user_baseline.keys():
                # delta = (cohort - baseline)/baseline
                if calc == "pcg":
                    pcg = (
                        dict_user_cluster[user] - dict_user_baseline[user]
                    ) / dict_user_baseline[user]

                elif calc == "delta":
                    pcg = dict_user_cluster[user] - dict_user_baseline[user]
                
                if len(exp_list.keys()) == 1:
                    dict_user_delta[f"{exp_str}"][user].append(pcg)
                else:
                    dict_user_delta[f"{dataset}:{exp_str}"][user].append(pcg)
            # calculate the average delta for each user
            for user in dict_user_baseline.keys():
                if len(exp_list.keys()) == 1:
                    dict_user_delta[f"{exp_str}"][user] = sum(
                        dict_user_delta[f"{exp_str}"][user]
                    ) / len(dict_user_delta[f"{exp_str}"][user])
                else:
                    dict_user_delta[f"{dataset}:{exp_str}"][user] = sum(
                        dict_user_delta[f"{dataset}:{exp_str}"][user]
                    ) / len(dict_user_delta[f"{dataset}:{exp_str}"][user])

        # end approaches loop
    # end dataset loop

    # prepare dataframe
    df_user_delta = pd.DataFrame.from_dict(dict_user_delta)
    df_user_delta = df_user_delta.stack().reset_index()
    df_user_delta.columns = ["user_id", "cohorts", "avg"]
    print(df_user_delta)
    print(df_user_delta.groupby("cohorts").agg({"avg": ["mean", "min", "max"]}))
    
    if bins != 0:
        df_user_delta = df_user_delta.mask(df_user_delta.applymap(str).eq("[]"))
        df_user_delta["avg_range"] = pd.cut(
            df_user_delta["avg"], bins=bins, include_lowest=True
        )
        print(df_user_delta)

        df_user_delta["avg_common"] = df_user_delta["avg_range"].apply(lambda x: x.mid)
        df_count = (
            df_user_delta.groupby(["cohorts", "avg_range"])
            .size()
            .reset_index(name="Occupants")
        )
        df_user_delta = pd.merge(df_user_delta, df_count, on=["cohorts", "avg_range"])

        print(df_user_delta)

        sizes = df_user_delta["Occupants"].unique()

        print(sizes)

    # plot
    rc("text.latex", preamble=r"\usepackage{cmbright}")
    rc("text", usetex=True)

    if metric == "f1_micro":
        metric_str = "F1-micro"

    # hardcoding order
    if len(exp_list.keys()) == 1:
        order = [
            "Cold start",
            "Warm start",
        ]
    else:
        order = [
            "Dorn:Dist-Cross",
            "Dorn:Cross",
            "Dorn:Life Satisfaction",
            "Dorn:Agreeableness",
            "SMC:Dist-Cross",
            "SMC:Cross",
        ]
    df_user_delta["cohorts"] = pd.Categorical(df_user_delta["cohorts"], order)
    df_user_delta["cohorts"] = df_user_delta["cohorts"].replace({
        "Dorn:Dist-Cross": "Dorn: Dist-Cross",
        "Dorn:Cross": "Dorn: Cross",
        "Dorn:Life Satisfaction": "Dorn: Life Satisfaction",
        "Dorn:Agreeableness": "Dorn: Agreeableness",
        "SMC:Dist-Cross": "SMC: Dist-Cross",
        "SMC:Cross": "SMC: Cross",
        }
    )
    
    df_user_delta = df_user_delta.sort_values(by=["cohorts"])

    figure, _ = plt.subplots(1, 1, figsize=figsize)

    if bins != 0:
        ax = sns.scatterplot(
            data=df_user_delta,
            x="avg_common",
            y="cohorts",
            size="Occupants",
            alpha=1,
            sizes={
                1: 100,
                2: 200,
                3: 300,
                4: 400,
                5: 500,
                6: 600,
                7: 700,
                8: 800,
                9: 900,
                10: 1000,
                15: 1500,
                17: 1700,
                18: 1800,
                19: 1900,
                20: 2000,
                21: 2100,
                23: 2300,
                24: 2400,
                25: 2500,
                30: 3000,
                31: 3100,
                32: 3200,
            },
            size_order=[1, 5, 20, 30],
        )
    else:
        ax = sns.scatterplot(
            data=df_user_delta, x="avg", y="cohorts", alpha=0.8, s=2000
        )
    plt.vlines(
        0,
        ymin=-0.5,
        ymax=len([j for i in exp_list.values() for j in i]) - 0.5,
        colors="k",
        linestyles="dashed",
        linewidth=2,
    )
    xlabel_str = (
        f"Average change in {metric_str}"
        if calc == "delta"
        else f"Average percentage [\%] change in {metric_str}"
    )
    ax.tick_params(length=20, direction="inout", labelsize=fontsize)
    ax.set_ylabel("Dataset:Cohort approaches", size=fontsize)
    ax.set_xlabel(xlabel_str, size=fontsize)
    if bins != 0:
        ax.legend(
            title=r"\# Occupants",
            title_fontsize=fontsize,
            fontsize=fontsize,
            frameon=False,
            bbox_to_anchor=(1, 0.75),
            ncol=1,
        )
    ax.set(frame_on=False)

    print(df_user_delta)

    # write text of improved users
    for cohort in df_user_delta["cohorts"].unique():
        print(f"For cohort approach {cohort}")      
        
        # numbers users that increased ( > 0 )
        print(f"Increased ---")
        val = df_user_delta[df_user_delta["cohorts"] == cohort]["avg"].values
        textstr_improved = (
            f"{sum(i > 0 for i in val)/len(val)*100 :.2f}% of participants improved"
        )
        print(textstr_improved)
        # values increased
        val_improved = df_user_delta[
            (df_user_delta["cohorts"] == cohort) & (df_user_delta["avg"] > 0)
        ]["avg"].values
        print(
            f"{sum(val_improved)/len(val_improved) :.2f}% averaged improved score"
        )
        print(f"{max(val_improved) :.2f}% max averaged improved score")
        print(f"{min(val_improved) :.2f}% min averaged improved score")
        
        # overall values
        print(f"Overall ---")
        print(
            f"{sum(val)/len(val) :.2f}% overall averaged score"
        )
        print(f"{max(val) :.2f}% max overall score")
        print(f"{min(val) :.2f}% min overall score")
        
        # numbers users that decreased ( < 0 )
        print(f"Decreased ---")
        textstr_decreased = (
            f"{sum(i < 0 for i in val)/len(val)*100 :.2f}% of participants decreased"
        )
        print(textstr_decreased)
        # values decreased
        val_decreased = df_user_delta[
            (df_user_delta["cohorts"] == cohort) & (df_user_delta["avg"] < 0)
        ]["avg"].values
        print(
            f"{sum(val_decreased)/len(val_decreased) :.2f}% averaged decreased score"
        )
        print(f"{max(val_decreased) :.2f}% min averaged decreased score")
        print(f"{min(val_decreased) :.2f}% max averaged decreased score")
        print("\n")
        
    plt.tight_layout()
    plt.show()
    figure.savefig(f"img/change-user-{model}_{metric}_{fig_name}.png")
    
    color_list = [
        '#c52a52', # dist cross
        '#93003a', # cross
        '#ffcab9', # life sat
        '#e75d6f', # agree
        '#c52a52', # dist cross
        '#93003a', # cross
    ]
    # box plot
    figure, _ = plt.subplots(1, 1, figsize=figsize)
    ax = sns.boxplot(
        data=df_user_delta, 
        x="avg", 
        y="cohorts",
        orient="h", 
        width=0.5,
#         color="darkgrey"
        palette=color_list
    )

    ax = sns.swarmplot(
        data=df_user_delta, 
        x="avg", 
        y="cohorts", 
        color=".25",
        size=10
    )
    
    plt.vlines(
        0,
        ymin=-0.5,
        ymax=len([j for i in exp_list.values() for j in i]) - 0.5,
        colors="k",
        linestyles="dashed",
        linewidth=2,
    )
    xlabel_str = (
        f"Average change in {metric_str}"
        if calc == "delta"
        else f"Average percentage [\%] change in {metric_str}"
    )
    ax.tick_params(length=20, direction="inout", labelsize=fontsize)
#     ax.set_ylabel("Dataset:Cohort approaches", size=fontsize)
    ax.set_ylabel("", size=fontsize)
    ax.set_xlabel(xlabel_str, size=fontsize)
    
    plt.tight_layout()
    plt.show()
    figure.savefig(f"img/change-user-violin-{model}_{metric}_{fig_name}.png")

    
def vote_by_user(
    dataframe,
    dataset="dorn",
    show_percentages=False,
    preference_label="thermal_cozie",
    fontsize=40,
):
    """
    Original code by Dr. Federico Tartarini
    https://github.com/FedericoTartarini
    """

    df = dataframe.copy()
    df[preference_label] = df[preference_label].map(
        {9.0: "Warmer", 10.0: "No Change", 11.0: "Cooler"}
    )
    _df = (
        df.groupby(["user_id", preference_label])[preference_label]
        .count()
        .unstack(preference_label)
    )
    _df.reset_index(inplace=True)

    df_total = _df.sum(axis=1)
    df_rel = _df[_df.columns[1:]].div(df_total, 0) * 100
    df_rel["user_id"] = _df["user_id"]

    # sort properly
    df_rel["user_id"] = df_rel["user_id"].str.replace(dataset, "").astype(int)
    df_rel = df_rel.sort_values(by=["user_id"], ascending=False)
    df_rel["user_id"] = dataset + df_rel["user_id"].astype(str)
    df_rel = df_rel.reset_index(drop=True)

    # plot a Stacked Bar Chart using matplotlib
    rc("text.latex", preamble=r"\usepackage{cmbright}")
    rc("text", usetex=True)

    df_rel.plot(
        x="user_id",
        kind="barh",
        stacked=True,
        mark_right=True,
        cmap=LinearSegmentedColormap.from_list(
            preference_label,
            [
                "tab:blue",
                "tab:green",
                "tab:red",
            ],
            N=3,
        ),
        width=0.95,
        figsize=(16, 16),
    )

    plt.legend(
        bbox_to_anchor=(0.5, 1.02),
        loc="center",
        borderaxespad=0,
        ncol=3,
        frameon=False,
        fontsize=fontsize,
    )
    sns.despine(left=True, bottom=True, right=True, top=True)

    plt.tick_params(labelsize=fontsize * 0.75)
    plt.xlabel(r"Percentage [\%]", size=fontsize)
    plt.ylabel("User ID", size=fontsize)

    if show_percentages:
        # add percentages
        for index, row in df_rel.drop(["user_id"], axis=1).iterrows():
            cum_sum = 0
            for ix, el in enumerate(row):
                if ix == 1:
                    plt.text(
                        cum_sum + el / 2 if not np.isnan(cum_sum) else el / 2,
                        index,
                        str(int(np.round(el, 0))) + "\%",
                        va="center",
                        ha="center",
                        size=fontsize * 0.6,
                    )
                cum_sum += el

    plt.tight_layout()
    plt.savefig(f"img/{dataset}_vote_dist.png", pad_inches=0, dpi=300)
    plt.show()


def compare_clusters(
    folder_name_list,
    plot_name_list,
    iterations=100,
    metric="SSI",
    k_range=range(2, 11),
    ylim=(-0.2, 0.6),
    fontsize=40,
    figsize=(16, 12),
):
    """Plots the average elbow plot for `metric` for multiple experiments"""
    folder_metrics = {}
    plot_title = ""
    name_i = 0
    figure, axis = plt.subplots(1, 1, figsize=figsize)

    rc("text.latex", preamble=r"\usepackage{cmbright}")
    rc("text", usetex=True)

    for folder_name in folder_name_list:
        metric_list = [0] * len(k_range)

        for i in range(0, iterations):  # load metrics for each iteration
            iter_metrics = load_variable(f"{folder_name}/iter{i}_cluster_metrics")
            metric_list = [a + b for a, b in zip(metric_list, iter_metrics[metric])]

        # check if the original files are in a subfolder
        if "/" in folder_name:
            pos = folder_name.find("/")  # same pos for both strings
            plot_title = f"{plot_title}-{folder_name[pos+1:]}"
        else:
            plot_title = f"{plot_title}-{folder_name}"

        # normalised by the number of iterations used
        folder_metrics = {folder_name: [number / iterations for number in metric_list]}
        metric_str = "Silhouette Score" if metric == "SSI" else metric
        # plot
        axis.plot(
            k_range,
            folder_metrics[folder_name],
            label=plot_name_list[name_i],
            linewidth=4,
        )
        axis.set_ylim(ylim)
        axis.set_xlabel(r"\textit{k}, number of cohorts", fontsize=fontsize)
        axis.set_ylabel(metric_str, fontsize=fontsize)
        axis.tick_params(length=10, direction="inout", labelsize=fontsize)
        axis.legend(fontsize=fontsize, ncol=2, frameon=False)

        name_i += 1

    figure.tight_layout()
    figure.savefig(
        f"img/{plot_title}_cluster_metrics.png", bbox_inches="tight", format="PNG"
    )

def user_increased_metadata(
    experiment_details,
    dict_metadata,
    metric="f1_micro",
    iterations=100,
    fontsize=40,
    figsize=(16, 6),
    fig_name="test",
):
    # experiment information
    model = experiment_details["model"]
    exp_list = experiment_details["exp_list"]
    exp_str_list = experiment_details["name_list"]
    assign_list = experiment_details["cluster_assign"]

    # {user: list(percentual change)}
    dict_user_delta = {}
    dict_user_delta_sex = {}
    dict_user_delta_height = {}
    dict_user_delta_weight = {}

    dict_user_delta_sex_worse = {}
    dict_user_delta_height_worse = {}
    dict_user_delta_weight_worse = {}

    
    # load cohort approaches for each datasets
    for dataset in exp_list.keys():
        # use idx delimiting the files for each dataset
        for exp, exp_str, assign in zip(
            exp_list[dataset], exp_str_list[dataset], assign_list[dataset]
        ):
          
            dict_user_delta[f"{dataset}:{exp_str}"] = defaultdict(list)
            dict_user_baseline = defaultdict(list)  # general-purpose model
            dict_user_cluster = defaultdict(list)  # cohort approach
            dict_user_delta_sex[f"{dataset}:{exp_str}"] = {}
            dict_user_delta_height[f"{dataset}:{exp_str}"] = {}
            dict_user_delta_weight[f"{dataset}:{exp_str}"] = {}
            dict_user_delta_sex_worse[f"{dataset}:{exp_str}"] = {}
            dict_user_delta_height_worse[f"{dataset}:{exp_str}"] = {}
            dict_user_delta_weight_worse[f"{dataset}:{exp_str}"] = {}
            
            for i in range(0, iterations):
                # baseline
                for user, value in load_variable(
                    f"{exp}/iter{i}_dict_baseline_acc_{model}_{metric}"
                ).items():
                    dict_user_baseline[user] = value

                # cohort approach performance
                for user, value in load_variable(
                    f"{exp}/iter{i}_dict_test_acc_{assign}_{metric}"
                ).items():
                    dict_user_cluster[user] = value
            # end iterations loop

            # calculate delta for each user on this iteration and append it
            for user in dict_user_baseline.keys():
                # delta = (cohort - baseline)/baseline
                pcg = (
                    dict_user_cluster[user] - dict_user_baseline[user]
                ) / dict_user_baseline[user]

                dict_user_delta[f"{dataset}:{exp_str}"][user].append(pcg)

            # calculate the average delta for each user
            for user in dict_user_baseline.keys():
                dict_user_delta[f"{dataset}:{exp_str}"][user] = sum(
                    dict_user_delta[f"{dataset}:{exp_str}"][user]
                    ) / len(dict_user_delta[f"{dataset}:{exp_str}"][user])
                
                # get metadata of users who are better off
                if dict_user_delta[f"{dataset}:{exp_str}"][user] > 0:
                    dict_user_delta_sex[f"{dataset}:{exp_str}"][user] = dict_metadata[dataset][dict_metadata[dataset]["user_id"] == user]["sex"].values[0]
                    dict_user_delta_height[f"{dataset}:{exp_str}"][user] = dict_metadata[dataset][dict_metadata[dataset]["user_id"] == user]["height"].values[0]
                    dict_user_delta_weight[f"{dataset}:{exp_str}"][user] = dict_metadata[dataset][dict_metadata[dataset]["user_id"] == user]["weight"].values[0]
                else: # <= 0
                    dict_user_delta_sex_worse[f"{dataset}:{exp_str}"][user] = dict_metadata[dataset][dict_metadata[dataset]["user_id"] == user]["sex"].values[0]
                    dict_user_delta_height_worse[f"{dataset}:{exp_str}"][user] = dict_metadata[dataset][dict_metadata[dataset]["user_id"] == user]["height"].values[0]
                    dict_user_delta_weight_worse[f"{dataset}:{exp_str}"][user] = dict_metadata[dataset][dict_metadata[dataset]["user_id"] == user]["weight"].values[0]
                    
            # analyze users' metadata
            print(f"Sex breakdown for {dataset}:{exp_str}:")
            print(Counter(list(dict_user_delta_sex[f"{dataset}:{exp_str}"].values())))
            print("worse off people:")
            print(Counter(list(dict_user_delta_sex_worse[f"{dataset}:{exp_str}"].values())))
            
            print(f"Height breakdown for {dataset}:{exp_str}:")
            print("mean: ", np.mean(list(dict_user_delta_height[f"{dataset}:{exp_str}"].values())))
            print("std: ", np.std(list(dict_user_delta_height[f"{dataset}:{exp_str}"].values())))
            print("worse off people:")
            print("mean: ", np.mean(list(dict_user_delta_height_worse[f"{dataset}:{exp_str}"].values())))
            print("std: ", np.std(list(dict_user_delta_height_worse[f"{dataset}:{exp_str}"].values())))
            
            print("Weight breakdown for {dataset}:{exp_str}:")
            print("mean: ", np.mean(list(dict_user_delta_weight[f"{dataset}:{exp_str}"].values())))
            print("std: ", np.std(list(dict_user_delta_weight[f"{dataset}:{exp_str}"].values())))
            print("worse off people:")
            print("mean: ", np.mean(list(dict_user_delta_weight_worse[f"{dataset}:{exp_str}"].values())))
            print("std: ", np.std(list(dict_user_delta_weight_worse[f"{dataset}:{exp_str}"].values())))
            print("\n")
        # end approaches loop
    # end dataset loop
 

def save_variable(file_name, variable):
    pickle.dump(variable, open(file_name + ".pickle", "wb"))


def load_variable(filename):
    with open(filename + ".pickle", "rb") as f:
        return pickle.load(f)
