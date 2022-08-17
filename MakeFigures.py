import copy
import scipy.stats
import oneWaySign as sign
import oneWayStrength as strength
import OneWayVsTwoWay as reciprocal
import createData as createData
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np
from sklearn.metrics import roc_curve, roc_auc_score


def convertToDic(df, models, metrics):
    """
    Convert the df to dictionary
    """
    dic = {}
    for metric in metrics:
        metric_df = df[
            (df['models'].isin(models)) & (
                df['Measurement'].isin([metric]))]
        dic[metric] = metric_df
    return dic


def mergeAllDf():
    strains_info = pd.read_csv('./Data/strains.csv')
    strain_classification = pd.merge(
        pd.read_csv('Data/Partial_models_sign_strain.csv').drop(columns=['Unnamed: 0'], axis=1), strains_info,
        how="inner", right_on='labeledStrain', left_on='index')
    strain_classification.rename(columns={'Group': 'color_by'}, inplace=True)

    # carbons classification info
    carbon_info = pd.read_csv("./Data/carbons.csv")
    carbon_classification = pd.merge(
        pd.read_csv('Data/Partial_models_sign_env.csv').drop(columns=['Unnamed: 0'], axis=1), carbon_info,
        how='left', right_on='env', left_on='index')
    carbon_classification.rename(columns={'Type_carbon': 'color_by'}, inplace=True)
    carbon_classification = carbon_classification[carbon_classification['index'] != 'Water']

    # strains regression info
    strain_regression = pd.merge(
        pd.read_csv('Data/Partial_models_strength_strain.csv').drop(columns=['Unnamed: 0'], axis=1),
        strains_info, how="inner", right_on='labeledStrain', left_on='index')
    strain_regression.rename(columns={'Group': 'color_by'}, inplace=True)

    # carbon regression info
    carbon_regression = pd.merge(
        pd.read_csv('Data/Partial_models_strength_env.csv').drop(columns=['Unnamed: 0'], axis=1),
        carbon_info, how='left', right_on='env', left_on='index')
    carbon_regression.rename(columns={'Type_carbon': 'color_by'}, inplace=True)
    carbon_regression = carbon_regression[~carbon_regression['index'].isin(['Water', 'Sucrose01'])]
    return strain_classification, carbon_classification, strain_regression, carbon_regression


def dfForFig4(type, score, df):
    """
    Concat the different partial models scores
    :param type: strain/carbon
    :param score: either mcc, accuracy, precision, f1, recall, nrmse, r2
    :param df: the partial models scores
    :return: the data in the needed shape.
    """

    if type == "strain":
        copy_model = "Phylogenetic"
    else:
        copy_model = "Metabolic"

    untrained_naive = pd.DataFrame(
        {'score': df[score + '_copy_model'], 'score_type': copy_model + " copy model",
         type: df['index'], 'family': df['color_by']})

    untrained_no_mono = pd.DataFrame(
        {'score': df[score + '_without_mono'], 'score_type': 'Without coculture or monoculture',
         type: df['index'], 'family': df['color_by']})

    untrained_mono = pd.DataFrame(
        {'score': df[score + '_with_mono'], 'score_type': 'Without coculture',
         type: df['index'], 'family': df['color_by']})

    fully_trained = pd.DataFrame({'score': df[score], 'score_type': 'With all data',
                                  type: df['index'], 'family': df['color_by']})

    full_data = pd.concat([pd.concat(
        [pd.concat([untrained_naive, untrained_no_mono], ignore_index=True), untrained_mono], ignore_index=True),
        fully_trained], ignore_index=True)
    return full_data


def plot_Fig2(sign_dic, y_test, strength_prediction):
    """
    Plot ans save figure 2
    """
    fig, axs = plt.subplots(1, 3, figsize=(12, 7), gridspec_kw={'width_ratios': [2, 1, 1]})
    models = ['Most frequent sign model', 'Metabolic threshold', 'XGBoost', 'Mono Growth threshold']
    metrics = ['Accuracy', 'MCC', 'Precision', 'Recall', 'F1']
    dic_df = convertToDic(sign_dic, models, metrics)

    sns.barplot(y=dic_df['Accuracy']['score'], x=dic_df['Accuracy']['models'], color="#5F5E80", ax=axs[1], )
    axs[1].set_yticks([0, 0.5, 1])
    axs[1].set_yticklabels([0, 0.5, 1], size=15)
    axs[1].grid(False)
    axs[1].spines["top"].set_visible(False)
    axs[1].spines["bottom"].set_visible(True)
    axs[1].spines["right"].set_visible(False)
    axs[1].spines["left"].set_visible(True)
    axs[1].set_title("Accuracy", fontsize=15)
    axs[1].set_xticklabels(models, rotation=90, fontsize=15)
    axs[1].set(xlabel='', ylabel='')

    sns.barplot(y=dic_df['MCC']['score'], x=dic_df['MCC']['models'], color="#5F5E80", ax=axs[2], )
    axs[2].set_yticks([0, 0.5, 1])
    axs[2].set_yticklabels('', size=15)
    axs[2].grid(False)
    axs[2].spines["top"].set_visible(False)
    axs[2].spines["bottom"].set_visible(True)
    axs[2].spines["right"].set_visible(False)
    axs[2].spines["left"].set_visible(True)
    axs[2].set_xticklabels(models, rotation=90, fontsize=15)
    axs[2].set_title("MCC", fontsize=15)
    axs[2].set(xlabel='', ylabel='')
    sns.despine()

    data = pd.DataFrame(np.transpose(np.array([y_test, strength_prediction])),
                        columns=['Effect', 'Prediction'])
    positive_effect_negative_prediction_index = ((data['Effect'] > 0) & (data['Prediction'] <= 0))
    negative_effect_positive_prediction_index = ((data['Effect'] <= 0) & (data['Prediction'] > 0))
    color = np.full(data.shape[0], 'correct')
    color[positive_effect_negative_prediction_index] = 'sign_changed'
    color[negative_effect_positive_prediction_index] = 'same_sign'
    data['colored_by'] = color
    color[negative_effect_positive_prediction_index] = 'sign_changed'
    data['colored_by'] = color

    sns.scatterplot(x="Effect", y="Prediction", hue="colored_by",
                    s=50, alpha=0.9, palette='bone',
                    data=data, ax=axs[0], legend=False)
    axs[0].axhline(0, lineStyle='--', color='black')
    axs[0].axvline(0, lineStyle='--', color='black')
    axs[0].set_xticks([-4, -2, 0, 2, 4])
    axs[0].set_yticks([-4, -2, 0, 2, 4])
    axs[0].set_yticklabels(axs[0].get_yticks(), size=15)
    axs[0].set_xticklabels(axs[0].get_xticks(), size=15)

    plt.figtext(0.34, 0.47, "NRMSE: " + str(round(strength.evaluate(strength_prediction, y_test)['RMSE'], 2)),
                fontsize=13)
    axs[0].set_ylabel('Predicted effect', fontsize=15)
    axs[0].set_xlabel('Measured effect', fontsize=15)
    plt.figtext(0.21, 0.96, "Strength prediction", fontsize=18)
    plt.figtext(0.63, 0.96, "Sign prediction", fontsize=18)
    plt.figtext(0.12, 0.92, "A", fontsize=15)
    plt.figtext(0.5, 0.92, "B", fontsize=15)
    plt.subplots_adjust(bottom=0.4, wspace=0.3)
    plt.show()


def plot_Fig1S(sign_dictionary, strength_dictionary, best_sign_prediction, X_test, y_test, y_test_sign, rf, logistic,
               kn, xgb):
    """
    Plot and save figures S1+S2
    """
    fig, axs = plt.subplots(2, 3, figsize=(12, 9), sharex='col', sharey='col')

    sign_models = ['Most frequent sign model', 'Metabolic threshold', 'XGBoost', 'Mono Growth threshold']
    sign_ml_models = ['Random forest', 'Logistic regression', 'K-nearest neighbors', 'Most frequent sign model',
                      'Metabolic threshold', 'XGBoost']
    strength_ml_models = ['Random forest', 'XGBoost', 'Linear regression', 'K-nearest neighbors']
    metrics = ['Accuracy', 'MCC', 'Precision', 'Recall', 'F1']
    strength_metrics = ['R2', 'RMSE']

    sign_dictionary_fig1 = convertToDic(sign_dictionary, sign_models, metrics)
    sign_dictionary_with_ml_models = convertToDic(sign_dictionary, sign_ml_models, metrics)
    strength_dctionary_with_ml_models = convertToDic(strength_dictionary, strength_ml_models, strength_metrics)

    sns.barplot(y=sign_dictionary_with_ml_models['Accuracy']['score'],
                x=sign_dictionary_with_ml_models['Accuracy']['models'], color="#5F5E80", ax=axs[0][0], )
    axs[0][0].set_ylabel("", fontsize=15)
    axs[0][0].set_xlabel("")
    axs[0][0].set_yticks([0, 0.5, 1])
    axs[0][0].set_yticklabels([0, 0.5, 1], size=15)
    axs[0][0].set_xticklabels("")
    axs[0][0].set_title("Accuracy", fontsize=15)

    sns.barplot(y=sign_dictionary_with_ml_models['MCC']['score'], x=sign_dictionary_with_ml_models['MCC']['models'],
                color="#5F5E80", ax=axs[0][1], )
    axs[0][1].set_ylabel("")
    axs[0][1].set_xlabel("")
    axs[0][1].set_yticks([0, 0.5, 1])
    axs[0][1].set_yticklabels('', size=15)
    axs[0][1].set_xticklabels('')
    axs[0][1].set_title("Matthew's correlation coefficient", fontsize=15)

    sns.barplot(y=sign_dictionary_with_ml_models['Precision']['score'],
                x=sign_dictionary_with_ml_models['Recall']['models'], color="#5F5E80", ax=axs[1][0], )
    axs[1][0].set_ylabel("", fontsize=15)
    axs[1][0].set_xlabel("")
    axs[1][0].set_yticks([0, 0.5, 1])
    axs[1][0].set_yticklabels([0, 0.5, 1], size=15)
    axs[1][0].set_xticklabels(sign_ml_models, rotation=90, fontsize=16)
    axs[1][0].set_title("Precision", fontsize=15)

    sns.barplot(y=sign_dictionary_with_ml_models['Recall']['score'],
                x=sign_dictionary_with_ml_models['Recall']['models'], color="#5F5E80", ax=axs[1][1], )
    axs[1][1].set_ylabel("")
    axs[1][1].set_xlabel("")
    axs[1][1].set_yticklabels('', size=15)
    axs[1][1].set_xticklabels(sign_ml_models, rotation=90, fontsize=16)
    axs[1][1].set_title("Recall", fontsize=15)

    sns.barplot(y=strength_dctionary_with_ml_models['R2']['score'], x=strength_dctionary_with_ml_models['R2']['models'],
                color="#5F5E80", ax=axs[0][2], )
    axs[0][2].set_ylabel("")
    axs[0][2].set_xlabel("")
    axs[0][2].set_yticklabels('', size=15)
    axs[0][2].set_xticklabels('')
    axs[0][2].set_title("R2", fontsize=15)

    sns.barplot(y=strength_dctionary_with_ml_models['RMSE']['score'],
                x=strength_dctionary_with_ml_models['RMSE']['models'], color="#5F5E80", ax=axs[1][2], )
    axs[1][2].set_ylabel("")
    axs[1][2].set_xlabel("")
    axs[1][2].set_yticklabels('', size=15)
    axs[1][2].set_xticklabels(strength_ml_models, rotation=90, fontsize=16)
    axs[1][2].set_title("NRMSE", fontsize=15)
    plt.subplots_adjust(bottom=0.33)

    plt.figtext(0.3, 0.96, "Sign prediction", fontsize=18)
    plt.figtext(0.69, 0.96, "Strength prediction", fontsize=18)
    plt.figtext(0.12, 0.92, "A", fontsize=15)
    plt.figtext(0.4, 0.92, "B", fontsize=15)
    plt.figtext(0.67, 0.92, "C", fontsize=15)
    sns.despine()
    plt.show()

    fig2, axs2 = plt.subplots(1, 4, figsize=(18, 7))
    y_pred_rf = rf.predict_proba(X_test)[:, 1]
    rf_fpr, rf_tpr, _ = roc_curve(y_test_sign, y_pred_rf)
    sns.lineplot(rf_fpr, rf_tpr, label='Random forest AUC:' + str(round(roc_auc_score(y_test_sign, y_pred_rf), 3)),
                 ax=axs2[3])

    y_pred_logistic = logistic.predict_proba(X_test)[:, 1]
    rf_fpr, rf_tpr, _ = roc_curve(y_test_sign, y_pred_logistic)
    sns.lineplot(rf_fpr, rf_tpr,
                 label='Logistic regression AUC:' + str(round(roc_auc_score(y_test_sign, y_pred_logistic), 3)),
                 ax=axs2[3])

    y_pred_kn = kn.predict_proba(X_test)[:, 1]
    rf_fpr, rf_tpr, _ = roc_curve(y_test_sign, y_pred_kn)
    sns.lineplot(rf_fpr, rf_tpr, palette="bone_r",
                 label='k-nearest neighbors AUC:' + str(round(roc_auc_score(y_test_sign, y_pred_kn), 3)), ax=axs2[3])

    y_pred_nb = xgb.predict_proba(X_test)[:, 1]
    rf_fpr, rf_tpr, _ = roc_curve(y_test_sign, y_pred_nb)
    sns.lineplot(rf_fpr, rf_tpr, palette="bone_r",
                 label='XGBoost AUC:' + str(round(roc_auc_score(y_test_sign, y_pred_nb), 3)), ax=axs2[3])
    axs2[3].set_xlabel('False positive rate', fontsize=15)
    axs2[3].set_ylabel('True positive rate', fontsize=15)
    plt.legend(frameon=False)
    axs2[3].set_yticks([0, 0.5, 1])
    sns.despine()

    sns.barplot(y=sign_dictionary_with_ml_models['Precision']['score'], x=sign_dictionary_fig1['Precision']['models'],
                color="#5F5E80", ax=axs2[0], )
    axs2[0].set_ylabel("", fontsize=15)
    axs2[0].set_xlabel("")
    axs2[0].set_yticks([0, 0.5, 1])
    axs2[0].set_yticklabels([0, 0.5, 1], size=15)
    axs2[0].set_xticklabels(sign_models, rotation=90, fontsize=18)
    axs2[0].set_title("Precision", fontsize=18)

    sns.barplot(y=sign_dictionary_with_ml_models['Recall']['score'], x=sign_dictionary_fig1['Recall']['models'],
                color="#5F5E80", ax=axs2[1], )
    axs2[1].set_ylabel("")
    axs2[1].set_xlabel("")
    axs2[1].set_yticks([0, 0.5, 1])
    axs2[1].set_yticklabels([0, 0.5, 1], size=15)
    axs2[1].set_yticklabels('', size=15)
    axs2[1].set_xticklabels(sign_models, rotation=90, fontsize=18)
    axs2[1].set_title("Recall", fontsize=18)

    true_pos_neg_labeld, true_nef_pos_pred_labeled = createFnFp(y_test, best_sign_prediction, X_test)
    sns.histplot(true_pos_neg_labeld['effect'], kde=False, bins=30, color="#79AEBF", ax=axs2[2])
    sns.histplot(true_nef_pos_pred_labeled['effect'], kde=False, bins=30, color='#D1988A', ax=axs2[2])
    sns.set_theme(style="white", palette="pastel")
    axs2[2].set_xlabel('Measured effect', fontsize=15)
    axs2[2].set_ylabel('', fontsize=15)
    axs2[2].set_yticks([0, 50, 100, 150])
    sns.set_theme(style="white", palette="pastel")
    plt.subplots_adjust(bottom=0.52)

    plt.figtext(0.13, 0.92, "A", fontsize=15)
    plt.figtext(0.52, 0.92, "B", fontsize=15)
    plt.figtext(0.72, 0.92, "C", fontsize=15)
    plt.show()


def createFnFp(y_test, prediction, X_test):
    """
    Extract false negative mistakes and false positive mistakes
    :param y_test: True y test labels.
    :param prediction: prediction.
    :param X_test: test set.
    :return:
    """
    X_test_copy = copy.copy(X_test)
    X_test_copy['prediction'] = prediction
    X_test_copy['effect'] = y_test
    X_test_copy['true'] = y_test

    # False negative
    true_pos_ind = X_test_copy['true'] > 0
    true_pos = X_test_copy[true_pos_ind]
    false_negative_index = true_pos['prediction'] <= 0
    false_negative = true_pos[false_negative_index]

    # False positive
    true_neg_index = X_test_copy['true'] <= 0
    neg_pos = X_test_copy[true_neg_index]
    false_positive = neg_pos['prediction'] > 0
    true_nef_pos_pred_labeled = neg_pos[false_positive]
    return false_negative, true_nef_pos_pred_labeled


def plot_Fig4(strain_strength_df_nrmse, carbon_strength_df_nrmse, strength_null_metrics):
    """
    Plot and save figure 4
    """
    f, ax = plt.subplots(1, 2, figsize=(15, 9), sharey="row")
    sns.boxplot(x="score_type", y="score", data=strain_strength_df_nrmse, width=.6, color='White', showfliers=False,
                ax=ax[0])
    phylogenetic_order = ["Phylogenetic copy model", "Without coculture \n or monoculture",
                          "Without coculture",
                          "With all data"]
    metabolic_order = ["Metabolic copy model", "Without coculture \n or monoculture",
                       "Without coculture", "With all data"]

    # Add in points to show each observation
    sns.stripplot(x="score_type", y="score", data=strain_strength_df_nrmse, s=7, hue="family", palette="bone",  # = ''
                  size=4, linewidth=0.6, color='0.4', edgecolor='black', ax=ax[0])
    ax[0].axhline(strength_null_metrics['RMSE'][0], lineStyle='--', color='black')
    ax[0].legend([], [], frameon=False)
    ax[0].set_xlabel('')
    ax[0].set_ylabel('NRMSE', fontsize=15)
    ax[0].set_yticks([0, 0.5, 1, 1.5, 2])
    ax[0].set_xticklabels(phylogenetic_order, rotation=90, fontsize=18)
    ax[0].xaxis.grid(False)

    sns.boxplot(x="score_type", y="score", data=carbon_strength_df_nrmse,
                width=.6, showfliers=False, color='White', ax=ax[1])

    # Add in points to show each observation
    sns.stripplot(x="score_type", y="score", data=carbon_strength_df_nrmse, s=7,
                  size=4, linewidth=0.6, color='0.4', edgecolor='black', ax=ax[1])
    ax[1].axhline(strength_null_metrics['RMSE'][0], lineStyle='--', color='black')
    ax[1].legend([], [], frameon=False)
    ax[1].set_xlabel('')
    ax[1].set_ylabel('')
    ax[1].set_xticklabels(metabolic_order, rotation=90, fontsize=18)
    ax[1].xaxis.grid(False)

    plt.subplots_adjust(bottom=0.42, right=0.78)
    plt.figtext(0.18, 0.96, "Untrained species", fontsize=18)
    plt.figtext(0.52, 0.96, "Untrained carbon environments", fontsize=18)
    plt.figtext(0.79, 0.68, "Average strength model", color="#A52A2A", fontsize=18)
    plt.figtext(0.12, 0.92, "C", fontsize=15)
    plt.figtext(0.503, 0.92, "D", fontsize=15)
    sns.despine()
    plt.show()


def plot_Fig5S(strain_sign_df_acc, strain_sign_df_recall, strain_sign_df_precision,
               carbon_sign_df_acc, carbon_sign_df_recall, carbon_sign_df_precision, sign_null_metrics):
    """
    Plot and safe figure S5
    """
    f, ax = plt.subplots(2, 3, figsize=(12, 10), sharex=True)

    models_names = ["Copy model", "Without coculture or monoculture",
                    "Without coculture",
                    "With all data"]
    sns.boxplot(x="score_type", y="score", data=strain_sign_df_acc,
                width=.6, showfliers=False, color='White', ax=ax[0][0])
    sns.stripplot(x="score_type", y="score", data=strain_sign_df_acc, s=7, hue="family", palette="bone",
                  size=4, linewidth=0.6, color='0.4', edgecolor='black', ax=ax[0][0])
    ax[0][0].axhline(sign_null_metrics['Accuracy'][0], lineStyle='--', color='black')
    ax[0][0].legend([], [], frameon=False)
    ax[0][0].set_xlabel('')
    ax[0][0].set_ylabel('Accuracy', fontsize=15)
    ax[0][0].set_yticks([0, 0.5, 1])
    ax[0][0].xaxis.grid(False)

    sns.boxplot(x="score_type", y="score", data=strain_sign_df_recall,
                width=.6, showfliers=False, color='White', ax=ax[0][1])
    sns.stripplot(x="score_type", y="score", data=strain_sign_df_recall, s=7, hue="family", palette="bone",
                  size=4, linewidth=0.6, color='0.4', edgecolor='black', ax=ax[0][1])
    ax[0][1].axhline(sign_null_metrics['Recall'][0], lineStyle='--', color='black')
    ax[0][1].legend([], [], frameon=False)
    ax[0][1].set_xlabel('')
    ax[0][1].set_ylabel('Recall', fontsize=15)
    ax[0][1].set_yticks([0, 0.5, 1])
    ax[0][1].xaxis.grid(False)

    sns.boxplot(x="score_type", y="score", data=strain_sign_df_precision,
                width=.6, showfliers=False, color='White', ax=ax[0][2])
    sns.stripplot(x="score_type", y="score", data=strain_sign_df_precision, s=7, hue="family", palette="bone",  # = ''
                  size=4, linewidth=0.6, color='0.4', edgecolor='black', ax=ax[0][2])
    ax[0][2].axhline(sign_null_metrics['Precision'][0], lineStyle='--', color='black')
    ax[0][2].legend([], [], frameon=False)
    ax[0][2].set_xlabel('')
    ax[0][2].set_ylabel('Precision', fontsize=15)
    ax[0][2].set_yticks([0, 0.5, 1])
    ax[0][2].xaxis.grid(False)

    sns.boxplot(x="score_type", y="score", data=carbon_sign_df_acc,
                width=.6, showfliers=False, color='White', ax=ax[1][0])
    sns.stripplot(x="score_type", y="score", data=carbon_sign_df_acc, s=7,
                  size=4, linewidth=0.6, color='0.4', edgecolor='black', ax=ax[1][0])
    ax[1][0].axhline(sign_null_metrics['Accuracy'][0], lineStyle='--', color='black')
    ax[1][0].legend([], [], frameon=False)
    ax[1][0].set_xlabel('')
    ax[1][0].set_ylabel('Accuracy', fontsize=15)
    ax[1][0].set_yticks([0, 0.5, 1])
    ax[1][0].set_xticklabels(models_names, rotation=90, fontsize=15)
    ax[1][0].xaxis.grid(False)

    sns.boxplot(x="score_type", y="score", data=carbon_sign_df_recall,
                width=.6, showfliers=False, color='White', ax=ax[1][1])
    sns.stripplot(x="score_type", y="score", data=carbon_sign_df_recall, s=7,
                  size=4, linewidth=0.6, color='0.4', edgecolor='black', ax=ax[1][1])
    ax[1][1].axhline(sign_null_metrics['Recall'][0], lineStyle='--', color='black')
    ax[1][1].legend([], [], frameon=False)
    ax[1][1].set_xlabel('')
    ax[1][1].set_ylabel('Recall', fontsize=15)
    ax[1][1].set_yticks([0, 0.5, 1])
    ax[1][1].set_xticklabels(models_names, rotation=90, fontsize=15)
    ax[1][1].xaxis.grid(False)

    sns.boxplot(x="score_type", y="score", data=carbon_sign_df_precision,
                width=.6, showfliers=False, color='White', ax=ax[1][2])
    sns.stripplot(x="score_type", y="score", data=carbon_sign_df_precision, s=7,
                  size=4, linewidth=0.6, color='0.4', edgecolor='black', ax=ax[1][2])
    ax[1][2].axhline(sign_null_metrics['Precision'][0], lineStyle='--', color='black')
    ax[1][2].legend([], [], frameon=False)
    ax[1][2].set_xlabel('')
    ax[1][2].set_ylabel('Precision', fontsize=15)
    ax[1][2].set_yticks([0, 0.5, 1])
    ax[1][2].set_xticklabels(models_names, rotation=90, fontsize=15)
    ax[1][2].xaxis.grid(False)

    plt.figtext(0.4, 0.96, "Untrained species", fontsize=18)
    plt.figtext(0.35, 0.6, "Untrained carbon environments", fontsize=18)

    plt.subplots_adjust(bottom=0.37, hspace=0.4)
    sns.despine()
    plt.show()


def createGenusType(data):
    data['family_type'] = 'betweenGroup'
    for family in data['group_strain'].unique():
        data.loc[((data['group_strain'] == data['group_closest_strain']) & (
                data['group_strain'] == family)), 'family_type'] = data['group_strain'] + '/' + data[
            'group_closest_strain']

    data['genus_type'] = 'betweenGroup'
    for genus in data['genus_strain'].unique():
        data.loc[((data['genus_strain'] == data['genus_closest_strain']) & (
                data['genus_strain'] == genus)), 'genus_type'] = data['genus_strain'] + '/' + data[
            'genus_closest_strain']
    return data


def plot_Fig5():
    """
    Plot and save figure 5
    """
    strains_groups = pd.read_csv('./Data/strains.csv')
    strains_groups['Genus'] = strains_groups['Fullname'].str.split(' ', expand=True)[0]
    strains_groups = strains_groups[['labeledStrain', 'Group', 'Genus']]

    # strength predictions
    strength = pd.read_csv("./Data/Naive_copy_strength_scores_per_distance.csv")
    null_metrics = pd.read_csv("./Data/full_metrics_strength.csv")
    strength = strength[['r2', 'rmse', 'strain', 'closest_strain', 'distance', 'ith']]

    strength = pd.merge(strength, strains_groups, how='left', left_on='strain', right_on=['labeledStrain']).rename(
        columns={'Group': 'group_strain', 'Genus': 'genus_strain'})
    strength = pd.merge(strength, strains_groups, how='left', left_on='closest_strain',
                        right_on=['labeledStrain']).rename(
        columns={'Group': 'group_closest_strain', 'Genus': 'genus_closest_strain'}).drop(
        columns=['labeledStrain_x', 'labeledStrain_y'], axis=1)

    strength = createGenusType(strength)
    f, ax = plt.subplots(figsize=(9, 6))
    corr = scipy.stats.pearsonr(strength['rmse'], strength['distance'])
    box = sns.scatterplot(x="distance", y="rmse", data=strength, hue="family_type",
                          palette=['#131F46', '#C37272', '#868EB2'])
    sns.move_legend(ax, "center right", bbox_to_anchor=(2.13, 0.70), title=None, frameon=False, ncol=1)
    box.axhline(null_metrics['RMSE'][0], lineStyle='--', color='black')
    ax.xaxis.grid(False)
    ax.set_xticks([0, 0.2, 0.4, 0.6, 0.8])
    ax.set_yticks([0, 0.5, 1, 1.5, 2])

    ax.set_xlabel("Distance", fontsize=15)
    plt.subplots_adjust(right=0.7)
    plt.figtext(0.71, 0.47, "Average strength model", color="#A52A2A", fontsize=15)
    ax.set_ylabel("NRMSE", fontsize=15)
    sns.despine()
    plt.show()


def plot_Fig4S():
    """
    Plot and save figure S4
    """
    f, ax = plt.subplots(figsize=(6, 6))
    sns.set(style='ticks', font_scale=2.5, rc={"lines.linewidth": 3})
    data = pd.read_csv("./Data/FigS4_data.csv")
    data = data[['PCA,OHE,properties', 'OHE']]
    data.index = ['Yes', 'No']
    data.columns = ['Yes', 'No']
    data.T.plot.bar(colormap='crest')
    plt.xticks(rotation=0)
    ax.xaxis.grid(False)
    sns.despine()
    plt.figtext(0.4, 0.96, "Untrained species", fontsize=18)
    plt.xlabel("Phylogenetic information available")
    plt.legend(title='Inferred metabolic\npathways available', bbox_to_anchor=(1.02, 1), loc='upper left',
               frameon=False)
    plt.show()


def plot_Fig6(y_test, strength_prediction, effect1, effect2, rmse_reciprocal, avg_nrmse_reciprocal, avg_one_nrmse,
              avg_one_r2, avg_multi_nrmse, avg_multi_r2):
    """
    Plot and save figure 6
    """
    rmse_reciprocal = pd.DataFrame({"nrmse": avg_nrmse_reciprocal, "model": 'reciprocal'}, index=[0])
    rmse_one_way = pd.DataFrame({"nrmse": avg_one_nrmse, "model": "One-way strength model"}, index=[0])
    df = pd.concat([rmse_one_way, rmse_reciprocal], ignore_index=True)
    df_r2 = pd.DataFrame(
        {'score': [avg_one_r2, avg_multi_r2], 'models': ['one-way model \n trained twice', 'two-way model']})
    df_rmse = pd.DataFrame(
        {'score': [avg_one_nrmse, avg_multi_nrmse], 'models': ['one-way model \n trained twice', 'two-way model']})

    f, ax = plt.subplots(1, 3, figsize=(14, 8))

    sns.barplot(y=df_rmse['score'], x=df_rmse['models'], palette="bone", ax=ax[0], )
    ax[0].set_ylim(0, 1)
    ax[0].set_yticks([0, 0.5, 1])
    ax[0].grid(False)
    ax[0].spines["top"].set_visible(False)
    ax[0].spines["bottom"].set_visible(True)
    ax[0].spines["right"].set_visible(False)
    ax[0].spines["left"].set_visible(True)
    ax[0].set_xticklabels(ax[0].get_xticklabels(), rotation=90, fontsize=18)
    ax[0].set(xlabel='', ylabel='')

    f.subplots_adjust(left=None, bottom=0.3, right=None, top=None, wspace=0.3, hspace=None)
    sns.despine()
    sns.scatterplot(x=effect2, y=effect1, s=50, alpha=0.3, color='#343C6C', ax=ax[1])
    ax[1].set_yticks([-8, -4, 0, 4, 8])
    ax[1].set_yticklabels([-8, -4, 0, 4, 8], size=15)
    ax[1].set_xticks([-8, -4, 0, 4, 8])
    ax[1].set_xticklabels([-8, -4, 0, 4, 8], size=15)
    ax[1].set_ylabel('Effect of B on A', fontsize=18)
    ax[1].set_xlabel('', fontsize=15)
    plt.figtext(0.45, 0.34, 'Effect of A on B', fontsize=18)

    sns.barplot(y=df['nrmse'], x=df['model'], palette="bone", ax=ax[2], )
    ax[2].set_ylim(0, 1)
    ax[2].set_yticks([0, 0.5, 1])
    ax[2].grid(False)
    ax[2].spines["top"].set_visible(False)
    ax[2].spines["bottom"].set_visible(True)
    ax[2].spines["right"].set_visible(False)
    ax[2].spines["left"].set_visible(True)
    ax[2].set_xlabel("")
    ax[2].set_ylabel("")
    ax[2].set_xticklabels(['No', 'Yes'], rotation=90, fontsize=18)
    ax[2].set_xlabel("Reciprocal effect as feature", fontsize=18)

    plt.figtext(0.12, 0.90, "A", fontsize=15)
    plt.figtext(0.4, 0.90, "B", fontsize=15)
    plt.figtext(0.68, 0.90, "C", fontsize=15)
    plt.subplots_adjust(bottom=0.43, wspace=0.3)
    plt.show()


def plot_Fig6S():
    """
    Plot and save figure S6
    """
    f, axs = plt.subplots(2, 2, figsize=(8, 8), sharex=True)
    sign_data = pd.read_csv('Data/Naive_copy_sign_per_distance.csv')
    strains_groups = pd.read_csv('./Data/strains.csv')
    strains_groups['Genus'] = strains_groups['Fullname'].str.split(' ', expand=True)[0]
    strains_groups = strains_groups[['labeledStrain', 'Group', 'Genus']]
    null_metrics = pd.read_csv("./Data/full_metrics_sign.csv")

    sign_data = pd.merge(sign_data, strains_groups, how='left', left_on='strain', right_on=['labeledStrain']).rename(
        columns={'Group': 'group_strain', 'Genus': 'genus_strain'})
    sign_data = pd.merge(sign_data, strains_groups, how='left', left_on='closest_strain',
                         right_on=['labeledStrain']).rename(
        columns={'Group': 'group_closest_strain', 'Genus': 'genus_closest_strain'}).drop(
        columns=['labeledStrain_x', 'labeledStrain_y'], axis=1)

    sign_data = createGenusType(sign_data)
    sns.scatterplot(x="distance", y="acc", data=sign_data, hue="family_type",
                    palette=['#131F46', '#C37272', '#868EB2'], ax=axs[0][0])
    sns.move_legend(axs[0][0], "center right", bbox_to_anchor=(2.13, 0.70), title=None, frameon=False, ncol=1)
    axs[0][0].axhline(null_metrics['Accuracy'][0], lineStyle='--', color='black')
    axs[0][0].xaxis.grid(False)
    axs[0][0].set_xticks([0, 0.2, 0.4, 0.6, 0.8])
    axs[0][0].set_yticks([0, 0.5, 1])
    axs[0][0].set_xlabel("Distance", fontsize=15)
    axs[0][0].set_ylabel("", fontsize=15)
    axs[0][0].set_title("Accuracy", fontsize=15)

    sns.scatterplot(x="distance", y="mcc", data=sign_data, hue="family_type",
                    palette=['#131F46', '#C37272', '#868EB2'], ax=axs[0][1])
    sns.move_legend(axs[0][1], "center right", bbox_to_anchor=(2.13, 0.70), title=None, frameon=False, ncol=1)
    axs[0][1].axhline(null_metrics['MCC'][0], lineStyle='--', color='black')
    axs[0][1].xaxis.grid(False)
    axs[0][1].set_xticks([0, 0.2, 0.4, 0.6, 0.8])
    axs[0][1].set_yticks([0, 0.5, 1])
    axs[0][1].set_xlabel("Distance", fontsize=15)
    axs[0][1].set_ylabel("", fontsize=15)
    axs[0][1].set_title("Matthews correlation coefficient", fontsize=15)

    sns.scatterplot(x="distance", y="Recall", data=sign_data, hue="family_type",
                    palette=['#131F46', '#C37272', '#868EB2'], ax=axs[1][0])
    sns.move_legend(axs[1][0], "center right", bbox_to_anchor=(2.13, 0.70), title=None, frameon=False, ncol=1)
    axs[1][0].axhline(null_metrics['Recall'][0], lineStyle='--', color='black')
    axs[1][0].xaxis.grid(False)
    axs[1][0].set_xticks([0, 0.2, 0.4, 0.6, 0.8])
    axs[1][0].set_yticks([0, 0.5, 1])
    axs[1][0].set_xlabel("Distance", fontsize=15)
    axs[1][0].set_ylabel("", fontsize=15)
    axs[1][0].set_title("Recall", fontsize=15)

    sns.scatterplot(x="distance", y="Precision", data=sign_data, hue="family_type",
                    palette=['#131F46', '#C37272', '#868EB2'], ax=axs[1][1])
    sns.move_legend(axs[1][1], "center right", bbox_to_anchor=(2.13, 0.70), title=None, frameon=False, ncol=1)
    axs[1][1].axhline(null_metrics['Precision'][0], lineStyle='--', color='black')
    axs[1][1].xaxis.grid(False)
    axs[1][1].set_xticks([0, 0.2, 0.4, 0.6, 0.8])
    axs[1][1].set_yticks([0, 0.5, 1])
    axs[1][1].set_xlabel("Distance", fontsize=15)
    axs[1][1].set_ylabel("", fontsize=15)
    axs[1][1].set_title("Precision", fontsize=15)

    sns.despine()
    plt.show()


def plot_Fig7S(avg_one_nrmse, avg_one_r2, avg_multi_nrmse, avg_multi_r2, report_one_way, report_two_way):
    """
    Plot and save figure S7
    """

    plt.subplots_adjust(bottom=0.5)
    df_r2 = pd.DataFrame(
        {'score': [avg_one_r2, avg_multi_r2], 'models': ['one-way model \n trained twice', 'two-way model']})
    df_rmse = pd.DataFrame(
        {'score': [avg_one_nrmse, avg_multi_nrmse], 'models': ['one-way model \n trained twice', 'two-way model']})
    df_accuracy = pd.DataFrame({'score': [report_one_way['Accuracy'], report_two_way['Accuracy']],
                                'models': ['one-way model \n trained twice', 'two-way model']})
    df_recall = pd.DataFrame(
        {'score': [report_one_way['Weighted_avg']['recall'], report_two_way['Weighted_avg']['recall']],
         'models': ['one-way model \n trained twice', 'two-way model']})
    df_precision = pd.DataFrame(
        {'score': [report_one_way['Weighted_avg']['precision'], report_two_way['Weighted_avg']['precision']],
         'models': ['one-way model \n trained twice', 'two-way model']})

    fig_7s2, axs_7s2 = plt.subplots(1, 3, sharey=True, figsize=(10, 7))
    plt.subplots_adjust(bottom=0.5)
    sns.barplot(y=df_accuracy['score'], x=df_accuracy['models'], palette="bone", ax=axs_7s2[0])
    axs_7s2[0].set_ylim(0, 1)
    axs_7s2[0].set_yticks([0, 0.5, 1])
    axs_7s2[0].set_yticklabels([0, 0.5, 1], size=15)
    axs_7s2[0].grid(False)
    axs_7s2[0].spines["top"].set_visible(False)
    axs_7s2[0].spines["bottom"].set_visible(True)
    axs_7s2[0].spines["right"].set_visible(False)
    axs_7s2[0].spines["left"].set_visible(True)
    axs_7s2[0].set_xticklabels(axs_7s2[0].get_xticklabels(), rotation=90, fontsize=18)
    axs_7s2[0].set_title("Accuracy", fontsize=15)
    axs_7s2[0].set(xlabel='', ylabel='')

    sns.barplot(y=df_precision['score'], x=df_precision['models'], palette="bone", ax=axs_7s2[1])
    axs_7s2[1].set_ylim(0, 1)
    axs_7s2[1].grid(False)
    axs_7s2[1].spines["top"].set_visible(False)
    axs_7s2[1].spines["bottom"].set_visible(True)
    axs_7s2[1].spines["right"].set_visible(False)
    axs_7s2[1].spines["left"].set_visible(True)
    axs_7s2[1].set_xticklabels(axs_7s2[1].get_xticklabels(), rotation=90, fontsize=18)
    axs_7s2[1].set_title("Weighted precision", fontsize=15)
    axs_7s2[1].set(xlabel='', ylabel='')

    sns.barplot(y=df_recall['score'], x=df_recall['models'], palette="bone", ax=axs_7s2[2])
    axs_7s2[2].set_ylim(0, 1)
    axs_7s2[2].grid(False)
    axs_7s2[2].spines["top"].set_visible(False)
    axs_7s2[2].spines["bottom"].set_visible(True)
    axs_7s2[2].spines["right"].set_visible(False)
    axs_7s2[2].spines["left"].set_visible(True)
    axs_7s2[2].set_xticklabels(axs_7s2[2].get_xticklabels(), rotation=90, fontsize=18)
    axs_7s2[2].set_title("Weighted recall", fontsize=15)
    axs_7s2[2].set(xlabel='', ylabel='')
    plt.show()


def plot_Fig9S(carbon_pca, strain_pca):
    """
    Plot and save figure S8
    """
    f, axs = plt.subplots(1, 2, figsize=(12, 6))
    sns.scatterplot(x="carbon_component_0", y="carbon_component_1",
                    s=50, alpha=0.9, color='#003366',
                    data=carbon_pca, ax=axs[0], legend=False)
    axs[0].set_ylabel('PC 2 (30%)', fontsize=15)
    axs[0].set_xlabel('PC 1 (50%)', fontsize=15)

    sns.scatterplot(x="phy_strain_component_0", y="phy_strain_component_1",
                    s=50, alpha=0.9, color='#003366',
                    data=strain_pca, ax=axs[1], legend=False)
    axs[1].set_ylabel('PC 2 (1%)', fontsize=15)
    axs[1].set_xlabel('PC 1 (97%)', fontsize=15)

    f.subplots_adjust(left=None, bottom=None, right=None, top=None, wspace=0.3, hspace=None)
    sns.despine()
    plt.figtext(0.12, 0.90, "A", fontsize=15)
    plt.figtext(0.56, 0.90, "B", fontsize=15)
    plt.figtext(0.15, 0.92, "Representation of carbons with PCA", fontsize=15)
    plt.figtext(0.59, 0.92, "Representation of species with PCA", fontsize=15)
    plt.savefig("./Fig9S.png", dpi=300)
    plt.show()


def main():
    # All data needed for figures
    sign_dictionary, sign_prediction, rf, logistic, kn, xgb, X_test, y_test, y_test_sign = sign.main()
    strength_dictionary, y_test, strength_prediction = strength.main()
    effect_1, effect_2, reciprocal_nrmse_2_1, reciprocal_r2_2_1, avg_nrmse_reciprocal, avg_one_nrmse, avg_one_r2, avg_multi_nrmse, avg_multi_r2, report_one_way, report_two_way = reciprocal.main()
    carbon_pca, strain_pca = createData.main()
    strain_classification, carbon_classification, strain_regression, carbon_regression = mergeAllDf()
    strength_null_metrics, sign_null_metrics = pd.read_csv("./Data/full_metrics_strength.csv"), pd.read_csv(
        "./Data/full_metrics_sign.csv")
    strain_strength_df_nrmse = dfForFig4("strain", "nrmse", strain_regression)
    carbon_strength_df_nrmse = dfForFig4("carbon", "nrmse", carbon_regression)
    strain_sign_df_acc = dfForFig4("strain", "accuracy", strain_classification)
    strain_sign_df_recall = dfForFig4("strain", "recall", strain_classification)
    strain_sign_df_precision = dfForFig4("strain", "precision", strain_classification)
    carbon_sign_df_acc = dfForFig4("carbon", "accuracy", carbon_classification)
    carbon_sign_df_recall = dfForFig4("carbon", "recall", carbon_classification)
    carbon_sign_df_precision = dfForFig4("carbon", "precision", carbon_classification)

    # plot_Fig2(sign_dictionary, y_test, strength_prediction)
    # plot_Fig1S(sign_dictionary, strength_dictionary, sign_prediction, X_test, y_test, y_test_sign, rf, logistic, kn,
    #            xgb)
    # plot_Fig4(strain_strength_df_nrmse, carbon_strength_df_nrmse, strength_null_metrics)
    # plot_Fig4S()
    # plot_Fig5()
    # plot_Fig5S(strain_sign_df_acc, strain_sign_df_recall, strain_sign_df_precision, carbon_sign_df_acc,
    #            carbon_sign_df_recall, carbon_sign_df_precision, sign_null_metrics)
    # plot_Fig6(y_test, strength_prediction, effect_1, effect_2, reciprocal_nrmse_2_1, avg_nrmse_reciprocal,
    #           avg_one_nrmse, avg_one_r2, avg_multi_nrmse, avg_multi_r2)
    # plot_Fig6S()
    # plot_Fig7S(avg_one_nrmse, avg_one_r2, avg_multi_nrmse, avg_multi_r2, report_one_way, report_two_way)
    # plot_Fig9S(carbon_pca, strain_pca)


if __name__ == "__main__":
    main()
