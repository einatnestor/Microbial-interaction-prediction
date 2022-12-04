import copy
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from scipy.stats import pearsonr
import xgboost
from sklearn.metrics import classification_report
import oneWayStrength as ce
from minepy import *
import shap as shap
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix
import seaborn as sns


def createBinary(vector):
    """
    Convert the array to binary values (values greater than 0 assigned with 1)
    :param vector: Vector
    :return: Binary vector
    """
    y = np.array(vector)
    positive_interaction = np.where(y > 0)[0]
    negative_interaction = np.where(y <= 0)[0]
    y[positive_interaction] = 1
    y[negative_interaction] = -1
    return y


def createmultiLabel(effect_vector):
    """
    Convert the array to multi-lab.
    :param effect_vector: Effect vector
    :return: multi-label vector (1 for mutualism, 2 for competition and 3 for parasitism)
    """
    effect_vector = effect_vector.transpose()
    multiclass = np.full(effect_vector.shape[1], 3)
    index_mut = np.where((np.array(effect_vector)[0] == 1) & (np.array(effect_vector)[1] == 1))
    multiclass[index_mut] = 1
    index_comp = np.where((np.array(effect_vector)[0] == -1) & (np.array(effect_vector)[1] == -1))
    multiclass[index_comp] = 2
    return multiclass


def compare_two_vs_one_binary(one_way_train, y_train_one_way, one_way_test, y_test_one_way, X_train, X_test,
                              y_train_effects, y_test_effects):
    """
    Compare the performance of one-way sign model predictions vs two-way sign (multi label) model predictions.
    :param one_way_train: One-way train set
    :param y_train_one_way: y (one-way) labels train set
    :param one_way_test: One-way test set
    :param y_test_one_way:  y (one-way) labels test set
    :param X_train: Two-way train set
    :param X_test: Two-way test set
    :param y_train_effects: y (two-way) labels train set
    :param y_test_effects: y (two-way) labels test set
    """
    y_train_effects.loc[y_train_effects["effect_1_2"] > 0, 'effect_1_2'], y_train_effects.loc[
        y_train_effects["effect_1_2"] <= 0, 'effect_1_2'] = 1, -1
    y_train_effects.loc[y_train_effects["effect_2_1"] > 0, 'effect_2_1'], y_train_effects.loc[
        y_train_effects["effect_2_1"] <= 0, 'effect_2_1'] = 1, -1

    y_test_effects.loc[y_test_effects["effect_1_2"] > 0, 'effect_1_2'], y_test_effects.loc[
        y_test_effects["effect_1_2"] <= 0, 'effect_1_2'] = 1, -1
    y_test_effects.loc[y_test_effects["effect_2_1"] > 0, 'effect_2_1'], y_test_effects.loc[
        y_test_effects["effect_2_1"] <= 0, 'effect_2_1'] = 1, -1

    y_train_one_way.loc[y_train_one_way > 0,], y_train_one_way.loc[y_train_one_way <= 0] = 1, -1
    y_test_one_way.loc[y_test_one_way > 0,], y_test_one_way.loc[y_test_one_way <= 0] = 1, -1

    one_way_model = xgboost.XGBClassifier(base_score=0.5, booster='gbtree', colsample_bylevel=1,
              colsample_bynode=1, colsample_bytree=1, enable_categorical=False,
              eta=0.3, gamma=0.001, gpu_id=-1, importance_type=None,
              interaction_constraints='', learning_rate=0.1, max_delta_step=2,
              max_depth=15, min_child_weight=11,
              monotone_constraints='()', n_estimators=168, n_jobs=8,
              num_parallel_tree=1, predictor='auto', random_state=0,
              reg_alpha=0.0001, reg_lambda=0.8888888888888888,
              scale_pos_weight=1, subsample=1, tree_method='auto',
              validate_parameters=1, verbosity=None).fit(
        one_way_train.drop(['Bug 1', 'Bug 2', 'Carbon'], axis=1), y_train_one_way)
    prediction_binary = one_way_model.predict(one_way_test.drop(['Bug 1', 'Bug 2', 'Carbon'], axis=1))

    two_way_model = RandomForestClassifier(bootstrap=False,class_weight="balanced_subsample", max_features='log2', min_samples_split=5, n_estimators=239)
    two_way_model.fit(X_train, y_train_effects)
    two_way_binary_effects = two_way_model.predict(X_test)
    effect_2_1_index = int(prediction_binary.shape[0] / 2)
    y_test_multi = createmultiLabel(y_test_effects)
    one_way_binary_effects = pd.DataFrame(
        np.array([prediction_binary[:effect_2_1_index], prediction_binary[effect_2_1_index:]])).transpose()
    multi_one_way_binary = createmultiLabel(one_way_binary_effects)
    multi_two_way_binary = createmultiLabel(two_way_binary_effects)
    cm = confusion_matrix(y_test_multi,multi_one_way_binary, labels=(1,2,3))


    group_counts = ["{0:0.0f}".format(value) for value in
                    cm.flatten()]

    group_percentages = ["{0:.2%}".format(value) for value in
                         cm.flatten() / np.sum(cm)]
    labels = [f"{v2}\n{v3}" for  v2, v3 in
              zip(group_counts, group_percentages)]
    labels = np.asarray(labels).reshape(3, 3)
    ax = sns.heatmap(cm,  cmap='Blues', fmt="", cbar=True,annot =labels  )
    ax.set_ylabel('Actual Values')
    ax.set_xlabel('Predicted Values')
    #plt.savefig("Figures_msystems/FigureS11.png", dpi=300)
    # plt.show()
    target_names = ['Mutualism',"Competition",'Parasitism']
    report_one = classification_report(y_test_multi, multi_one_way_binary, target_names=target_names, output_dict=True)
    report_two = classification_report(y_test_multi, multi_two_way_binary, target_names=target_names, output_dict=True)
    return {'Accuracy': report_one['accuracy'], 'Weighted_avg': report_one['weighted avg']}, {
        'Accuracy': report_two['accuracy'], 'Weighted_avg': report_two['weighted avg']}


def preformanceWithReciprocal(X_train, X_test, y_train_effects, y_test_effects):
    """
    Compare the performance of model with the reciprocal effect as feature to a model without it.
    :param X_train: Train set.
    :param X_test: Test set.
    :param y_train_effects: y train labels.
    :param y_test_effects: y test labels.
    :return: Average of NRMSE and r2 for each type of model.
    """

    model_2_1_effect = xgboost.XGBRegressor(base_score=0.6, booster='gbtree', colsample_bylevel=1,
                 colsample_bynode=1, colsample_bytree=1, enable_categorical=False,
                 eta=0.5, gamma=0, gpu_id=-1, importance_type=None,
                 interaction_constraints='', learning_rate=0.1, max_delta_step=2,
                 max_depth=7, min_child_weight=9,
                 monotone_constraints='()', n_estimators=421, n_jobs=8,
                 num_parallel_tree=1, predictor='auto', random_state=0,
                 reg_alpha=0.1, reg_lambda=0.2222222222222222, scale_pos_weight=1,
                 subsample=0.8, tree_method='exact', validate_parameters=1,
                 verbosity=None)

    model_1_2_effect = xgboost.XGBRegressor(base_score=0.6, booster='gbtree', colsample_bylevel=1,
                 colsample_bynode=1, colsample_bytree=1, enable_categorical=False,
                 eta=0.5, gamma=0, gpu_id=-1, importance_type=None,
                 interaction_constraints='', learning_rate=0.1, max_delta_step=2,
                 max_depth=7, min_child_weight=9,
                 monotone_constraints='()', n_estimators=421, n_jobs=8,
                 num_parallel_tree=1, predictor='auto', random_state=0,
                 reg_alpha=0.1, reg_lambda=0.2222222222222222, scale_pos_weight=1,
                 subsample=0.8, tree_method='exact', validate_parameters=1,
                 verbosity=None)
    X_train_for_2_1, X_test_for_2_1 = copy.copy(X_train), copy.copy(X_test)

    X_train_for_2_1['reciprocal_effect'], X_test_for_2_1['reciprocal_effect'] = y_train_effects['effect_1_2'], \
                                                                                y_test_effects['effect_1_2']

    X_train_for_1_2, X_test_for_1_2 = copy.copy(X_train), copy.copy(X_test)
    X_train_for_1_2['reciprocal_effect'], X_test_for_1_2['reciprocal_effect'] = y_train_effects['effect_2_1'], \
                                                                                y_test_effects['effect_2_1']

    model_2_1_effect.fit(X_train_for_2_1, y_train_effects['effect_2_1'])
    prediction_2_1_effect = model_2_1_effect.predict(X_test_for_2_1)

    model_1_2_effect.fit(X_train_for_1_2, y_train_effects['effect_1_2'])
    prediction_1_2_effect = model_1_2_effect.predict(X_test_for_1_2)

    evaluate_2_1 = ce.evaluate(prediction_2_1_effect, y_test_effects['effect_2_1'])
    rmse_2_1, r2_2_1 = evaluate_2_1['RMSE'], evaluate_2_1['R2']

    evaluate_1_2 = ce.evaluate(prediction_1_2_effect, y_test_effects['effect_1_2'])
    rmse_1_2, r2_1_2 = evaluate_1_2['RMSE'], evaluate_1_2['R2']

    avg_nrmse = np.mean([rmse_2_1, rmse_1_2])
    avg_r2 = np.mean([r2_2_1, r2_1_2])

    # SHAP values
    #explainer = shap.Explainer(model_2_1_effect)
    #shap_values = explainer(X_test_for_2_1)
    #shap.summary_plot(shap_values,max_display=10, show=False)
    #shap.plots.waterfall(shap_values[3], max_display=10, show=False)
    #plt.savefig("Figures_msystems/FigureS9", dpi=300)
    return rmse_2_1, rmse_1_2, r2_2_1, avg_nrmse, avg_r2


def compareTwoVsOneEffect(one_way_train, y_train_one_way, one_way_test, y_test_one_way, X_train, X_test,
                          y_train_effects, y_test_effects):
    """
    Compare between two-way model performance and one-way model performance used twice.
    :param one_way_train: Train set for one-way predictions
    :param y_train_one_way: y train labels for one-way predictions
    :param one_way_test: Test set for one-way predictions
    :param y_test_one_way: y test labels for one-way predictions
    :param X_train: Train set for two-way predictions
    :param X_test: Test set for two-way predictions
    :param y_train_effects: y train labels for two-way predictions
    :param y_test_effects: y test labels for two-way predictions
    :return: Average of NRMSE and r2 for each type of model
    """
    one_way_model = xgboost.XGBRegressor(base_score=0.6, booster='gbtree', colsample_bylevel=1,
                 colsample_bynode=1, colsample_bytree=1, enable_categorical=False,
                 eta=0.5, gamma=0, gpu_id=-1, importance_type=None,
                 interaction_constraints='', learning_rate=0.1, max_delta_step=2,
                 max_depth=7, min_child_weight=9,
                 monotone_constraints='()', n_estimators=421, n_jobs=8,
                 num_parallel_tree=1, predictor='auto', random_state=0,
                 reg_alpha=0.1, reg_lambda=0.2222222222222222, scale_pos_weight=1,
                 subsample=0.8, tree_method='exact', validate_parameters=1,
                 verbosity=None).fit(one_way_train.drop(['Bug 1', 'Bug 2', 'Carbon'], axis=1), y_train_one_way)

    one_way_prediction = one_way_model.predict(one_way_test.drop(['Bug 1', 'Bug 2', 'Carbon'], axis=1))
    effect_2_1_index = int(one_way_prediction.shape[0] / 2)

    two_way_model = RandomForestRegressor(bootstrap=False, max_features='log2', min_samples_split=4, n_estimators=160)
    two_way_model.fit(X_train, y_train_effects)
    two_way_prediction = two_way_model.predict(X_test)

    rmse_2_1 = ce.evaluate(one_way_prediction[:effect_2_1_index], y_test_effects['effect_2_1'])['RMSE']
    rmse_1_2 = ce.evaluate(one_way_prediction[effect_2_1_index:], y_test_effects['effect_1_2'])['RMSE']

    r2_2_1 = ce.evaluate(one_way_prediction[:effect_2_1_index], y_test_effects['effect_2_1'])['R2']
    r2_1_2 = ce.evaluate(one_way_prediction[effect_2_1_index:], y_test_effects['effect_1_2'])['R2']

    avg_one_nrmse = np.mean([rmse_2_1, rmse_1_2])
    avg_one_r2 = np.mean([r2_2_1, r2_1_2])

    rmse_2_1_with_multi = ce.evaluate(two_way_prediction.transpose()[1], y_test_effects['effect_2_1'])['RMSE']
    rmse_1_2_with_multi = ce.evaluate(two_way_prediction.transpose()[0], y_test_effects['effect_1_2'])['RMSE']

    r2_2_1_with_multi = ce.evaluate(two_way_prediction.transpose()[1], y_test_effects['effect_2_1'])['R2']
    r2_1_2_with_multi = ce.evaluate(two_way_prediction.transpose()[0], y_test_effects['effect_1_2'])['R2']

    avg_multi_nrmse = np.mean([rmse_2_1_with_multi, rmse_1_2_with_multi])
    avg_multi_r2 = np.mean([r2_2_1_with_multi, r2_1_2_with_multi])
    return avg_one_nrmse, avg_one_r2, avg_multi_nrmse, avg_multi_r2


def create_one_way_data(X_train, X_test, y_train_effect_2_1, y_test_effect_2_1):
    """
    Create one-way dataset (train + test) where each effect in the two-way data will appear twice in the one-way dataset.
    :param X_train: Two-way train set
    :param X_test: Two-way test set
    :param y_train_effect_2_1: y train label of effect of bug 2 on bug 1
    :param y_test_effect_2_1: y train label of effect of bug 1 on bug 2
    """

    # train
    X_train['effect'] = y_train_effect_2_1
    copy = X_train[['Bug 1', 'Bug 2', 'Carbon']]
    full_one_way = pd.read_csv("Data_msystems/Features.csv").drop(['Unnamed: 0'], axis=1)
    copy = pd.merge(copy, full_one_way, how="inner", left_on=['Bug 1', 'Bug 2', 'Carbon'],
                    right_on=['Bug 2', 'Bug 1', 'Carbon']).drop(['Bug 1_x', 'Bug 2_x'], axis=1).rename(columns={'Bug 1_y': 'Bug 1', 'Bug 2_y': 'Bug 2', '2 on 1: Effect': 'effect'})
    one_way_train = pd.concat([X_train.drop(['1 on 2: Effect'], axis=1), copy],ignore_index=True)
    y_train = one_way_train['effect']
    one_way_train = one_way_train.drop(['effect'], axis=1)

    # test
    X_test['effect'] = y_test_effect_2_1
    copy = X_test[['Bug 1', 'Bug 2', 'Carbon']]
    full_one_way = pd.read_csv("Data_msystems/Features.csv").drop(['Unnamed: 0'], axis=1)
    copy = pd.merge(copy, full_one_way, how="inner", left_on=['Bug 1', 'Bug 2', 'Carbon'],
                    right_on=['Bug 2', 'Bug 1', 'Carbon']).drop(['Bug 1_x', 'Bug 2_x'], axis=1).rename(
        columns={'Bug 1_y': 'Bug 1', 'Bug 2_y': 'Bug 2', '2 on 1: Effect': 'effect'})
    one_way_test = pd.concat([X_test.drop(['1 on 2: Effect'], axis=1), copy])
    y_test = one_way_test['effect']
    one_way_test = one_way_test.drop(['effect'], axis=1)
    return one_way_train, y_train, one_way_test, y_test


def main():
    no_duplicate = pd.read_csv("Data_msystems/no_duplicates.csv")
    effect_2_1, X = no_duplicate[['2 on 1: Effect']], no_duplicate.drop(['Unnamed: 0', '2 on 1: Effect'], axis=1)
    X_train, X_test, y_train_effect_2_1, y_test_effect_2_1 = train_test_split(X, effect_2_1, test_size=0.2,
                                                                              random_state=66)
    one_way_train, y_train_one_way, one_way_test, y_test_one_way = create_one_way_data(X_train, X_test,
                                                                                       y_train_effect_2_1,
                                                                                       y_test_effect_2_1)
    with_duplicates = pd.read_csv("Data_msystems/Features_two_way.csv").drop(['Unnamed: 0'], axis=1)
    y_train_effects = pd.DataFrame()
    y_train_effects['effect_1_2'], y_train_effects['effect_2_1'] = X_train['1 on 2: Effect'], y_train_effect_2_1
    y_test_effects = pd.DataFrame()
    y_test_effects['effect_1_2'], y_test_effects['effect_2_1'] = X_test['1 on 2: Effect'], y_test_effect_2_1
    X_train, X_test = X_train.drop(['Carbon', 'Bug 1', 'Bug 2', '1 on 2: Effect', 'effect'], axis=1), X_test.drop(
        ['Carbon', 'Bug 1', 'Bug 2', '1 on 2: Effect', 'effect'], axis=1)
    with_duplicates = with_duplicates[
        ((with_duplicates['Carbon'] != 'Water') & (with_duplicates['monoGrow_x'] != with_duplicates['monoGrow_y']))]

    # correlations #
    # Pearson correlation
    corr, p = pearsonr(with_duplicates['1 on 2: Effect'], with_duplicates['2 on 1: Effect'])
    # mic
    mine = MINE()
    mine.compute_score(with_duplicates['1 on 2: Effect'], with_duplicates['2 on 1: Effect'])
    mic = mine.mic()

    # Strength prediction
    avg_one_nrmse, avg_one_r2, avg_multi_nrmse, avg_multi_r2 = compareTwoVsOneEffect(one_way_train,
                                                                                                    y_train_one_way,
                                                                                                    one_way_test,
                                                                                                    y_test_one_way,
                                                                                                    X_train, X_test,
                                                                                                    y_train_effects,
                                                                                                    y_test_effects)
    nrmse_2_1_partial, nrmse_1_2_partial, r2_2_1_partial, avg_nrmse_reciprocal, avg_r2 = preformanceWithReciprocal(
        X_train, X_test, y_train_effects,
        y_test_effects)
    # Sign prediction
    report_one_way, report_two_way = compare_two_vs_one_binary(one_way_train, y_train_one_way, one_way_test,
                                                               y_test_one_way, X_train, X_test, y_train_effects,
                                                               y_test_effects)

    return with_duplicates['1 on 2: Effect'], with_duplicates[
        '2 on 1: Effect'], nrmse_2_1_partial, r2_2_1_partial, avg_nrmse_reciprocal, avg_one_nrmse, avg_one_r2, avg_multi_nrmse, avg_multi_r2, report_one_way, report_two_way


if __name__ == "__main__":
    main()
