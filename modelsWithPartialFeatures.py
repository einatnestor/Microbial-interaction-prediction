import pandas as pd
import numpy as np
from createData import createPCA
from sklearn.metrics import accuracy_score, precision_score, mean_squared_error, matthews_corrcoef, recall_score
from oneWaySign import evaluate
from oneWayStrength import evaluate as evaluate_effect
import naiveModelStrain as nms
import naiveModelEnvironment as nme
import csv
import xgboost
from sklearn.decomposition import PCA
import random

STRAINS = ['PR2', 'PR1', 'PAr', 'EC', 'EL', 'LA', 'RP2', 'RP1', 'EA', 'SF1', 'BI', 'KA', 'CF', 'PAg2', 'PAg1', 'PAl',
           'PAg3', 'PH', 'PK', 'PP']

CARBONS = ['ArabinoseD',
           'ArabinoseL',
           'Xylose',
           'Ribose',
           'Rhamnose',
           'Fructose',
           'Galactose',
           'Glucose',
           'Glucose01',
           'Mannose',
           'GlcNAc',
           'Acetate',
           'Pyruvate',
           'Pyruvate01',
           'Fumarate',
           'Succinate',
           'Citrate',
           'Glycerol',
           'Glycerol01',
           'Mannitol',
           'Sorbitol',
           'Alanine',
           'Serine',
           'Proline',
           'Proline01',
           'Glutamine',
           'Arginine',
           'Trehalose',
           'Cellobiose',
           'Maltose',
           'Sucrose',
           'Sucrose01',
           'Lactose',
           'Raffinose',
           'Melezitose',
           'Arabinogalactan',
           'Isoleucine',
           'Uridine',
           'Mix',
           'Water']


def readTable(df, type):
    """
    Separate the data to features and labels.
    :param df: the dataset
    :param type: the kind of labels
    :return: X,y ,carbon pca
    """
    y = df['2 on 1: Effect']
    y = np.array(y)
    carbon_pca = df[['Carbon', 'carbon_component_0', 'carbon_component_1', 'carbon_component_2', 'carbon_component_3']]
    carbon_pca = carbon_pca.drop_duplicates()
    df = df.drop(
        columns=['2 on 1: Effect', 'carbon_component_0', 'carbon_component_1', 'carbon_component_2',
                 'carbon_component_3'], axis=1)
    if type == "classification":
        positive_interaction = np.where(y > 0)[0]
        negative_interaction = np.where(y <= 0)[0]
        y[positive_interaction] = 1
        y[negative_interaction] = -1
    return df, np.array(y), carbon_pca


def calcDistanceCarbon(index, info, names, ith_closest):
    """
    Find the ith closest carbon
    :param index: the index of the current carbon
    :param info:  metabolic pca
    :param names: carbons' names
    :param ith_closest: the ith wanted value
    :return:
    """
    # calculate according to metabolic pca
    info = np.array(info[['carbon_component_0', 'carbon_component_1', 'carbon_component_2', 'carbon_component_3']])
    c1 = info[index]
    distance_arr = []
    names_arr = []
    for i in range(info.shape[0]):
        if i != index:
            c2 = info[i]
            e_distance = np.sqrt(np.sum(np.power(c2 - c1, 2)))
            distance_arr.append(e_distance)
            names_arr.append(names[i])

    distances_df = pd.DataFrame({"carbon": names_arr, "distance": distance_arr})
    distances_df = distances_df.sort_values('distance')
    distances_df = np.array(distances_df.iloc[ith_closest])
    return distances_df[0], distances_df[1]


def readPhylogeneticMatrix():
    """
    resf phylogenetic distance matrix
    :return: Phylogenetic distance matrix.
    """
    table = pd.read_csv("./Data/pairwiseDistances-21Isolates&Ecoli.csv")
    table = table.rename(columns={"Unnamed: 0": "strain"})
    table = table.drop([9, 11])
    table = table.drop(columns=["SF2", "CB"])
    table['new_index'] = [i for i in range(20)]
    table = table.set_index(['new_index'])
    return table


def calcDistance(type, ith_closest):
    """
    find the ith closest strain/carbon.
    :param type: strain/carbon
    :param ith_closest: the ith wanted value
    :return: distance_df
    """
    distance_rr, closest = [], []
    # distance for carbon - using Metabolic pca, for strains phylogenetic distance matrix
    if type == "carbon":
        metabolic_distance_matrix = createPCA("carbon", 4)
        names = metabolic_distance_matrix['carbon']
        for i in range(metabolic_distance_matrix.shape[0]):
            name, distance = calcDistanceCarbon(i, metabolic_distance_matrix, names, ith_closest)
            distance_rr.append(distance)
            closest.append(name)
    else:
        phylogenetic_distance_matrix = readPhylogeneticMatrix()
        names = phylogenetic_distance_matrix['strain']
        phylogenetic_distance_matrix = phylogenetic_distance_matrix.replace(0, float("inf")).drop(['strain'], axis=1)
        closest, distance_rr = phylogenetic_distance_matrix.idxmin(axis=1), phylogenetic_distance_matrix.min(axis=1)

    if type == "carbon":
        distance_df = pd.DataFrame({"carbon": names, "closest_carbon": closest, "distance": distance_rr})

    else:
        distance_df = pd.DataFrame({"strain": names, "closest_strain": closest, "distance": distance_rr})
    return distance_df


def runAllStrainsMetrics(X, y, modelType, mono):
    """
    Run partial strain models
    :param X: features
    :param y: labels
    :param modelType: classification/regression
    :param mono: bool for mono features
    :return: metrics df
    """
    if modelType == 'classification':
        metricsPerStrain = runClassification(X, y, mono)

    else:  # regression
        metricsPerStrain = runRegression(X, y, mono)

    return metricsPerStrain


def runAllCarbonsMetrics(X, y, modelType, mono):
    """
    Run partial carbon models
    :param X: Features
    :param y: labels
    :param modelType: classification/regression
    :param mono: bool for mono features
    :return: metrics df
    """
    if modelType == 'classification':
        metricsPerCarbon = runClassificationCarbon(X, y, mono)

    else:
        metricsPerCarbon = runRegressionCarbon(X, y, mono)

    return metricsPerCarbon


def createArray(allOtherCarbons):
    """
    Cretae array of the input carbons
    """
    otherArr = []
    for index in allOtherCarbons:
        otherArr.append(CARBONS[index])
    return otherArr


def findindex(carbon):
    """
    Find index of the carbon to remove.
    """
    index = CARBONS.index(carbon)
    allCabrbons = [i for i in range(40)]
    return [index], allCabrbons[:index] + allCabrbons[index + 1:]


def createMetricsSign(type, mono, acc_arr_rf, mcc_arr_rf, recall_arr_rf, precision_arr_rf,
                      acc_arr_naive_predictor, mcc_arr_naive_predictor, recall_arr_naive_predictor,
                      precision_arr_naive_predictor):
    """
    Create a df of partial sign scores (for strain or carbon)
    """
    name = "_without_mono"
    if mono:
        name = "_with_mono"
        if type == "strain":
            metrics = pd.DataFrame({'strain': STRAINS, "accuracy" + name: acc_arr_rf,
                                    "mcc" + name: mcc_arr_rf,
                                    "recall" + name: recall_arr_rf,
                                    "precision" + name: precision_arr_rf,
                                    "accuracy_copy_model": acc_arr_naive_predictor,
                                    "mcc_copy_model": mcc_arr_naive_predictor,
                                    "recall_copy_model": recall_arr_naive_predictor,
                                    "precision_copy_model": precision_arr_naive_predictor})
        else:
            metrics = pd.DataFrame({'carbon': CARBONS, "accuracy" + name: acc_arr_rf,
                                    "mcc" + name: mcc_arr_rf,
                                    "recall" + name: recall_arr_rf,
                                    "precision" + name: precision_arr_rf,
                                    "accuracy_copy_model": acc_arr_naive_predictor,
                                    "mcc_copy_model": mcc_arr_naive_predictor,
                                    "recall_copy_model": recall_arr_naive_predictor,
                                    "precision_copy_model": precision_arr_naive_predictor})
    else:
        if type == "strain":
            metrics = pd.DataFrame({'strain': STRAINS, "accuracy" + name: acc_arr_rf,
                                    "mcc" + name: mcc_arr_rf,
                                    "recall" + name: recall_arr_rf,
                                    "precision" + name: precision_arr_rf})
        else:
            metrics = pd.DataFrame({'carbon': CARBONS, "accuracy" + name: acc_arr_rf,
                                    "mcc" + name: mcc_arr_rf,
                                    "recall" + name: recall_arr_rf,
                                    "precision" + name: precision_arr_rf})

    return metrics


def createMetricsStrength(type, mono, rmse_arr_rf, r2_arr_rf, rmse_arr_naive, r2_arr_naive):
    """
    Create a df of partial strength scores (for strain or carbon)
    """
    name = "_without_mono"
    if mono:
        name = "_with_mono"
        if type == "strain":
            metrics = pd.DataFrame({'strain': STRAINS, "nrmse" + name: rmse_arr_rf,
                                    "r2" + name: r2_arr_rf,
                                    "nrmse_copy_model": rmse_arr_naive,
                                    "r2_copy_model": r2_arr_naive})
        else:
            metrics = pd.DataFrame({'carbon': CARBONS, "nrmse" + name: rmse_arr_rf,
                                    "r2" + name: r2_arr_rf,
                                    "nrmse_copy_model": rmse_arr_naive,
                                    "r2_copy_model": r2_arr_naive})
    else:
        if type == "strain":
            metrics = pd.DataFrame({'strain': STRAINS, "nrmse" + name: rmse_arr_rf,
                                    "r2" + name: r2_arr_rf})
        else:
            metrics = pd.DataFrame({'carbon': CARBONS, "nrmse" + name: rmse_arr_rf,
                                    "r2" + name: r2_arr_rf})
    return metrics


def createPcaForCarbron(dimensions, carbon, new_carbon, allOtherCarbons):
    """
    Create a PCA without a specific carbon source. (new carbon)
    """
    carbon_table = pd.read_csv("Data/environments_profiles.csv")
    withoutNew = carbon_table.iloc[allOtherCarbons, 1:]
    newCarbon = carbon_table.iloc[new_carbon, 1:]
    pca = PCA()
    principalComponents = pca.fit_transform(withoutNew)
    newCarbonComponent = pca.transform(newCarbon)
    pca_components = np.array(principalComponents).transpose()[:dimensions]
    newCarbonComponent = np.array(newCarbonComponent).transpose()[:dimensions]
    pca_df = pd.DataFrame()
    newCarbon = pd.DataFrame()
    otherCarbons = createArray(allOtherCarbons)
    pca_df['carbon'] = otherCarbons
    newCarbon['carbon'] = [carbon]
    index = 0
    while index < dimensions:
        pca_df["carbon_component_" + str(index)] = pca_components[index]
        newCarbon["carbon_component_" + str(index)] = newCarbonComponent[index]
        index += 1
    return pd.concat([pca_df, newCarbon], ignore_index=True)


def createTeststrain(X, y, strain):
    """
    Create train and test sets for a given strain
    :param X: dataset
    :param y: labels
    :param strain: a strain to include in train and remove from test
    :return: X_train, X_test, y_train, y_test
    """
    X['y'] = y
    X_test = X[(X['Bug 1'] == strain) | (X['Bug 2'] == strain)]
    y_test = X_test['y']
    X_test = X_test.drop(["y"], axis=1)
    X_train = X[(X['Bug 1'] != strain) & (X['Bug 2'] != strain)]
    y_train = X_train['y']
    X_train = X_train.drop(["y"], axis=1)
    return X_train, X_test, y_train, y_test


def createTestCarbon(X, y, carbon):
    """
    Create a test set including a given environment.
    """
    X['y'] = y
    X_test = X[(X['carbon'] == carbon)]
    y_test = X_test['y']
    X_test = X_test.drop(["y"], axis=1)
    X_train = X[X['carbon'] != carbon]
    y_train = X_train['y']
    X_train = X_train.drop(["y"], axis=1)
    return X_train, X_test, y_train, y_test


def metricPerStrain(X_train, y_train, X_test, y_test):
    model = xgboost.XGBClassifier(base_score=0.5, booster='gbtree',
                                  colsample_bynode=1,
                                  eta=0.3, gamma=0.001,
                                  interaction_constraints='', learning_rate=0.1, max_delta_step=0,
                                  max_depth=15, min_child_weight=12,
                                  n_estimators=182,
                                  predictor='auto', random_state=0,
                                  reg_alpha=1e-05, reg_lambda=0.6666666666666666,
                                  scale_pos_weight=1, subsample=1, tree_method='exact')
    model.fit(X_train, y_train)
    prediction = model.predict(X_test)
    scores = evaluate(prediction, y_test)
    return scores


def FullyTrainedStrengthStrain(X_train, y_train, X_test, y_test):
    """
    Performance of the fully train strength model (and null models) on a test set, goruped by strain/environment.
    """
    model = xgboost.XGBRegressor(base_score=0.2, booster='gbtree', colsample_bylevel=0.4,
                                 colsample_bynode=1, colsample_bytree=1,
                                 eta=0.3, gamma=0.01,
                                 learning_rate=0.1, max_delta_step=2,
                                 max_depth=9, min_child_weight=11,
                                 n_estimators=342, n_jobs=8,
                                 num_parallel_tree=1, predictor='auto', random_state=0,
                                 reg_alpha=0.1, reg_lambda=0.2222222222222222, scale_pos_weight=1,
                                 subsample=0.8, tree_method='exact').fit(X_train, y_train)

    prediction = model.predict(X_test)
    scores = evaluate_effect(prediction, y_test)
    return scores


def runClassification(X, y, mono):
    """
    Train partial sign predictions. For each strain, a train + test sets are created, and the phylogenetic pca
    is changing as well (find PC without the strain information)
    :param X: full features table
    :param y: labels
    :param mono: bool for addition of monogrow features
    :return: metrics df
    """

    distance_matrix_strains = readPhylogeneticMatrix()
    acc_arr_rf, mcc_arr_rf, precision_arr_rf, recall_arr_rf, acc_arr_naive_predictor, mcc_arr_naive_predictor, precision_arr_naive_predictor, recall_arr_naive_predictor = [], [], [], [], [], [], [], []

    for strain in STRAINS:
        df_copy = X
        if not mono:
            df_copy = df_copy.drop(columns=['monoGrow_x', 'monoGrow_y', 'monoGrow24_x', 'monoGrow24_y', 'metDis'],
                                   axis=1)
            pca_carbon_no_strain = createPCA("carbon", 4, strain)
        else:
            pca_carbon_no_strain = createPCA("carbon", 4)
        df_copy = pd.merge(df_copy, pca_carbon_no_strain, how='left', right_on=['carbon'],
                           left_on=['Carbon']).drop(columns=['Carbon'], axis=1)

        X_train_without_strain, X_test_without_strain, y_train_without_strain, y_test_without_strain = createTeststrain(
            df_copy, y, strain)
        X_train_without_strain_x_rf, X_test_without_strain_x_rf = X_train_without_strain.drop(
            ['Bug 1', 'Bug 2', 'carbon'], axis=1), X_test_without_strain.drop(['Bug 1', 'Bug 2', 'carbon'], axis=1)

        scores = metricPerStrain(X_train_without_strain_x_rf,
                                 y_train_without_strain,
                                 X_test_without_strain_x_rf,
                                 y_test_without_strain)

        naive_model = nms.naiveModel(X_train_without_strain, y_train_without_strain, strain, distance_matrix_strains)
        first_y_true, first_y_prediction = naive_model.predict(X_test_without_strain, y_test_without_strain, 0)
        scores_naive = evaluate(first_y_prediction, first_y_true)

        acc_arr_rf.append(scores['Accuracy'])
        mcc_arr_rf.append(scores['MCC'])
        recall_arr_rf.append(scores['Recall'])
        precision_arr_rf.append(scores['Precision'])

        acc_arr_naive_predictor.append(scores_naive['Accuracy'])
        mcc_arr_naive_predictor.append(scores_naive['MCC'])
        recall_arr_naive_predictor.append(scores_naive['Recall'])
        precision_arr_naive_predictor.append(scores_naive['Precision'])
        df_copy = pd.DataFrame()

    metrics_per_strain = createMetricsSign("strain", mono, acc_arr_rf, mcc_arr_rf, recall_arr_rf,
                                           precision_arr_rf, acc_arr_naive_predictor,
                                           mcc_arr_naive_predictor, recall_arr_naive_predictor,
                                           precision_arr_naive_predictor)

    return metrics_per_strain


def runRegression(X, y, mono):
    """
    Train partial strength predictions. For each strain, a train + test sets are created, and the phylogenetic pca
    is changing as well (find PC without the strain information)
    :param X: full features table
    :param y: labels
    :param mono: bool for addition of monogrow features
    :return: metrics df
    """
    distance_matrix_strains = readPhylogeneticMatrix()
    rmse_arr_rf, r2_arr_rf, rmse_arr_naive, r2_arr_naive = [], [], [], []

    for strain in STRAINS:
        df_copy = X
        if not mono:
            df_copy = df_copy.drop(columns=['monoGrow_x', 'monoGrow_y', 'monoGrow24_x', 'monoGrow24_y', 'metDis'],
                                   axis=1)
            pca_carbon_no_strain = createPCA("carbon", 4, strain)

        else:
            pca_carbon_no_strain = createPCA("carbon", 4)

        df_copy = pd.merge(df_copy, pca_carbon_no_strain, how='left', right_on=['carbon'],
                           left_on=['Carbon']).drop(columns=['Carbon'], axis=1)

        X_train_without_strain_x, X_test_without_strain_x, y_train_without_strain_x, y_test_without_strain_x = createTeststrain(
            df_copy, y, strain)

        scores = FullyTrainedStrengthStrain(X_train_without_strain_x.drop(['Bug 1', 'Bug 2', 'carbon'], axis=1),
                                            y_train_without_strain_x,
                                            X_test_without_strain_x.drop(['Bug 1', 'Bug 2', 'carbon'], axis=1),
                                            y_test_without_strain_x)

        naive_model = nms.naiveModel(X_train_without_strain_x, y_train_without_strain_x, strain,
                                     distance_matrix_strains)
        naive_y_true, naive_y_prediction = naive_model.predict(X_test_without_strain_x, y_test_without_strain_x, 0)
        scores_naive = evaluate_effect(naive_y_prediction, naive_y_true)

        rmse_arr_rf.append(scores['RMSE'])
        r2_arr_rf.append(scores['R2'])

        rmse_arr_naive.append(scores_naive['RMSE'])
        r2_arr_naive.append(scores_naive['R2'])
        df_copy = pd.DataFrame()

    metrics_per_strain = createMetricsStrength("strain", mono, rmse_arr_rf, r2_arr_rf, rmse_arr_naive, r2_arr_naive)
    return metrics_per_strain


def runClassificationCarbon(X, y, mono):
    """
    Train partial sign predictions. For each carbon, a train + test sets are created, and the metabolic pca
    is changing as well (find PC without the carbon information)
    :param X: full features table
    :param y: labels
    :param mono: bool for addition of monogrow features
    :return: metrics df
    """
    acc_arr_rf, mcc_arr_rf, precision_arr_rf, recall_arr_rf, acc_arr_naive_predictor, mcc_arr_naive_predictor, precision_arr_naive_predictor, recall_arr_naive_predictor = [], [], [], [], [], [], [], []

    if mono:
        pca_carbon = createPCA("carbon", 4)
    for carbon in CARBONS:
        newCarbonindex, OtherCarbonsIndex = findindex(carbon)
        df_copy = X
        if not mono:
            pca_carbon = createPcaForCarbron(4, carbon, newCarbonindex, OtherCarbonsIndex)
        df_copy = pd.merge(df_copy, pca_carbon, how='left', right_on=['carbon'], left_on=['Carbon']).drop(
            columns=['Carbon'])

        if not mono:
            df_copy = df_copy.drop(columns=['monoGrow_x', 'monoGrow_y', 'monoGrow24_x', 'monoGrow24_y', 'metDis'],
                                   axis=1)
        X_train_without_carbon, X_test_without_carbon, y_train_without_carbon, y_test_without_carbon = createTestCarbon(
            df_copy, y, carbon)
        scores = metricPerStrain(X_train_without_carbon.drop(['Bug 1', 'Bug 2', 'carbon'], axis=1),
                                 y_train_without_carbon,
                                 X_test_without_carbon.drop(['Bug 1', 'Bug 2', 'carbon'], axis=1),
                                 y_test_without_carbon)

        naive_model = nme.naiveModel(X_train_without_carbon, y_train_without_carbon, carbon, pca_carbon)
        first_y_true, first_y_prediction = naive_model.predict(X_test_without_carbon, y_test_without_carbon, 0)
        scores_naive = evaluate(
            first_y_prediction, first_y_true)

        acc_arr_rf.append(scores['Accuracy'])
        mcc_arr_rf.append(scores['MCC'])
        recall_arr_rf.append(scores['Recall'])
        precision_arr_rf.append(scores['Precision'])

        acc_arr_naive_predictor.append(scores_naive['Accuracy'])
        mcc_arr_naive_predictor.append(scores_naive['MCC'])
        recall_arr_naive_predictor.append(scores_naive['Recall'])
        precision_arr_naive_predictor.append(scores_naive['Precision'])

        # df_copy = pd.DataFrame()
    metrics_per_carbon = createMetricsSign("carbon", mono, acc_arr_rf, mcc_arr_rf, recall_arr_rf,
                                           precision_arr_rf, acc_arr_naive_predictor,
                                           mcc_arr_naive_predictor, recall_arr_naive_predictor,
                                           precision_arr_naive_predictor)
    return metrics_per_carbon


def runRegressionCarbon(X, y, mono):
    """
    Train partial strength predictions. For each carbon, a train + test sets are created, and the metabolic pca
    is changing as well (find PC without the carbon information)
    :param X: full features table
    :param y: labels
    :param mono: bool for addition of monogrow features
    :return: metrics df
    """
    rmse_arr_rf, r2_arr_rf, rmse_arr_naive, r2_arr_naive = [], [], [], []
    if mono:
        pca_carbon = createPCA("carbon", 4)
    for carbon in CARBONS:
        newCarbonindex, OtherCarbonsIndex = findindex(carbon)
        df_copy = X
        if not mono:
            pca_carbon = createPcaForCarbron(4, carbon, newCarbonindex, OtherCarbonsIndex)
        df_copy = pd.merge(df_copy, pca_carbon, how='left', right_on=['carbon'],
                           left_on=['Carbon']).drop(columns=['Carbon'])
        if not mono:
            df_copy = df_copy.drop(columns=['monoGrow_x', 'monoGrow_y', 'monoGrow24_x', 'monoGrow24_y', 'metDis'],
                                   axis=1)
        X_train_without_carbon_x, X_test_without_carbon_x, y_train_without_carbon_x, y_test_without_carbon_x = createTestCarbon(
            df_copy, y, carbon)
        X_train_without_carbon_x_rf = X_train_without_carbon_x.drop(['Bug 1', 'Bug 2', 'carbon'], axis=1)
        X_test_without_carbon_x_rf = X_test_without_carbon_x.drop(['Bug 1', 'Bug 2', 'carbon'], axis=1)
        scores = FullyTrainedStrengthStrain(X_train_without_carbon_x_rf,
                                            y_train_without_carbon_x,
                                            X_test_without_carbon_x_rf,
                                            y_test_without_carbon_x)
        naive_model = nme.naiveModel(X_train_without_carbon_x, y_train_without_carbon_x, carbon, pca_carbon)
        naive_y_true, naive_y_prediction = naive_model.predict(X_test_without_carbon_x, y_test_without_carbon_x, 0)
        scores_naive = evaluate_effect(naive_y_prediction, naive_y_true)
        rmse_arr_rf.append(scores['RMSE'])
        r2_arr_rf.append(scores['R2'])
        rmse_arr_naive.append(scores_naive['RMSE'])
        r2_arr_naive.append(scores_naive['R2'])
        df_copy = pd.DataFrame()

    metrics_per_carbon = createMetricsStrength("carbon", mono, rmse_arr_rf, r2_arr_rf, rmse_arr_naive, r2_arr_naive)
    return metrics_per_carbon


def signNaiveAccIthClosest(X, y):
    """
    Sign predictions using phylogenetic copy model.
    :param X: Features
    :param y: labels
    :return:
    """
    allDistancesdf = pd.DataFrame(
        {'acc': [], 'mcc': [], 'Recall': [], 'Precision': [], 'strain': [], 'closest_strain': [], 'distance': [],
         'ith': []})
    distance_matrix = readPhylogeneticMatrix()
    for strain in STRAINS:
        df_copy = X
        pca_carbon_no_strain = createPCA("carbon", 4, strain)
        df_copy = pd.merge(df_copy, pca_carbon_no_strain, how='left', right_on=['carbon'],
                           left_on=['Carbon']).drop(columns=['Carbon'], axis=1)
        X_train_without_strain_x, X_test_without_strain_x, y_train_without_strain_x, y_test_without_strain_x = createTeststrain(
            df_copy, y, strain)
        naive_model = nms.naiveModel(X_train_without_strain_x, y_train_without_strain_x, strain, distance_matrix)
        tmpArrAcc, tmpArrMcc, tmpArrRecall, tmpArrPrecision = [], [], [], []
        for i in range(len(STRAINS) - 1):
            y_true, y_prediction = naive_model.predict(X_test_without_strain_x, y_test_without_strain_x, i)
            ith_distance, ith_strain = naive_model.getInfo()
            scores_naive = evaluate(y_true, y_prediction)
            tmpArrAcc.append(scores_naive['Accuracy'])
            tmpArrMcc.append(scores_naive['MCC'])
            tmpArrRecall.append(scores_naive['Recall'])
            tmpArrPrecision.append(scores_naive['Precision'])
            new_line = {'acc': scores_naive['Accuracy'], 'mcc': scores_naive['MCC'], 'Recall': scores_naive['Recall'],
                        'Precision': scores_naive['Precision'], 'strain': strain, 'closest_strain': ith_strain[0],
                        'distance': ith_distance[0], 'ith': i}
            allDistancesdf = allDistancesdf.append(new_line, ignore_index=True)
    # allDistancesdf.to_csv("Naive_copy_sign_per_distance.csv")
    return


def strengthNaiveAccIthClosest(X, y):
    """
    Strength predictions using phylogenetic copy model.
    :param X: Features
    :param y: labels
    :return:
    """
    allDistancesdf = pd.DataFrame(
        {'r2': [], 'rmse': [], 'strain': [], 'closest_strain': [], 'distance': [], 'ith': []})
    distance_matrix = readPhylogeneticMatrix()
    for strain in STRAINS:
        df_copy = X
        df_copy['y'] = y
        pca_carbon_no_strain = createPCA("carbon", 4, strain)
        df_copy = pd.merge(df_copy, pca_carbon_no_strain, how='left', right_on=['carbon'],
                           left_on=['Carbon']).drop(columns=['Carbon'], axis=1)
        X_train_without_strain_x, X_test_without_strain_x, y_train_without_strain_x, y_test_without_strain_x = createTeststrain(
            df_copy, y, strain)
        naive_model = nms.naiveModel(X_train_without_strain_x, y_train_without_strain_x, strain, distance_matrix)
        tmpArrR2, tmpArrRMSE = [], []
        for i in range(len(STRAINS) - 1):
            tmp_y_true, tmp_y_prediction = naive_model.predict(X_test_without_strain_x, y_test_without_strain_x, i)

            ith_distance, ith_strain = naive_model.getInfo()
            scores_naive = evaluate_effect(tmp_y_prediction, tmp_y_true)
            tmpArrR2.append(scores_naive['RMSE'])
            tmpArrRMSE.append(scores_naive['R2'])
            new_line = {'r2': scores_naive['R2'], 'rmse': scores_naive['RMSE'], 'strain': strain,
                        'closest_strain': ith_strain[0],
                        'distance': ith_distance[0], 'ith': i}
            allDistancesdf = allDistancesdf.append(new_line, ignore_index=True)
    # allDistancesdf.to_csv("Naive_copy_strength_scores_per_distance.csv")
    return


def saveSign(type, index_df, metrics_per_strain_mono, metrics_per_strain_no_mono, full_metrics):
    """
    Save sign partial models scores
    """
    pairplot_strain = pd.merge(pd.merge(index_df, metrics_per_strain_mono, how="inner", right_on=type,
                                        left_on="index"), metrics_per_strain_no_mono, how="inner",
                               right_on=type,
                               left_on="index")

    pairplot_strain = pairplot_strain[
        ['index', 'accuracy', 'mcc', 'recall', 'precision', 'accuracy_with_mono',
         'mcc_with_mono', 'recall_with_mono',
         'precision_with_mono',
         'accuracy_without_mono',
         'mcc_without_mono', 'recall_without_mono',
         'precision_without_mono',
         'accuracy_copy_model', 'mcc_copy_model', 'recall_copy_model', 'precision_copy_model']]
    pairplot_strain.to_csv("Paper_full_Classification_xgb_" + type + ".csv")

    with open("full_metrics_" + type + "_classification.csv", 'w') as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=['Accuracy', 'Precision', 'Recall', 'f1', 'MCC'])
        writer.writeheader()
        writer.writerow(full_metrics)


def saveStrength(type, index_df, metrics_per_strain_mono, metrics_per_strain_no_mono, full_metrics):
    """
    Save strength partial models scores
    """
    pairplot_strain = pd.merge(pd.merge(index_df, metrics_per_strain_mono, how="inner", right_on=type,
                                        left_on="index"), metrics_per_strain_no_mono, how="inner",
                               right_on=type,
                               left_on="index")

    pairplot_strain = pairplot_strain[
        ['index', 'nrmse', 'nrmse_null', 'nrmse_with_mono', 'nrmse_without_mono', 'nrmse_copy_model']]
    pairplot_strain.to_csv("Paper_full_Regression_" + type + ".csv")

    with open("full_metrics_" + type + "_regression.csv", "w") as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=['R2', 'RMSE'])
        writer.writeheader()
        writer.writerow(full_metrics)


def testUknown(X, y, untrainedType, modelType):
    """
    Train and predict unmeasured strain/environment using partial models.
    :param X: Features
    :param y: y labels
    :param untrainedType: Strain/environment
    :param modelType: sign/strength
    :return:
    """
    # sign prediction
    if modelType == 'classification':
        strain_df, carbon_df, full_metrics = signMetricasGroupByStrain()

        if untrainedType == 'strain':
            distance_info_strain = calcDistance("strain", 0)
            metrics_per_strain_mono = runAllStrainsMetrics(X, y, modelType, True)
            metrics_per_strain_no_mono = runAllStrainsMetrics(X, y, modelType, False)
            signNaiveAccIthClosest(X, y)
            # saveClassification(untrainedType, strain_df, metrics_per_strain_mono, metrics_per_strain_no_mono,
            #                   full_metrics)

        elif untrainedType == 'carbon':
            distance_info_carbon = calcDistance("carbon", 0)
            metrics_per_carbon_mono = runAllCarbonsMetrics(X, y, modelType, True)
            metrics_per_carbon_no_mono = runAllCarbonsMetrics(X, y, modelType, False)
            # saveClassification(untrainedType, carbon_df, metrics_per_carbon_mono, metrics_per_carbon_no_mono,
            #                   full_metrics)

    # strength prediction
    else:
        strain_df, carbon_df, full_metrics = strengthMetricasGroupByStrain()

        if untrainedType == 'strain':
            strengthNaiveAccIthClosest(X, y)
            distance_info_strain = calcDistance("strain", 0)
            metrics_per_strain_mono = runAllStrainsMetrics(X, y, modelType, True)
            metrics_per_strain_no_mono = runAllStrainsMetrics(X, y, modelType, False)
            # saveRegression(untrainedType, strain_df, metrics_per_strain_mono, metrics_per_strain_no_mono, full_metrics)

        elif untrainedType == 'carbon':
            distance_info_carbon = calcDistance("carbon", 0)
            metrics_per_carbon_mono = runAllCarbonsMetrics(X, y, modelType, True)
            metrics_per_carbon_no_mono = runAllCarbonsMetrics(X, y, modelType, False)
            # saveRegression(untrainedType, carbon_df, metrics_per_carbon_mono, metrics_per_carbon_no_mono,full_metrics)


def create_binary(y):
    """
    Create sign labels
    :param y: effect vector
    :return: sign effect vector
    """
    y_binary = np.full(y.shape[0], 1)
    y_binary[np.where(y <= 0)[0]] = -1
    return y_binary


def fullMetricsGroupedByStrainOrEnv(X_test):
    """
    Calculates all metrics for the full model (sign prediction)
    :param X_test: A df contains all needed information
    :return: strain df, carbon df
    """
    strain_names, carbon_names = [], []

    acc_strain, acc_null_strain = [], []
    acc_carbon, acc_null_carbon = [], []

    mcc_strain, mcc_null_strain = [], []
    mcc_carbon, mcc_null_carbon = [], []

    recall_strain, recall_null_strain = [], []
    recall_carbon, recall_null_carbon = [], []

    precision_strain, precision_null_strain = [], []
    precision_carbon, precision_null_carbon = [], []

    for name, group in X_test.groupby('strain_x'):
        mcc_strain.append(matthews_corrcoef(group['y'], group['y_pred']))
        acc_strain.append(accuracy_score(group['y'], group['y_pred']))
        recall_strain.append(recall_score(group['y'], group['y_pred']))
        precision_strain.append(precision_score(group['y'], group['y_pred']))

        mcc_null_strain.append(matthews_corrcoef(group['y'], group['neg_pred']))
        acc_null_strain.append(accuracy_score(group['y'], group['neg_pred']))
        recall_null_strain.append(recall_score(group['y'], group['neg_pred']))
        precision_null_strain.append(precision_score(group['y'], group['neg_pred']))
        strain_names.append(name)

    for name, group in X_test.groupby('Carbon'):
        mcc_carbon.append(matthews_corrcoef(group['y'], group['y_pred']))
        acc_carbon.append(accuracy_score(group['y'], group['y_pred']))
        recall_carbon.append(recall_score(group['y'], group['y_pred']))
        precision_carbon.append(precision_score(group['y'], group['y_pred']))

        mcc_null_carbon.append(matthews_corrcoef(group['y'], group['neg_pred']))
        acc_null_carbon.append(accuracy_score(group['y'], group['neg_pred']))
        recall_null_carbon.append(recall_score(group['y'], group['neg_pred']))
        precision_null_carbon.append(precision_score(group['y'], group['neg_pred']))
        carbon_names.append(name)

    mcc_strain_df = pd.DataFrame({"index": strain_names, "mcc": mcc_strain})
    mcc_carbon_df = pd.DataFrame({"index": carbon_names, "mcc": mcc_carbon})

    acc_strain_df = pd.DataFrame({"index": strain_names, "accuracy": acc_strain})
    acc_carbon_df = pd.DataFrame({"index": carbon_names, "accuracy": acc_carbon})

    recall_strain_df = pd.DataFrame({"index": strain_names, "recall": recall_strain})
    recall_carbon_df = pd.DataFrame({"index": carbon_names, "recall": recall_carbon})

    precision_strain_df = pd.DataFrame({"index": strain_names, "precision": precision_strain})
    precision_carbon_df = pd.DataFrame({"index": carbon_names, "precision": precision_carbon})

    mcc_null_strain_df = pd.DataFrame({"index": strain_names, "mcc": mcc_null_strain})
    mcc_null_carbon_df = pd.DataFrame({"index": carbon_names, "mcc": mcc_null_carbon})

    accuracy_null_strain_df = pd.DataFrame({"index": strain_names, "accuracy_null": acc_null_strain})
    accuracy_null_carbon_df = pd.DataFrame({"index": carbon_names, "accuracy_null": acc_null_carbon})

    recall_null_strain_df = pd.DataFrame({"index": strain_names, "recall_null": recall_null_strain})
    recall_null_carbon_df = pd.DataFrame({"index": carbon_names, "recall_null": recall_null_carbon})

    precision_null_strain_df = pd.DataFrame({"index": strain_names, "precision_null": precision_null_strain})
    precision_null_carbon_df = pd.DataFrame({"index": carbon_names, "precision_null": precision_null_carbon})

    strain_df = pd.merge(pd.merge(pd.merge(acc_strain_df, mcc_strain_df), recall_strain_df), precision_strain_df)
    carbon_df = pd.merge(pd.merge(pd.merge(acc_carbon_df, mcc_carbon_df), recall_carbon_df), precision_carbon_df)
    return strain_df, carbon_df


def fullyTraindMetricsSign(X_test):
    """
    Calculates all metrics for the full model (strength prediction)
    :param X_test: A df contains all needed information
    :return: strain df, carbon df
    """
    rmse_strain, rmse_null_strain = [], []
    strain_names = []

    rmse_carbon, rmse_null_carbon = [], []
    carbon_names = []

    for name, group in X_test.groupby('strain_x'):
        rmse_strain.append(mean_squared_error(group['y'], group['y_pred'], squared=False) / np.std(group['y']))
        rmse_null_strain.append(
            mean_squared_error(group['y'], group['mean_prediction'], squared=False) / np.std(group['y']))
        strain_names.append(name)

    for name, group in X_test.groupby('carbon'):
        rmse_carbon.append(mean_squared_error(group['y'], group['y_pred'], squared=False) / np.std(group['y']))
        rmse_null_carbon.append(
            mean_squared_error(group['y'], group['mean_prediction'], squared=False) / np.std(group['y']))
        carbon_names.append(name)

    rmse_strain_df = pd.DataFrame({"index": strain_names, "nrmse": rmse_strain})
    rmse_carbon_df = pd.DataFrame({"index": carbon_names, "nrmse": rmse_carbon})

    rmse_null_strain_df = pd.DataFrame({"index": strain_names, "nrmse_null": rmse_null_strain})
    rmse_null_carbon_df = pd.DataFrame({"index": carbon_names, "nrmse_null": rmse_null_carbon})

    strain_df = pd.merge(rmse_strain_df, rmse_null_strain_df)
    carbon_df = pd.merge(rmse_carbon_df, rmse_null_carbon_df)
    return strain_df, carbon_df


def signMetricasGroupByStrain():
    """
    Train and predict sign using the fully trained model
    :return: Scores for sign predictions
    """
    X_train, y_train, X_test, y_test = pd.read_csv("./Data/X_train_partial.csv"), pd.read_csv(
        "./Data/y_train_partial.csv"), pd.read_csv(
        "./Data/X_test_partial.csv"), pd.read_csv("./Data/y_test_partial.csv")
    y_train, y_test = np.array(y_train), np.array(y_test).reshape(-1)
    X_train['y_train'], X_test['y_test'] = y_train, y_test

    # remove water
    X_train, X_test = X_train[X_train['Carbon'] != 'Water'], X_test[X_test['Carbon'] != 'Water']
    y_train, y_test = X_train['y_train'], X_test['y_test']

    X_test_info = X_test[['Bug 1', 'Carbon']]
    X_train, X_test = X_train.drop(['Carbon', 'Bug 1', 'Bug 2', 'y_train'], axis=1), X_test.drop(
        ['Carbon', 'Bug 1', 'Bug 2', 'y_test'],
        axis=1)
    y_test = create_binary(y_test)
    y_train = create_binary(y_train)

    model = xgboost.XGBClassifier(base_score=0.5, booster='gbtree',
                                  colsample_bynode=1,
                                  eta=0.3, gamma=0.001,
                                  interaction_constraints='', learning_rate=0.1, max_delta_step=0,
                                  max_depth=15, min_child_weight=12,
                                  n_estimators=182,
                                  predictor='auto', random_state=0,
                                  reg_alpha=1e-05, reg_lambda=0.6666666666666666,
                                  scale_pos_weight=1, subsample=1, tree_method='exact')
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)

    X_test['y'], X_test['y_pred'], X_test['strain_x'], X_test['Carbon'] = y_test, y_pred, X_test_info['Bug 1'], \
                                                                          X_test_info['Carbon']
    X_test = X_test[["Carbon", "strain_x", "y", "y_pred"]]
    X_test['neg_pred'] = np.full(X_test.shape[0], -1)

    strain_df, carbon_df = fullMetricsGroupedByStrainOrEnv(X_test)
    return strain_df, carbon_df, evaluate(np.full(y_test.shape[0], -1), y_test)


def strengthMetricasGroupByStrain():
    """
    Train and predict strength using the fully trained model
    :return: scores for strength predictions
    """

    X_train, y_train, X_test, y_test = pd.read_csv("./Data/X_train_partial.csv"), pd.read_csv(
        "./Data/y_train_partial.csv"), pd.read_csv(
        "./Data/X_test_partial.csv"), pd.read_csv("./Data/y_test_partial.csv")
    y_train, y_test = np.array(y_train), np.array(y_test).reshape(-1)
    X_train['y_train'], X_test['y_test'] = y_train, y_test
    # remove water
    X_train, X_test = X_train[X_train['Carbon'] != 'Water'], X_test[X_test['Carbon'] != 'Water']
    y_train, y_test = X_train['y_train'], X_test['y_test']

    X_test_info = X_test[['Bug 1', 'Carbon']]
    X_train, X_test = X_train.drop(['Carbon', 'Bug 1', 'Bug 2', 'y_train'], axis=1), X_test.drop(
        ['Carbon', 'Bug 1', 'Bug 2', 'y_test'],
        axis=1)

    model = xgboost.XGBRegressor(base_score=0.2, booster='gbtree', colsample_bylevel=0.4,
                                 colsample_bynode=1, colsample_bytree=1,
                                 eta=0.3, gamma=0.01,
                                 learning_rate=0.1, max_delta_step=2,
                                 max_depth=9, min_child_weight=11,
                                 n_estimators=342, n_jobs=8,
                                 num_parallel_tree=1, predictor='auto', random_state=0,
                                 reg_alpha=0.1, reg_lambda=0.2222222222222222, scale_pos_weight=1,
                                 subsample=0.8, tree_method='exact').fit(X_train, y_train)
    mean_model = y_train.mean()
    mean_prediction = np.full(y_test.shape[0], mean_model)

    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)

    X_test['carbon'] = X_test_info['Carbon']
    X_test['strain_x'] = X_test_info['Bug 1']
    X_test['y'] = y_test
    X_test['y_pred'] = y_pred

    X_test = X_test[
        ["carbon", "strain_x", "y", "y_pred"]]
    X_test['mean_prediction'] = mean_prediction
    strain_df, carbon_df = fullyTraindMetricsSign(X_test)
    return strain_df, carbon_df, evaluate_effect(mean_prediction, y_test)


def main():
    random.seed(10)
    df = pd.read_csv("./Data/Features.csv").drop(['Unnamed: 0'], axis=1)

    # Create datasets for sign/strength predictions
    X, y, carbon_pca = readTable(df, "classification")
    X_r, y_r, carbon_pca = readTable(df, "regression")

    # train partial models
    #testUknown(X_r, y_r, 'strain', 'regression')
    testUknown(X_r, y_r, 'carbon', 'regression')
    testUknown(X, y, 'strain', 'classification')
    testUknown(X, y, 'carbon', 'classification')
    strengthNaiveAccIthClosest(X_r, y_r)


if __name__ == "__main__":
    main()
