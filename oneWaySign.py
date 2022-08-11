import numpy as np
import pandas as pd
import shap as shap
import random
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import RandomizedSearchCV
from sklearn.metrics import confusion_matrix, accuracy_score, recall_score, precision_score, matthews_corrcoef, f1_score
import pickle
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
import xgboost
import matplotlib.pyplot as plt

MODELS = ['Random forest', 'Logistic regression', 'K-nearest neighbors', 'Most frequent sign model',
          'Metabolic threshold', 'XGBoost', 'Mono Growth threshold']
metricsDictionay = {'Accuracy': accuracy_score, 'Precision': precision_score, 'Recall': recall_score, 'f1': f1_score,
                    'MCC': matthews_corrcoef}


class RandomForest():
    def __init__(self, X, y):
        """
        Constructor
        :param X: Train set
        :param y: y train labels
        """
        self.model = RandomForestClassifier()
        self.X = X
        self.y = y

    def train(self):
        """
        Find the best model out a wide parameters range, and save it.
        """
        self.hyperParameterTuning()
        pickle.dump(self.model, open('random_forest_sign.sav', 'wb'))

    def hyperParameterTuning(self):
        """
        Find the best model out a wide parameters range, and save it.
        """
        n_estimators = [int(x) for x in np.linspace(start=100, stop=400, num=100)]
        max_features = ['auto', 'sqrt', 'log2']
        class_weight = ['balanced', 'balanced_subsample', None]
        min_samples_split = [int(x) for x in np.linspace(2, 20, num=18)]
        bootstrap = [True, False]
        criterion = ['gini']
        random_grid = {'n_estimators': n_estimators,
                       'max_features': max_features,
                       'class_weight': class_weight,
                       'min_samples_split': min_samples_split,
                       'bootstrap': bootstrap,
                       'criterion': criterion}

        # find best model out of all range
        rf_random = RandomizedSearchCV(estimator=self.model, param_distributions=random_grid,
                                       n_iter=500, cv=5,
                                       verbose=2, n_jobs=-1)
        # find best model out of all range
        self.model = rf_random.fit(self.X, self.y)
        print(rf_random.best_estimator_)


class xgboost_classifier():

    def __init__(self, X, y):
        """
        Constructor
        :param X: Train set
        :param y: y train labels
        """
        self.model = xgboost.XGBClassifier()
        self.X = X
        self.y = y

    def train(self):
        """
        Find the best model out a wide parameters range, and save it.
        """
        self.hyperParameterTuning()
        pickle.dump(self.model, open('sign_model.sav', 'wb'))

    def hyperParameterTuning(self):
        """
        Find the best model using random grid search.
        """
        XGR = xgboost.XGBClassifier()
        max_depth = [int(x) for x in np.linspace(start=7, stop=15, num=7)]
        gamma = [0, 0.1, 0.01, 0.001]
        eta = [0, 0.1, 0.2, 0.3, 0.5]
        max_delta_step = [0, 1, 2]
        subsample = [0, 0.1, 0.3, 0.5, 0.8, 1]
        predictor = ['auto']
        booster = ['gbtree']
        learning_rate = [0.1, ]
        reg_alpha = [0.1, 0.001, 0.0001, 0.00001]
        min_child_weight = [int(x) for x in np.linspace(start=5, stop=15, num=10)]
        n_estimators = [int(x) for x in np.linspace(start=100, stop=200, num=30)]
        reg_lambda = [x for x in np.linspace(start=0.0, stop=1.0, num=10)]
        tree_method = ['exact', 'auto', 'approx']

        random_grid = {'max_depth': max_depth,
                       'gamma': gamma,
                       'eta': eta,
                       'subsample': subsample,
                       'min_child_weight': min_child_weight,
                       'booster': booster,
                       'max_delta_step': max_delta_step,
                       'learning_rate': learning_rate,
                       'predictor': predictor,
                       'reg_alpha': reg_alpha,
                       'tree_method': tree_method,
                       'n_estimators': n_estimators,
                       'reg_lambda': reg_lambda}

        # find best model out of all range
        random = RandomizedSearchCV(scoring="neg_mean_squared_error", estimator=XGR,
                                    param_distributions=random_grid,
                                    n_iter=300, cv=5,
                                    verbose=2, n_jobs=-1)
        # fit the best model
        random.fit(self.X, self.y)
        self.model = random
        print(random.best_estimator_)


class Logistic():

    def __init__(self, X, y):
        """
        Constructor
        :param X: Train set
        :param y: y train labels
        """
        self.model = LogisticRegression()
        self.X = X
        self.y = y

    def train(self):
        """
        Find the best model out a wide parameters range, and save it.
        """
        self.hyperParameterTuning()
        pickle.dump(self.model, open('logistic_sign.sav', 'wb'))

    def hyperParameterTuning(self):
        """
        Find the best model using random grid search.
        """
        C = [0, 0.001, 0.1, 0.5, 1, 1.5, 2]
        intercept_scaling = [0, 1, 2]
        l1_ratio = [None, 0, 0.01, 0.001, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1]
        max_iter = [int(x) for x in np.linspace(5, 200, num=100)]
        multi_class = ['auto']
        penalty = ['l1', 'l2', 'none']

        random_grid = {'C': C,
                       'intercept_scaling': intercept_scaling,
                       'l1_ratio': l1_ratio,
                       'max_iter': max_iter,
                       'multi_class': multi_class,
                       'penalty': penalty}

        # find best model out of all range
        log = RandomizedSearchCV(estimator=self.model, param_distributions=random_grid,
                                 n_iter=500, cv=5,
                                 verbose=2, n_jobs=-1)
        # fit the best model
        self.model = log.fit(self.X, self.y)
        print(log.best_estimator_)


class kNeighbors():
    def __init__(self, X, y):
        """
        Constructor
        :param X: Train set
        :param y: y train labels
        """
        self.model = KNeighborsClassifier()
        self.X = X
        self.y = y

    def train(self):
        """
        Find the best model out a wide parameters range, and save it.
        """
        self.hyperParameterTuning()
        pickle.dump(self.model, open('KNeighbors_sign.sav', 'wb'))

    def hyperParameterTuning(self):
        """
        Find the best model using random grid search.
        """
        algorithm = ['auto', 'ball_tree', 'kd_tree', 'brute']
        leaf_size = [int(x) for x in np.linspace(1, 100, num=50)]
        metric = ['minkowski']
        metric_params = [None]
        n_neighbors = [int(x) for x in np.linspace(1, 100, num=70)]
        p = [1, 2]
        weights = ['uniform', 'distance']

        random_grid = {'algorithm': algorithm,
                       'leaf_size': leaf_size,
                       'metric': metric,
                       'metric_params': metric_params,
                       'n_neighbors': n_neighbors,
                       'p': p,
                       'weights': weights}

        # find best model out of all range
        kn = RandomizedSearchCV(estimator=self.model, param_distributions=random_grid,
                                n_iter=500, cv=5,
                                verbose=2, n_jobs=-1)
        # fit the best model
        self.model = kn.fit(self.X, self.y)
        print(kn.best_estimator_)


class threshold():
    """
    Naive one-feature model according to one feature threshold.
    """

    def __init__(self, feature_vector, name):
        """
        Constructor
        :param feature_vector: A feature to choose threshold from
        :param name: feature's name
        """
        self.feature_name = name
        self.threshold = 0
        self.feature_vector = feature_vector
        self.max_threshold = np.max(self.feature_vector)
        self.min_threshold = np.min(self.feature_vector)

    def fit_threshold(self, vector, threshold):
        """
        Fit the best threshold (accuracy on train set)
        :param vector: Feature
        :param threshold: Threshold value for feature
        :return: prediction
        """
        prediction = np.full(vector.shape[0], -1)
        above_threshold = np.where(vector < threshold)
        prediction[above_threshold] = 1
        return prediction

    def fit(self, y):
        """
        Find the best threshold according to the train set.
        :param y: True labels.
        :return:
        """
        thresholds = [int(x) for x in np.linspace(self.min_threshold, self.max_threshold, num=100)]
        max_accuracy = accuracy_score(y,
                                      self.fit_threshold(self.feature_vector, self.min_threshold))
        max_threshold = self.min_threshold
        for threshold in thresholds:
            y_pred = self.fit_threshold(self.feature_vector, threshold)
            tmp_accuracy = accuracy_score(y, y_pred)
            if tmp_accuracy > max_accuracy:
                max_accuracy = tmp_accuracy
                max_threshold = threshold
        self.threshold = max_threshold

    def predict(self, X):
        """
        Predict using the best threshold
        :param X: Dataset.
        :return: Prediction
        """
        metabolic_distance = X[self.feature_name]
        return self.fit_threshold(metabolic_distance, self.threshold)


def readTable(df, effect):
    '''
    Seperate the table into y label and samples.
    :param df: data
    :param effect: can be either Eba (Effect of B on A) or Eab
    :return: X, y, and carbons df
    '''
    carbon = df['Carbon']
    if effect == "Eba":
        y = df['2 on 1: Effect']
        df = df.drop(columns=['Unnamed: 0', '2 on 1: Effect', 'Carbon', 'Bug 1', 'Bug 2'], axis=1)
    else:
        y = df['1 on 2: Effect']
        df = df.drop(columns=['Unnamed: 0', '1 on 2: Effect', 'Carbon', 'Bug 1', 'Bug 2'], axis=1)
    y = np.array(y)
    return df, np.array(y), carbon


def shapExplainer(model, X_train):
    explainer = shap.Explainer(model, X_train)
    shap_values = explainer(X_train)
    shap.summary_plot(shap_values, X_train, max_display=10, plot_type="bar", show=False)  #
    fig = plt.gcf()
    # plt.savefig("./figS3A.png", dpi=300)


def evaluateModels(X_train, X_test, y_train, y_test):
    """
    Evaluate all models.
    :param X_train: Train set.
    :param X_test:  Test set.
    :param y_train: y train labels.
    :param y_test:  y test labels.
    :return: dictionary with models evaluated.
    """
    random_forest = pickle.load(open('./models/sign_models/random_forest_sign.sav', 'rb'))
    rfPrediction = random_forest.predict(X_test)
    rfScores = evaluate(rfPrediction, y_test)

    logistic_regression = pickle.load(open('./models/sign_models/logistic_sign.sav', 'rb'))
    logPrediction = logistic_regression.predict(X_test)
    logScores = evaluate(logPrediction, y_test)

    k_nearest_heighbors = pickle.load(open('./models/sign_models/KNeighbors_sign.sav', 'rb'))
    knPrediction = k_nearest_heighbors.predict(X_test)
    knScores = evaluate(knPrediction, y_test)

    null_prediction = np.full((y_test.shape[0]), -1)
    nullScores = evaluate(null_prediction, y_test)

    metabolic_threshold = threshold(X_train['metDis'], 'metDis')
    metabolic_threshold.fit(y_train)
    met_prediction = metabolic_threshold.predict(X_test)
    metScores = evaluate(met_prediction, y_test)

    monogrowth_threshold = threshold(X_train['monoGrow_x'], 'monoGrow_x')
    monogrowth_threshold.fit(y_train)
    mono_prediction = monogrowth_threshold.predict(X_test)
    monoScores = evaluate(mono_prediction, y_test)

    xgb = pickle.load(open('./models/sign_models/sign_model.sav', 'rb'))
    xgb_prediction = xgb.predict(X_test)
    xgbScores = evaluate(xgb_prediction, y_test)

    modelsDic = {}
    modelsDic['rf'] = rfScores
    modelsDic['logistic'] = logScores
    modelsDic['kn'] = knScores
    modelsDic['neg'] = nullScores
    modelsDic['met'] = metScores
    modelsDic['xgb'] = xgbScores
    modelsDic['monogrow_x'] = monoScores
    return modelsDic, xgb_prediction, random_forest, logistic_regression, k_nearest_heighbors, xgb


def evaluate(prediction, trueLable):
    """
    Evaluate predictions, return dictionary with scores.
    """
    scores = {}
    for metric in metricsDictionay:
        score = metricsDictionay[metric](prediction, trueLable)
        scores[metric] = score
    return scores


def createDf(dic):
    """
    Create dataframe with all the models and their performance.
    """
    acc = []
    mcc = []
    f1 = []
    recall = []
    precision = []
    for model in dic:
        for score in dic[model]:
            if score == 'Accuracy':
                acc.append(dic[model][score])
            elif score == 'Precision':
                precision.append(dic[model][score])
            elif score == 'Recall':
                recall.append(dic[model][score])
            elif score == 'f1':
                f1.append(dic[model][score])
            else:
                mcc.append(dic[model][score])
    df = pd.DataFrame()
    df1 = pd.DataFrame()
    df2 = pd.DataFrame()
    df3 = pd.DataFrame()
    df4 = pd.DataFrame()
    df['models'] = MODELS
    df['score'] = acc
    df['Measurement'] = ['Accuracy'] * df.shape[0]

    df2['models'] = MODELS
    df2['score'] = f1
    df2['Measurement'] = ['F1'] * df.shape[0]

    df4['models'] = MODELS
    df4['score'] = f1
    df4['Measurement'] = ['Precision'] * df.shape[0]

    df3['models'] = MODELS
    df3['score'] = recall
    df3['Measurement'] = ['Recall'] * df.shape[0]

    df1['models'] = MODELS
    df1['score'] = mcc
    df1['Measurement'] = ['MCC'] * df.shape[0]

    df = pd.concat([df, df1])
    df = pd.concat([df, df2])
    df = pd.concat([df, df3])
    df = pd.concat([df, df4])
    return df


def createBinary(y):
    """
    Convert effect vector to sign vector. Negative interaction is labeled with -1 and positive with 1.
    :param y: effect vector
    :return: sign vector
    """
    negative_interaction = np.where(y <= 0)[0]
    y_sign = np.full(y.shape[0], 1)
    y_sign[negative_interaction] = -1
    return y_sign


def main():
    random.seed(10)
    X_train, y_train, X_test, y_test = pd.read_csv("Data/X_train.csv").drop(
        ['Unnamed: 0', 'Bug 1', 'Bug 2', 'Carbon'], axis=1), pd.read_csv("Data/y_train.csv").drop(
        ['Unnamed: 0'], axis=1), pd.read_csv(
        "Data/X_test.csv").drop(['Unnamed: 0', 'Bug 1', 'Bug 2', 'Carbon'], axis=1), pd.read_csv(
        "Data/y_test.csv").drop(['Unnamed: 0'], axis=1)
    y_train, y_test = np.array(y_train), np.array(y_test)
    y_test_sign, y_train_sign = createBinary(y_test), createBinary(y_train)
    # Evaluate models
    dic, best_sign_prediction, random_forest, logistic_regression, k_nearest_neighbors, xgb = evaluateModels(X_train,
                                                                                                             X_test,
                                                                                                             y_train_sign,
                                                                                                             y_test_sign)
    sign_prediction_df = createDf(dic)
    # S3
    # shapExplainer(xgb, X_test)
    return sign_prediction_df, best_sign_prediction, random_forest, logistic_regression, k_nearest_neighbors, xgb, X_test, y_test, y_test_sign


if __name__ == "__main__":
    main()
