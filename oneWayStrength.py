import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import shap as shap
from sklearn.neighbors import KNeighborsRegressor
import xgboost
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import RandomizedSearchCV, train_test_split
from sklearn.metrics import mean_squared_error, r2_score
import pickle
from sklearn.linear_model import Ridge
metricsDictionay = {'RMSE': mean_squared_error, 'R2': r2_score}


class kNeighbors_regressor():
    def __init__(self, X, y):
        """
        Constructor
        :param X: Train set
        :param y: y train labels
        """
        self.model = KNeighborsRegressor()
        self.X = X
        self.y = y

    def train(self):
        """
        Find the best model out a wide parameters range, and save it.
        """
        self.hyperParameterTuning()
        pickle.dump(self.model, open('models/strength_models/KNeighbors_strength.sav', 'wb'))

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
        kn = RandomizedSearchCV(estimator=self.model, scoring='neg_root_mean_squared_error', param_distributions=random_grid,
                                n_iter=500, cv=5,
                                verbose=2, n_jobs=-1)
        # fit the best model
        kn.fit(self.X, self.y)
        self.model =kn.best_estimator_
        print(kn.best_estimator_)


class ridge_regression():
    def __init__(self, X, y):
        """
        Constructor
        :param X: Train set
        :param y: y train labels
        """
        self.model = Ridge()
        self.X = X
        self.y = y

    def train(self):
        """
        Find the best model out a wide parameters range, and save it.
        """
        self.hyperParameterTuning()
        pickle.dump(self.model, open('models/strength_models/ridge_regression.sav', 'wb'))

    def hyperParameterTuning(self):
        """
        Find the best model using random grid search.
        """
        alpha = [0,0.1,0.5,1,1.5,2]
        random_grid = {'alpha': alpha}
        # find best model out of all range
        ridge = RandomizedSearchCV(estimator=self.model, scoring='neg_root_mean_squared_error', param_distributions=random_grid,
                                n_iter=500, cv=5,
                                verbose=2, n_jobs=-1)
        # fit the best model
        ridge.fit(self.X, self.y)
        self.model = ridge.best_estimator_
        print(ridge.best_estimator_)


class random_forest_regressor():
    def __init__(self, X, y):
        """
        Constructor
        :param X: Train set
        :param y: y train labels
        """
        self.model = RandomForestRegressor()
        self.X = X
        self.y = y

    def train(self):
        """
        Find the best model out a wide parameters range, and save it.
        """
        self.hyperParameterTuning()
        pickle.dump(self.model, open('models/strength_models/random_forest_strength.sav', 'wb'))

    def hyperParameterTuning(self):
        """
        Find the best model using random grid search.
        """
        n_estimators = [int(x) for x in np.linspace(start=100, stop=400, num=100)]
        max_features = ['auto', 'sqrt', 'log2']
        min_samples_leaf = [int(x) for x in np.linspace(2, 25, num=10)]
        bootstrap = [True, False]
        random_grid = {'n_estimators': n_estimators,
                       'max_features': max_features,
                       'min_samples_split': min_samples_leaf,
                       'bootstrap': bootstrap}

        # find best model out of all range
        rf_random = RandomizedSearchCV(scoring="neg_root_mean_squared_error", estimator=self.model,
                                       param_distributions=random_grid,
                                       n_iter=500, cv=5,
                                       return_train_score=True,
                                       verbose=2, n_jobs=-1)
        # fit the best model
        rf_random.fit(self.X, self.y)
        self.model = rf_random.best_estimator_
        print(rf_random.best_estimator_)


class xgb():
    def __init__(self, X, y):
        """
        Constructor
        :param X: Train set
        :param y: y train labels
        """
        self.model = xgboost.XGBRegressor()
        self.X = X
        self.y = y

    def train(self):
        """
        Find the best model out a wide parameters range, and save it.
        """
        self.hyperParameterTuning()
        pickle.dump(self.model, open('models/strength_models/strength_model.sav', 'wb'))

    def hyperParameterTuning(self):
        """
        Find the best model using random grid search.
        """

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
        n_estimators = [int(x) for x in np.linspace(start=200, stop=500, num=20)]
        reg_lambda = [x for x in np.linspace(start=0.0, stop=1.0, num=10)]
        scale_pos_weight = [1]
        tree_method = ['exact', 'auto', 'approx']
        base_score = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8]
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
                       'scale_pos_weight': scale_pos_weight,
                       'tree_method': tree_method,
                       'n_estimators': n_estimators,
                       'reg_lambda': reg_lambda,
                       'base_score': base_score}

        # find best model out of all range
        xgb = RandomizedSearchCV(scoring="neg_root_mean_squared_error", estimator=self.model,
                                 param_distributions=random_grid,
                                 n_iter=200, cv=5,
                                 verbose=2, n_jobs=-1)
        # fit the best model
        self.model = xgb.fit(self.X, self.y)
        print(xgb.best_estimator_)


def evaluate(prediction, trueLable):
    """
    Evaluate strength prediction
    :param prediction: Prediction of a model
    :param trueLable: True labels (y test)
    :return: A dictionary with NRMSE (normalized using sd) and R^2
    """
    scores = {}
    for metric in metricsDictionay:
        if metric == 'RMSE':
            score = metricsDictionay[metric](trueLable, prediction, squared=False) / np.std(
                trueLable)
        else:
            score = metricsDictionay[metric](trueLable, prediction)
        scores[metric] = score
    return scores


def readTable(df, type):
    '''
    Separate the table into y label and samples.
    '''
    carbon = df['Carbon']
    if type == "one-way":
        y = df['2 on 1: Effect']
    elif type == "two-way":
        y = [df['2 on 1: Effect'], df['1 on 2: Effect']]

    df = df.drop(columns=['Unnamed: 0', '2 on 1: Effect'], axis=1)
    y = np.array(y)
    return df, np.array(y), carbon


def evaluateModels(X_train, X_test, y_train, y_test, shap):
    """
    Evaluate all models.
    :param X_train: Train set
    :param X_test:  Test set
    :param y_train: y train labels
    :param y_test:  y test labels
    :param shap: boolean variable for calculating SHAP values
    :return: dictionary with models evaluated
    """
    # Model was too heavy to upload to github
    rf = pickle.load(open('models/strength_models/random_forest_strength.sav', 'rb'))#RandomForestRegressor(bootstrap=False, max_features='sqrt', min_samples_split=4, n_estimators=378)
    rfPrediction = rf.predict(X_test)
    rfScores = evaluate(rfPrediction, y_test)

    XGR = pickle.load(open('models/strength_models/strength_model.sav', 'rb'))
    if shap:
        plot_shap(XGR, X_test)

    xgprediction = XGR.predict(X_test)
    xgScores = evaluate(xgprediction, y_test)

    linear_reg = pickle.load(open('models/strength_models/ridge_regression.sav', 'rb'))
    linear_prediction = linear_reg.predict(X_test)
    linScores = evaluate(linear_prediction, y_test)

    kn_neighbors = pickle.load(open('models/strength_models/KNeighbors_strength.sav', 'rb'))
    kn_prediction = kn_neighbors.predict(X_test)
    knScores = evaluate(kn_prediction, y_test)

    average = np.mean(y_train)
    null_prediction = np.full((y_test.shape[0]), average)
    nullScores = evaluate(null_prediction, y_test)

    modelsDic = {}
    modelsDic['rf'] = rfScores
    modelsDic['XGB'] = xgScores
    modelsDic['lin'] = linScores
    modelsDic['kn'] = knScores
    modelsDic['null'] = nullScores
    return modelsDic, xgprediction


def plotComparison(dic):
    """
    Create a df from dictionary (for figures)
    :param dic:  Dictionary with all model's evaluation
    :return: A dataframe with all models' evaluation
    """
    rmse = []
    r2 = []
    for model in dic:
        for score in dic[model]:
            if score == 'RMSE':
                rmse.append(dic[model][score])
            elif score == 'R2':
                r2.append(dic[model][score])
    df = pd.DataFrame()
    df1 = pd.DataFrame()
    df2 = pd.DataFrame()
    df1['models'] = ['Random forest', 'XGBoost', 'Linear regression', 'K-nearest neighbors', 'Average strength effect']
    df1['score'] = rmse
    df1['Measurement'] = ['RMSE'] * df1.shape[0]
    df2['models'] = ['Random forest', 'XGBoost', 'Linear regression', 'K-nearest neighbors', 'Average strength effect']
    df2['score'] = r2
    df2['Measurement'] = ['R2'] * df2.shape[0]
    df = pd.concat([df, df1])
    df = pd.concat([df, df2])
    return df


def plot_shap(XGR, X_test):
    """
    Plot SHAP values.
    :param XGR: the best sign strength model
    :param X_test: Test set
    """
    shap.initjs()
    explainer = shap.Explainer(XGR.best_estimator_)
    shap_values = explainer(X_test)
    shap.summary_plot(shap_values,max_display=10, show=False)
    #shap.plots.waterfall(shap_values[1], max_display=10, show=False)
    plt.savefig("Figures_msystems/Figure3", dpi=300)



def main():
    features = pd.read_csv("Data_msystems/Features.csv")
    features = features[features['Carbon'] != 'Water']
    X,y = features[['monoGrow_x','monoGrow_y','monoGrow24_x','monoGrow24_y','metDis','carbon_component_0','carbon_component_1','carbon_component_2','carbon_component_3','phy_strain_component_0_x','phy_strain_component_1_x','phy_strain_component_0_y','phy_strain_component_1_y']],features['2 on 1: Effect']
    X_train, X_test, y_train,y_test = train_test_split(X,y,test_size =0.2,random_state=10)
    dic, best_strength_prediction = evaluateModels(X_train, X_test, y_train, y_test, False)
    df = plotComparison(dic)

    return df, y_test, best_strength_prediction


if __name__ == "__main__":
    main()
