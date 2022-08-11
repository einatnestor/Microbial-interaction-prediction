import numpy as np
import pandas as pd


class naiveModel():
    def __init__(self, X, y, environment, carbons_pca):
        """
        Constructor
        :param X: Train set
        :param y: y train labels
        :param environment: the carbon the model will predict interactions in
        :param carbons_pca: Metabolic profiles pca
        """
        self.X_train = X
        self.X_train['y'] = y
        self.pca = carbons_pca
        self.carbon = environment

    def predict(self, X_test, y_test, ith_closest_env):
        """
        Predict according to the metabolic closest carbon with the same interacting strains.
        :param X_test: Test set
        :param y_test:  y test labels
        :param ith_closest_env: The ith closest environment ( 0 = closest, 1= second closest straon, ... 18 = the most
        different environment)
        :return: Prediction for X test
        """
        ith_closest_strain_pd = self.calcDistance(ith_closest_env)
        X_test['y'] = y_test
        X_test = X_test[['Bug 1', 'Bug 2', 'carbon', 'y']]
        X_test = pd.merge(X_test, ith_closest_strain_pd, how='left', left_on=['carbon'], right_on=['carbon'])
        X_test = X_test[['Bug 1', 'Bug 2', 'carbon', 'y', 'closest_carbon', 'distance']]

        full_test_prediction = pd.merge(X_test, self.X_train, how='inner',
                                        left_on=['Bug 1', 'Bug 2', 'closest_carbon'],
                                        right_on=['Bug 1', 'Bug 2',
                                                  'carbon'])
        full_test_prediction = full_test_prediction[['Bug 1', 'Bug 2', 'carbon_x', 'y_x', 'y_y']].rename(
            columns={"y_x": "y_true", "y_y": "y_pred"})

        return full_test_prediction['y_true'], full_test_prediction['y_pred']

    def calcDistance(self, ith_closest):
        """
        Create a df with the information of the ith closest environment to each environment. (the chosen environment is according to
        metabolic profiles table).
        :param ith_closest: The ith closest environment ( 0 = closest, 1= second closest straon, ... 18 = the most
        different environment)
        :return:  Dataframe with the ith closest environment info (distance, name) for each environment.
        """
        pca = self.pca
        names = pca['carbon']
        pca = np.transpose(np.transpose(pca)[1:5])
        data_pairplot = pd.DataFrame()
        distance, closest, avg_distance, random_close_distance = [], [], [], []

        for i in range(pca.shape[0]):
            name, distnce = self.calcDistanceStrain(i, pca, names, ith_closest)
            distance.append(distnce)
            closest.append(name)

        data_pairplot['carbon'] = names
        data_pairplot['closest_carbon'] = closest
        data_pairplot['distance'] = distance

        return data_pairplot

    def calcDistanceStrain(self, index, pca, names, ith_closest):
        """
        Find the ith closest strain from the metabolic profiles matrix.
        :param i: The index of the environment from environments name array
        :param strain: The name of the strain in strains name array
        :param matrix: Metabolic profiles table
        :param ith_closest: The ith closest strain
        :return: The name and distance of the ith environment name.
        """
        pca = np.array(pca)
        c1 = pca[index]
        distance_arr = []
        names_arr = []
        for i in range(pca.shape[0]):
            if i != index:
                c2 = pca[i]
                e_distance = np.sqrt(np.sum(np.power(c2 - c1, 2)))
                distance_arr.append(e_distance)
                names_arr.append(names[i])

        distances_df = pd.DataFrame()
        distances_df['strain'] = names_arr
        distances_df['distance'] = distance_arr
        distances_df = distances_df.sort_values('distance')
        distances_df = np.array(distances_df.iloc[ith_closest])
        return distances_df[0], distances_df[1]
