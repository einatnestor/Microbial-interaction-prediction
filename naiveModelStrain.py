import numpy as np
import pandas as pd

STRAINS = ['PR2', 'PR1', 'PAr', 'EC', 'EL', 'LA', 'RP2', 'RP1', 'EA', 'SF1', 'BI', 'KA', 'CF', 'PAg2', 'PAg1', 'PAl',
           'PAg3', 'PH', 'PK', 'PP']


class naiveModel():

    def __init__(self, X, y, strain, distance_matrix):
        """
        Constructor
        :param X: Train set
        :param y: y train labels
        :param strain: the strain the model will predict interactions on
        :param distance_matrix: phylogenetic distance matrix
        """
        self.X_train = X
        self.X_train['y'] = y
        self.strain = strain
        self.distance_matrix = distance_matrix

    def getInfo(self):
        return self.ith_distance, self.ith_closest

    def predict(self, X_test, y_test, ith_closest_strain):
        """
        Predict according to the phylogenetic closest strain with the same interacting strain and environment
        :param X_test: Test set
        :param y_test: y test labels
        :param ith_closest_strain: The ith closest strain ( 0 = closest, 1= second closest straon, ... 18 = the most
        different strain)
        :return: Prediction for X test
        """
        ith_closest_strain_pd = self.calcDistance(ith_closest_strain)
        ith_closest = ith_closest_strain_pd[ith_closest_strain_pd['strain'] == self.strain]
        self.ith_distance, self.ith_closest = np.array(ith_closest['distance']), np.array(ith_closest['closest_strain'])
        X_test['y'] = y_test
        X_test = X_test[['Bug 1', 'Bug 2', 'carbon', 'y']]
        X_test = pd.merge(X_test, ith_closest_strain_pd, how='left', left_on=['Bug 1'], right_on=['strain'])
        X_test = X_test[['Bug 1', 'Bug 2', 'carbon', 'y', 'closest_strain', 'distance']].rename(
            columns={"closest_strain": "closest_strain_of_bug_1", "distance": "distance_bug_1"})
        X_test = pd.merge(X_test, ith_closest_strain_pd, how='left', left_on=['Bug 2'], right_on=['strain']).rename(
            columns={"closest_strain": "closest_strain_of_bug_2", "distance": "distance_bug_2"})
        X_test = X_test[['Bug 1', 'closest_strain_of_bug_1', 'distance_bug_1', 'Bug 2', 'distance_bug_2', 'carbon', 'y',
                         'closest_strain_of_bug_2']]
        X_test_as_Bug_1 = X_test[X_test['Bug 1'] == self.strain]
        X_test_as_Bug_2 = X_test[X_test['Bug 2'] == self.strain]
        find_prediction_closest_is_bug_1 = pd.merge(X_test_as_Bug_1, self.X_train, how='inner',
                                                    left_on=['closest_strain_of_bug_1', 'Bug 2', 'carbon'],
                                                    right_on=['Bug 1', 'Bug 2',
                                                              'carbon'])  # effect when tested strain is bug 1
        find_prediction_closest_is_bug_2 = pd.merge(X_test_as_Bug_2, self.X_train, how='inner',
                                                    left_on=['Bug 1', 'closest_strain_of_bug_2', 'carbon'],
                                                    right_on=['Bug 1', 'Bug 2',
                                                              'carbon'])  # effect when tested strain is bug 2

        x_test_prediction = pd.concat([find_prediction_closest_is_bug_1, find_prediction_closest_is_bug_2], axis=0)
        x_test_prediction = x_test_prediction[['Bug 1_x', 'Bug 2', 'carbon', 'y_x', 'y_y']].rename(
            columns={"y_x": "y_true", "y_y": "y_pred"})

        return x_test_prediction['y_true'], x_test_prediction['y_pred']

    def calcDistance(self, ith_closest):
        """
        Create a df with the information of the ith closest strain to each strain. (the chosen strain is according to
        phylogenetic table).
        :param ith_closest: The ith closest strain ( 0 = closest, 1= second closest straon, ... 18 = the most
        different strain)
        :return: Dataframe with the ith closest strain info (distance, name) for each strain.
        """
        names = self.distance_matrix['strain']
        data_pairplot = pd.DataFrame()
        distance, closest, avg_distance, random_close_distance = [], [], [], []
        matrix = self.distance_matrix
        for i, strain in enumerate(STRAINS):
            name, distnce = self.calcDistanceStrain(i, strain, matrix, ith_closest)
            distance.append(distnce)
            closest.append(name)
        data_pairplot['strain'] = names
        data_pairplot['closest_strain'] = closest
        data_pairplot['distance'] = distance
        return data_pairplot

    def calcDistanceStrain(self, i, strain, matrix, ith_closest):
        """
        Find the ith closest strain from the phylogenetic distances matrix.
        :param i: The index of the strain from strains name array
        :param strain: The name of the strain in strains name array
        :param matrix: Phylogenetic distance matrix
        :param ith_closest: The ith closest strain
        :return: the name and distance of the ith closest strain
        """
        strain_row = matrix.iloc[:, [0, i + 1]]
        strain_row = strain_row.drop([i]).sort_values(strain)
        all_names, all_ditances = np.array(strain_row['strain']), np.array(strain_row[strain])
        return all_names[ith_closest], all_ditances[ith_closest]
