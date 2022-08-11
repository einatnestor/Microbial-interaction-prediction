import numpy as np
import pandas as pd
from sklearn.decomposition import PCA

CARBONS = ['ArabinoseD', 'ArabinoseL',
           'Xylose', 'Ribose', 'Rhamnose', 'Fructose', 'Galactose', 'Glucose', 'Glucose01', 'Mannose', 'GlcNAc',
           'Acetate',
           'Pyruvate', 'Pyruvate01', 'Fumarate', 'Succinate', 'Citrate', 'Glycerol', 'Glycerol01', 'Mannitol',
           'Sorbitol',
           'Alanine', 'Serine', 'Proline', 'Proline01', 'Glutamine', 'Arginine', 'Trehalose', 'Cellobiose', 'Maltose',
           'Sucrose',
           'Sucrose01', 'Lactose', 'Raffinose', 'Melezitose', 'Arabinogalactan', 'Isoleucine', 'Uridine', 'Mix',
           'Water']

STRAINS = ['PR2', 'PR1', 'PAr', 'EC', 'EL', 'LA', 'RP2', 'RP1', 'EA', 'SF1', 'BI', 'KA', 'CF', 'PAg2', 'PAg1', 'PAl',
           'PAg3', 'PH', 'PK', 'PP']

REMOVE_COLUMNS = ['Strength',
                  'Type',
                  '1 on 2: 25th Percentile', '1 on 2: 75th Percentile',
                  '2 on 1: Replicates',
                  '2 on 1: 25th Percentile', '2 on 1: 75th Percentile',
                  'Class', '1 on 2: Replicates']


def createTable(type):
    """
    Create table fro pca
    :param type: carbon/ strain
    :return: df
    """
    # create carbon table
    if type == "carbon":
        table = pd.read_csv("Data/environments_profiles.csv").drop(['carbon'], axis=1)

    # create phylo distance
    elif type == "strain":
        table = pd.read_csv("./Data/pairwiseDistances-21Isolates&Ecoli.csv").drop(['Unnamed: 0'], axis=1)
        table = table.drop([9, 11])
        table = table.drop(columns=["SF2", "CB"])
    return table


def createPCA(type, dimensions, strain=None):
    """
    Create PCA data. Can either be strains' pca (from phylogenetic distance matrix) or carbons' pca
    (from monoculture growth df)
    :param type: carbon/ strain
    :param dimensions: number of dimension for pca
    :param strain: a strain to remove.
    :return: pca df
    """
    table = createTable(type)

    if type == 'carbon' and strain != None:
        table = table.drop([strain], axis=1)

    # fit pca and find the components
    pca = PCA(dimensions)
    principalComponents = pca.fit_transform(table)
    pca_components = np.array(principalComponents).transpose()[:dimensions]
    pca_df = pd.DataFrame()
    if type == "carbon":
        pca_df['carbon'] = CARBONS
    else:
        pca_df['strain'] = STRAINS
    index = 0
    while index < dimensions:
        if type == "carbon":
            pca_df["carbon_component_" + str(index)] = pca_components[index]
        else:
            pca_df["phy_strain_component_" + str(index)] = pca_components[index]
        index += 1
    return pca_df


def mergeAllFeatures(processed_data_table, monoGrow, metabolic_distance, monoGrow24):
    """
    Merge all features into a single table.
    :param processed_data_table: dataset of effect measurments.
    :param monoGrow: monogrow table
    :param metabolic_distance: metabolic distance table
    :param monoGrow24: normalized monogrow table
    :return:
    """
    full_data = pd.merge(processed_data_table, monoGrow, how='inner', left_on=['Bug 1', 'Carbon'],
                         right_on=['labeledStrain', 'env'])
    full_data = pd.merge(full_data, metabolic_distance, how='inner', left_on=['Bug 1', 'Bug 2'],
                         right_on=['Bug 1', 'Bug 2'])
    full_data = full_data.drop(['labeledStrain', 'env'], axis=1).rename(columns={'monoGrow': 'monoGrow_x'})
    full_data = pd.merge(full_data, monoGrow, how='inner', left_on=['Bug 2', 'Carbon'],
                         right_on=['labeledStrain', 'env']).drop(['labeledStrain', 'env'], axis=1).rename(
        columns={'monoGrow': 'monoGrow_y'})
    full_data = pd.merge(full_data, monoGrow24, right_on=['labeledStrain', 'env'],
                         left_on=['Bug 1', 'Carbon']).drop(['labeledStrain', 'env'], axis=1).rename(
        columns={'monoGrow24': 'monoGrow24_x'})
    full_data = pd.merge(full_data, monoGrow24, right_on=['labeledStrain', 'env'],
                         left_on=['Bug 2', 'Carbon']).drop(['labeledStrain', 'env'], axis=1).rename(
        columns={'monoGrow24': 'monoGrow24_y'})
    full_data = full_data.drop(REMOVE_COLUMNS, axis=1)
    return full_data


def addMetabolicPhylogeneticFeatures(df, pca_df_carbon, pca_df_strainPhy):
    """
    Add metabolic and phylogenetic pca features
    :param df: Features
    :param pca_df_carbon: Carbons' PCA
    :param pca_df_strainPhy: Strains' PCA
    :return: df with all features.
    """
    df = pd.merge(df, pca_df_carbon, how='left', left_on=['Carbon'], right_on=['carbon']).drop(['carbon'], axis=1)
    df = pd.merge(df, pca_df_strainPhy, how='inner', left_on=['Bug 1'], right_on=['strain']).drop(['strain'], axis=1)
    df = pd.merge(df, pca_df_strainPhy, how='inner', left_on=['Bug 2'], right_on=['strain']).drop(['strain'], axis=1)
    return df


def normelized_monoGrow(df, column_name):
    """
    Create normalized monoculture
    :param df: Unnormalized monogrow
    :param column_name: column name to normalize (monogrow or monogrow 24)
    :return: normalized df.
    """
    strains = df['labeledStrain']
    env = df['env']
    normalized_mono_df = pd.DataFrame()
    for name, group in df.groupby(['labeledStrain']):
        normalized_mono = (group[column_name] - group[column_name].mean()) / group[column_name].std()
        normalized_mono = pd.DataFrame({column_name: normalized_mono})
        normalized_mono_df = pd.concat([normalized_mono_df, normalized_mono], ignore_index=True)
    normalized_mono_df['labeledStrain'], normalized_mono_df['env'] = strains, env
    return normalized_mono_df


def CombineAllFeatures(type):
    """
    Combine all features into a single dataset.
    :param type: one way/ two-way
    :return: Full dataset, carbon pca, strain phylogenetic pca, and normalized monogrow
    """
    processed_data_table = pd.read_csv("./Data/processed_data_table.csv")
    monoGrow = pd.read_csv('./Data/monoGrow.csv')
    monoGrow_normelized = normelized_monoGrow(monoGrow, 'monoGrow')
    metabolic_distance = pd.read_csv('./Data/metabolicDistanceTable.csv')
    monoGrow24 = pd.read_csv("./Data/monoGrow24.csv", index_col=0)
    monoGrow24_normelized = normelized_monoGrow(monoGrow24, 'monoGrow24')

    # Merge between all datasets
    full_data = mergeAllFeatures(processed_data_table, monoGrow_normelized, metabolic_distance, monoGrow24_normelized)

    # Add carbons pca and strains pca
    pca_df_carbon = createPCA("carbon", 4)
    pca_df_strainPhy = createPCA("strain", 2)
    full_data = addMetabolicPhylogeneticFeatures(full_data, pca_df_carbon, pca_df_strainPhy)

    if type == 'one_way':
        full_data = full_data.drop(['1 on 2: Effect'], axis=1)
        full_data = full_data.dropna()
    else:
        full_data = full_data.dropna()
    return full_data, pca_df_carbon, pca_df_strainPhy, monoGrow_normelized, monoGrow24_normelized


def main():
    file_name = "Features"
    type = 'one_way'
    df, pca_df_carbon, pca_df_strainPhy, monoGrow_normelized, monoGrow24_normelized = CombineAllFeatures(type)
    #df.to_csv(file_name + ".csv")
    return pca_df_carbon, pca_df_strainPhy


if __name__ == "__main__":
    main()
