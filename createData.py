import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

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

REMOVE_COLUMNS = ['Strength', 'Type', '1 on 2: 25th Percentile', '1 on 2: 75th Percentile', '2 on 1: Replicates',
                  '2 on 1: 25th Percentile', '2 on 1: 75th Percentile', 'Class', '1 on 2: Replicates']


def createTable(type):
    """
    Create table for PCA. Can either be carbons' metabolic profiles or phylogenetic distance matrix.
    :param type: carbon/ strain
    :return: Data_old for PCA
    """
    # create carbon table
    if type == "carbon":
        table = pd.read_csv("Data_msystems/metabolic_profiles.csv").drop(['carbon'], axis=1)
        coulmns_names = table.columns
        table  = StandardScaler().fit_transform(table)
        table = pd.DataFrame(table,columns= coulmns_names)
    # create phylo distance
    elif type == "strain":
        table = pd.read_csv("./Data_msystems/pairwiseDistances-21Isolates&Ecoli.csv").drop(['Unnamed: 0'], axis=1)
        table = table.drop([9, 11])
        table = table.drop(columns=["SF2", "CB"])
    return table


def createPCA(type, dimensions, strain=None):
    """
    Create PCA data. Can either be strains' pca (from phylogenetic distance matrix) or carbons' pca
    (from monoculture growth df)
    :param type: Carbon/ strain
    :param dimensions: Number of dimension for pca
    :param strain: A strain to remove.
    :return: PCA table
    """
    table = createTable(type)
    if type == 'carbon' and strain != None:
        table = table.drop([strain], axis=1)
    # fit pca and find the components
    pca = PCA(dimensions)
    principalComponents = pca.fit_transform(table)
    sum = np.sum(pca.explained_variance_ratio_)
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
    :param processed_data_table: dataset of effect measurement.
    :param monoGrow: Monoculture growth yield (after 72h)
    :param metabolic_distance: metabolic distance table
    :param monoGrow24: Monoculture growth yield (after 24h)
    :return: Merged table
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
    :return: Merged table
    """
    df = pd.merge(df, pca_df_carbon, how='left', left_on=['Carbon'], right_on=['carbon']).drop(['carbon'], axis=1)
    df = pd.merge(df, pca_df_strainPhy, how='inner', left_on=['Bug 1'], right_on=['strain']).drop(['strain'], axis=1)
    df = pd.merge(df, pca_df_strainPhy, how='inner', left_on=['Bug 2'], right_on=['strain']).drop(['strain'], axis=1)
    return df


def normelized_monoGrow(df, column_name):
    """
    Create normalized monoculture
    :param df: Unnormalized monoculture growth yield
    :param column_name: column name to normalize (monogrow or monogrow 24)
    :return: normalized monoculture growth yield
    """
    normalized_mono_df = pd.DataFrame()
    for name, group in df.groupby(['labeledStrain']):
        normalized_mono = (group[column_name] - group[column_name].mean())/group[column_name].std()
        normalized_mono = pd.DataFrame({column_name: normalized_mono})
        normalized_mono['labeledStrain'],normalized_mono['env']= name,group['env']
        normalized_mono_df = pd.concat([normalized_mono_df, normalized_mono], ignore_index=True)
    return normalized_mono_df


def normelized_metabolicDistance(df, column_name):
    """
    Create normalized monoculture
    :param df: Unnormalized metabolic distance
    :param column_name: column name to normalize (monogrow or monogrow 24)
    :return: normalized monoculture growth yield
    """
    normalized_md_df = pd.DataFrame()
    for name, group in df.groupby(['Bug 2']):
        normalized_md = (group[column_name] - group[column_name].min())/(group[column_name].max() - group[column_name].min())
        normalized_md = pd.DataFrame({column_name: normalized_md})
        normalized_md['Bug 2'],normalized_md['Bug 1']= name,group['Bug 1']
        normalized_md_df = pd.concat([normalized_md_df, normalized_md], ignore_index=True)
    return normalized_md_df

def CombineAllFeatures(type):
    """
    Combine all features into a single dataset.
    :param type: One way/ two-way
    :return: Full dataset, carbon pca, strain phylogenetic pca, and normalized monogrow
    """
    processed_data_table = pd.read_csv("./Data_msystems/processed_data_table.csv")
    monoGrow = pd.read_csv('Data_msystems/monoGrow.csv')
    monoGrow_normelized = normelized_monoGrow(monoGrow, 'monoGrow')
    metabolic_distance = pd.read_csv('./Data_msystems/metabolicDistanceTable.csv')
    metabolic_distance_normalized = normelized_metabolicDistance(metabolic_distance,'metDis')
    monoGrow24 = pd.read_csv("Data_msystems/monoGrow24.csv", index_col=0)
    monoGrow24_normelized = normelized_monoGrow(monoGrow24, 'monoGrow24')

    # Merge between all datasets
    full_data = mergeAllFeatures(processed_data_table, monoGrow_normelized, metabolic_distance_normalized, monoGrow24_normelized)

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


def metabolicDistance(uniquelabeled,fullMonoTable):
    """
    Create a metabolic distance table.
    :param uniquelabeled: strains' names
    :param fullMonoTable: monoculture df
    :return: A new table of unnormalized metabolic distances
    """
    metabolic_distance_df = pd.DataFrame()
    for strain in uniquelabeled:
        labeledStrain= fullMonoTable[fullMonoTable['labeledStrain'] == strain]
        labeledStrainMono = np.asarray(labeledStrain['monoGrow'])
        for otherStrain in uniquelabeled:
            otherStraindf = fullMonoTable[fullMonoTable['labeledStrain'] == otherStrain]
            otherStrainMono = np.asarray(otherStraindf['monoGrow'])
            metDistance = otherStrainMono-labeledStrainMono
            metDistance = np.sqrt(np.sum(np.power(metDistance,2)))
            new_line = {'Bug 1': strain, 'metDis': metDistance, 'Bug 2': otherStrain}
            metabolic_distance_df = metabolic_distance_df.append(new_line, ignore_index=True)
    metabolic_distance_df.to_csv("Data_msystems/metabolicDistanceTable.csv")
    return metabolic_distance_df

def metabolic_profiles(monoGrow):
    """
    Create a metabolic profiles table.
    :param monoGrow: monoculture df
    :return: A new table of metabolic profiles(for each environment)
    """
    metabolic_profiles_df = pd.DataFrame()
    for carbon in CARBONS:
        a = metabolic_profiles_df.columns
        mono_carbon = monoGrow[monoGrow['env'] == carbon]
        strains = np.array(mono_carbon['labeledStrain']).transpose()
        if 'strain' not in metabolic_profiles_df.columns:
            metabolic_profiles_df['strain'] = strains
        mono = np.array(mono_carbon['monoGrow']).transpose()
        metabolic_profiles_df[carbon] = mono

    metabolic_profiles_df = metabolic_profiles_df.transpose()
    metabolic_profiles_df = pd.DataFrame(metabolic_profiles_df)
    metabolic_profiles_df.to_csv("Data_msystems/metabolic_profiles.csv")
    return metabolic_profiles_df

def main():
    type = 'one_way'
    df, pca_df_carbon, pca_df_strainPhy, monoGrow_normelized, monoGrow24_normelized = CombineAllFeatures(type)
    return pca_df_carbon, pca_df_strainPhy


if __name__ == "__main__":
    main()
