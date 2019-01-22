import pandas as pd
from src.get_molecules import *
import numpy as np
from chembl_webresource_client.new_client import new_client

def get_compound_descriptors(df, col_compound_id='compound_id', col_smiles='smiles', need_to_generate_similes=True):
    """
    :param df: pandas data frame
    :return: return df with columns with numeric descriptors for compounds
    """

    if need_to_generate_similes:
        # remove rows without a CHEMBL identifier
        df = df[~df[col_compound_id].isnull()]
        df.index = range(df.shape[0])


        # Read CHEMBL data mapping InChIKeys to InChI and SMILES identifiers which will then be read by RDKit
        # to generate molecular descriptors.
        '''
        chembl_reps = pd.read_csv('data/chembl_24_1_chemreps.txt.gz', sep='\t').set_index('chembl_id')
        compound_reps = chembl_reps.loc[df[col_compound_id], :]
        chembl_reps['chembl_id'] = chembl_reps.index
        compound_reps.index = range(compound_reps.shape[0])

        finaldf = df.join(compound_reps, rsuffix='_chembl').rename(columns={'canonical_smiles': col_smiles})'''


        # Geting smiles with chwmble api

        molecule = new_client.molecule

        id_smiles = {}

        def get_smiles(line):
            compound_id = line[col_compound_id]

            if compound_id in id_smiles:
                return id_smiles[compound_id]

            #try to get a chemble correspondence
            try:
                dic = molecule.get(compound_id)
            except:
                dic = None

            if dic != None:
                #print(dic)
                #if dic['molecule_structures']['canonical_smiles'] != None:
                try:
                    id_smiles[compound_id] = dic['molecule_structures']['canonical_smiles']
                    return id_smiles[compound_id]
                except:
                    return np.nan
            else:
                return np.nan

        df[col_smiles] = df.apply(get_smiles, axis=1)

        finaldf = df.copy()

    else:
        finaldf = df.copy()

    # Get the unique compounds from the dataframe above to speed up the generation of molecular descriptors.
    unique_compounds = finaldf[[col_compound_id, col_smiles]].drop_duplicates().dropna()

    # Generate one dataframe ($n$ by $2$ shape) for each molecular descriptor type.
    descriptors = ['Morgan', 'MACCS', 'MolLogP', 'AtomPair'] #, '2Dfingerprint']#, 'TPSA']#['Morgan', 'MACCS', 'MolLogP']

    descriptor_dfs = list(map(lambda x: df_for_molecular_descriptors(unique_compounds, x, col_smiles), descriptors))

    descriptor_dfs[0].head()

    # Combine all three dataframes into a single long dataframe with all features
    tdfs = []

    # descriptors that are represented as integers
    integer_descriptors = ['Morgan', 'MACCS']

    for descriptor, df in zip(descriptors, descriptor_dfs):

        # if the descriptor is a list of features (such as Morgan - 1024 values per molecule)
        if df[descriptor].dtype == 'O':

            # make a new dataframe from each descriptor
            feature_df = pd.DataFrame.from_dict(dict(zip(df.index, df[descriptor].values))).T

            # convert numbers to integer if needed
            if descriptor in integer_descriptors:
                feature_df = feature_df.astype(int)

            # set column names as descriptorname_n
            feature_df.columns = [descriptor + "_" + str(i) for i in range(feature_df.shape[1])]
            feature_df[col_smiles] = df.smiles

        else:
            feature_df = df

        tdfs.append(feature_df)

    # join all dataframes in tdfs by smiles
    final_descriptor_df = pd.concat([df.set_index(col_smiles) for df in tdfs], axis=1, join='inner').reset_index()

    # Merge the molecular descriptor dataframe with the original DTC data,
    # drop columns with all NA values and remove unnecessary columns.
    training_df = pd.merge(finaldf, final_descriptor_df, on=col_smiles, how='right').dropna(how='all', axis=1)

    # Remove rows without valid units or null values
    #good_units = training_df.standard_units == 'NM'
    #has_values = ~training_df.standard_value.isnull() already done

    #training_df = training_df[good_units & has_values]

    return training_df
