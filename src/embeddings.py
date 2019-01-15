from mol2vec.features import mol2alt_sentence, MolSentence, DfVec, sentences2vec
from gensim.models import Word2Vec
from gensim.models import word2vec
from rdkit import Chem
import pandas as pd
import numpy as np


def loadTrainingData():
    df = pd.read_csv('../data/KD_data.csv').iloc[:,1:]
    df = df[~df['compound_id'].isnull()]
    df.index = range(df.shape[0]) # resets index
    chembl_reps = pd.read_csv('../data/chembl_24_1_chemreps.txt.gz', sep='\t').set_index('chembl_id')
    compound_reps = chembl_reps.loc[df['compound_id'],:]
    chembl_reps['chembl_id'] = chembl_reps.index
    compound_reps.index = range(compound_reps.shape[0])
    finaldf = df.join(compound_reps, rsuffix='_chembl').rename(columns={'canonical_smiles':'smiles'})
    return finaldf

def loadTestData():
    df = pd.read_csv('../data/round_1_template.csv')
    df.rename(columns={'Compound_SMILES':'smiles'}, inplace=True)
    return df
    
def generateEmbeddings(original_df, trained_model):
    unique_compounds = original_df[['smiles']].drop_duplicates().dropna()
    smiles = list(unique_compounds['smiles'])
    smiles = [x.split(';')[0] for x in smiles]
    # SMILES to Mol
    molecules = [Chem.MolFromSmiles(x) for x in smiles]
    # Load previously trained mol2vec model
    model = Word2Vec.load(trained_model)
    # Convert molecules to sentences and then to embeddings
    sentences = [mol2alt_sentence(x, 1) for x in molecules]
    vectors = [DfVec(x) for x in sentences2vec(sentences, model, unseen='UNK')]
    vec_df = pd.DataFrame(data=np.array([x.vec for x in vectors]))
    vec_df.columns = ['mol2vec_' + str(x+1) for x in vec_df.columns.values]
    vec_df.index = unique_compounds.index.values # confirm that order in smiles_df is maintained in vec_df in previous steps
    # Embeddings with 100 dimensions instead of 300! using model provided in Notebooks repository; model provided in examples doesn't unpickle...
    embeddings_df = pd.concat([unique_compounds, vec_df], axis=1)
    df = original_df.merge(embeddings_df, how='right', on="smiles").dropna(how='all', axis=1)
    return df

def saveEmbeddings(df, output_file, training_set=True):
    if training_set:
        extra_cols = ['assay_description', 'title', 'journal', 'doc_type', 'annotation_comments', 'pubmed_id', 'detection_tech', 'assay_cell_line'] # extra_cols in training set
        df = df[[col for col in df.columns if col not in extra_cols]]
        # Remove rows without valid units or null values
        good_units = df.standard_units == 'NM'
        has_values = ~df.standard_value.isnull()
        df = df[good_units & has_values]
    else:
        df.rename(columns={'smiles':'Compound_SMILES'}, inplace=True)
    df.to_csv(output_file, index=False)
    return df

# training_df = pd.merge(finaldf, final_descriptor_df, on='smiles', how='right').dropna(how='all', axis=1)

if __name__ == '__main__':
    smi = loadTrainingData()
    emb_df = generateEmbeddings(smi, 'model_300dim.pkl')
    saveEmbeddings(emb_df, '/home/dbaptista/Dropbox/Drug_Kinase_DREAM/data/compound_embeddings_train.csv', training_set=True)
    smi2 = loadTestData()
    emb_df2 = generateEmbeddings(smi2, 'model_300dim.pkl')
    saveEmbeddings(emb_df2, '/home/dbaptista/Dropbox/Drug_Kinase_DREAM/data/compound_embeddings_test.csv', training_set=False)