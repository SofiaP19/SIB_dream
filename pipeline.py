import first_analysis
import enzyme_descriptors
import compound_descriptors
from DreamMLModels import *
import time
start_time = time.time()/60

# IMPORT DATA TO PANDAS OBEJECT:
print('\n IMPORT DATA TO PANDAS OBEJECT \n')
data = pd.read_csv('data/DtcDrugTargetInteractions.csv', sep=',')

# a small one first to see if everithyng runs:
#data = pd.read_csv('data/dreamInteractions_reduzido.csv', sep=',', index_col=0)

# ANALYZE THE DATA AND DELETE SOME ROWS WITHOUT INTEREST:
print('\n ANALYZE THE DATA AND DELETE SOME ROWS WITHOUT INTEREST \n')
first_analysis.look_at_data(data)
# Delete the rows withouth interest and make kd tranformations and conversions (lead with units too)
data = first_analysis.first_clean_and_transformations(data)


#COMPLETE COMPOUND ID WITH COMPOUND NAME
print('\n TRYING TO COMPLETE DATA \n')
data = first_analysis.complete_cols(data, 'compound_id', 'compound_name')
#see if there were any changes
print('Null values for column: \n')
print(data.isnull().sum())
print('\n')


#COMPLETE COUMPOUND ID WITH STANDARD_INCHI_KEY
data = first_analysis.complete_cols(data, 'compound_id', 'standard_inchi_key')
#see if there were any changes
print('Null values for column: \n')
print(data.isnull().sum())
print('\n')


#COMPLETE STANDARD_INCHI_KEY WITH COMPOUND ID
data = first_analysis.complete_cols(data, 'standard_inchi_key', 'compound_id')
#see if there were any changes
print('Null values for column: \n')
print(data.isnull().sum())
print('\n')


# GET COMPOUND DESCRIPTORS
print('\n GET COMPOUND DESCRIPTORS \n')
data = compound_descriptors.get_compound_descriptors(data, col_compound_id='compound_id', col_smiles='smiles', need_to_generate_similes=True)

# GET ENZYME SEQUENCES, GO TERMS AND DOMAINS
print('\n GET ENZYME UNIPROT INFO \n')
data = enzyme_descriptors.add_uniprot_info_cols(data, 'target_id')

# GET ENZYME FEATURES FROM SEQUENCES
print('\n GET ENZYME FEATURES FROM SEQUENCES \n')
data = enzyme_descriptors.add_ProFet_features(data)

#print(data.isnull().sum())
first_analysis.look_at_data(data)


data.to_csv('data/DtcDrugTargetInteractions_complete_to_train.csv', sep=',')


# data = data[data.target_id != 'P08485'] # we cant get features to this row
# data.isnull().sum()


#TRAINING MODELS
print('\n TRAINING MODELS \n')

#CREATE INSTANCES FOR THE MODEL CLASS
dream = DreamMLModels(data,(10,5)) # 10 lines to the test set (5 to evaluate the final model)
dream_feat_sel = DreamMLModels(data,(10,5), feature_selection=True)

#CREATE AND TEST MODELS
#create and evaluate all the models
print('\n *************************** ML Models ************************************\n')
modelos = dream.create_test_eval('ALL')

#for mod in modelos:
    #print(mod) # ------> (model designation, model fit, predicted values)

print("\n With feature selection: \n")
dream_feat_sel.create_test_eval('ALL')

print('\n FEATURE SELECTION WITH RF \n')
#Selecting the top features with random forests (only ProFet features)
print(dream.RF_feature_selection(100,True))



print("Execution time: " + str((time.time() - start_time)/60) + 's')