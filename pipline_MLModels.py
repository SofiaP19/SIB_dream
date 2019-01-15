import pandas as pd
from DreamMLModels import *
#/Users/martagomes/Documents/Bioinformatica/3Semestre/Sistemas_Inteligentes/Trabalho/dream_project/data
data = pd.read_csv('data/dados_para_treinar_reduzidos.csv', index_col=0)

#PREPARE THE DATA FRAME TO TRAIN MODELS
#data = prepare_to_ML.clean(data)


#CREATE INSTANCES FOR THE MODEL CLASS
dream = DreamMLModels(data,(50,20)) # 50 lines to the test set (20 to evaluate the final model)
dream_feat_sel = DreamMLModels(data,(50,20), feature_selection=True)

#CREATE AND TEST MODELS

#modelos = dream.create_test_eval('ALL')

#for mod in modelos:
    #print(mod) # ------> (model designation, model fit, predicted values)

#print("\n With feature selection: \n")
#dream_feat_sel.create_test_eval('ALL')


#Selecting the top features with random forests
print(dream.RF_feature_selection(50,True))




