from DreamMLModels import *
import time
import pandas as pd

data = pd.read_csv('data/dados_para_treinar_reduzidos.csv',index_col=0)


#TRAINING MODELS
print('\n TRAINING MODELS \n')

#CREATE INSTANCES FOR THE MODEL CLASS
dream = DreamMLModels(data,(100,50)) # 10 lines to the test set (5 to evaluate the final model)
#dream_feat_sel = DreamMLModels(data,(100,50), feature_selection=True)

start_time = time.time()
#CREATE AND TEST MODELS
#create and evaluate all the models
print('\n *************************** ML Models ************************************\n')
modelos = dream.create_test_eval('ALL')

print("Execution time: " + str((time.time() - start_time)) + 's')


#LR funciona
#NNT - não... não funciona com os dados divididos em treino e teste, só funciona com os dados do x e do y todos
#SVM - funciona
#RF -funciona