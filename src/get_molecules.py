import pandas as pd
import numpy as np
np.set_printoptions(threshold=np.nan)
from itertools import chain
from rdkit import Chem, DataStructs
from rdkit.Chem import Descriptors, AllChem, rdMolDescriptors
from rdkit.Chem.Pharm2D import Gobbi_Pharm2D, Generate

# import pickle

'''Molecules using Rdkit'''




def df_for_molecular_descriptors(df_with_chembl_and_canonical_smiles=pd.DataFrame(), descriptors='', smiles_column_name='smiles'):
	# todo later change this to the same structure as the proteins() function
	# todo or maybe not, it is working this way and could be difficult since some don't need the
	# todo ConvertToNumpyArray(desc, arr) part
	# todo probably it is better, due to the memory problems that we had to have them separated
	# todo if it is really necessary, maybe, but as it is right now is fine
	'''beginning of auxiliary functions'''

	def _create_matrix_with_data(df, column, from_csv=True):
		def __convert_str_to_numpy(string):
			if string is not np.nan:
				return np.fromstring(''.join(string).replace('[', '').replace(']', '').replace(' ', ','), sep=',')
		if from_csv==True:
			df[column] = df[column].apply(lambda x: __convert_str_to_numpy(x))
		i = 0
		length = [len(df[column][i]) if isinstance(df[column][i], np.ndarray) else None for i in
				  range(0, len(df[column]))]
		new_df = pd.DataFrame(np.concatenate(
			[x.reshape(1, -1) if isinstance(x, np.ndarray) else np.array([np.nan] * length[0]).reshape(1, -1) for x in
			 (df[column]).tolist()], axis=0)).set_index(df.iloc[:, 0])
		return new_df

	def __fromBinaryToText(bytess):
		return ''.join([str(bin(bytess[i]))[2:] for i in range(len(bytess))])

	def _getMorganWithTry(molecule):
		'''
		function to deal with the possible errors with the function GetMorganFingerprintAsBitVect
		:param molecule: canonical smiles to compute
		:return: Morgan FingerPrint
		'''
		try:
			arr = np.zeros((1,))
			desc = rdMolDescriptors.GetMorganFingerprintAsBitVect(Chem.MolFromSmiles(str(molecule)), 2,
																  1024)
			DataStructs.ConvertToNumpyArray(desc, arr)
		except Exception as e:
			print(e)
			# print('error ' + str(molecule))
			arr = np.nan
		return arr

	def _getMACCSKeys(molecule):
		'''
		function to deal with the possible errors with the function GetMACCSKeysFingerprint()
		:param molecule: canonical smiles to compute
		:return: MACCSKeys
		'''
		try:
			arr = np.zeros((1,))
			desc = rdMolDescriptors.GetMACCSKeysFingerprint(Chem.MolFromSmiles(str(molecule)))
			DataStructs.ConvertToNumpyArray(desc, arr)
		except Exception as e:
			print(e)
			# print('error ' + str(molecule))
			arr = np.nan
		return arr

	def _getAtomPairFingerPrint(molecule):
		'''
		function to deal with the possible errors with the function GetHashedAtomPairFingerprint()
		:param molecule: canonical smiles to compute
		:return: HashedAtomPairFingerprint
		'''
		try:
			arr = np.zeros((1,))
			desc = rdMolDescriptors.GetHashedAtomPairFingerprint(Chem.MolFromSmiles(str(molecule)))
			DataStructs.ConvertToNumpyArray(desc, arr)
		except Exception as e:
			print(e)
			# print('error ' + str(molecule))
			arr = np.nan
		return arr

	def _get2DFingerprint(molecule):
		'''
		function to deal with the possible errors with the function Gen2DFingerprint
		:param molecule: canonical smiles to compute
		:return: Morgan FingerPrint
		'''
		try:
			# arr = np.zeros((1,))
			desc = Generate.Gen2DFingerprint(Chem.MolFromSmiles(str(molecule)), Gobbi_Pharm2D.factory)
			arr = np.array(desc)
		except Exception as e:
			print(e)
			# print('error ' + str(molecule))
			arr = np.nan
		return arr

	def _getTSPA(molecule):
		'''
		function to deal with the possible errors with the function TSPA()
		:param molecule: canonical smiles to compute
		:return: HashedAtomPairFingerprint
		'''
		try:
			desc = Descriptors.TPSA(Chem.MolFromSmiles(str(molecule)))
		except Exception as e:
			print(e.args)
			print('error ' + str(molecule))
			desc = np.nan
		return desc

	def _getMolLogP(molecule):
		'''
		function to deal with the possible errors with the function TSPA()
		:param molecule: canonical smiles to compute
		:return: HashedAtomPairFingerprint
		'''
		try:
			desc = Descriptors.MolLogP(Chem.MolFromSmiles(str(molecule)))
		except:
			print('error ' + str(molecule))
			desc = np.nan
		return desc

	'''end of auxiliary functions'''

	# df_with_chembl_and_canonical_smiles.index = df_with_chembl_and_canonical_smiles['molecule_chembl_id']
	df = pd.DataFrame(data=None, index=df_with_chembl_and_canonical_smiles.index)
	# df[smiles_column_name] = df_with_chembl_and_canonical_smiles['canonical_smiles']
	df[smiles_column_name] = df_with_chembl_and_canonical_smiles[smiles_column_name]
	if descriptors == ('Morgan'):
		print('Morgan')
		df['Morgan'] = df[smiles_column_name].apply(lambda x: _getMorganWithTry(x))
	elif descriptors == ('MACCS'):
		print('MACCS')
		df['MACCS'] = df[smiles_column_name].apply(lambda x: _getMACCSKeys(x))
	elif descriptors == ('AtomPair'):
		print('AtomPair')
		df['AtomPair'] = df[smiles_column_name].apply(lambda x: _getAtomPairFingerPrint(x))
	elif descriptors == ('2Dfingerprint'):
		print('2Dfingerprint')
		df['2Dfingerprint'] = df[smiles_column_name].apply(lambda x: _get2DFingerprint(x))
	elif descriptors == ('TPSA'):
		print('TSPA')
		df['TSPA'] = df[smiles_column_name].apply(lambda x: _getTSPA(x))
	elif descriptors == ('MolLogP'):
		print('MolLogP')
		df['MolLogP'] = df[smiles_column_name].apply(lambda x: _getMolLogP(x))
	# return _create_matrix_with_data(df,descriptor, False)
	return df

if __name__ == '__main__':
	# chembl = pd.read_csv('chembl23_selected_frac30.csv')
	zinc_targets_1 = pd.read_csv('targets_1.csv', index_col=0)
	zinc_anti_targets_1 = pd.read_csv('anti_targets_1.csv', index_col=0)
	# descriptors = ['Morgan', 'MACCS', 'AtomPair', 'TSPA', 'MolLogP']
	# descriptors = ['Morgan', 'TSPA', 'MolLogP']  # this is just for right now, for all the
	# the fingerprints at the same time, use the list above
	descriptors = ['Morgan','MACCS']
	descriptors = ['Morgan']
	for descriptor in descriptors:
		asd = df_for_molecular_descriptors(zinc_anti_targets_1, descriptor)
		asd.to_csv(path_or_buf='anti_targets_1_' + descriptor + '.csv')
		# del (asd)
	to_redo = pd.read_csv('anti_targets_1_Morgan.csv')
	final_df = _create_matrix_with_data(to_redo, 'Morgan', True)
	final_df.to_csv('anti_targets_1_Morgan_3.csv')
	# asd = df_for_molecular_descriptors(chembl, 'Morgan')
	# asd.to_csv('Morgan_3_test.csv')
	#
	# qwer = pd.read_csv('MACCS.txt')

	'''Molecule testing'''
	# m = Chem.MolFromSmiles(str(chembl.iloc[0][3]))
	# a = Generate.Gen2DFingerprint(m, Gobbi_Pharm2D.factory)
	# desc = rdMolDescriptors.GetHashedAtomPairFingerprint(m)
	# arr = np.zeros((1,))
	# DataStructs.ConvertToNumpyArray(b, arr)
