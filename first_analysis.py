import pandas as pd
import kd_transformations


def look_at_data(data):
    """
    :type data: pandas data frame
    """
    print('Data head:\n')
    print(data.head())
    print('\n')
    print('Data info:\n')
    print(data.info())
    print('\n')
    print('Null values for column: \n')
    print(data.isnull().sum())
    print('\n')

def complete_cols(data, col_to_complete, col_with):
    """
    COMPLETE col_to_complete COM col_with
    :param data: pandas data frame
    :param col_to_complete: name of the column to complete
    :param col_with: name of the column to complete with
    :return: the data frame with the col_to_complete complete ()
    """
    dic = {}

    def complete_dic(line):
        if line[col_with] not in dic and pd.notnull(line[col_to_complete]) and pd.notnull(line[col_with]):
            dic[line[col_with]] = line[col_to_complete]

        return None

    data.apply(complete_dic, axis=1)
    dic = pd.Series(dic)

    def complete(line):
        if (line[col_with] in dic) and not (pd.notnull(line[col_to_complete])):
            return dic[line[col_with]]
        else:
            return line[col_to_complete]

    data[col_to_complete] = data.apply(complete, axis=1)
    return data

def first_clean_and_transformations(data):
    data = data.dropna(subset=['standard_value'])
    data = kd_transformations.filter_standard_type(data, keep=True, to_keep = ['KD','kd'])
    # Leaving only NM units
    data = kd_transformations.filter_standard_units(data) #just NM
    # Conversion to pkd values in the correct unites
    data = kd_transformations.conversion(data)

    return data



