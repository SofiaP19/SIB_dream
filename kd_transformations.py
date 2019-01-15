import numpy as np
import re
import pandas as pd


def filter_standard_type(data, regex="(?i)(kd)", keep = False, to_keep = ["PKD", "LOGKD", "LOG KD", "LOG 1/KD", "Kd", "KD\'", "-LOG KD", "KD"]):
    """
    :param data: pandas data frame
    :param regex: regular expression to use as filter
    :return: data online with lines that matche regex
    :keep: indicates if is supose to filter according the values of list
    :list: list of standard types to keep

    "(?i)(kd)" #faz match com qualquer kd (KD,Kd,kD,kd) em qualquer indice da string
    """
    if keep == False:
        def regex_filter(line):
            val=line.standard_type
            if val:
                val = str(val)
                mo = re.search(regex,val)
                if mo:
                    return True
                else:
                    return False
            else:
                return False

        data = data[data.apply(regex_filter, axis=1)]

    else:
        data = data.loc[data['standard_type'].isin(to_keep)]

    return data


def filter_standard_units(data, to_keep=["NM"]):
    data = data.loc[data['standard_units'].isin(to_keep)]
    return data


def conversion(data):

    def conversion_pkd(value):
        res = value/(10**9)
        res2 = -np.log10(res)
        return res2

    def conversion_to_kd(line):
        s_type = line.standard_type
        value = line.standard_value

        if str(s_type) == 'LOG KD' or str(s_type) == 'LOGKD':
            return conversion_pkd(10**value)

        elif str(s_type) == '-LOG KD':
            return conversion_pkd(-10**value)

        elif str(s_type) == 'LOG 1/KD':
            return conversion_pkd(10**(-value))

        elif str(s_type) == 'PKD':
            return value

        else:
            return conversion_pkd(value)


    data['standard_value'] = data.apply(conversion_to_kd, axis=1)
    return data

