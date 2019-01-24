from bs4 import BeautifulSoup
import requests
import numpy as np
import pickle
from ProFET_master.ProFET.feat_extract.FeatureGen import Get_Protein_Feat

'''
def getUniProtInfo(uniprotId):
    url = 'http://www.uniprot.org/uniprot/' + uniprotId + '.xml'
    try:
        response = requests.get(url)
    except:
        return np.nan
    soup = BeautifulSoup(response.text, 'lxml')
    seq = soup.findAll('sequence')
    res = np.nan
    for s in seq:
        if s.text:
            res = s.text.replace('\n', '')
    return res


def add_sequence_col(data, uniProtId_col_name):
    """
    :param data: pandas data frame
    :param uniProtId_col_name: name of the column containing uniprot ids
    :return:  -> data with a extra column sequence
    """
    id_seqs = {}

    def get_sequence(line):
        uniprotIds = line[uniProtId_col_name].split(',')
        uniprotId = str(uniprotIds[0].strip())
        # print(uniprotId)
        if uniprotId in id_seqs:
            return id_seqs[uniprotId]
        else:
            #print(uniprotId)
            res = getUniProtInfo(uniprotId)
            id_seqs[uniprotId] = res
            return res

    # remove rows withouth uniProtId_col_name
    data = data[~data[uniProtId_col_name].isnull()]
    #data.loc[:,'sequence'] = data.apply(get_sequence, axis=1)
    data['sequence'] = data.apply(get_sequence, axis=1)
    return data'''


def get_domains_list(data, uniProtId_col_name='target_id'):

    domains_list = []

    def get_prot_domains(uniprotId):
        url = 'http://www.uniprot.org/uniprot/' + uniprotId + '.xml'
        try:
            response = requests.get(url)
        except:
            return None

        soup = BeautifulSoup(response.text, 'lxml')
        # Domains
        domains_value = []
        domains = soup.findAll('feature', {'type': 'domain'})
        for dom in domains:
            if dom not in domains_list:
                domains_list.append(dom['description'])
                #domains_value.append(dom['description'])
        return None

    visited_ids = []

    def get_values(line):
        uniprotIds = line[uniProtId_col_name].split(',')
        uniprotId = str(uniprotIds[0].strip())
        # print(uniprotId)
        # Get the values for sequence, go terms and domains
        if uniprotId in visited_ids:
            return None
        else:
            get_prot_domains(uniprotId)

    data = data[~data[uniProtId_col_name].isnull()]
    data.apply(get_values, axis=1)

    #print(len(domains_list))
    with open('data/Domains_list.pkl', 'wb') as f:
        pickle.dump(domains_list, f)

    return True

def getUniProtInfo(uniprotId):
    url = 'http://www.uniprot.org/uniprot/' + uniprotId + '.xml'
    try:
        response = requests.get(url)
    except:
        return np.nan
    soup = BeautifulSoup(response.text, 'lxml')
    # Sequence
    seq = soup.findAll('sequence')
    sequence_value = np.nan
    for s in seq:
        if s.text:
            sequence_value = s.text.replace('\n', '')
    # GO terms
    """
    gos_value = []
    gos = soup.findAll('dbreference', {'type': 'GO'})
    for go in gos:
        gos_value.append(go['id'])"""
    # Domains
    domains_value = []
    domains = soup.findAll('feature', {'type': 'domain'})
    for dom in domains:
        domains_value.append(dom['description'])
    return (sequence_value, domains_value)




def add_uniprot_info_cols(data, uniProtId_col_name='target_id'):
    """
    :param data: pandas data frame
    :param uniProtId_col_name: name of the column containing uniprot ids
    :return:  -> data with extra columns sequence, go terms and domains
    """
    id_values = {}
    """
    with open('data/GOterms_list.pkl', 'rb') as f:
        GOterms_list = pickle.load(f)"""

    with open('data/Domains_list.pkl', 'rb') as f:
        Domains_list = pickle.load(f)

    def get_values(line):
        uniprotIds = line[uniProtId_col_name].split(',')
        new_line = line.copy()
        uniprotId = str(uniprotIds[0].strip())
        # print(uniprotId)
        # Get the values for sequence, go terms and domains
        if uniprotId in id_values:
            new_line = id_values[uniprotId]
        else:
            #print(uniprotId)
            res = getUniProtInfo(uniprotId)

            # add columns
            # ----sequence
            new_line['sequence'] = res[0]

            # ----GO terms
            """
            for go in GOterms_list:
                if go in res[1]:
                    new_line[go] = 1
                    res[1].remove(go)
                else:
                    new_line[go] = 0"""

            # ----Domains
            # isto está a demorar bué, como otimizar??!
            for dom in Domains_list:
                if dom in res[1]:
                    new_line[dom] = 1
                    res[1].remove(dom)  # para ir diminuindo o tempo de procura
                else:
                    new_line[dom] = 0
            id_values[uniprotId] = new_line
        return new_line

    # remove rows withouth uniProtId_col_name
    data = data[~data[uniProtId_col_name].isnull()]
    data = data.apply(get_values, axis=1)
    return data


def add_ProFet_features(data):
    """
    :param data: pandas data frame
    :return: data with enzime features
    Admit that sequences are in the column 'sequence'
    """

    #remove the lines with no sequence information
    data = data[~data.sequence.isnull()]

    # dictionary to save the features of sequences
    seq_feat = {}

    ######ATUALIZAR VALORES DAS CHAVES E APRAMETROS DO GET_PROTEIN_FEAT

    # get list with name of profet features
    with open('profFet_keys_list.pkl', 'rb') as f:
        chaves = pickle.load(f)

    def complete_feature(line):
        seq = line.sequence

        if seq in seq_feat:
            line = seq_feat[seq]

        else:
            try:
                dic = Get_Protein_Feat(seq, SeqReducedAlph='ofer14', ReducedK=2, GetSimpleFeatSet=True,
                                       GetExtraScaleFeatSet=True, aaParamScaleWindow=7, ExtraScaleWindow=17,
                                       GetSubSeqSegs=True, SubSeqSegs=3, GetTriLetterGroupAlphKFreq=True,
                                       TriLetterGroupAlphK=5, GetSeqReducedGroups=True, SeqReducedGroups='ofer_w8',
                                       GetSeqReducedAlph=True, GetCTDFeatSet=True, GetPTMFeatSet=True,
                                       GetDBCleavageFeatSet=True, split_N=False, split_C=False, N_TAIL_REGION=30, C_TAIL_REGION=30)
            except:
                dic = None

            if dic != None:
                #print("Get_Protein_Feat done :D !")
                for chave in chaves:
                    if chave in dic:
                        line[chave] = dic[chave]
                    else:
                        line[chave] = np.nan
            else:
                #print("Error in Get_Protein_Feat :(")
                for chave in chaves:
                    line[chave] = np.nan

            seq_feat[seq] = line

        return line

    data_protein_feature = data.apply(complete_feature, axis=1)

    return data_protein_feature



if __name__ == "__main__":
    #print(sys.path)

    #To test ProFet and define chaves values
    seq = 'MKKFFDSRREQGGSGLGSGSSGGGGSTSGLGSGYIGRVFGIGRQQVTVDEVLAEGGFAIVFLVRTSNGMKCALKRMFVNNEHDLQVCKREIQIMRDLSGHKNIVGYIDSSINNVSSGDVWEVLILMDFCRGGQVVNLMNQRLQTGFTENEVLQIFCDTCEAVARLHQCKTPIIHRDLKVENILLHDRGHYVLCDFGSATNKFQNPQTEGVNAVEDEIKKYTTLSYRAPEMVNLYSGKIITTKADIWALGCLLYKLCYFTLPFGESQVAICDGNFTIPDNSRYSQDMHCLIRYMLEPDPDKRPDIYQVSYFSFKLLKKECPIPNVQNSPIPAKLPEPVKASEAAAKKTQPKARLTDPIPTTETSIAPRQRPKAGQTQPNPGILPIQPALTPRKRATVQPPPQAAGSSNQPGLLASVPQPKPQAPPSQPLPQTQAKQPQAPPTPQQTPSTQAQGLPAQAQATPQHQQQLFLKQQQQQQQPPPAQQQPAGTFYQQQQAQTQQFQAVHPATQKPAIAQFPVVSQGGSQQQLMQNFYQQQQQQQQQQQQQQLATALHQQQLMTQQAALQQKPTMAAGQQPQPQPAAAPQPAPAQEPAIQAPVRQQPKVQTTPPPAVQGQKVGSLTPPSSPKTQRAGHRRILSDVTHSAVFGVPASKSTQLLQAAAAEASLNKSKSATTTPSGSPRTSQQNVYNPSEGSTWNPFDDDNFSKLTAEELLNKDFAKLGEGKHPEKLGGSAESLIPGFQSTQGDAFATTSFSAGTAEKRKGGQTVDSGLPLLSVSDPFIPLQVPDAPEKLIEGLKSPDTSLLLPDLLPMTDPFGSTSDAVIEKADVAVESLIPGLEPPVPQRLPSQTESVTSNRTDSLTGEDSLLDCSLLSNPTTDLLEEFAPTAISAPVHKAAEDSNLISGFDVPEGSDKVAEDEFDPIPVLITKNPQGGHSRNSSGSSESSLPNLARSLLLVDQLIDL'
    dic = Get_Protein_Feat(seq, SeqReducedAlph='ofer14', ReducedK=2, GetSimpleFeatSet=True, GetExtraScaleFeatSet=False,
                           aaParamScaleWindow=7, ExtraScaleWindow=17, GetSubSeqSegs=False, SubSeqSegs=3,
                           GetTriLetterGroupAlphKFreq=False, TriLetterGroupAlphK=5, GetSeqReducedGroups=False,
                           SeqReducedGroups='ofer_w8', GetSeqReducedAlph=False, GetCTDFeatSet=False,
                           GetPTMFeatSet=False, GetDBCleavageFeatSet=False, split_N=False, split_C=False,
                           N_TAIL_REGION=30, C_TAIL_REGION=30)

    print(dic)
    #print(dic.keys())