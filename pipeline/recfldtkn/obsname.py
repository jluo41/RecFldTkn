import hashlib
import base64

def get_consistent_short_hash(input_data, length=10):
    # Ensure the input is in string form
    input_str = str(input_data)
    # Hash the input using SHA-256
    hash_object = hashlib.sha256(input_str.encode('utf-8'))
    # Get the hash digest as bytes
    hash_bytes = hash_object.digest()
    # Encode the hash in base64 to shorten it
    short_hash_base64 = base64.urlsafe_b64encode(hash_bytes).decode('utf-8')
    # Truncate or slice the base64 hash to the desired length
    short_hash = short_hash_base64[:length]
    return short_hash

def parse_RecObsName(RecObsName):
    # A RecObsName can be:
    #   1. RecName
    #   2. RecName-Ckpd
    #   3. RecName-FieldName
    #   4. RecName-Ckpd-FldName
    element_list = RecObsName.split('-')
    d = {}
    if len(element_list) == 1:
        d['RecName'] = element_list[0]
        d['CkpdName'] = None
        d['FldName'] = None 

    elif len(element_list) == 2:
        second_element = element_list[1]
        if 'Af' in second_element or 'Bf' in second_element or 'In' in second_element:
            d['RecName'] = element_list[0]
            d['CkpdName'] = element_list[1]
            d['FldName'] = None
        else:
            d['RecName'] = element_list[0]
            d['CkpdName'] = None
            d['FldName'] = element_list[1]
    
    elif len(element_list) == 3:
        d['RecName'] = element_list[0]
        d['CkpdName'] = element_list[1]
        d['FldName'] = element_list[2]
    return d


def parse_CaseObsName(CaseObsName):
    d  = {i.split('.')[0]: i.split('.')[-1] for i in CaseObsName.split('_')}
    RecObsNames = d['ro'].split('&')
    CaseTkn = d['ct']
    return RecObsNames, CaseTkn

def parse_CaseFeatName(CaseFeatName):
    d  = {i.split('.')[0]: i.split('.')[-1] for i in CaseFeatName.split('_')}
    COList_hash = d['ro'].split('&')
    name_CaseGamma = d['cf']
    return COList_hash, name_CaseGamma

def convert_RONameList_to_COName(RecObsNames, CaseTkn):
    RecObsNames = sorted([i for i in RecObsNames])
    CaseObsName = 'ro.' + '&'.join(RecObsNames) + '_' + 'ct.' + CaseTkn
    return CaseObsName

def convert_CONameList_to_CFName(case_observations, CaseFeatTkn):
    CO_list_hash = get_consistent_short_hash(tuple(sorted(case_observations)))
    CaseFeatName = 'cf.' + CaseFeatTkn + '_' + 'co.' + CO_list_hash
    return CaseFeatName

def get_tokenizer_name_for_CaseObsName(CaseObsName):
    RecObsNames, CaseTkn = parse_CaseObsName(CaseObsName)
    parsed_list = [parse_RecObsName(RecObsName) for RecObsName in RecObsNames]
    field_list = list(set([d['FldName'] for d in parsed_list if d['FldName'] is not None]))
    record_list = list(set([d['RecName'] for d in parsed_list]))

    if len(field_list) == 0:
        tokenizer_name = CaseTkn
    else:
        assert len(field_list) == 1
        assert len(record_list) == 1
        tokenizer_name = record_list[0] + '-' + field_list[0]
    return tokenizer_name


def convert_case_observations_to_co_to_observation(case_observations):
    co_to_CaseObsName = {i.split(':')[0]: i.split(':')[1] for i in case_observations}
    co_to_CaseObsNameInfo = {}
    for co, CaseObsName in co_to_CaseObsName.items():
        Record_Observations_List = CaseObsName.split('_')[0].replace('ro.', '').split('&')
        CaseTkn = CaseObsName.split('_')[1].replace('ct.', '')
        co_to_CaseObsNameInfo[co] = {'RecObsName_List': Record_Observations_List, 'name_CasePhi': CaseTkn}
    return co_to_CaseObsName, co_to_CaseObsNameInfo


def get_RecNameList_and_FldTknList(co_to_CaseObsNameInfo):
    RecNameList_All = []
    CkpdNameList_All = []
    FldTknList_All = []
    CasePhiList = []

    for co, CaseObsNameInfo in co_to_CaseObsNameInfo.items():
        RecObsName_List = CaseObsNameInfo['RecObsName_List']
        name_CasePhi = CaseObsNameInfo['name_CasePhi']
        RecNameList = [parse_RecObsName(i)['RecName'] for i in RecObsName_List]
        CkpdNameList = [parse_RecObsName(i)['CkpdName'] for i in RecObsName_List]
        FldNameList = [parse_RecObsName(i)['RecName'] + '-' + parse_RecObsName(i)['FldName'] 
                       for i in RecObsName_List if parse_RecObsName(i)['FldName'] is not None]
        
        RecNameList_All = RecNameList_All + [i for i in RecNameList if i is not None]
        CkpdNameList_All = CkpdNameList_All + [i for i in CkpdNameList if i is not None]
        FldTknList_All = FldTknList_All + [i for i in FldNameList if i is not None]
        CasePhiList.append(name_CasePhi)
        
    PipelineInfo = {
        'RecNameList': list(set(RecNameList_All)), 
        'CkpdNameList': list(set(CkpdNameList_All)), 
        'FldTknList': list(set(FldTknList_All)), 
        'CasePhiList': list(set(CasePhiList))
    }
    for k, v in PipelineInfo.items():
        PipelineInfo[k] = sorted(v) 
    return PipelineInfo