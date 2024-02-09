

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


def convert_RecObsName_and_CaseTkn_to_CaseObsName(RecObsNames, CaseTkn):
    RecObsNames = sorted([i for i in RecObsNames])
    CaseObsName = 'ro.' + '&'.join(RecObsNames) + '_' + 'ct.' + CaseTkn
    return CaseObsName

def parse_CaseObsName(CaseObsName):
    d  = {i.split('.')[0]: i.split('.')[-1] for i in CaseObsName.split('_')}
    RecObsNames = d['ro'].split('&')
    CaseTkn = d['ct']
    return RecObsNames, CaseTkn

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
