cf_to_QueryCaseFeatConfig = {
    'EgmBf1Y':  {
        'case_observations': [
            'Bf1yInvRN:ro.PInv-Bf1Y_ct.RecNum', # CO
            'Bf1yRxRN:ro.Rx-Bf1Y_ct.RecNum',
            'Bf1yEgmClickRN:ro.EgmClick-Bf1Y_ct.RecNum',
        ],
        
        
        'name_CaseGamma': 'CatUnseqTknsOneTS', # CF
        
        
        'tkn_name_list': [
            'Bf1yInvRN:recnum',
            'Bf1yRxRN:recnum',
            'Bf1yEgmClickRN:recnum',
            'Bf1yInvRN:recspan',
            'Bf1yRxRN:recspan',
            'Bf1yEgmClickRN:recspan',
        ],
    }, 

    'RxEgmAf1W': {
        'case_observations': [
            'Af1wEdu:ro.EgmEdu-Af1Wlft_ct.RxEgmInfo',  
            'Af1wRmd:ro.EgmRmd-Af1Wlft_ct.RxEgmInfo',
            'Af1wCpy:ro.EgmCopay-Af1Wlft_ct.RxEgmInfo',
        ],
        'name_CaseGamma': 'CatUnseqTknsOneTS',

        'tkn_name_list': [
            'Af1wEdu:RxEgmNum',
            'Af1wRmd:RxEgmNum',
            'Af1wCpy:RxEgmNum',
        ]
    }, 


    'InvEgmAf1W': {
        'case_observations': [

            'Af1wClick:ro.EgmClick-Af1Wlft_ct.InvEgmInfo',  
            'Af1wAuthen:ro.EgmAuthen-Af1Wlft_ct.InvEgmInfo',
            'Af1wCallPhm:ro.EgmCallPharm-Af1Wlft_ct.InvEgmInfo',

            'Af1wEdu:ro.EgmEdu-Af1Wlft_ct.InvEgmInfo',  
            'Af1wRmd:ro.EgmRmd-Af1Wlft_ct.InvEgmInfo',
            'Af1wCpy:ro.EgmCopay-Af1Wlft_ct.InvEgmInfo',

        ],
        'name_CaseGamma': 'CatUnseqTknsOneTS',
        'tkn_name_list': [
            'Af1wClick:InvEgmNum',
            'Af1wAuthen:InvEgmNum',
            'Af1wCallPhm:InvEgmNum',
            'Af1wEdu:InvEgmNum',
            'Af1wRmd:InvEgmNum',
            'Af1wCpy:InvEgmNum',
        ]
    }



}