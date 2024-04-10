cf_to_CaseFeatConfig = {

    'RxLevel.DemoInvRx.ObsPnt': { 
        'case_observations':  [
            # PDemo
            'PDemo:ro.P-Demo_ct.InCaseInfo',
            'PZip3Demo:ro.P-Zip3DemoNume_ct.InCaseInfo',
            'PZip3Econ:ro.P-Zip3EconNume_ct.InCaseInfo',
            'PZip3House:ro.P-Zip3HousingNume_ct.InCaseInfo',
            'PZip3Social:ro.P-Zip3SocialNume_ct.InCaseInfo',

            # InCase of Inv
            'InvInCase1:ro.PInv-ObsPnt-Info_ct.InCaseInfo',

            # InCase of Rx
            'RxInCase1:ro.Rx-ObsPnt-CmpCate_ct.InCaseInfo',
            'RxInCase2:ro.Rx-ObsPnt-InsCate_ct.InCaseInfo',
            'RxInCase3:ro.Rx-ObsPnt-ServiceCate_ct.InCaseInfo',
            'RxInCase4:ro.Rx-ObsPnt-SysCate_ct.InCaseInfo',
            'RxInCase5:ro.Rx-ObsPnt-QuantNume_ct.InCaseInfo',
            'RxInCase6:ro.Rx-ObsPnt-TherEqCate_ct.InCaseInfo',
            'RxInCase7:ro.Rx-ObsPnt-DrugBasicCate_ct.InCaseInfo',
            'RxInCase8:ro.Rx-ObsPnt-PhmBasicCate_ct.InCaseInfo',

            # InObs
            'RxObsPnt1:ro.Rx-ObsPnt-CmpCate_ct.AggMeanFeat',
            'RxObsPnt2:ro.Rx-ObsPnt-InsCate_ct.AggMeanFeat',
            'RxObsPnt3:ro.Rx-ObsPnt-ServiceCate_ct.AggMeanFeat',
            'RxObsPnt4:ro.Rx-ObsPnt-SysCate_ct.AggMeanFeat',
            'RxObsPnt5:ro.Rx-ObsPnt-QuantNume_ct.AggMeanFeat',
            'RxObsPnt6:ro.Rx-ObsPnt-TherEqCate_ct.AggMeanFeat',
            'RxObsPnt7:ro.Rx-ObsPnt-DrugBasicCate_ct.AggMeanFeat',
            'RxObsPnt8:ro.Rx-ObsPnt-PhmBasicCate_ct.AggMeanFeat',
            'RxObsPntNum:ro.Rx-ObsPnt_ct.RecNum',

        ],
        'name_CaseGamma': 'CatUnseqTknsOneTS',
    },


    'InvLevel.DemoInvRx.ObsPnt': { 
        'case_observations':  [
            # PDemo
            'PDemo:ro.P-Demo_ct.InCaseInfo',
            'PZip3Demo:ro.P-Zip3DemoNume_ct.InCaseInfo',
            'PZip3Econ:ro.P-Zip3EconNume_ct.InCaseInfo',
            'PZip3House:ro.P-Zip3HousingNume_ct.InCaseInfo',
            'PZip3Social:ro.P-Zip3SocialNume_ct.InCaseInfo',

            # InCase of Inv
            'InvInCase1:ro.PInv-ObsPnt-Info_ct.InCaseInfo',

            # InObs
            'RxObsPnt1:ro.Rx-ObsPnt-CmpCate_ct.AggMeanFeat',
            'RxObsPnt2:ro.Rx-ObsPnt-InsCate_ct.AggMeanFeat',
            'RxObsPnt3:ro.Rx-ObsPnt-ServiceCate_ct.AggMeanFeat',
            'RxObsPnt4:ro.Rx-ObsPnt-SysCate_ct.AggMeanFeat',
            'RxObsPnt5:ro.Rx-ObsPnt-QuantNume_ct.AggMeanFeat',
            'RxObsPnt6:ro.Rx-ObsPnt-TherEqCate_ct.AggMeanFeat',
            'RxObsPnt7:ro.Rx-ObsPnt-DrugBasicCate_ct.AggMeanFeat',
            'RxObsPnt8:ro.Rx-ObsPnt-PhmBasicCate_ct.AggMeanFeat',
            'RxObsPntNum:ro.Rx-ObsPnt_ct.RecNum',

        ],
        'name_CaseGamma': 'CatUnseqTknsOneTS',
    },


    'RxLevel.Egm.Af1W': {
        'case_observations': [
            'Af1wEdu:ro.EgmEdu-Af1Wlft_ct.RxEgmInfo',  
            'Af1wRmd:ro.EgmRmd-Af1Wlft_ct.RxEgmInfo',
            'Af1wCpy:ro.EgmCopay-Af1Wlft_ct.RxEgmInfo',
        ],
        'name_CaseGamma': 'CatUnseqTknsOneTS',
    }, 


    'InvLevel.Egm.Af1W': {
        'case_observations': [

            'Af1wClick:ro.EgmClick-Af1Wlft_ct.InvEgmInfo',  
            'Af1wAuthen:ro.EgmAuthen-Af1Wlft_ct.InvEgmInfo',
            'Af1wCallPhm:ro.EgmCallPharm-Af1Wlft_ct.InvEgmInfo',

            'Af1wEdu:ro.EgmEdu-Af1Wlft_ct.InvEgmInfo',  
            'Af1wRmd:ro.EgmRmd-Af1Wlft_ct.InvEgmInfo',
            'Af1wCpy:ro.EgmCopay-Af1Wlft_ct.InvEgmInfo',

        ],
        'name_CaseGamma': 'CatUnseqTknsOneTS',
    }
}
