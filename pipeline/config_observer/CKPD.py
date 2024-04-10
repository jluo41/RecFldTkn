ckpd_to_CkpdObsConfig = {
    'ObsPnt': {
        'DistStartToPredDT': -0.01, # 0.01 min away from ObsDT (PredDT)
        'DistEndToPredDT': 0.01,
        'TimeUnit': 'min',
        'StartIdx5Min': 0,
        'EndIdx5Min': 0, 
    },
   'Bf24H': {
        'DistStartToPredDT': -24,
        'DistEndToPredDT': 0.01,
        'TimeUnit': 'h',
        'StartIdx5Min': -288,
        'EndIdx5Min': 0
    },
    'Bf1M': {
        'DistStartToPredDT': -24 * 30,
        'DistEndToPredDT': 0.01,
        'TimeUnit': 'h',
        'StartIdx5Min': -288 * 30,
        'EndIdx5Min': 0
    },
    'Bf1Y': {
        'DistStartToPredDT': -365 * 1,
        'DistEndToPredDT': - 0.000001,
        'TimeUnit': 'D',
        'StartIdx5Min': -288 * 365,
        'EndIdx5Min': -1
    },
    'Bf1YIn': {
        'DistStartToPredDT': -365 * 1,
        'DistEndToPredDT': 0.000001,
        'TimeUnit': 'D',
        'StartIdx5Min': -288 * 365,
        'EndIdx5Min': 1
    },
    'Bf10Y': {
        'DistStartToPredDT': - 365 * 10,
        'DistEndToPredDT': - 0.000001, # excluding observation DT.
        'TimeUnit': 'D',
        'StartIdx5Min': -288 * 365 * 10,
        'EndIdx5Min': 1
    },
    'Af2H': {
        'DistStartToPredDT': 1,
        'DistEndToPredDT': 121,
        'TimeUnit': 'min',
        'StartIdx5Min': 1,
        'EndIdx5Min': 24, 
    }, 
    'Af1W': {
        'DistStartToPredDT': 0.0001,
        'DistEndToPredDT': 7,
        'TimeUnit': 'D',
        'StartIdx5Min': 1,
        'EndIdx5Min': 7 * 24 * 12, 
    }, 
    'Af1Wlft': {
        'DistStartToPredDT': -0.0001,
        'DistEndToPredDT': 7,
        'TimeUnit': 'D',
        'StartIdx5Min': 1,
        'EndIdx5Min': 7 * 24 * 12, 
    }, 
    'Af1Y': {
        'DistStartToPredDT': 0.0001,
        'DistEndToPredDT': 365,
        'TimeUnit': 'D',
        'StartIdx5Min': 1,
        'EndIdx5Min': 365 * 24 * 12, 
    }, 
    'PntAf1Y2M': {
        'DistStartToPredDT':365 - 30 * 2,
        'DistEndToPredDT': 365 + 30 * 2,
        'TimeUnit': 'D',
        'StartIdx5Min': (365 - 30 * 2) * 24 * 12,
        'EndIdx5Min': (365 + 30 * 2) * 24 * 12, 
    },
    'PntAf1Y3M': {
        'DistStartToPredDT':365 - 30 * 3,
        'DistEndToPredDT': 365 + 30 * 3,
        'TimeUnit': 'D',
        'StartIdx5Min': (365 - 30 * 3) * 24 * 12,
        'EndIdx5Min': (365 + 30 * 3) * 24 * 12, 
    }
}