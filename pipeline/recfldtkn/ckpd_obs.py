Ckpd_ObservationS = {
    'InObs': {
        'CkpdName': 'InCase',
        'DistStartToPredDT': -0.01, # 0.01 min away from ObsDT (PredDT)
        'DistEndToPredDT': 0.01,
        'TimeUnit': 'min',
        'StartIdx5Min': 0,
        'EndIdx5Min': 0, 
    },
   'Bf24H': {
        'CkpdName': 'Bf24H',
        'DistStartToPredDT': -24,
        'DistEndToPredDT': 0.01,
        'TimeUnit': 'H',
        'StartIdx5Min': -288,
        'EndIdx5Min': 0
    },
    'Bf1M': {
        'CkpdName': 'Bf1M',
        'DistStartToPredDT': -24 * 30,
        'DistEndToPredDT': 0.01,
        'TimeUnit': 'H',
        'StartIdx5Min': -288 * 30,
        'EndIdx5Min': 0
    },
    'Af2H': {
        'CkpdName': 'Af2H',
        'DistStartToPredDT': 1,
        'DistEndToPredDT': 121,
        'TimeUnit': 'min',
        'StartIdx5Min': 1,
        'EndIdx5Min': 24, 
    }, 
    'Af1W': {
        'CkpdName': 'Af1W',
        'DistStartToPredDT': 0.0001,
        'DistEndToPredDT': 7,
        'TimeUnit': 'D',
        'StartIdx5Min': 1,
        'EndIdx5Min': 7 * 24 * 12, 
    }
}