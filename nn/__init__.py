
def load_model_instance_from_nn(model_type):
    if model_type == 'XGBClassifierV1':
        from .xgboost.instance_xgboost import XGBClassifierInstance
        return XGBClassifierInstance
    elif model_type == 'BanditV1':
        from .bandit.instance_bandit import BanditInstance
        return BanditInstance
    else:
        raise ValueError(f"Model type {model_type} not found")
