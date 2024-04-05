import numpy as np

# 各种衡量指标
def RMSE_func(gt,pred):
    return np.sqrt(np.mean((gt-pred)**2))

def MAE_func(gt,pred):
    return abs(gt-pred).mean()


def Rel_func2(gt,pred):
    """rel in paper SimCol3D using median

    Args:
        gt (_type_): _description_
        pred (_type_): _description_

    Returns:
        _type_: _description_
    """
    mask = gt>0
    return np.median(abs(gt-pred)/(gt+1e-5))

