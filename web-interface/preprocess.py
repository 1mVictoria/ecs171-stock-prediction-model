import numpy as np

def make_feature_vector(form_data):
    """
    Build a 16-element array matching the RF’s feature_names_in_ order.
    """
    try:
        close       = float(form_data["close"])
        volume      = float(form_data["volume"])
        ret_1d      = float(form_data["ret_1d"])
        ret_2d      = float(form_data["ret_2d"])
        momentum_5d = float(form_data["momentum_5d"])
        ma_5d       = float(form_data["ma_5d"])
        vol_5d      = float(form_data["vol_5d"])
        accel       = float(form_data["accel"])
        rsi_14      = float(form_data["rsi_14"])

        eps         = float(form_data["Earnings Per Share"])
        total_rev   = float(form_data["Total Revenue"])
        net_inc     = float(form_data["Net Income"])

        total_esg   = float(form_data["Total ESG Risk score"])
        env_risk    = float(form_data["Environment Risk Score"])
        social_risk = float(form_data["Social Risk Score"])
        gov_risk    = float(form_data["Governance Risk Score"])

    except KeyError as ke:
        raise KeyError(f"Missing field: {ke}")
    except ValueError as ve:
        raise ValueError(f"Invalid number: {ve}")

    # Return in exactly the same order the model expects:
    return np.array([
        close, volume,
        ret_1d, ret_2d, momentum_5d,
        ma_5d, vol_5d, accel, rsi_14,
        eps, total_rev, net_inc,
        total_esg, env_risk, social_risk, gov_risk
    ])
