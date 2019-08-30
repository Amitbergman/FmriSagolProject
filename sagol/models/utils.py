AVAILABLE_MODELS = {'svr': {'kernel': str, 'C': float, 'gamma': float},
                    'bagging_regressor': {'n_estimators': int},
                    'nusvr': {'kernel': str, 'C': float, 'gamma': float}}

AVAILABLE_3D_MODELS = set([])

AVAILABLE_OPTIONS = {'svr': {'kernel': set(['rbf', 'linear', 'poly'])},
                     'nusvr': {'kernel': set(['rbf', 'linear', 'poly'])}}


def get_model_params(model_name, model, params_to_get=None):
    params_to_get = params_to_get or AVAILABLE_MODELS[model_name]
    all_params = model.get_params(deep=False)
    return {param: all_params[param] for param in params_to_get}


def is_valid_param(model_name, param, value):
    t = AVAILABLE_MODELS[model_name][param]
    if value == '':
        return True, value
    elif t == str:
        try:
            float(value)
            is_valid = False
        except ValueError:
            is_valid = True
    else:
        try:
            converted_val = t(value)
            value = converted_val
            is_valid = True
        except ValueError:
            is_valid = False
    if model_name in AVAILABLE_OPTIONS and param in AVAILABLE_OPTIONS[model_name] and \
            value not in AVAILABLE_OPTIONS[model_name][param]:
        is_valid = False
    return is_valid, value
