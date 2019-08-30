AVAILABLE_MODELS = {'svr': {'kernel': str, 'C': float, 'gamma': float},
                    'bagging_regressor': {'n_estimators': int},
                    'nusvr': {'kernel': str, 'C': float, 'gamma': float}}


def get_model_params(model_name, model, params_to_get=None):
    params_to_get = params_to_get or AVAILABLE_MODELS[model_name]
    all_params = model.get_params(deep=False)
    return {param: all_params[param] for param in params_to_get}
