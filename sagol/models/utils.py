import torch.nn as nn

AVAILABLE_MODELS = {'svr': {'kernel': str, 'C': float, 'gamma': float},
                    'nusvr': {'kernel': str, 'C': float, 'gamma': float},
                    'bagging_regressor': {'n_estimators': int},
                    'lasso': {'alpha': float}, 
                    'cnn': {'kernel_size': int, 'filters': (list, int), 'max_pool_size': int, 'max_pool_every': int,
                            'batch_norm_every': int, 'hidden_dimensions': (list, int), 'out_dimension': int,
                            'hidden_dimensions_of_regressor': (list, int), 'learning_rate': float, 'beta_1': float,
                            'beta_2': float, 'eps': float, 'weight_decay': float, 'amsgrad': bool, 'loss_function': str,
                            'batch_size': int, 'epochs': int, 'early_stopping': int}}

AVAILABLE_OPTIONS = {'svr': {'kernel': {'': '', 'rbf': 'rbf', 'linear': 'linear', 'poly': 'poly'}},
                     'nusvr': {'kernel': {'': '', 'rbf': 'rbf', 'linear': 'linear', 'poly': 'poly'}},
                     'cnn': {'kernel_size': {3: 3, 5: 5, 7: 7, 9: 9},
                             'loss_function': {'mse': nn.MSELoss(), 'l1': nn.L1Loss(), 'smooth l1': nn.SmoothL1Loss()}}}

NOT_VALID_EMPTY_PARAMETERS = {'cnn': set(['filters', 'learning_rate', 'epochs'])}

DEFAULT_PARAMETERS = {'cnn': {'kernel_size': 3, 'max_pool_size': 2, 'max_pool_every': 1, 'batch_norm_every': 1,
                              'hidden_dimensions': [], 'out_dimension': 10, 'hidden_dimensions_of_regressor': [],
                              'beta_1': 0.9, 'beta_2': 0.999, 'eps': 1e-8, 'weight_decay': 0,
                              'amsgrad': False, 'loss_function': 'mse', 'batch_size': 32, 'early_stopping': -1}}

LIST_FORMAT_NOT_EMPTY_REMARK = 'format is: "x y z...", cannot be empty'
LIST_FORMAT_DEFAULT_IS_EMPTY_REMARK = 'format is: "x y z...", default is empty - no hidden dimensions'

REMARKS = {'cnn': {'filters': LIST_FORMAT_NOT_EMPTY_REMARK,
                   'hidden_dimensions': LIST_FORMAT_DEFAULT_IS_EMPTY_REMARK,
                   'hidden_dimensions_of_regressor': LIST_FORMAT_DEFAULT_IS_EMPTY_REMARK}}

AVAILABLE_3D_MODELS = ['cnn']


def get_model_params(model_name, model, params_to_get=None):
    params_to_get = params_to_get or AVAILABLE_MODELS[model_name]
    all_params = model.get_params(deep=False)
    return {param: all_params[param] for param in params_to_get}


def is_valid_param(model_name, param, value, t=None, force_not_empty=False):
    t = t or AVAILABLE_MODELS[model_name][param]
    additional_check = True
    if isinstance(t, tuple):
        t, t_inner = t[0], t[1]
    if value == '':
        if force_not_empty:
            return False, value
        if model_name in NOT_VALID_EMPTY_PARAMETERS and param in NOT_VALID_EMPTY_PARAMETERS[model_name]:
            return False, value
        if model_name in DEFAULT_PARAMETERS and param in DEFAULT_PARAMETERS[model_name]:
            value = DEFAULT_PARAMETERS[model_name][param]
        is_valid = True
        additional_check = False
    elif t == bool:
        if value == 'True' or value == 'true':
            value = True
            is_valid = True
        elif value == 'False' or value == 'false':
            value = False
            is_valid = True
        else:
            is_valid = False
    elif t == str:
        try:
            float(value)
            is_valid = False
        except ValueError:
            is_valid = True
    elif t == list:
        l = value.split(' ')
        inner_values = []
        for elem in l:
            is_inner_valid, inner_value = is_valid_param(model_name, param, elem, t=t_inner, force_not_empty=True)
            if not is_inner_valid:
                return False, value
            inner_values.append(inner_value)
        value = inner_values
        return True, value
    else:
        try:
            converted_val = t(value)
            value = converted_val
            is_valid = True
        except ValueError:
            is_valid = False

    if is_valid and model_name in AVAILABLE_OPTIONS and param in AVAILABLE_OPTIONS[model_name]:
        if value not in AVAILABLE_OPTIONS[model_name][param]:
            is_valid = False
        else:
            value = AVAILABLE_OPTIONS[model_name][param][value]

    if is_valid and additional_check and model_name in ADDITIONAL_CHECKS and param in ADDITIONAL_CHECKS[model_name]:
        for func in ADDITIONAL_CHECKS[model_name][param]:
            if not func(value):
                is_valid = False
                break
    return is_valid, value


def get_parameter_remark(model_name, param):
    if model_name in REMARKS and param in REMARKS[model_name]:
        return REMARKS[model_name][param]
    elif model_name in NOT_VALID_EMPTY_PARAMETERS and param in NOT_VALID_EMPTY_PARAMETERS[model_name]:
        return 'cannot be empty'
    elif model_name in DEFAULT_PARAMETERS and param in DEFAULT_PARAMETERS[model_name]:
        return 'default is: ' + str(DEFAULT_PARAMETERS[model_name][param])
    else:
        return 'empty for grid search'


def is_positive(value):
    return value > 0


def is_non_negative(value):
    return value >= 0


def between_zero_and_one(value):
    return 1 >= value >= 0


ADDITIONAL_CHECKS = {'svr': {'C': [is_positive], 'gamma': [is_positive]},
                     'nusvr': {'C': [is_positive], 'gamma': [is_positive]},
                     'bagging_regressor': {'n_estimators': [is_positive]},
                     'cnn': {'kernel_size': [is_positive], 'filters': [is_positive], 'max_pool_size': [is_positive],
                             'max_pool_every': [is_positive], 'batch_norm_every': [is_positive],
                             'hidden_dimensions': [is_positive], 'out_dimension': [is_positive],
                             'hidden_dimensions_of_regressor': [is_positive], 'learning_rate': [between_zero_and_one],
                             'beta_1': [between_zero_and_one], 'beta_2': [between_zero_and_one], 'eps': [between_zero_and_one],
                             'weight_decay': [between_zero_and_one], 'batch_size': [is_positive], 'epochs': [is_positive]}}
