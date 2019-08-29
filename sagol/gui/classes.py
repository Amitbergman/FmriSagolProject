from sagol.gui.globals import STATE
from sagol.run_models import AVAILABLE_MODELS
from sagol.run_models import generate_flattened_models


class UntrainedModel:
    def __init__(self, name):
        self.name = name
        if name in STATE['trained_models'].models:
            self.parameters = {param: [AVAILABLE_MODELS[name][param], v] for
                               param, v in STATE['trained_models'].parameters[name].items()}
        else:
            self.parameters = {param: [t, ''] for param, t in AVAILABLE_MODELS[name].items()}


class UntrainedModels:
    def __init__(self):
        self.models = {name: UntrainedModel(name) for name in AVAILABLE_MODELS}

    def generate_models(self, model_names, is_train_only):
        data = STATE['experiment_data_after_split']
        trained_models = STATE['trained_models']
        ylabels = trained_models.ylabels
        roi_paths = trained_models.roi_paths
        shape = trained_models.shape
        model_params = {self.models[name]: {param: self.models[name].parameters[param] for
                                            param in self.models[name].parameters} for name in model_names}
        trained_models = generate_flattened_models(model_names=model_names, experiment_data_after_split=data, ylabels=ylabels,
                                  roi_paths=roi_paths, shape=shape, model_params=model_params, is_train_only=is_train_only)
        STATE['trained_models'].set_models(trained_models)
